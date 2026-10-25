"""Approach B: the cheap LLM reads every news ONCE and returns its economic consequences (who is affected, channel,
direction, status). Cached per news id and reused for all companies and roles; matching to a company is deterministic.
Not run yet (needs LLM_API_KEY)."""
import copy, hashlib, json, sys
from pathlib import Path
from typing import Literal

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.llm import ChatModels, JsonlCache, PromptLibrary, structured
from common.news_corpus import article_fingerprint
from common.news_chunks import split_into_chunks

PROMPTS = PromptLibrary(Path(__file__).with_name('prompts'))


class Affected(BaseModel):
    who: str = Field(description='отрасль или тип бизнеса, как в ОКВЭД')
    okved_hint: str = Field('', description='код ОКВЭД 2-4 знака или пусто')
    region: str = ''
    country: str = ''
    direction: Literal['+', '-', '?'] = '?'
    channel: str = Field(description='спрос|издержки|цены|регулирование|налоги|ставка|курс|логистика|конкуренция|финансирование')
    mechanism: str = Field(description='как именно влияет')
    horizon: str = Field('месяцы', description='дни|месяцы|год+')


class Entities(BaseModel):
    companies: list[str] = []
    regions: list[str] = []
    countries: list[str] = []
    commodities: list[str] = []
    regulators: list[str] = []


class ExtractedFact(BaseModel):
    text: str = Field(description='факт, явно утверждаемый в статье; сохраняй статус плана/слуха/прогноза')
    quote: str = Field(description='дословный непрерывный фрагмент статьи, подтверждающий факт')
    participants: list[str] = Field(default_factory=list)
    industries: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)
    commodities: list[str] = Field(default_factory=list)
    change: str = Field('', description='что изменилось; сохраняй статус плана, слуха или прогноза')


class NewsConsequences(BaseModel):
    facts: list[ExtractedFact] = Field(default_factory=list)
    economic: bool = Field(description='есть ли последствия для бизнеса')
    event: str = Field(description='одно предложение: что произошло')
    status: Literal['fact', 'plan', 'proposal', 'rumor', 'forecast']
    significance: int = Field(ge=0, le=3)
    affected: list[Affected] = []
    entities: Entities = Entities()


def extraction_version() -> str:
    signature = 'consequences-v4-search-facts' + PROMPTS.text('consequences') + json.dumps(NewsConsequences.model_json_schema(), sort_keys=True)
    return hashlib.sha256(signature.encode()).hexdigest()


class ConsequenceExtractor:
    def __init__(self, model: BaseChatModel | None = None, cache: JsonlCache | None = None):
        self.chain = PROMPTS.chat('consequences') | structured(model or ChatModels().cheap(), NewsConsequences)
        self.cache = cache or JsonlCache('consequences')

    def extract(self, news: dict) -> dict:
        """news: a row of news.parquet as dict. Returns NewsConsequences as a plain dict (that is what the cache stores)."""
        key = self._key(news)
        return self._bind(self.cache.get_or_compute(key, lambda: self._extract(news)), news)

    async def aextract(self, news: dict) -> dict:
        """extract() with the model calls awaited; the cache is checked first, a computed value is stored the same way."""
        key = self._key(news)
        value = self.cache.get(key)
        if value is None:
            value = self._merge(news, [(start, text, (await self.chain.ainvoke({'title': news['title'], 'text': text})).model_dump())
                                       for start, end, text in self._chunks(news)])
            self.cache.put(key, value)
        return self._bind(value, news)

    @staticmethod
    def _key(news: dict) -> str:
        return hashlib.sha256((extraction_version() + article_fingerprint(news)).encode()).hexdigest()

    @staticmethod
    def _bind(value: dict, news: dict) -> dict:
        result = copy.deepcopy(value)
        # Content cache survives ID migrations; source identities always come from this input row.
        for fact in result['facts']:
            fact['evidence'].update(source=f"news:{news['id']}", url=news.get('url') or '')
        return result

    @staticmethod
    def _chunks(news: dict):
        return split_into_chunks(news.get('text') or '', chunk_size=3000, overlap=250) or [(0, 0, '')]

    def _extract(self, news: dict) -> dict:
        return self._merge(news, [(start, text, self.chain.invoke({'title': news['title'], 'text': text}).model_dump())
                                  for start, end, text in self._chunks(news)])

    @staticmethod
    def _merge(news: dict, answers: list[tuple]) -> dict:
        outputs = []
        for start, text, value in answers:
            facts, rejected = [], []
            for fact in value['facts']:
                quote = fact['quote']
                offset = text.find(quote) if quote.strip() else -1
                if offset < 0:
                    rejected.append(dict(fact, reason='quote_not_in_source_chunk'))
                    continue
                facts.append(dict(fact, evidence=dict(source=f"news:{news['id']}", quote=quote,
                             start=start + offset, end=start + offset + len(quote), url=news.get('url') or '')))
            value.update(facts=facts, rejected_facts=rejected)
            outputs.append(value)
        result = dict(max(outputs, key=lambda x: (x['economic'], x['significance'])))
        for key in ('facts', 'affected', 'rejected_facts'):
            unique = {json.dumps(item, ensure_ascii=False, sort_keys=True): item for out in outputs for item in out[key]}
            result[key] = list(unique.values())
        result['entities'] = {k: sorted({v for out in outputs for v in out['entities'][k]}) for k in result['entities']}
        return result


def match_consequences(conseq: dict | None, company: dict) -> tuple[float, str]:
    """Deterministic join of extracted consequences with company features: no LLM per (news, company) pair.
    company: {'okveds': {code: weight}, 'regions': set, 'countries': set, 'channels': set} built from facts.
    Returns (score, why)."""
    if not conseq or not conseq.get('economic'):
        return 0.0, ''
    best, why = 0.0, ''
    for a in conseq.get('affected', []):
        hint = (a.get('okved_hint') or '').strip()
        w = max((v for code, v in company['okveds'].items() if hint and code.startswith(hint[:2]) and
                 (len(hint) <= 2 or code.startswith(hint))), default=0)
        w = max(w, 0.5 if a.get('region') in company['regions'] else 0, 0.5 if a.get('country') in company['countries'] else 0,
                0.4 if a.get('channel') in company['channels'] else 0)
        s = w * (1 + conseq.get('significance', 1)) / 4
        if s > best:
            best, why = s, f'{a.get("who")}: {a.get("mechanism")} ({a.get("channel")}, {a.get("direction")})'
    return best, why
