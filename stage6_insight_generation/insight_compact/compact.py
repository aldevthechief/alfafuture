"""Screen version of a finished insight: short facts, «why it matters» as thesis + direct effect, «important if»
conditions, one-line bank theses. A second, cheap writer over the stage-6 Insight only (no new evidence); the full
insight stays the source of truth. Sources are not trusted from the model: each line names the insight items it comes
from (ids) and code takes their sources."""
import json, re, sys
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from langchain_core.messages import AIMessage, HumanMessage

from common.llm import PromptLibrary, StructuredOutputError, structured
from common.llm.log import log_tool

PROMPTS = PromptLibrary(Path(__file__).with_name('prompts'))
IDS = re.compile(r'\s*\((?:[a-z]\d+[,;\s]*)+\)')  # «(h2)», «(e1, w2)»: the model sometimes cites ids in the text
WORD = re.compile(r'\w{4,}')
NUMBER = re.compile(r'\d+(?:[.,]\d+)?')
# the bank's confidential view of the client (24.10 review): ranks and peer comparisons always; revenue, payments and
# counterparties only next to a figure (a year is not one) — «рост выручки», «проверить контрагентов» are fine
RANK = re.compile(r'ОКВЭД|\d+-?[еймо]{1,2} мест|среди \d+|крупнейш\w* (?:по выручке|частн|в сегмент|своего|в отрасл)', re.I)
_KEY = r'(?:выручк|(?:исходящ|входящ)\w* плат[её]ж|контрагент|покупател|поставщик|дебитор)\w*'
_FIG = r'(?<!\d)(?!(?:19|20)\d\d(?!\d))\d+(?:[.,]\d+)?(?!\d|[.,]\d|-[а-я])'  # a whole number; not a year, not «22-го»
COMPANY_DATA = re.compile(rf'{_KEY}\W+(?:\w+\W+){{0,4}}?{_FIG}|{_FIG}\W+(?:\w+\W+){{0,4}}?{_KEY}', re.I)  # within 4 words
BLOCKS = {'what_happened': 'Что произошло', 'why_it_matters': 'Почему важно', 'risks': 'Риски',
          'opportunities': 'Возможности', 'bank_view': 'Позиция Банка', 'hypotheses': 'Гипотезы'}


class Line(BaseModel):
    ids: list[str] = Field(default=[], description='id пунктов инсайта, из которых взята строка')
    text: str

    @model_validator(mode='before')
    @classmethod
    def plain(cls, v):  # GLM at low reasoning returns bank_view as plain strings: ids are restored in with_sources
        return {'text': v} if isinstance(v, str) else v


class Labeled(Line):
    label: str


class CompactInsight(BaseModel):
    title: str
    lead: str
    facts: list[Line] = Field(min_length=1)
    why: list[Labeled] = Field(min_length=1, max_length=3)
    important_if: list[str] = Field(min_length=2, max_length=3)
    risks: list[Labeled] = []
    opportunities: list[Labeled] = []
    actions: list[str] = []
    bank_view: list[Line] = []


def render(insight: dict) -> str:
    out = [f'Заголовок: {insight.get("title", "")}']
    for key, name in BLOCKS.items():
        items = insight.get(key) or []
        if items:
            out += [f'## {name}', *[f'- {c.get("id", "")}: {c["text"]}' for c in items]]
    actions = insight.get('action_details') or [{'id': f'a{n}', 'text': t} for n, t in enumerate(insight.get('actions') or [], 1)]
    if actions:
        out += ['## Действия', *[f'- {a["id"]}: {a["text"]}' for a in actions]]
    if insight.get('ai_conclusion'):
        out += ['## Итоговый вывод', insight['ai_conclusion']]
    return '\n'.join(out)


def with_sources(compact: CompactInsight, insight: dict) -> dict:
    """Model output + `sources` per line, taken from the cited insight items; unknown ids are dropped."""
    sources = {c['id']: c.get('sources') or [] for key in BLOCKS for c in insight.get(key) or [] if c.get('id')}
    out = json.loads(IDS.sub('', compact.model_dump_json()))
    origin = {'facts': 'what_happened', 'why': 'why_it_matters', 'risks': 'risks', 'opportunities': 'opportunities',
              'bank_view': 'bank_view'}
    for key, block in origin.items():
        claims = [c for c in insight.get(block) or [] if c.get('id')]
        for line in out[key]:
            line['ids'] = [i for i in line['ids'] if i in sources]
            if not line['ids'] and claims:  # no ids given: the block item sharing most words
                words = set(WORD.findall(line['text'].lower()))
                line['ids'] = [max(claims, key=lambda c: len(words & set(WORD.findall(c['text'].lower()))))['id']]
            line['sources'] = list(dict.fromkeys(s for i in line['ids'] for s in sources[i]))
    return out


def problems(compact: dict, insight: dict) -> list[str]:
    """What code can see for sure: confidential company data, numbers absent from the insight, uncovered facts."""
    source = render(insight).replace(' ', ' ')
    known = {n.replace(',', '.') for n in NUMBER.findall(source)}
    lines = [compact['title'], compact['lead'], *compact['important_if'], *compact['actions'],
             *[f'{x.get("label", "")} {x["text"]}' for k in ('facts', 'why', 'risks', 'opportunities', 'bank_view') for x in compact[k]]]
    sentences = [x for t in lines for x in re.split(r'(?<=[.;!?])\s+', t)]
    out = [f'конфиденциальные данные о компании или сравнение с другими: «{t}»' for t in sentences
           if RANK.search(t) or COMPANY_DATA.search(t)]
    out += [f'числа {n} нет в инсайте: «{t}»' for t in lines for n in NUMBER.findall(t) if n.replace(',', '.') not in known]
    facts = {c['id'] for c in insight.get('what_happened') or [] if c.get('id')}
    if missing := facts - {i for x in compact['facts'] for i in x['ids']}:
        out.append(f'нет строки facts для пунктов {", ".join(sorted(missing))}')
    return out


class InsightCompactor:
    """compact -> code check -> if issues, ONE fix in the same dialog; still failing -> None (the UI shows the full insight)."""
    def __init__(self, model):
        self.chain = structured(model, CompactInsight)

    async def acompact(self, insight: dict) -> dict | None:
        dialog = [HumanMessage(PROMPTS.render('compact', role=insight.get('role') or '', insight=render(insight)))]
        for attempt in range(2):
            try:
                draft = await self.chain.ainvoke(dialog)
            except StructuredOutputError as e:
                log_tool('compact_no_schema', {'title': insight.get('title')}, str(e))
                return None
            out = with_sources(draft, insight)
            if not (issues := problems(out, insight)):
                return out
            log_tool('compact_issues', {'title': insight.get('title'), 'attempt': attempt}, '\n'.join(issues))
            dialog += [AIMessage(draft.model_dump_json()), HumanMessage(
                'Исправь и верни ответ целиком:\n' + '\n'.join(f'- {i}' for i in issues))]
        return None


if __name__ == '__main__':  # smoke: python -X utf8 -m stage6_insight_generation.insight_compact.compact <job.json> [n]
    import asyncio
    from common.llm import ChatModels
    job = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    rows = [r for r in job['result']['results'] if r.get('status') == 'ready' and r.get('insight')]
    row = rows[int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 0]
    model = ChatModels().agent(0.0).model_copy(update={'reasoning_effort': 'low'}) if '--agent' in sys.argv else ChatModels().cheap()
    print(json.dumps(asyncio.run(InsightCompactor(model).acompact(row['insight'])), ensure_ascii=False, indent=1))
