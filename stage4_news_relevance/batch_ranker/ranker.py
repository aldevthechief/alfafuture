"""Ranker 1, approach «LLM + packs»: the cheap LLM sees a short brief of the company (one LLM summary of its stage-2 facts)
and a pack of candidate titles + leads, and answers keep / drop for each. Recall first: drop only what clearly cannot
matter. Packs mix generators (shuffled) so that generator weights can be compared later; pack size is a parameter to
compare against one-news-per-call (backlog.md items 3–4). Not run yet (needs LLM_API_KEY)."""
import hashlib
import random
import sys
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.llm import ChatModels, JsonlCache, PromptLibrary, structured

PROMPTS = PromptLibrary(Path(__file__).with_name('prompts'))


class Brief(BaseModel):
    text: str = Field(description='5–8 строк: чем занимается компания, масштаб, ключевые контрагенты и их отрасли, '
                                  'продукты банка, регионы, ВЭД, группа; только факты')


class Verdict(BaseModel):
    i: int = Field(description='номер новости в пакете')
    keep: bool = Field(description='true, если новость может повлиять на бизнес компании')
    channel: str = Field('', description='одно-два слова: через что влияет (ставка, цены на сырьё, спрос, регулирование, …)')


class PackVerdicts(BaseModel):
    verdicts: list[Verdict]


def lead(news: dict, n: int = 300) -> str:
    """RSS summary when present; else the text without the repeated title (Kommersant starts the text with it)."""
    s, title, text = (v if isinstance(v, str) else '' for v in (news.get('summary'), news.get('title'), news.get('text')))  # parquet rows carry NaN
    s = s.strip()
    if len(s) >= 40 and s != title:
        return s[:n]
    return (text[len(title):] if text.startswith(title) else text).strip()[:n]


class BatchRanker:
    def __init__(self, model: BaseChatModel | None = None, cache: JsonlCache | None = None, pack: int = 25, seed: int = 0):
        model = model or ChatModels().cheap(max_tokens=2000)
        self.brief_chain = PROMPTS.chat('brief') | structured(model, Brief)
        self.rank_chain = PROMPTS.chat('rank_pack') | structured(model, PackVerdicts)
        self.cache = cache or JsonlCache('company_brief')
        self.pack, self.seed = pack, seed

    def brief(self, facts: list[dict]) -> str:
        text = '\n'.join(f"[{f['cat']}] {f['text']}" for f in facts)
        def make():  # the model sometimes skips the tool call (None): retry, then fall back to the first-level facts
            for _ in range(2):
                r = self.brief_chain.invoke({'facts': text})
                if r is not None:
                    return r.text
            return '\n'.join(f['text'] for f in facts if f.get('depth', 1) == 1)[:1500]
        return self.cache.get_or_compute(hashlib.sha1(text.encode()).hexdigest(), make)

    def rank(self, brief: str, items: list[dict], role: str | None = None) -> dict:
        """items: {'id', 'title', 'lead'}. Returns {id: {'keep', 'channel'}}; an item the model skipped is kept (recall first)."""
        items = items[:]
        random.Random(self.seed).shuffle(items)
        out = {}
        for s in range(0, len(items), self.pack):
            pack = items[s:s + self.pack]
            lines = '\n'.join(f"{k}. {it['title']} — {it['lead']}" for k, it in enumerate(pack, 1))
            role_line = f'Читатель: {role}.' if role else ''  # shortcut: role in ranker 1 is to be tested (backlog.md item 2)
            got = {v.i: v for v in self.rank_chain.invoke({'brief': brief, 'role': role_line, 'news': lines}).verdicts}
            for k, it in enumerate(pack, 1):
                v = got.get(k)
                out[it['id']] = {'keep': v.keep if v else True, 'channel': v.channel if v else ''}
        return out
