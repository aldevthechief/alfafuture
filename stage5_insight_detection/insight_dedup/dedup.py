"""Duplicate gate between the judges and the writer (24.10): positive judge decisions and the bank-analytics insights of
one run go to ONE LLM call that groups the candidates a reader would see as repeats (same event or bank forecast AND
same consequence for the company). The writer then runs once per group. Evaluation on labelled pairs of real runs:
experiments/insight_dedup/README.md."""
import sys
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.llm import PromptLibrary, structured

PROMPTS = PromptLibrary(Path(__file__).with_name('prompts'))


class Group(BaseModel):
    ids: list[str] = Field(description='id кандидатов-дублей, два и более')
    common: str = Field('', description='что у них общего, одной фразой')


class Groups(BaseModel):
    groups: list[Group] = []


def render(item: dict) -> str:
    """A candidate as the gate sees it: news — the news titles and the judge's company link and mechanism;
    analytics — the insight title and its thesis."""
    if item['kind'] == 'analytics':
        return f"[{item['id']}] аналитика Банка: {item['title']}\n  вывод: {item['thesis'][:600]}"
    news = '; '.join(t for t in item['news_titles'] if t)
    return (f"[{item['id']}] новости: {news}\n  связь с компанией: {item['company_link'][:400]}"
            f"\n  механизм: {item['mechanism'][:600]}")


class DedupGate:
    def __init__(self, model):
        self.chain = structured(model, Groups)

    async def agroups(self, items: list[dict], role: str) -> list[list[str]]:
        """Groups of duplicate ids (each id at most once, unknown ids dropped); fewer than two candidates: no call."""
        if len(items) < 2:
            return []
        return self._clean(await self.chain.ainvoke(self._prompt(items, role)), items)

    def groups(self, items: list[dict], role: str) -> list[list[str]]:
        """agroups() for a caller outside the news graph's event loop (the merge node): the async client of the model
        keeps connections of the loop that opened them, a second asyncio.run fails with «Event loop is closed»."""
        if len(items) < 2:
            return []
        return self._clean(self.chain.invoke(self._prompt(items, role)), items)

    @staticmethod
    def _prompt(items: list[dict], role: str) -> str:
        return PROMPTS.render('dedup', role=role, candidates='\n'.join(render(i) for i in items))

    @staticmethod
    def _clean(answer: Groups, items: list[dict]) -> list[list[str]]:
        known, seen, out = {i['id'] for i in items}, set(), []
        for group in answer.groups:
            ids = [i for i in dict.fromkeys(group.ids) if i in known and i not in seen]
            if len(ids) > 1:
                out.append(ids)
                seen |= set(ids)
        return out
