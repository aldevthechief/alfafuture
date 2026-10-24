"""Frozen evidence of one insight: the news, company facts and bank fragments it was written from.

A discussion of the insight (feedback_api) answers only from this snapshot and from explicit read-only lookups, so the
answer can cite the same ids the card cites: news:<id>, fact:<N> (the pipeline's numbering of the company facts),
bank ids as the result gives them (bank:<file>:p<page>.<n> or bank:<report>:<page>).
`from_result()` rebuilds the snapshot from a saved pipeline result without any LLM call.
"""
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Source(BaseModel):
    model_config = ConfigDict(extra='forbid')


class NewsEvidence(Source):
    id: int = Field(ge=0)
    date: str = Field(max_length=100)
    source: str = Field(max_length=300)
    title: str = Field(max_length=2000)
    text: str = Field(max_length=200000)
    url: str | None = Field(default=None, max_length=4000)


class FactEvidence(Source):
    id: str
    text: str = Field(max_length=8000)


class BankEvidence(Source):
    id: str = Field(max_length=500)
    report: str = Field(max_length=1000)
    page: int | None = None
    date: str = Field('', max_length=100)
    stale: bool = False
    text: str = Field(max_length=100000)


class EvidenceSnapshot(Source):
    version: Literal[2] = 2
    captured_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    company_id: str = Field(min_length=1, max_length=100)
    role: str = Field(min_length=1, max_length=100)
    news: list[NewsEvidence] = Field(default_factory=list, max_length=12)
    facts: list[FactEvidence] = Field(default_factory=list, max_length=512)
    calculations: list[FactEvidence] = Field(default_factory=list, max_length=128)
    bank: list[BankEvidence] = Field(default_factory=list, max_length=48)

    @model_validator(mode='after')
    def validate_ids(self):
        if [f.id for f in self.facts] != [f'fact:{i}' for i in range(1, len(self.facts) + 1)]:
            raise ValueError('Facts must keep the pipeline numbering: fact:1, fact:2, ...')
        if len({n.id for n in self.news}) != len(self.news):
            raise ValueError('Duplicate news IDs in snapshot')
        if len(self.model_dump_json()) > 2_000_000:
            raise ValueError('Evidence snapshot exceeds 2 MB')
        return self

    def sources(self) -> dict[str, str]:
        out = {f'news:{n.id}': f'{n.date}, {n.source}: {n.title}\n{n.text[:1500]}' for n in self.news}
        out |= {f.id: f.text for f in self.facts}
        out |= {f.id: f.text for f in self.calculations}
        for b in self.bank:
            out[b.id] = f'{b.report}, {b.date}{", УСТАРЕЛ" if b.stale else ""}{f", с. {b.page}" if b.page else ""}\n{b.text[:1200]}'
        return out

    def render(self) -> str:
        s = self.sources()
        parts = ['## Новости', *(f'{k} {s[k]}' for k in s if k.startswith('news:')),
                 '## Факты о компании', *(f'{f.id} {f.text}' for f in self.facts),
                 *(['## Расчёты кодом (сценарии помечены как ГИПОТЕЗА)',
                    *(f.text for f in self.calculations)] if self.calculations else []),
                 '## Аналитика Банка', *([f'{b.id} {s[b.id]}' for b in self.bank] or ['(нет)'])]
        return '\n\n'.join(parts)

    @classmethod
    def from_result(cls, result: dict, company_id: str, role: str, *, news_lookup=None, facts=None) -> 'EvidenceSnapshot':
        """news_lookup(id) -> {id, date, source, title, text, url}. Facts: `result['evidence_facts']` (the exact lines the
        run's writer numbered fact:1..N) when the pipeline saved them; otherwise `facts` (records rebuilt now — the
        numbering can differ from the run if the tables changed since)."""
        if news_lookup is None:
            from common.search import news_pool
            pool = news_pool().set_index('id', drop=False)
            news_lookup = lambda i: pool.loc[i].to_dict() if i in pool.index else None
        lines = result.get('evidence_facts')
        if lines is None:
            if facts is None:
                from stage2_company_facts.deterministic_facts.facts import build
                facts = build([company_id], use_profiles=True)[company_id]
            lines = [f'[{f["cat"]}] {f["text"]}' for f in facts]
        insight = result.get('insight') or result.get('draft') or {}
        ids = [result.get('news_id'), *insight.get('trigger', {}).get('news_ids', []),
               *(result.get('decision') or {}).get('related_news_ids', [])[:3]]
        for block in ('what_happened', 'why_it_matters', 'risks', 'opportunities', 'bank_view'):
            for claim in insight.get(block, []):
                ids += [int(s.split(':', 1)[1]) for s in claim.get('sources', []) if s.startswith('news:') and s[5:].isdigit()]
        news = []
        for i in dict.fromkeys(int(x) for x in ids if x is not None):
            n = news_lookup(i)
            if n:
                news.append(NewsEvidence(id=i, date=str(n.get('date', ''))[:100], source=str(n.get('source', ''))[:300],
                                         title=str(n.get('title', ''))[:2000], text=str(n.get('text', ''))[:200000],
                                         url=n.get('url')[:4000] if isinstance(n.get('url'), str) else None))
        bank = {}
        for b in [*result.get('judge_bank', []), *result.get('calculation_bank', []), *result.get('bank', [])]:
            bid = b.get('id') or f'bank:{b.get("report")}:{b.get("page")}'
            bank.setdefault(bid, BankEvidence(id=bid[:500], report=str(b.get('report', ''))[:1000],
                                              page=b.get('page') if isinstance(b.get('page'), int) else None,
                                              date=str(b.get('date', ''))[:100], stale=bool(b.get('stale', False)),
                                              text=str(b.get('text', ''))[:100000]))
        from stage5_insight_detection.analyst_loop.facts import Fact
        calculations = [FactEvidence(id=f['id'], text=Fact(**f).render() + f' [источник: {f.get("source", "")}]')
                        for f in result.get('calculations', [])]
        return cls(company_id=company_id, role=role, news=news[:12], bank=list(bank.values())[:48], calculations=calculations,
                   facts=[FactEvidence(id=f'fact:{i}', text=str(t)[:8000]) for i, t in enumerate(lines, 1)][:512])
