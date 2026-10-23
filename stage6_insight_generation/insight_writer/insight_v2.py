"""Insight contract v2: a superset of `insight.Insight` (v1) for insights that do not need a news item.

- v1 JSON loads into v2 as it is (every new field is optional or has a default), and `to_v1` projects v2 back; nothing of a news
  insight is lost either way (`from_v1` adds ids and the trigger, preserving origin). Bank-thesis results use the same migration.
- The unit is a `Claim` with its `Evidence` (a quote with the document, page and offsets inside the fragment `source`): the same
  for a news fact and for a statement from the bank's report. A news is one kind of `trigger`; an analytics insight has reports.
- Hypotheses keep their matching: every claim has an `id`; a hypothesis lists in `supports` the ids of the claims, actions or the impact
  that rest on it and says what confirms it (`check`); the claims, actions and the impact carry the reverse link in `depends_on`.
- `review` keeps what the code checked (warnings, the verification statuses of the quotes) and what the agent answered (`accepted`).
- `AnalysisRun` is the container of one (company, role) run, shared by the news judge and the analytics agent.
"""
import hashlib
from typing import Literal

from pydantic import BaseModel, Field

from stage6_insight_generation.insight_writer import insight as v1


class Evidence(BaseModel):
    """A quote: v1 `SourceExcerpt` (source, quote, start, end, url) + where it stands and how it was verified."""
    source: str = Field(description='news:<id> | bank:<файл>:p<стр>.<n> (фрагмент, в котором стоит цитата)')
    quote: str
    start: int | None = Field(None, description='смещение начала в тексте источника (у отчёта — внутри фрагмента)')
    end: int | None = None
    url: str = ''
    title: str = ''
    date: str = ''
    page: int | None = None
    kind: Literal['text', 'table', 'picture'] = 'text'
    status: str = Field('', description='exact | fuzzy | relocated | table_row | picture | not_found …: итог проверки цитаты кодом')
    cited_page: int | None = Field(None, description='страница, которую назвал автор, если код её исправил')


class Claim(BaseModel):
    text: str
    sources: list[str] = Field(description='news:<id> | bank:<отчёт>:<стр> | fact:<номер>')
    evidence: list[Evidence] = Field(default_factory=list)
    kind: Literal['fact', 'bank', 'ai', 'hypothesis'] = 'ai'
    unconfirmed: bool = False
    id: str = Field('', description='e/b/w/r/o/h + номер; ставит код (assign_ids)')
    depends_on: list[str] = Field(default_factory=list, description='id гипотез, на которых держится утверждение')
    check: str = Field('', description='только у гипотезы: что её подтвердит')
    supports: list[str] = Field(default_factory=list, description='только у гипотезы: id утверждений, действий и оценки, которые на ней держатся')


class Action(BaseModel):
    id: str = ''
    text: str
    because: str = ''
    depends_on: list[str] = Field(default_factory=list)


class Impact(v1.Impact):
    basis: list[str] = Field(default_factory=list, description='на чём оценка: fact:N, id утверждений')
    depends_on: list[str] = Field(default_factory=list)


class Trigger(BaseModel):
    news_ids: list[int] = Field(default_factory=list)
    reports: list[dict] = Field(default_factory=list, description='[{title, date}]: отчёты Банка, на которых стоит инсайт')


class Review(BaseModel):
    warnings: list[str] = Field(default_factory=list)
    accepted: list[dict] = Field(default_factory=list, description='[{id, reason}]: замечания, которые автор проверил и не учёл, с причиной')
    verification: dict[str, int] = Field(default_factory=dict, description='сколько цитат по статусам: exact, fuzzy, relocated…')


class Insight(BaseModel):
    """Same fields as v1 (same names and meaning) + the additions below."""
    title: str
    role: str
    what_happened: list[Claim] = Field(default_factory=list, description='ФАКТЫ из новостей; у аналитического инсайта пусто: событие в title и trigger')
    why_it_matters: list[Claim] = Field(default_factory=list, description='факты о компании (kind=fact) и вывод AI')
    risks: list[Claim] = []
    opportunities: list[Claim] = []
    actions: list[str] = Field(default_factory=list, description='как в v1; у аналитического инсайта «текст — потому что …» из action_details')
    bank_view: list[Claim] = []
    ai_conclusion: str = ''
    confidence: Literal['high', 'medium', 'low'] = 'medium'
    bank_product: str = ''
    impact: Impact | None = None
    hypotheses: list[Claim] = []
    not_bank_position: bool = False
    disclaimer: str = 'Не является инвестиционной рекомендацией.'
    # additions
    id: str = ''
    company_id: str = ''
    origin: Literal['news', 'bank'] = Field('news', description='news; bank — событие это позиция аналитики Банка (то же значение, что у Insight.origin в ветке vladimir-all)')
    kind: Literal['event', 'forecast', 'view_change', 'recommendation', 'calendar', 'product'] = 'event'
    channel: str = ''
    event_date: str = ''
    role_fit: int | None = None
    trigger: Trigger = Field(default_factory=Trigger)
    action_details: list[Action] = Field(default_factory=list)
    review: Review = Field(default_factory=Review)


class AnalysisRun(BaseModel):
    """One (company, role) run of an engine: the insights and what the run learned besides them."""
    company_id: str
    role: str
    as_of: str
    engine: Literal['news_judge', 'analyst_loop', 'bank_analytics']
    model: str = ''
    insights: list[Insight] = Field(default_factory=list)
    covered: list[str] = Field(default_factory=list)
    rejected: list[str] = Field(default_factory=list)
    open_requests: list[str] = Field(default_factory=list)
    cost_rub: float | None = None
    trace_session: str = ''
    key: str = Field('', description='хеш входов прогона (роль, факты, отчёты): по нему кэш считается свежим или устаревшим')
    error: str = ''
    cached: bool = False
    stale_cache: bool = False


BLOCKS = (('what_happened', 'e'), ('bank_view', 'b'), ('why_it_matters', 'w'), ('risks', 'r'), ('opportunities', 'o'), ('hypotheses', 'h'))


def assign_ids(ins: Insight) -> Insight:
    """Ids by position (e1, b1, w1, r1, o1, h1, a1, imp) where there is none; depends_on is rebuilt from the `supports` of the hypotheses."""
    for block, prefix in BLOCKS:
        for n, claim in enumerate(getattr(ins, block), 1):
            claim.id = claim.id or f'{prefix}{n}'
    for n, action in enumerate(ins.action_details, 1):
        action.id = action.id or f'a{n}'
    if not ins.id:
        ins.id = hashlib.sha1('|'.join([ins.company_id, ins.role, ins.kind, ins.title]).encode()).hexdigest()[:12]
    holders = {c.id: c for block, _ in BLOCKS for c in getattr(ins, block)} | {a.id: a for a in ins.action_details}
    if ins.impact:
        holders['imp'] = ins.impact
    for h in ins.hypotheses:
        for target in h.supports:
            if target in holders and h.id not in holders[target].depends_on:
                holders[target].depends_on.append(h.id)
    return ins


def from_v1(old: v1.Insight, company_id: str = '', news_ids: list[int] = (), reports: list[dict] = ()) -> Insight:
    """An insight of either origin as v2, retaining all v1 content and source references."""
    new = Insight.model_validate(old.model_dump() | {'company_id': company_id, 'kind': 'event',
                                                     'trigger': {'news_ids': list(news_ids), 'reports': list(reports)}})
    new.action_details = [Action(text=t) for t in new.actions]
    return assign_ids(new)


def to_v1(new: Insight) -> v1.Insight:
    """The v1 projection for the consumers of the old contract (v2-only fields are not carried)."""
    return v1.Insight.model_validate(new.model_dump(include=set(v1.Insight.model_fields)))


def news_ids_of(result: dict) -> list[int]:
    """Numeric ids of the news an insight cites (news:<id> in its claims) plus the news of the result itself."""
    ids = {int(x) for x in _walk_news(result.get('insight') or result.get('draft') or {})}
    if isinstance(result.get('news_id'), (int, str)) and str(result['news_id']).isdigit():
        ids.add(int(result['news_id']))
    return sorted(ids)


def _walk_news(value):
    if isinstance(value, str) and value.startswith('news:') and value[5:].isdigit():
        yield value[5:]
    elif isinstance(value, list):
        for x in value:
            yield from _walk_news(x)
    elif isinstance(value, dict):
        for x in value.values():
            yield from _walk_news(x)


def migrate_result(result: dict, company_id: str) -> dict:
    """A pipeline result of the news engines with its `insight` / `draft` in contract v2. v2 holds every v1 field, so the readers of
    the old result (cards, feed) see what they saw; an insight that does not parse as v1 is left as it is."""
    out = dict(result)
    for key in ('insight', 'draft'):
        raw = result.get(key)
        if isinstance(raw, dict) and 'trigger' not in raw:
            try:
                origin = result.get('origin') or raw.get('origin') or result.get('event', {}).get('origin', 'news')
                reports = []
                for b in result.get('bank', []):
                    report = {'title': b.get('report'), 'date': b.get('date') or ''}
                    if report['title'] and report not in reports:
                        reports.append(report)
                new = from_v1(v1.Insight.model_validate(raw | {'origin': origin}), company_id, news_ids_of(result), reports)
                # Keep the review with the card too, while leaving the full original review/audit/verdict intact.
                warnings = [*(result.get('review') or {}).get('issues', []),
                            *[w.get('text', '') for w in result.get('audit', [])]]
                new.review.warnings = list(dict.fromkeys(w for w in warnings if w))
                out[key] = new.model_dump()
            except Exception as e:  # shortcut: keep the old shape, say why; upgrade when a result of the loop engine fails here
                out.setdefault('migration_errors', []).append(f'{key}: {type(e).__name__}')
    return out
