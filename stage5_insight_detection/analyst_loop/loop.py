"""The analyst loop: one event × one company × one role, from the bank's analytics to an insight or a refusal.

    event card ─> bank match (2) ─> back to the news pool (3) ─> consequences (5) ─> risks / opportunities (6)
        ^                                                                                  │
        └──── requests for missing data (4) <──── grounds check (7) <── model explains ────┘
                                                        │
                                                        └─> verdict (8): brief -> writer -> audit | refusal

Numbers are computed by code; the model is called twice at most: `assess` explains the computed consequences and may
ask for missing data, the stage-6 writer formulates the insight from the brief. Untrusted text reaches prompts only via
common/safety.render_untrusted. Every step is a line in the JSONL journal (common/trace.py).
"""
import json, sys
from pathlib import Path
from datetime import date, datetime
from typing import Literal

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.llm import ChatModels, PromptLibrary, StructuredOutputError, structured
from common.safety import UNTRUSTED_NOTE, render_untrusted, suspicious
from common.text import stems
from common.chunk_search import article_context
from common.trace import AgentTrace
from stage5_insight_detection.analyst_loop import grounds, impact, risks
from stage5_insight_detection.analyst_loop.compute import compute
from stage5_insight_detection.analyst_loop.audit import audit
from stage5_insight_detection.analyst_loop.dossier import Dossier
from stage5_insight_detection.analyst_loop.enrich import Enricher
from stage5_insight_detection.analyst_loop.event import EventCard
from stage5_insight_detection.analyst_loop.pool import NewsPool
from stage5_insight_detection.analyst_loop.profile import CompanyProfile
from stage5_insight_detection.analyst_loop.data_requests import Answer, DataRequest, RequestBroker
from stage5_insight_detection.analyst_loop.verdict import InsightBrief, decide
from stage5_insight_detection.bank_analytics.library import BankLibrary
from stage5_insight_detection.bank_analytics.match import BankMatch, BankMatcher
from stage5_insight_detection.bank_analytics.review import CANDIDATES, BankReviewer, bank_positions
from stage5_insight_detection.bank_analytics.digest import BudgetExceeded
from stage6_insight_generation.insight_writer.writer import Evidence, InsightWriter

from stage2_company_facts.profile.context import CONTEXT_CATEGORY, compact_context, render_context

PROMPTS = PromptLibrary(Path(__file__).with_name('prompts'))
# These warnings mean a published claim cannot be verified. Cosmetic warnings and corrected layers are non-blocking.
BLOCKING_AUDIT_CODES = {'unknown_ref', 'no_source'}


class Explained(BaseModel):
    consequence_id: str
    text: str
    refs: list[str]
    kind: Literal['ai', 'hypothesis'] = 'ai'


class Assessment(BaseModel):
    """Explanations of the computed consequences and requests for missing data."""
    model_config = ConfigDict(title='assess')
    explanations: list[Explained] = []
    missing: list[DataRequest] = Field([], description='чего не хватает для вывода')


class BriefEvidence(Evidence):
    """Evidence rendered with the loop's own ids (fact:<slug>, bank:<file>:p<page>.<n>), all untrusted text wrapped."""

    def __init__(self, dossier: Dossier, brief: InsightBrief):
        super().__init__(news=list(dossier.news().values()), facts=[],
                         bank=[dict(m.dict(), text=m.fragment.text) for m in dossier.bank],
                         origin=getattr(dossier.event, 'origin', 'news'))
        self.dossier, self.brief = dossier, brief

    def render(self) -> str:
        d = self.dossier
        news = [render_untrusted(ref, r['title'], article_context(r), {'источник': r.get('source'), 'дата': r.get('date')}, body_limit=3500)
                for ref, r in d.news().items()]
        computed = d.facts()
        facts = [computed[f['id']].render() for f in self.brief.facts + self.brief.hypotheses if f['id'] in computed] + \
                [f'fact:stage2_{i} {t}' for i, t in enumerate(d.extra_facts, 1)]

        if d.profile_context:
            facts.append(compact_context(d.profile_context, facts))

        bank = [m.prompt_item() for m in d.bank]
        positions = bank_positions(d.bank)  # the reports' own views in short, for the big picture; cited only via bank:…

        return '\n\n'.join([UNTRUSTED_NOTE, '## Новости', *news, '## Факты о компании (посчитаны кодом)', *facts,
                            '## Аналитика Банка', *(bank or ['(нет)']),
                            *(['## Позиция Банка в этих отчётах (выжимка, для общей картины; ссылаться только на фрагменты bank:…)',
                               '\n'.join(f'- {p}' for p in positions)] if positions else [])])


class AnalystLoop:
    def __init__(self, library: BankLibrary, pool: NewsPool, model: BaseChatModel | None = None,
                 writer: InsightWriter | None = None, trace: AgentTrace | None = None, max_rounds: int = 3,
                 company_facts: list[dict] | None = None, explain: bool = True, reviewer: BankReviewer | None = None):
        self.library, self.pool = library, pool
        self.matcher = BankMatcher(library)
        self.reviewer = reviewer  # the bank analyst agent over the code's wider pick (bank_analytics/review.py), optional
        self.enricher = Enricher(pool, self.matcher)
        self.model = model
        self.assessor = None
        if explain:
            self.assessor = PROMPTS.chat('assess') | structured(model or ChatModels().agent(), Assessment)
        self.writer = writer
        self.trace = trace or AgentTrace()
        self.max_rounds, self.company_facts = max_rounds, company_facts or []

    def run(self, row: dict | None, profile: CompanyProfile, role: str, scores: dict | None = None, *,
            event: EventCard | None = None, pinned: tuple | list = (), deltas: dict | None = None) -> dict:
        try:
            return self._run(row, profile, role, scores, event=event, pinned=pinned, deltas=deltas)
        except StructuredOutputError as error:
            self.trace.log('end', status='model_error', schema=error.schema, attempts=error.attempts)
            return {'news_id': event.news_id if event else str(row['id']),
                    'title': event.title if event else row['title'], 'origin': event.origin if event else 'news',
                    'company': profile.id, 'role': role,
                    'status': 'model_error', 'error': {'stage': 'analyst_loop', 'schema': error.schema,
                                                       'attempts': error.attempts, 'message': str(error)},
                    'verdict': {'status': 'reject', 'reasons': [str(error)]},
                    'trace': str(self.trace.path) if self.trace.path else None, 'run_id': self.trace.run_id}

    def _run(self, row: dict | None, profile: CompanyProfile, role: str, scores: dict | None = None, *,
             event: EventCard | None = None, pinned: tuple | list = (), deltas: dict | None = None) -> dict:
        t = self.trace
        event = event or EventCard.from_news(row, scores, self.pool.plausibility(row['id']))
        t.log('start', event=event.id, company=profile.id, role=role, title=event.title[:200])
        if suspicious(event.title) or suspicious(event.text):
            t.log('injection_suspected', ref=event.id)
        t.log('event', topics=event.topics, moves=event.moves, figures=len(event.figures), importance=event.importance,
              plausibility=event.plausibility, sources=event.sources)
        d = Dossier(event, profile, role, deltas=deltas or {})
        broker = RequestBroker(self._handlers(d), t)
        d.profile_context = render_context(self.company_facts)

        with t.timed('bank_match') as rec:  # task 2
            if self.reviewer is None:
                d.matches = self.matcher.select(event, profile)
            else:
                wide = self.matcher.select(event, profile, k=CANDIDATES)
                try:
                    d.matches, review = self.reviewer.review(event, profile, wide)
                except BudgetExceeded as e:  # the budget is spent: the code's pick, said so in the trace
                    d.matches, review = wide[:5], {'verdict': f'агент не вызван: {e}', 'decisions': [], 'by': 'code'}
                rec['review'] = review
            d.matches = list(pinned) + [m for m in d.matches if m.fragment.id not in {p.fragment.id for p in pinned}]
            rec['matches'] = [{'id': m.fragment.id, 'score': m.score, 'reason': m.reason} for m in d.matches]
            rec['verdict'] = 'аналитики Банка нет' if not d.matches else f'{len(d.matches)} фрагментов'
            rec['disagrees'] = [{'id': m.fragment.id, 'topics': [t for t, v in m.parts.get('direction', {}).items() if v == 'расходится']}
                                for m in d.matches if 'расходится' in m.parts.get('direction', {}).values()]
        with t.timed('enrich') as rec:  # task 3
            d.enrichment = self.enricher.enrich(event, d.matches, profile)
            rec.update(confirmations=[r['id'] for r in d.enrichment.confirmations], updates=len(d.enrichment.updates),
                       extra_bank=[m.fragment.id for m in d.enrichment.extra_bank], queries=d.enrichment.queries)

        report, explained = None, False
        for round_no in range(1, self.max_rounds + 1):
            self._compute(d)
            report = grounds.check(d)  # task 7
            t.log('grounds', round=round_no, **report.dict())
            asks = list(report.requests)
            if report.sufficient and self.assessor is not None and not explained:
                asks += self._explain(d)
                explained = True
            answers = broker.dispatch(asks)  # task 4
            if not any(a.ok for a in answers):
                break
        else:
            # Data received in the final allowed round must reach the verdict too.
            self._compute(d)
            report = grounds.check(d)
            t.log('grounds', round=self.max_rounds + 1, final=True, **report.dict())
        verdict = decide(d, report, broker.open)  # task 8
        t.log('verdict', status=verdict.status, reasons=verdict.reasons, open_requests=[r.model_dump() for r in broker.open])
        result = {'news_id': event.news_id, 'origin': event.origin, 'title': event.title, 'event': event.dict(), 'role': role, 'company': profile.id,
                  'news': [{k: r.get(k) for k in ('id', 'title', 'source', 'date', 'url')} for ref, r in d.news().items() if ref != event.id],
                  'bank': [m.dict() for m in d.bank], 'consequences': [c.dict() for c in d.consequences],
                  'risks': [r.dict() for r in d.risks], 'grounds': report.dict(), 'verdict': verdict.dict(),
                  'open_requests': [r.model_dump() for r in broker.open], 'trace': str(t.path) if t.path else None,
                  'run_id': t.run_id}
        if verdict.status == 'insight' and self.writer is not None:
            with t.timed('write') as rec:
                insight, review = self.writer.write(profile.id, role, verdict.brief.render(), BriefEvidence(d, verdict.brief))
                rec['review_ok'] = review.ok
            if not review.ok:
                result.update(draft=insight.model_dump(), review=review.model_dump(), status='review_rejected')
                result['verdict'] = dict(result['verdict'], status='reject', reasons=[
                    'Инсайт не прошёл проверку: ' + '; '.join(review.issues)])
                t.log('end', status='review_rejected')
                return result
            sources = {ref: f'{r["title"]}. {r.get("text", "")}' for ref, r in d.news().items()}
            sources |= {m.fragment.id: m.fragment.text for m in d.bank}
            insight, warnings = audit(insight, verdict.brief, sources)
            t.log('audit', warnings=warnings, unconfirmed=sum(c.unconfirmed for c in
                                                               insight.what_happened + insight.why_it_matters + insight.risks +
                                                               insight.opportunities + insight.bank_view))
            blocking = [w for w in warnings if w['code'] in BLOCKING_AUDIT_CODES]
            if blocking:
                result.update(draft=insight.model_dump(), review=review.model_dump(), audit=warnings,
                              status='audit_rejected')
                result['verdict'] = dict(result['verdict'], status='reject', reasons=[
                    f'Финальный аудит: {w["code"]} ({w["section"]}): {w["text"]}' for w in blocking])
                t.log('end', status='audit_rejected')
                return result
            result |= {'insight': insight.model_dump(), 'review': review.model_dump(), 'audit': warnings, 'status': 'ready'}
        t.log('end', status=verdict.status)
        return result

    def _compute(self, d: Dossier) -> None:
        before = [(c.id, c.sign, c.score, c.estimate.dict() if c.estimate else None) for c in d.consequences]
        with self.trace.timed('compute') as rec:  # task 5
            d.consequences, d.computed = impact.assess(d.event, d.profile, d.bank,
                                                     d.enrichment.independent_confirmations(d.event), deltas=d.deltas)
            rec['consequences'] = [{'id': c.id, 'topic': c.topic, 'sign': c.sign, 'level': c.level, 'score': c.score,
                                    'estimate': c.estimate.text() if c.estimate else None,
                                    'level_why': c.level_why} for c in d.consequences]
            rec['facts'] = [{'id': f.id, 'value': f.value, 'unit': f.unit, 'formula': f.formula, 'source': f.source, 'kind': f.kind}
                            for f in d.computed]
        d.risks = risks.split(d.consequences, d.bank)  # task 6
        after = [(c.id, c.sign, c.score, c.estimate.dict() if c.estimate else None) for c in d.consequences]
        if before != after:
            d.explanations.clear()
        self.trace.log('risks', items=[{'id': r.id, 'type': r.type, 'basis': r.basis, 'bank_disagrees': r.bank_disagrees,
                                        'roles': r.roles} for r in d.risks])

    def _explain(self, d: Dossier) -> list[DataRequest]:
        facts = d.facts()
        used = sorted({f for c in d.consequences for f in c.facts} | {f.id for f in d.requested})
        inputs = {'role': d.role, 'card': d.profile.card(), 'note': UNTRUSTED_NOTE,
                  'event': render_untrusted(d.event.id, d.event.title, d.event.text[:2000],
                                            {'источник': d.event.source, 'дата': d.event.date}),
                  'bank': '\n'.join(m.prompt_item(page=False) for m in d.bank) or '(нет)',
                  'facts': '\n'.join(facts[f].render() for f in used if f in facts),
                  'consequences': '\n'.join(f'{c.id} [{c.level}] {c.topic}: {c.mechanism}, знак {c.sign:+d}; ссылки '
                                            f'{", ".join(c.facts + c.bank_refs + c.news_refs)}' for c in d.consequences),
                  'risks': '\n'.join(f'{r.id} {r.type} basis={r.basis}: {r.text}' for r in d.risks)}
        if d.profile_context:
            inputs['card'] += '\n\n' + compact_context(d.profile_context, [inputs['card'], inputs['facts']])
        with self.trace.timed('assess') as rec:
            a: Assessment = self.assessor.invoke(inputs)
            refs = d.refs()
            for e in a.explanations:
                ok_refs = [r for r in e.refs if r in refs]
                d.explanations[e.consequence_id] = {'text': e.text, 'refs': ok_refs, 'kind': e.kind,
                                                    'dropped_refs': [r for r in e.refs if r not in refs]}
            rec.update(explained=len(a.explanations), missing=[m.model_dump() for m in a.missing])
        return [m.model_copy(update={'origin': 'model'}) for m in a.missing]

    def _handlers(self, d: Dossier) -> dict:
        def bank(r: DataRequest) -> Answer:
            known = {m.fragment.id for m in d.bank}
            hits = self.matcher.select(d.event, d.profile, k=3, exclude=known, query=r.query)
            d.enrichment.extra_bank += hits
            return Answer(r, bool(hits), ', '.join(m.fragment.id for m in hits) or 'ничего не найдено')

        def news(r: DataRequest) -> Answer:
            known = set(d.news())
            rows = [x for x in self.pool.search(r.query, 5) if f'news:{x["id"]}' not in known][:3]
            d.enrichment.confirmations += rows
            return Answer(r, bool(rows), ', '.join(f'news:{x["id"]}' for x in rows) or 'ничего не найдено')

        def company(r: DataRequest) -> Answer:
            keys = set(stems(r.query))
            rows = [f['text'] for f in self.company_facts
                    if f['cat'] != CONTEXT_CATEGORY and keys & set(stems(f['text']))][:5]
            d.extra_facts += [x for x in rows if x not in d.extra_facts]
            return Answer(r, bool(rows), '; '.join(rows)[:300] or 'нет таких фактов')

        def calculation(r: DataRequest) -> Answer:
            facts, why = compute(r.query, d, len(d.requested) + 1)
            d.requested.extend(facts)
            return Answer(r, bool(facts), '; '.join(f.render() for f in facts) or why,
                          payload=[f.dict() for f in facts])

        return {'bank_analytics': bank, 'news': news, 'company_facts': company, 'computation': calculation}


if __name__ == '__main__':
    import argparse
    from stage5_insight_detection.analyst_loop.pool import RelatedNewsPool
    from stage2_company_facts.deterministic_facts.facts import build
    ap = argparse.ArgumentParser(description='analyst loop for one news item and company (needs LLM_API_KEY)')
    ap.add_argument('company_id')
    ap.add_argument('news_id', type=int)
    ap.add_argument('--role', default='cfo')
    a = ap.parse_args()
    models = ChatModels()
    loop = AnalystLoop(BankLibrary.load(), RelatedNewsPool(), models.agent(), InsightWriter(models.agent()),
                       company_facts=build([a.company_id])[a.company_id])
    pool = loop.pool
    print(json.dumps(loop.run(pool.get(a.news_id), CompanyProfile.from_tables(a.company_id), a.role), ensure_ascii=False,
                     indent=1, default=str))
