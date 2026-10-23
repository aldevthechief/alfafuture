"""The agent's candidate as an `Insight` of contract v2 (stage6 insight_v2.py), by code only: no model call, nothing is rewritten.

- every bank claim is a `Claim(kind='bank')` with its quotes as `Evidence`; `source` is the id of the fragment the quote stands in
  (`bank:<file>:p<page>.<n>`) and `start`/`end` are offsets inside that fragment (the page, `bank:<file>:p<page>`, for a table row or
  a reading of a picture); the quote is the text of the page at that place when the code found it;
- the company facts named are `Claim(kind='fact')` with the text of the fact, `link_inference` is an `ai` claim, the mechanism goes to
  `risks` / `opportunities` by `effect` (`mixed`: to why_it_matters);
- the hypotheses keep their matching: `supports` of the agent (`bank_claim:N`, `link`, `mechanism`, `action:N`, `impact`) becomes ids of
  the claims, actions and the impact, and the reverse `depends_on` is filled (insight_v2.assign_ids).
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stage2_company_facts.profile.context import source_facts
from stage5_insight_detection.analytics_insights.agent import AnalyticsCandidate, Finish
from stage6_insight_generation.insight_writer import insight_v2 as v2


def locate_fragment(rows: list[dict], report: str, page: int, start: int | None, end: int | None) -> tuple[str, int | None, int | None]:
    """(fragment id, start, end inside the fragment) of a quote found at `start:end` of the page text (fragments of kinds text / table
    joined by a blank line, as quotes.page_text does); the page id and no offsets when the quote has no place."""
    frs = [f for f in rows if f['report'] == report and f['page'] == page and f['kind'] in ('text', 'table')]
    page_id = None
    cursor = 0
    for f in frs:
        page_id = page_id or f['id'].rsplit('.', 1)[0]
        n = len(f['text'])
        if start is not None and cursor <= start < cursor + n:
            return f['id'], start - cursor, min(end if end is not None else n, cursor + n) - cursor
        cursor += n + 2
    if page_id is None:
        stem = next((f['id'].split(':')[1] for f in rows if f['report'] == report), '')
        page_id = f'bank:{stem}:p{page}'
    return page_id, None, None


def candidate_to_insight(c: AnalyticsCandidate, company_id: str, role: str, rows: list[dict], facts: list[dict]) -> v2.Insight:
    facts = source_facts(facts)  # same fact:N numbering as AnalyticsTools, never cite profile hypotheses
    bank_view, claim_sources, statuses, reports = [], [], Counter(), {}
    for claim in c.bank_claims:
        evidence = []
        for e in claim.evidence:
            source, start, end = locate_fragment(rows, e.report, e.page, e.start, e.end)
            evidence.append(v2.Evidence(source=source, quote=e.source_excerpt if e.source_excerpt and e.kind != 'table' else e.quote, start=start, end=end,
                                        title=e.report, date=e.date, page=e.page, kind=e.kind, status=e.status, cited_page=e.cited_page))
            statuses[e.status or 'unchecked'] += 1
            reports[(e.report, e.date)] = {'title': e.report, 'date': e.date}
        sources = list(dict.fromkeys(x.source for x in evidence))
        claim_sources.append(sources)
        bank_view.append(v2.Claim(kind='bank', text=claim.text, sources=sources, evidence=evidence))
    bank_sources = list(dict.fromkeys(s for ss in claim_sources for s in ss))
    fact_claims = [v2.Claim(kind='fact', text=facts[i - 1]['text'], sources=[f'fact:{i}']) for i in c.company_facts if 1 <= i <= len(facts)]
    fact_sources = [f'fact:{i}' for i in c.company_facts]
    why = [*fact_claims, v2.Claim(kind='ai', text=c.link_inference, sources=fact_sources + bank_sources)]
    mechanism = v2.Claim(kind='ai', text=c.mechanism, sources=fact_sources + bank_sources)
    risks, opportunities = [], []
    if c.effect == 'risk':
        risks.append(mechanism)
        where = f'r1'
    elif c.effect == 'opportunity':
        opportunities.append(mechanism)
        where = 'o1'
    else:
        why.append(mechanism)
        where = f'w{len(why)}'

    def target(token: str) -> str:
        token = token.strip()
        if token.startswith('bank_claim:'):
            return f'b{token.split(":")[1]}'
        if token.startswith('action:'):
            return f'a{token.split(":")[1]}'
        return {'link': f'w{len(fact_claims) + 1}', 'mechanism': where, 'impact': 'imp'}.get(token, token)

    hypotheses = [v2.Claim(kind='hypothesis', text=a.claim, sources=[], check=a.check, supports=[target(t) for t in a.supports]) for a in c.assumptions]
    impact = None if c.impact is None else v2.Impact(
        level=c.impact.level, criteria=c.impact.criteria, estimate=c.impact.estimate, basis=c.impact.basis,
        sources=[x for t in c.impact.basis for x in ([t] if t.startswith('fact:') else claim_sources[int(t.split(':')[1]) - 1]
                                                       if t.startswith('bank_claim:') and t.split(':')[1].isdigit() and 1 <= int(t.split(':')[1]) <= len(claim_sources) else [])])
    accepted = {w.id: w.reason for w in c.accepted_warnings}
    ins = v2.Insight(
        title=c.title, role=role, company_id=company_id, origin='bank', kind=c.kind, channel=c.channel, event_date=c.event_date,
        role_fit=c.role_fit, confidence=c.confidence, bank_product=c.product, ai_conclusion=c.conclusion,
        what_happened=[], bank_view=bank_view, why_it_matters=why, risks=risks, opportunities=opportunities, hypotheses=hypotheses,
        actions=[f'{a.text} — потому что {a.because}' if a.because else a.text for a in c.actions],
        action_details=[v2.Action(text=a.text, because=a.because) for a in c.actions], impact=impact,
        trigger=v2.Trigger(reports=list(reports.values())),
        review=v2.Review(warnings=c.warnings, accepted=[{'id': i, 'reason': r} for i, r in accepted.items()], verification=dict(statuses)))
    return v2.assign_ids(ins)


def run_to_analysis(insights: list[v2.Insight], company_id: str, role: str, as_of: str, finish: Finish | None, model: str = '',
                    cost_rub: float | None = None, trace_session: str = '') -> v2.AnalysisRun:
    return v2.AnalysisRun(company_id=company_id, role=role, as_of=as_of, engine='bank_analytics', model=model, insights=insights,
                          covered=finish.covered if finish else [], rejected=finish.rejected if finish else [],
                          open_requests=finish.open_questions if finish else [], cost_rub=cost_rub, trace_session=trace_session)
