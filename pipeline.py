"""End-to-end run for one company and reader role (stage 1 output, data/news/news.parquet, is the input pool).

2 company profile (stage2_company_facts/profile: data/companies/profiles.jsonl or built on the fly) -> 3 news search -> 4 relevance + consequences + plausibility -> 5 insight judge (related news, bank view)
-> 6 insight for the role, self-checked.
python -X utf8 pipeline.py U_68622 --role cfo [--candidates 30 --judge 5] [--ocr]   (needs LLM_API_KEY)
--ocr: before the run, OCR the bank analytics PDFs (vision model, cached per page) and rebuild the bank library.
-> data/insights/<company>_<role>.json

Insights from the bank's own analytics (no news needed) run in parallel with the news engines and join the same result list
(stage5_insight_detection/analytics_insights): --source news|analytics|both (default both). The run of a (company, role) is served
from the committed cache (analytics_insights/cache/) while its inputs are unchanged, otherwise the analytics agent runs (~3-5 ₽, GLM);
--analytics-cache-only never calls a model, --analytics-refresh ignores the cache, --analytics-share is the quota of analytics insights
among all (at least 1, at most that share). Every `insight` of the saved results is in contract v2 (insight_writer/insight_v2.py),
a superset of the old one: news insights keep all their fields.

The judge engine uses retrieval v2 by default (standard generators + generator factory over e5 + qwen + BM25; the factory
runs GLM once per company, ~4 ₽, then is cached): --retrieval hybrid = standard generators over FRIDA + TF-IDF,
lexical = the facet search. --candidates limits distinct events in v2 / hybrid mode. Event membership is prepared/rebuilt automatically from the corpus.

CLI/API only run the judge engine. The legacy run_loop() and its modules remain for direct development use;
shared computations and audit are still used by judge. PDF preparation runs before the judge graph.
"""
import argparse, contextvars, hashlib, json, math, operator, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, TYPE_CHECKING, TypedDict

import pandas as pd

from stage2_company_facts.profile.integration import load_company
from stage2_company_facts.profile.context import CONTEXT_CATEGORY

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.data import DATA
from common.llm import ChatModels, StructuredOutputError
from common.news_corpus import StaleNewsReferences, bind_result, save_results
from common.roles import roles
from common.search import corpus_fingerprint, news_pool
from common.news_snapshot import pinned_run, news_snapshot
from common.personalization import Personalization, load_personalization
from common.safety import render_untrusted
from stage2_company_facts.deterministic_facts.facts import build
from stage3_news_search.retrieval import candidates
from stage4_news_relevance.consequence_extraction.consequences import ConsequenceExtractor
from stage4_news_relevance.relevance_scoring.relevance import RelevanceScorer
from stage5_insight_detection.bank_analytics.search import BankAnalytics
from stage5_insight_detection.insight_judge.judge import InsightJudge
from stage5_insight_detection.news_enrichment.related_news import RelatedNews
from stage6_insight_generation.insight_writer.insight_v2 import migrate_result
from stage6_insight_generation.insight_writer.writer import Evidence, InsightWriter

if TYPE_CHECKING:
    from stage5_insight_detection.bank_analytics.prepare import PreparationOptions as BankPreparationOptions, PreparedBank


class _NewsState(TypedDict, total=False):
    """Per-run state of the news engine (stages 2-6). `rows` keeps the retrieval input order; the Send tasks
    append out of order, so every fan-out item carries its index and the joins sort by it — the result rows are
    exactly what the thread-pool implementation produced."""
    rows: list[dict]
    assessed: Annotated[list[tuple], operator.add]   # (idx, priority, news, stage4 | error)
    top: list[tuple]                                  # (pos, priority, news, stage4), ranked
    judge_rows: Annotated[list[tuple], operator.add]  # (pos, result row)
    news_results: Annotated[list[dict], operator.add]


def scoped_personalization(value: Personalization | None, company_id: str, role: str) -> Personalization | None:
    if value is None:
        return None
    if (value.company_id, value.role) != (company_id, role):
        raise ValueError('Personalization company and role must match this analysis')
    return value.model_copy(deep=True) if not value.empty() else None


def personalization_context(value: Personalization | None) -> str:
    if value is None or not (value.notes or value.topic_weights):
        return ''
    return render_untrusted('user_context', 'Со слов пользователя: не проверенные факты и не позиция Банка',
                            value.render(), body_limit=16000)


class InsightPipeline:
    def __init__(self, cheap: BaseChatModel, agent: BaseChatModel, consequences: ConsequenceExtractor | None = None,
                 bank: BankAnalytics | None = None, related: RelatedNews | None = None):
        self.relevance = RelevanceScorer(cheap)
        self.consequences = consequences or ConsequenceExtractor(cheap)
        self.related = related or RelatedNews()
        self.bank = bank
        self.judge = InsightJudge(agent, self.related, bank)
        self.writer = InsightWriter(agent)
        self._graph = None
        self._params: dict = {}

    @pinned_run
    def run(self, company_id: str, role: str, n_candidates: int = 30, n_judge: int = 5,
            retrieval: str = 'v2', workers: int = 1, personalization: Personalization | None = None) -> list[dict]:
        """LangGraph news engine: stage 4 and stages 5-6 run as parallel `Send` tasks (max_concurrency=workers);
        the joins rank by the retrieval input index, so workers=1 keeps the call order deterministic for scripted
        test models exactly as the thread pool did."""
        pers = scoped_personalization(personalization, company_id, role)
        self._params = {'company_id': company_id, 'role': role, 'n_judge': n_judge,
                        'personalization': pers, 'user_context': personalization_context(pers)}
        rows = self._prepare(company_id, role, n_candidates, retrieval)
        state = self.news_graph().invoke({'rows': rows}, config={'max_concurrency': max(1, workers)})
        return state['news_results']

    def news_graph(self):
        """Stages 2-6 as a compiled LangGraph graph (see langgraph-e2e-design.md): prepare is done by run(),
        assess fans out per news, rank joins, judge+write fans out per top news, collect joins."""
        if getattr(self, '_graph', None) is None:
            g = StateGraph(_NewsState)
            g.add_node('assess_one', self._assess_node)
            g.add_node('rank', self._rank_node)
            g.add_node('judge_one', self._judge_node)
            g.add_node('collect', self._collect_node)
            g.add_conditional_edges(START, self._fan_assess, ['assess_one', 'rank'])
            g.add_edge('assess_one', 'rank')
            g.add_conditional_edges('rank', self._fan_judge, ['judge_one', 'collect'])
            g.add_edge('judge_one', 'collect')
            g.add_edge('collect', END)
            self._graph = g.compile()
        return self._graph

    @staticmethod
    def _fan_assess(state: _NewsState) -> list:
        # a fresh copy of this thread's context per task (the Langfuse session, the news snapshot), as the
        # thread-pool version did; each task enters its own copy, so parallel tasks never share a Context
        return [Send('assess_one', {'idx': i, 'news': r, 'ctx': contextvars.copy_context()})
                for i, r in enumerate(state['rows'])] or ['rank']

    @staticmethod
    def _fan_judge(state: _NewsState) -> list:
        return [Send('judge_one', {'pos': p, 'priority': pr, 'news': n, 'stage4': s, 'ctx': contextvars.copy_context()})
                for p, (pr, n, s) in enumerate(state['top'])] or ['collect']

    def _prepare(self, company_id: str, role: str, n_candidates: int, retrieval: str) -> list[dict]:
        if isinstance(self.related, RelatedNews):
            self.related.refresh()
        facts = build([company_id], use_profiles=True)[company_id]                     # stage 2
        self._params.update(facts=facts,
                            key_facts=[f'[{f["cat"]}] {f["text"]}' for f in facts if f['depth'] == 1],
                            # the judge reads every fact through its company_facts tool, so the writer and the audit
                            # get all of them too (key facts only -> the checker rejected group and payment figures)
                            all_facts=[f'[{f["cat"]}] {f["text"]}' for f in facts])
        shortlist, _ = candidates(company_id, role, top=n_candidates, facts=facts, method=retrieval)  # stage 3
        pers = self._params['personalization']
        if pers is not None:
            shortlist = shortlist[~shortlist['id'].astype(int).isin(pers.excluded_news_ids)]
            if self._params['user_context']:
                self._params['key_facts'].append(self._params['user_context'])
        if isinstance(self.related, RelatedNews) and shortlist.attrs.get(
                'corpus_fingerprint', corpus_fingerprint(self.related.news)) != corpus_fingerprint(self.related.news):
            raise StaleNewsReferences('News changed during retrieval; restart the batch')
        return shortlist.to_dict('records')

    def _record(self, news: dict, **fields) -> dict:
        result = {'news_id': news['id'], 'title': news['title'], **fields}
        result['matched_fragments'] = news.get('matched_fragments', [])
        if 'event_id' in news:
            result['retrieval'] = {k: news[k] for k in ('event_id', 'event_news_ids', 'generators')}
        if isinstance(self.related, RelatedNews):
            result = bind_result(result, self.related.news, allow_unresolved=True)
        return result

    @staticmethod
    def _model_error(error: StructuredOutputError, stage: str) -> dict:
        return {'stage': stage, 'schema': error.schema, 'attempts': error.attempts, 'message': str(error)}

    def _assess_node(self, item: dict) -> dict:
        """Stage 4 for one news: (idx, priority, news, stage4) — priority None marks a model error."""
        idx, news = item['idx'], item['news']
        p = self._params

        def body(_):
            plausibility = self.related.event_plausibility(news['id'])
            stage = 'relevance'
            try:
                relevance = self.relevance.score(news, '\n'.join(p['key_facts']), p['role'])
                stage = 'consequences'
                stage4 = {'matched_fragments': news.get('matched_fragments', []), 'relevance': relevance.model_dump(),
                          'consequences': self.consequences.extract(news), 'plausibility': plausibility}
                priority = relevance.score * (0.5 if plausibility is None else plausibility)
                if p['personalization'] is not None:
                    stage4['personalization_factor'] = p['personalization'].factor(news, relevance.channel)
                    priority *= stage4['personalization_factor']
                    if p['user_context']:
                        stage4['user_context'] = p['user_context']
            except Exception as error:  # a dropped connection or a timeout loses this news, not the whole batch
                return None, news, (self._model_error(error, stage) if isinstance(error, StructuredOutputError)
                                    else {'stage': stage, 'type': type(error).__name__, 'message': str(error)[:500]})
            return priority, news, stage4

        priority, news, out = item['ctx'].run(body, None)
        return {'assessed': [(idx, priority, news, out)]}

    def _rank_node(self, state: _NewsState) -> dict:
        """Join of the assess fan-out in the retrieval input order; stable ranking = the thread-pool order."""
        assessed = sorted(state['assessed'], key=lambda t: t[0])
        error_rows, scored = [], []
        for idx, priority, news, out in assessed:
            if priority is None:
                error_rows.append(self._record(news, status='model_error', error=out))
            else:
                scored.append((idx, priority, news, out))
        top = [t for t in sorted(scored, key=lambda t: (-t[1], t[0]))[:self._params['n_judge']] if t[1] > 0]
        return {'news_results': error_rows, 'top': [(priority, news, stage4) for _, priority, news, stage4 in top]}

    def _judge_node(self, item: dict) -> dict:
        """Stages 5-6 for one top news: judge -> writer -> checker -> audit, as one result row."""
        result, news = item['ctx'].run(self._judge_body, item)
        return {'judge_rows': [(item['pos'], self._record(news, **result))]}

    def _judge_body(self, item: dict) -> tuple[dict, dict]:
        p = self._params
        company_id, role, facts, all_facts = p['company_id'], p['role'], p['facts'], p['all_facts']
        news, stage4 = item['news'], item['stage4']
        result, stage = {'stage4': stage4}, 'judge'
        try:
            decision = self.judge.decide(news['id'], company_id, facts, role, stage4)  # stage 5
            result['decision'] = decision.model_dump()
            if decision.bank_evidence:
                result['judge_bank'] = decision.bank_evidence
            calculations = decision.calculation_facts
            if calculations:
                result['calculations'] = calculations
                result['calculation_bank'] = decision.calculation_bank
            if decision.is_insight:                                                   # stage 6
                # Preserve the original numbering for discussion even if company tables later change.
                result['evidence_facts'] = list(all_facts)
                related = [self.related.get(i) for i in decision.related_news_ids[:3] if i in self.related.row_of]
                # Reuse the exact sources read by the judge (hits, neighbours, compared tables).
                # A fresh mechanism query is only a fallback when the judge read no bank material.
                bank = decision.bank_evidence or (self.bank.search(decision.mechanism) if self.bank else [])
                bank = list({b.get('id') or f'bank:{b["report"]}:{b["page"]}': b
                             for b in [*bank, *decision.calculation_bank]}.values())
                stage = 'writer'
                insight, review = self.writer.write(company_id, role, decision.model_dump_json(),
                                                    Evidence(news=[news, *related], facts=all_facts, bank=bank,
                                                             calculations=calculations,
                                                             full_bank_ids={b['id'] for b in [*decision.bank_evidence,
                                                                                            *decision.calculation_bank]}))
                result |= {'insight' if review.ok else 'draft': insight.model_dump(),
                           'review': review.model_dump(), 'status': 'ready' if review.ok else 'review_rejected'}
                from common.evidence import EvidenceSnapshot
                # Capture the actual run's sources now; importing the card must not reread changed company/news data.
                writer_news = {int(n['id']): n for n in [news, *related]}
                result['evidence'] = EvidenceSnapshot.from_result(
                    dict(result, news_id=news['id'], bank=bank), company_id, role,
                    news_lookup=writer_news.get,
                    facts=facts).model_dump(mode='json')
                if review.ok:
                    from stage5_insight_detection.analyst_loop.audit import audit
                    from stage5_insight_detection.analyst_loop.verdict import InsightBrief
                    evidence_facts = [dict(id=f'fact:{i}', value='', unit='', formula='', label=f)
                                      for i, f in enumerate(all_facts, 1) if facts[i - 1]['cat'] != CONTEXT_CATEGORY]
                    evidence_facts += [f for f in calculations if f['kind'] == 'fact']
                    hypotheses = [f for f in calculations if f['kind'] == 'hypothesis']
                    banks = [dict(b, id=b.get('id') or f'bank:{b["report"]}:{b["page"]}',
                                  freshness=b.get('freshness', 'stale' if b['stale'] else 'fresh')) for b in bank]
                    sources = {f'news:{n["id"]}': f'{n["title"]}. {n.get("text", "")}' for n in [news, *related]}
                    sources |= {b['id']: b['text'] for b in banks}
                    brief = InsightBrief(refs=[*sources, *[f['id'] for f in evidence_facts + hypotheses]], facts=evidence_facts,
                        consequences=[], risks=[], bank=banks, hypotheses=hypotheses,
                        impact=insight.impact.model_dump() if insight.impact else
                            {'level': 'low', 'criteria': 'В режиме judge количественное влияние кодом не рассчитано'},
                        not_bank_position=not any(b.get('own_view') and b['freshness'] in ('fresh', 'aging') for b in banks), warnings=[])
                    insight, warnings = audit(insight, brief, sources)
                    result.update(insight=insight.model_dump(), audit=warnings)
                    if any(w['code'] in {'unknown_ref', 'no_source'} for w in warnings):
                        result.update(draft=result.pop('insight'), status='audit_rejected')
        except Exception as error:  # a dropped connection or a timeout loses this news, not the whole batch
            result.update(status='model_error', error=self._model_error(error, stage) if isinstance(error, StructuredOutputError)
                          else {'stage': stage, 'type': type(error).__name__, 'message': str(error)[:500]})
        return result, news

    def _collect_node(self, state: _NewsState) -> dict:
        rows = [row for _, row in sorted(state['judge_rows'], key=lambda t: t[0])]
        return {'news_results': rows}


@pinned_run
def run_loop(company_id: str, role: str, models: ChatModels, n_candidates: int = 30, n_judge: int = 5,
             retrieval: str = 'v2', bank_review: bool = False, bank_budget: float | None = None,
             preparation: 'BankPreparationOptions | None' = None,
             prepared_bank: 'PreparedBank | None' = None, workers: int = 1,
             cards_dir: Path | None = None, personalization: Personalization | None = None) -> list[dict]:
    from common.dedup import dedup_news
    from common.trace import AgentTrace
    from stage4_news_relevance.news_classifier.classifier import NewsClassifier
    from stage5_insight_detection.analyst_loop.loop import AnalystLoop
    from stage5_insight_detection.analyst_loop.pool import RelatedNewsPool
    from stage5_insight_detection.analyst_loop.profile import CompanyProfile
    from stage5_insight_detection.bank_analytics.library import BankLibrary
    from stage5_insight_detection.bank_analytics.review import BankReviewer
    from stage6_insight_generation.insight_card.render import render
    pers = scoped_personalization(personalization, company_id, role)
    trace = AgentTrace()
    prepared = prepared_bank
    if prepared is not None:
        trace.log('bank_preparation', **prepared.report)
    elif preparation is not None and preparation.enabled:
        from stage5_insight_detection.bank_analytics.prepare import prepare_bank
        prepared = prepare_bank(preparation, models, trace=trace)
    pool = RelatedNewsPool()
    stage2 = load_company(company_id)
    profile = CompanyProfile(**stage2.card)
    facts = stage2.facts
    shortlist, _ = candidates(company_id, role, top=n_candidates, facts=facts, method=retrieval)
    source_news = getattr(getattr(pool, 'related', None), 'news', None)
    if isinstance(source_news, pd.DataFrame):
        if shortlist.attrs.get('corpus_fingerprint', corpus_fingerprint(pool.related.news)) != corpus_fingerprint(pool.related.news):
            raise StaleNewsReferences('News changed during retrieval; restart the batch')
    rows = shortlist.to_dict('records')
    if pers is not None:
        excluded = set(pers.excluded_news_ids)
        rows = [r for r in rows if int(r['id']) not in excluded]
    if retrieval == 'lexical':
        rows = dedup_news(rows)
    rows_by_id = {str(r['id']): r for r in rows}
    trace.log('shortlist', candidates=len(shortlist), after_dedup=len(rows))
    card = profile.card()
    if context := personalization_context(pers):
        card += '\n\n' + context
    scored = NewsClassifier(models.cheap(max_tokens=1500), trace=trace).classify(rows, card)
    scored = [s for s in scored if role in s['affected_roles'] or not s['affected_roles']] or scored
    if pers is not None:
        for score in scored:
            score['personalization_factor'] = pers.factor(rows_by_id[str(score['id'])], score.get('channel', ''))
            score['importance'] *= score['personalization_factor']
    top = sorted(scored, key=lambda s: -s['importance'])[:n_judge]
    library = prepared.library if prepared is not None else BankLibrary.refresh()
    agent = models.agent()  # default reasoning and the 64k cap: a small cap cut GLM's answers (reasoning counts toward it)
    loop = AnalystLoop(library, pool, agent, InsightWriter(agent), trace=trace,
                       company_facts=facts, reviewer=BankReviewer(budget=bank_budget) if bank_review else None)
    results = []
    card_traces = {}
    shared_steps = list(trace.steps())
    def analyze(s):
        row = rows_by_id[str(s['id'])]
        local_trace = trace if workers == 1 else AgentTrace()
        active_loop = loop if workers == 1 else AnalystLoop(library, pool, agent, InsightWriter(agent),
            trace=local_trace, company_facts=facts, reviewer=BankReviewer(budget=bank_budget) if bank_review else None)
        result = active_loop.run(row, profile, role, s)
        if pers is not None:
            result['personalization_factor'] = s['personalization_factor']
        result['matched_fragments'] = row.get('matched_fragments', [])
        if prepared is not None:
            result['bank_preparation'] = prepared.report
        if 'event_id' in row:
            result['retrieval'] = {k: row[k] for k in ('event_id', 'event_news_ids', 'generators')}
        if isinstance(source_news, pd.DataFrame):
            result = bind_result(result, source_news, allow_unresolved=True)
        steps = shared_steps + list(local_trace.steps()) if workers > 1 else None
        return result, steps
    if workers < 1:
        raise ValueError('workers must be positive')
    with ThreadPoolExecutor(workers) as executor:
        jobs=[executor.submit(contextvars.copy_context().run,analyze,score) for score in top]
        for job in jobs:
            result, steps = job.result()
            results.append(result)
            if steps is not None:
                card_traces[str(result['news_id'])] = steps
        if isinstance(source_news, pd.DataFrame):
            results = [bind_result(r, source_news, allow_unresolved=True) for r in results]
    cards = cards_dir if cards_dir is not None else DATA / 'insights' / 'cards'
    cards.mkdir(parents=True, exist_ok=True)
    for r in results:
        source_id = r['thesis']['id'].replace(':', '_') if r.get('origin') == 'bank' else r['news_id']
        (cards / f'{company_id}_{role}_{source_id}.html').write_text(
            render(json.loads(json.dumps(r, default=str)), card_traces.get(str(r.get('news_id')), trace.steps())), encoding='utf-8')
    return results


def optional_bank() -> BankAnalytics | None:
    try:
        return BankAnalytics()
    except FileNotFoundError as e:
        print(f'warning: {e}; insights will have no bank view')
        return None


def run_analytics(company_id: str, role: str, models: ChatModels, *, cache_only: bool = False, refresh: bool = False,
                  strict: bool = False, bank: BankAnalytics | None = None):
    """Insights from the bank's analytics: (AnalysisRun | None, error). Never raises: the news results must not depend on it."""
    from stage5_insight_detection.analytics_insights.service import analytics_run
    try:
        facts = load_company(company_id).facts
        return analytics_run(company_id, role, facts, models=models, bank=bank or BankAnalytics(), cache_only=cache_only,
                              refresh=refresh, strict=strict), ''
    except Exception as e:
        return None, f'{type(e).__name__}: {e}'


def analytics_rows(run, error: str, news_results: list[dict], share: float, cache_only: bool) -> list[dict]:
    """The rows of the analytics part of the result list: the insights within the quota next to the ready news insights + the run row.
    No run (error, or cache only without a cached run): no rows, the reason is printed; the news results stay as they are."""
    from stage5_insight_detection.analytics_insights.service import result_rows, select
    if run is None:
        print('analytics: ' + (error or ('cache only: no cached run for this company and role' if cache_only else 'no run')))
        return []
    ready = sum(1 for r in news_results if r.get('status') == 'ready' and r.get('insight'))
    return result_rows(run, select(run.insights, ready, share))


def budget_arg(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError('budget must be finite and non-negative')
    return result


class E2EState(_NewsState, total=False):
    company_id: str
    role: str
    n_candidates: int
    n_judge: int
    retrieval: str
    workers: int
    personalization: dict | None
    analytics: object          # AnalysisRun | None of the parallel bank-analytics branch
    analytics_error: str
    results: list[dict]        # final rows: news (contract v2) + analytics within the quota
    out: str


def build_e2e_graph(pipe: InsightPipeline, *, with_analytics: bool, run_analytics_fn, analytics_rows_fn, migrate_fn,
                    save_fn, checkpointer=None):
    """The full judge-engine E2E as one LangGraph graph (langgraph-e2e-design.md): the news engine (itself a graph)
    and the bank-analytics branch start together and join at `merge`; `save` is the last node. The CLI injects its
    own callables, so the graph also runs in tests with plain functions."""
    g = StateGraph(E2EState)
    # the branches may run in parallel worker threads: each enters its own copy of the invoking thread's context
    ctx_news, ctx_analytics = contextvars.copy_context(), contextvars.copy_context()

    def news_engine(s: E2EState) -> dict:
        kwargs = {}
        if s.get('personalization') is not None:
            kwargs['personalization'] = Personalization.model_validate(s['personalization'])
        return ctx_news.run(lambda: {'news_results': pipe.run(s['company_id'], s['role'], s['n_candidates'], s['n_judge'],
                                                              retrieval=s['retrieval'], workers=s['workers'], **kwargs)})

    def analytics_node(s: E2EState) -> dict:
        run, error = ctx_analytics.run(lambda: run_analytics_fn())
        return {'analytics': run, 'analytics_error': error}

    def merge(s: E2EState) -> dict:
        results = [migrate_fn(r) for r in s['news_results']]  # contract v2 (a superset of the old one)
        return {'results': results + (analytics_rows_fn(s['analytics'], s['analytics_error'], results)
                                      if with_analytics else [])}

    def save(s: E2EState) -> dict:
        return {'out': save_fn(s['results'])}

    g.add_node('news_engine', news_engine)
    g.add_node('merge', merge)
    g.add_node('save', save)
    g.add_edge(START, 'news_engine')
    if with_analytics:
        g.add_node('analytics', analytics_node)
        g.add_edge(START, 'analytics')  # parallel branch, joins at merge
        g.add_edge('analytics', 'merge')
    g.add_edge('news_engine', 'merge')
    g.add_edge('merge', 'save')
    g.add_edge('save', END)
    return g.compile(checkpointer=checkpointer)  # a checkpointer + config['configurable']['thread_id'] = job id -> resume after a crash


def _save_results(company_id: str, role: str, source_news, rows: list[dict], user_id: str | None = None):
    suffix = '_' + hashlib.sha256(user_id.encode()).hexdigest()[:12] if user_id else ''
    out = DATA / 'insights' / f'{company_id}_{role}{suffix}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    if corpus_fingerprint(source_news) != corpus_fingerprint(news_pool()):
        raise StaleNewsReferences('News changed while the pipeline was running; restart the batch before saving')
    save_results(out, rows, source_news, DATA / 'news')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('company_id')
    ap.add_argument('--role', default='cfo', choices=list(roles()))
    ap.add_argument('--user-id', default=None, help='apply this user\'s feedback for this company and role')
    ap.add_argument('--candidates', type=int, default=30)
    ap.add_argument('--judge', type=int, default=5)
    ap.add_argument('--workers', type=int, default=8, help='judge engine: news processed in parallel (LLM calls in flight)')
    ap.add_argument('--engine', choices=['judge'], default='judge', help='judge is the only public engine')
    ap.add_argument('--retrieval', choices=['v2', 'v3', 'hybrid', 'lexical'], default='v2',
                    help='v2 = standard generators + generator factory (e5 + qwen + BM25); v3 = v2 + news-first channel generators; '
                         'hybrid = FRIDA + TF-IDF; lexical = facet search')
    ap.add_argument('--source', choices=['news', 'analytics', 'both'], default='both',
                    help='news = news engines only; analytics = insights from the bank analytics only; both = in parallel, one result list')
    ap.add_argument('--analytics-cache-only', action='store_true', help='analytics: only the cache (a stale one too), never a model call')
    ap.add_argument('--analytics-strict-cache', action='store_true',
                    help='analytics: a cache made for other facts or reports counts as stale and the agent runs again '
                         '(MVP default: the cache wins, no quota spent)')
    ap.add_argument('--analytics-refresh', action='store_true', help='analytics: ignore the cache and run the agent')
    ap.add_argument('--analytics-share', type=float, default=0.4, help='analytics: at most this share of all ready insights (and at least one)')
    ap.add_argument('--ocr', action='store_true', help='OCR bank analytics PDFs and rebuild the bank library first')
    ap.add_argument('--bank-charts', action='store_true', help='full OCR and verify/digitize charts before judge')
    ap.add_argument('--bank-tables', action='store_true', help='full OCR and reconcile table cells before judge')
    ap.add_argument('--bank-digest', action='store_true', help='full OCR and checked report digests before judge')
    ap.add_argument('--bank-prepare', action='store_true', help='full OCR, charts, tables and digests before judge')
    ap.add_argument('--chart-mode', choices=['all', 'selective', 'off'], default=None,
                    help='chart selection during bank preparation (default BANK_CHARTS or all)')
    ap.add_argument('--ocr-budget', type=budget_arg, default=None, help='OCR stage budget, rubles; full preparation default 15')
    ap.add_argument('--charts-budget', type=budget_arg, default=None, help='chart stage budget, rubles; default 8')
    ap.add_argument('--tables-budget', type=budget_arg, default=None, help='table stage budget, rubles; default 7')
    ap.add_argument('--digest-budget', type=budget_arg, default=None, help='digest stage budget, rubles; default 3')
    a = ap.parse_args()
    try:
        personalization = load_personalization(a.user_id, a.company_id, a.role) if a.user_id else None
    except ValueError as error:
        ap.error(str(error))
    do_charts, do_tables, do_digest = a.bank_charts or a.bank_prepare, a.bank_tables or a.bank_prepare, a.bank_digest or a.bank_prepare
    prepare_requested = do_charts or do_tables or do_digest
    if a.chart_mode is not None and not (prepare_requested or a.ocr):
        ap.error('--chart-mode requires --ocr/--bank-charts/--bank-tables/--bank-digest/--bank-prepare')
    for flag, value, enabled in (('--ocr-budget', a.ocr_budget, a.ocr or prepare_requested),
                                 ('--charts-budget', a.charts_budget, do_charts),
                                 ('--tables-budget', a.tables_budget, do_tables),
                                 ('--digest-budget', a.digest_budget, do_digest)):
        if value is not None and not enabled:
            ap.error(f'{flag} requires its preparation stage')
    source_news = news_pool()
    with news_snapshot(source_news):
        usage = UsageMetadataCallbackHandler()
        models = ChatModels(callbacks=[usage])
        preparation = None
        prepared_bank = None
        if prepare_requested or a.ocr:
            from stage5_insight_detection.bank_analytics.prepare import PreparationOptions
            preparation = PreparationOptions(ocr=True, charts=do_charts, tables=do_tables, digests=do_digest, chart_mode=a.chart_mode,
                ocr_budget=15 if a.ocr_budget is None else a.ocr_budget,
                charts_budget=8 if a.charts_budget is None else a.charts_budget,
                tables_budget=7 if a.tables_budget is None else a.tables_budget,
                digest_budget=3 if a.digest_budget is None else a.digest_budget)
            # Publish prepared fragments before the parallel agents open their search/library snapshots.
            from stage5_insight_detection.bank_analytics.prepare import prepare_bank
            prepared_bank = prepare_bank(preparation, models)
        bank = optional_bank()
        def save_run(rows):
            if prepared_bank is not None:
                for row in rows:
                    row['bank_preparation'] = prepared_bank.report
            return _save_results(a.company_id, a.role, source_news, rows, a.user_id)
        if a.source == 'analytics':
            run, error = contextvars.copy_context().run(lambda: run_analytics(
                a.company_id, a.role, models, cache_only=a.analytics_cache_only, refresh=a.analytics_refresh,
                strict=a.analytics_strict_cache, bank=bank))
            results = analytics_rows(run, error, [], a.analytics_share, a.analytics_cache_only)
            out = save_run(results)
        else:  # judge engine: the whole E2E is one LangGraph graph, the analytics branch runs in parallel with the news
            def _run_analytics():
                return contextvars.copy_context().run(lambda: run_analytics(
                    a.company_id, a.role, models, cache_only=a.analytics_cache_only, refresh=a.analytics_refresh,
                    strict=a.analytics_strict_cache, bank=bank))

            graph = build_e2e_graph(
                InsightPipeline(models.cheap(), models.agent(), bank=bank),
                with_analytics=a.source == 'both',
                run_analytics_fn=_run_analytics,
                analytics_rows_fn=lambda run, error, rows: analytics_rows(run, error, rows, a.analytics_share,
                                                                         a.analytics_cache_only),
                migrate_fn=lambda r: migrate_result(r, a.company_id),
                save_fn=save_run)
            state = graph.invoke({'company_id': a.company_id, 'role': a.role, 'n_candidates': a.candidates,
                                  'n_judge': a.judge, 'retrieval': a.retrieval, 'workers': a.workers,
                                  'personalization': personalization.model_dump() if personalization is not None else None},
                                 config={'max_concurrency': max(1, a.workers)})
            results, out = state['results'], state['out']
        print(f'{out}: {sum("insight" in r for r in results)} insights, '
              f'{sum(r.get("status") == "model_error" for r in results)} model errors, '
              f'{sum(r.get("source") == "bank_analytics" and "insight" in r for r in results)} from the bank analytics, '
              f'{len(results)} results; tokens {usage.usage_metadata}')
