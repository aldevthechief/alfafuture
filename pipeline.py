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

Both engines use retrieval v2 by default (standard generators + generator factory over e5 + qwen + BM25; the factory
runs GLM once per company, ~4 ₽, then is cached): --retrieval hybrid = standard generators over FRIDA + TF-IDF,
lexical = the facet search. --candidates limits distinct events in v2 / hybrid mode. Event membership is prepared/rebuilt automatically from the corpus.

--engine loop: stages 4-6 by the analyst loop (stage5_insight_detection/analyst_loop): the shortlist is deduplicated,
scored by the batch classifier (scores only), and the top news go through bank match -> news pool -> computed
consequences -> grounds check -> data requests -> verdict -> writer -> audit; journal in data/traces/<run>.jsonl.
"""
import argparse, contextvars, json, math, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from stage2_company_facts.profile.integration import load_company
from stage2_company_facts.profile.context import CONTEXT_CATEGORY

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.language_models import BaseChatModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.data import DATA
from common.llm import ChatModels, StructuredOutputError
from common.news_corpus import StaleNewsReferences, bind_result, save_results
from common.roles import roles
from common.search import corpus_fingerprint, news_pool
from common.news_snapshot import pinned_run, news_snapshot
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


class InsightPipeline:
    def __init__(self, cheap: BaseChatModel, agent: BaseChatModel, consequences: ConsequenceExtractor | None = None,
                 bank: BankAnalytics | None = None, related: RelatedNews | None = None):
        self.relevance = RelevanceScorer(cheap)
        self.consequences = consequences or ConsequenceExtractor(cheap)
        self.related = related or RelatedNews()
        self.bank = bank
        self.judge = InsightJudge(agent, self.related, bank)
        self.writer = InsightWriter(agent)
        
    @pinned_run
    def run(self, company_id: str, role: str, n_candidates: int = 30, n_judge: int = 5,
            retrieval: str = 'v2', workers: int = 1) -> list[dict]:
        """News are independent: stage 4 and stages 5-6 run in a pool of `workers` threads.
        workers=1 keeps the call order deterministic for scripted test models."""
        if isinstance(self.related, RelatedNews):
            self.related.refresh()
        facts = build([company_id], use_profiles=True)[company_id]                 # stage 2
        key_facts = [f'[{f["cat"]}] {f["text"]}' for f in facts if f['depth'] == 1]
        # the judge reads every fact through its company_facts tool, so the writer and the audit get all of them too
        # (with key facts only, the writer's checker rejected group and payment figures the judge had cited: U_68622, 22.10)
        all_facts = [f'[{f["cat"]}] {f["text"]}' for f in facts]
        shortlist, _ = candidates(company_id, role, top=n_candidates, facts=facts, method=retrieval)  # stage 3
        if isinstance(self.related, RelatedNews) and shortlist.attrs.get(
                'corpus_fingerprint', corpus_fingerprint(self.related.news)) != corpus_fingerprint(self.related.news):
            raise StaleNewsReferences('News changed during retrieval; restart the batch')
        scored, results = [], []

        def record(news: dict, **fields) -> None:
            result = {'news_id': news['id'], 'title': news['title'], **fields}
            result['matched_fragments'] = news.get('matched_fragments', [])
            if 'event_id' in news:
                result['retrieval'] = {k: news[k] for k in ('event_id', 'event_news_ids', 'generators')}
            if isinstance(self.related, RelatedNews):
                result = bind_result(result, self.related.news, allow_unresolved=True)
            results.append(result)

        def model_error(error: StructuredOutputError, stage: str) -> dict:
            return {'stage': stage, 'schema': error.schema, 'attempts': error.attempts, 'message': str(error)}

        def assess(news: dict) -> tuple:                                           # stage 4
            plausibility = self.related.event_plausibility(news['id'])
            stage = 'relevance'
            try:
                relevance = self.relevance.score(news, '\n'.join(key_facts), role)
                stage = 'consequences'
                stage4 = {'matched_fragments': news.get('matched_fragments', []), 'relevance': relevance.model_dump(), 'consequences': self.consequences.extract(news),
                          'plausibility': plausibility}
            except StructuredOutputError as error:
                return None, news, model_error(error, stage)
            return relevance.score * (0.5 if plausibility is None else plausibility), news, stage4

        def judge(item: tuple) -> dict:
            priority, news, stage4 = item
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
                if decision.is_insight:                                           # stage 6
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
                            consequences=[], risks=[], bank=banks, hypotheses=hypotheses, impact=insight.impact.model_dump() if insight.impact else
                                {'level': 'low', 'criteria': 'В режиме judge количественное влияние кодом не рассчитано'},
                            not_bank_position=not any(b.get('own_view') and b['freshness'] in ('fresh', 'aging') for b in banks), warnings=[])
                        insight, warnings = audit(insight, brief, sources)
                        result.update(insight=insight.model_dump(), audit=warnings)
                        if any(w['code'] in {'unknown_ref', 'no_source'} for w in warnings):
                            result.update(draft=result.pop('insight'), status='audit_rejected')
            except StructuredOutputError as error:
                result.update(status='model_error', error=model_error(error, stage))
            return result

        def threaded(fn, items):  # each task runs in a copy of this thread's context: the Langfuse session lives there
            return pool.map(lambda job: job[0].run(fn, job[1]), [(contextvars.copy_context(), x) for x in items])

        with ThreadPoolExecutor(max(1, workers)) as pool:
            for priority, news, out in threaded(assess, shortlist.to_dict('records')):
                if priority is None:
                    record(news, status='model_error', error=out)
                else:
                    scored.append((priority, news, out))
            top = [s for s in sorted(scored, key=lambda s: -s[0])[:n_judge] if s[0] > 0]
            for item, result in zip(top, threaded(judge, top)):
                record(item[1], **result)
        return results


@pinned_run
def run_loop(company_id: str, role: str, models: ChatModels, n_candidates: int = 30, n_judge: int = 5,
             retrieval: str = 'v2', bank_review: bool = False, bank_budget: float | None = None,
             preparation: 'BankPreparationOptions | None' = None, bank_insights_enabled: bool = False,
             prepared_bank: 'PreparedBank | None' = None, workers: int = 1, cards_dir: Path | None = None) -> list[dict]:
    from common.dedup import dedup_news
    from common.trace import AgentTrace
    from stage4_news_relevance.news_classifier.classifier import NewsClassifier
    from stage5_insight_detection.analyst_loop.loop import AnalystLoop
    from stage5_insight_detection.analyst_loop.pool import RelatedNewsPool
    from stage5_insight_detection.analyst_loop.profile import CompanyProfile
    from stage5_insight_detection.bank_analytics.library import BankLibrary
    from stage5_insight_detection.bank_analytics.review import BankReviewer
    from stage6_insight_generation.insight_card.render import render
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
    if retrieval == 'lexical':
        rows = dedup_news(rows)
    rows_by_id = {str(r['id']): r for r in rows}
    trace.log('shortlist', candidates=len(shortlist), after_dedup=len(rows))
    scored = NewsClassifier(models.cheap(max_tokens=1500), trace=trace).classify(rows, profile.card())
    scored = [s for s in scored if role in s['affected_roles'] or not s['affected_roles']] or scored
    top = sorted(scored, key=lambda s: -s['importance'])[:n_judge]
    library = prepared.library if prepared is not None else BankLibrary.refresh()
    agent = models.agent()  # default reasoning and the 64k cap: a small cap cut GLM's answers (reasoning counts toward it)
    loop = AnalystLoop(library, pool, agent, InsightWriter(agent), trace=trace,
                       company_facts=facts, reviewer=BankReviewer(budget=bank_budget) if bank_review else None)
    results = []
    def analyze(s):
        row = rows_by_id[str(s['id'])]
        active_loop = loop if workers == 1 else AnalystLoop(library, pool, agent, InsightWriter(agent),
            trace=AgentTrace(), company_facts=facts, reviewer=BankReviewer(budget=bank_budget) if bank_review else None)
        result = active_loop.run(row, profile, role, s)
        result['matched_fragments'] = row.get('matched_fragments', [])
        if prepared is not None:
            result['bank_preparation'] = prepared.report
        if 'event_id' in row:
            result['retrieval'] = {k: row[k] for k in ('event_id', 'event_news_ids', 'generators')}
        if isinstance(source_news, pd.DataFrame):
            result = bind_result(result, source_news, allow_unresolved=True)
        return result
    if workers < 1:
        raise ValueError('workers must be positive')
    with ThreadPoolExecutor(workers) as executor:
        jobs=[executor.submit(contextvars.copy_context().run,analyze,score) for score in top]
        results=[job.result() for job in jobs]
    if bank_insights_enabled:
        from stage5_insight_detection.analyst_loop.loop import bank_insights
        from stage5_insight_detection.bank_analytics.theses import ShownLog
        results.extend(bank_insights(loop, profile, role, results, ShownLog()))
        if isinstance(source_news, pd.DataFrame):
            results = [bind_result(r, source_news, allow_unresolved=True) for r in results]
    cards = cards_dir if cards_dir is not None else DATA / 'insights' / 'cards'
    cards.mkdir(parents=True, exist_ok=True)
    for r in results:
        source_id = r['thesis']['id'].replace(':', '_') if r.get('origin') == 'bank' else r['news_id']
        (cards / f'{company_id}_{role}_{source_id}.html').write_text(
            render(json.loads(json.dumps(r, default=str)), trace.steps()), encoding='utf-8')
    return results


def optional_bank() -> BankAnalytics | None:
    try:
        return BankAnalytics()
    except FileNotFoundError as e:
        print(f'warning: {e}; insights will have no bank view')
        return None


def run_analytics(company_id: str, role: str, models: ChatModels, *, cache_only: bool = False, refresh: bool = False, bank: BankAnalytics | None = None):
    """Insights from the bank's analytics: (AnalysisRun | None, error). Never raises: the news results must not depend on it."""
    from stage5_insight_detection.analytics_insights.service import analytics_run
    try:
        facts = load_company(company_id).facts
        return analytics_run(company_id, role, facts, models=models, bank=bank or BankAnalytics(), cache_only=cache_only, refresh=refresh), ''
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


def validate_sources(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Select one bank-only generator before constructing models or touching analytics caches."""
    if args.bank_insights and (args.engine != 'loop' or args.source != 'news'):
        parser.error('--bank-insights requires --engine loop --source news; '
                     '--source analytics/both uses the separate analytics agent')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('company_id')
    ap.add_argument('--role', default='cfo', choices=list(roles()))
    ap.add_argument('--candidates', type=int, default=30)
    ap.add_argument('--judge', type=int, default=5)
    ap.add_argument('--workers', type=int, default=8, help='judge engine: news processed in parallel (LLM calls in flight)')
    ap.add_argument('--engine', choices=['judge', 'loop'], default='judge', help='loop = analyst loop of stage 5')
    ap.add_argument('--retrieval', choices=['v2', 'v3', 'hybrid', 'lexical'], default='v2',
                    help='v2 = standard generators + generator factory (e5 + qwen + BM25); v3 = v2 + news-first channel generators; '
                         'hybrid = FRIDA + TF-IDF; lexical = facet search')
    ap.add_argument('--source', choices=['news', 'analytics', 'both'], default='both',
                    help='news = news engines only; analytics = insights from the bank analytics only; both = in parallel, one result list')
    ap.add_argument('--analytics-cache-only', action='store_true', help='analytics: only the cache (a stale one too), never a model call')
    ap.add_argument('--analytics-refresh', action='store_true', help='analytics: ignore the cache and run the agent')
    ap.add_argument('--analytics-share', type=float, default=0.4, help='analytics: at most this share of all ready insights (and at least one)')
    ap.add_argument('--ocr', action='store_true', help='OCR bank analytics PDFs and rebuild the bank library first')
    ap.add_argument('--bank-charts', action='store_true', help='loop: full OCR and verify/digitize charts before analysis')
    ap.add_argument('--bank-tables', action='store_true', help='loop: full OCR and reconcile table cells before analysis')
    ap.add_argument('--bank-digest', action='store_true', help='loop: full OCR and checked report digests before analysis')
    ap.add_argument('--bank-prepare', action='store_true', help='loop: full OCR, charts, tables and digests in this run')
    ap.add_argument('--ocr-budget', type=budget_arg, default=None, help='OCR stage budget, rubles; full preparation default 15')
    ap.add_argument('--charts-budget', type=budget_arg, default=None, help='chart stage budget, rubles; default 8')
    ap.add_argument('--tables-budget', type=budget_arg, default=None, help='table stage budget, rubles; default 7')
    ap.add_argument('--digest-budget', type=budget_arg, default=None, help='digest stage budget, rubles; default 3')
    ap.add_argument('--bank-review', action='store_true', help='loop: additionally filter bank fragments with the bank analyst agent')
    ap.add_argument('--bank-insights', action='store_true', help='loop with --source news: use the alternative bank-thesis generator')
    ap.add_argument('--bank-budget', type=float, default=None, help='bank reviewer budget in rubles for this run')
    a = ap.parse_args()
    validate_sources(ap, a)
    do_charts, do_tables, do_digest = a.bank_charts or a.bank_prepare, a.bank_tables or a.bank_prepare, a.bank_digest or a.bank_prepare
    prepare_requested = do_charts or do_tables or do_digest
    if prepare_requested and a.engine != 'loop':
        ap.error('--bank-prepare/--bank-charts/--bank-tables/--bank-digest require --engine loop')
    for flag, value, enabled in (('--ocr-budget', a.ocr_budget, a.ocr or prepare_requested),
                                 ('--charts-budget', a.charts_budget, do_charts),
                                 ('--tables-budget', a.tables_budget, do_tables),
                                 ('--digest-budget', a.digest_budget, do_digest)):
        if value is not None and not enabled:
            ap.error(f'{flag} requires its preparation stage')
    if a.bank_review and a.engine != 'loop':
        ap.error('--bank-review requires --engine loop')
    if a.bank_budget is not None and (not a.bank_review or a.bank_budget < 0):
        ap.error('--bank-budget requires --bank-review and must be non-negative')
    if a.ocr and not prepare_requested:
        from stage5_insight_detection.bank_analytics.library import BankLibrary
        from stage5_insight_detection.bank_analytics.ocr import AnalyticsOCR
        BankLibrary.refresh(ocr=AnalyticsOCR(**({'budget': a.ocr_budget} if a.ocr_budget is not None else {})), rebuild=True)
    source_news = news_pool()
    with news_snapshot(source_news):
        usage = UsageMetadataCallbackHandler()
        models = ChatModels(callbacks=[usage])
        preparation = None
        prepared_bank = None
        if prepare_requested:
            from stage5_insight_detection.bank_analytics.prepare import PreparationOptions
            preparation = PreparationOptions(charts=do_charts, tables=do_tables, digests=do_digest,
                ocr_budget=15 if a.ocr_budget is None else a.ocr_budget,
                charts_budget=8 if a.charts_budget is None else a.charts_budget,
                tables_budget=7 if a.tables_budget is None else a.tables_budget,
                digest_budget=3 if a.digest_budget is None else a.digest_budget)
            # Publish prepared fragments before the parallel agents open their search/library snapshots.
            from stage5_insight_detection.bank_analytics.prepare import prepare_bank
            prepared_bank = prepare_bank(preparation, models)
        bank = optional_bank() if a.source != 'news' or a.engine == 'judge' else None
        analytics = None
        if a.source != 'news':  # in parallel with the news engines: the Langfuse session lives in the copied context
            analytics = ThreadPoolExecutor(1)
            future = analytics.submit(contextvars.copy_context().run, lambda: run_analytics(
                a.company_id, a.role, models, cache_only=a.analytics_cache_only, refresh=a.analytics_refresh, bank=bank))
        if a.source == 'analytics':
            results = []
        elif a.engine == 'loop':
            results = run_loop(a.company_id, a.role, models, a.candidates, a.judge, retrieval=a.retrieval,
                               bank_review=a.bank_review, bank_budget=a.bank_budget, preparation=preparation,
                               bank_insights_enabled=a.bank_insights, prepared_bank=prepared_bank, workers=a.workers)
        else:
            results = InsightPipeline(models.cheap(), models.agent(), bank=bank).run(
                a.company_id, a.role, a.candidates, a.judge, retrieval=a.retrieval, workers=a.workers)
        results = [migrate_result(r, a.company_id) for r in results]  # contract v2 (a superset of the old one)
        if analytics is not None:
            run, error = future.result()
            analytics.shutdown()
            results += analytics_rows(run, error, results, a.analytics_share, a.analytics_cache_only)
        out = DATA / 'insights' / f'{a.company_id}_{a.role}.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        if corpus_fingerprint(source_news) != corpus_fingerprint(news_pool()):
            raise StaleNewsReferences('News changed while the pipeline was running; restart the batch before saving')
        save_results(out, results, source_news, DATA / 'news')
        print(f'{out}: {sum("insight" in r for r in results)} insights, '
              f'{sum(r.get("status") == "model_error" for r in results)} model errors, '
              f'{sum(r.get("source") == "bank_analytics" and "insight" in r for r in results)} from the bank analytics, '
              f'{len(results)} results; tokens {usage.usage_metadata}')
