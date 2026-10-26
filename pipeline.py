"""End-to-end run for one company and reader role (stage 1 output, data/news/news.parquet, is the input pool).

2 company profile (stage2_company_facts/profile: data/companies/profiles.jsonl or built on the fly) -> 3 news search -> 4 relevance + consequences + plausibility -> 5 insight judge (related news, bank view)
-> 6 insight for the role, self-checked.
python -X utf8 pipeline.py U_68622 --role cfo [--candidates 15 --judge 15] [--ocr]   (needs LLM_API_KEY)
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
import argparse, asyncio, contextvars, hashlib, json, math, operator, os, sys
from datetime import date
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
from common.news_corpus import StaleNewsReferences, article_fingerprint, bind_result, save_results
from common.stage_cache import StageCache, digest, model_id, prompts_version, code_version
from common import progress
from common.roles import roles
from common.search import corpus_fingerprint, news_pool
from common.news_snapshot import pinned_run, news_snapshot
from common.personalization import Personalization, load_personalization
from common.safety import render_untrusted
from stage2_company_facts.deterministic_facts.facts import build
from stage3_news_search.retrieval import candidates
from stage4_news_relevance.consequence_extraction.consequences import ConsequenceExtractor
from stage4_news_relevance.relevance_scoring.relevance import RelevanceScorer
from stage4_news_relevance.relevance_scoring.relevance import PROMPTS as RELEVANCE_PROMPTS, Relevance
from stage5_insight_detection.bank_analytics.search import BankAnalytics
from stage5_insight_detection.insight_dedup.dedup import DedupGate
from stage5_insight_detection.insight_judge.judge import InsightJudge
from stage5_insight_detection.insight_judge.judge import PROMPTS as JUDGE_PROMPTS, InsightDecision
from stage5_insight_detection.news_enrichment.related_news import RelatedNews
from stage6_insight_generation.insight_writer.insight_v2 import migrate_result as migrate_v2
from stage6_insight_generation.business_value import enrich
from common.credibility import for_result
from stage6_insight_generation.insight_writer.writer import Evidence, InsightWriter
from stage6_insight_generation.insight_writer.writer import PROMPTS as WRITER_PROMPTS
from stage6_insight_generation.insight_writer.insight import Insight, Review

if TYPE_CHECKING:
    from stage5_insight_detection.bank_analytics.prepare import PreparationOptions as BankPreparationOptions, PreparedBank


class _NewsState(TypedDict, total=False):
    """Per-run state of the news engine (stages 2-6). `rows` keeps the retrieval input order; the Send tasks
    append out of order, so every fan-out item carries its index and the joins sort by it — the result rows are
    exactly what the thread-pool implementation produced."""
    rows: list[dict]
    assessed: Annotated[list[tuple], operator.add]   # (idx, priority, news, stage4 | error)
    top: list[tuple]                                  # (pos, priority, news, stage4), ranked
    decided: Annotated[list[tuple], operator.add]     # (pos, news, result, decision | None) after the judge
    to_write: list[tuple]                             # the positive decisions left by the duplicate gate
    judge_rows: Annotated[list[tuple], operator.add]  # (pos, result row)
    news_results: Annotated[list[dict], operator.add]


def scoped_personalization(value: Personalization | None, company_id: str, role: str) -> Personalization | None:
    if value is None:
        return None
    if (value.company_id, value.role) != (company_id, role):
        raise ValueError('Personalization company and role must match this analysis')
    return value.model_copy(deep=True) if not value.empty() else None


def personalization_context(value: Personalization | None) -> str:
    if value is None or not (value.notes or value.topic_weights or value.answers):
        return ''
    return render_untrusted('user_context', 'Со слов пользователя: не проверенные факты и не позиция Банка',
                            value.render(), body_limit=16000)


def migrate_result(result: dict, company_id: str) -> dict:
    return enrich(migrate_v2(result,company_id),company_id)


class InsightPipeline:
    def __init__(self, cheap: BaseChatModel, agent: BaseChatModel, consequences: ConsequenceExtractor | None = None,
                 bank: BankAnalytics | None = None, related: RelatedNews | None = None):
        self.relevance = RelevanceScorer(cheap)
        self.consequences = consequences or ConsequenceExtractor(cheap)
        self.related = related or RelatedNews()
        self.bank = bank
        self.judge = InsightJudge(agent, self.related, bank)
        self.writer = InsightWriter(agent)
        # one call per run that merges duplicate positive decisions before the writer (experiments/insight_dedup)
        low = ({'reasoning_effort': 'low'} if 'reasoning_effort' in getattr(type(agent), 'model_fields', {}) else {})
        self.dedup = DedupGate(agent.model_copy(update=low) if low else agent)
        self._graph = None
        self._params: dict = {}
        self.stages={name:StageCache(name) for name in ('relevance','judge','writer')}
        self._stage_cache_on=os.environ.get('STAGE_CACHE','1')!='0'
        root=Path(__file__).resolve().parent
        implementation=code_version(Path(__file__),*(root/path for path in (
            'common/stage_cache.py','common/safety.py','common/data.py',
            'stage5_insight_detection/insight_judge/judge.py','stage5_insight_detection/insight_judge/calculations.py',
            'stage5_insight_detection/analyst_loop/compute.py','stage6_insight_generation/insight_writer/writer.py')))
        self._version={
            'relevance':digest('integrated-v1',implementation,prompts_version(RELEVANCE_PROMPTS),model_id(cheap)),
            'judge':digest('integrated-v1',implementation,prompts_version(JUDGE_PROMPTS),model_id(agent),InsightDecision.model_json_schema()),
            'writer':digest('integrated-v1',implementation,prompts_version(WRITER_PROMPTS),model_id(agent),Insight.model_json_schema(),Review.model_json_schema()),
        }
        self._bank_signature=None

    @pinned_run
    def run(self, company_id: str, role: str, n_candidates: int = 30, n_judge: int = 5,
            retrieval: str = 'v2', workers: int = 1, personalization: Personalization | None = None) -> list[dict]:
        """LangGraph news engine: stage 4 and stages 5-6 run as parallel `Send` tasks (max_concurrency=workers);
        the joins rank by the retrieval input index, so workers=1 keeps the call order deterministic for scripted
        test models exactly as the thread pool did."""
        pers = scoped_personalization(personalization, company_id, role)
        self._bank_signature=None
        self._params = {'company_id': company_id, 'role': role, 'n_judge': n_judge,
                        'personalization': pers, 'user_context': personalization_context(pers)}
        progress.enter('collecting')
        rows = self._prepare(company_id, role, n_candidates, retrieval)
        progress.enter('analyzing',total=len(rows))
        # one event loop per run: the LLM calls of all news are coroutines in flight together, no thread waits on a model
        state = asyncio.run(self.news_graph().ainvoke({'rows': rows}, config={'max_concurrency': max(1, workers)}))
        return state['news_results']

    def news_graph(self):
        """Stages 2-6 as a compiled LangGraph graph (see langgraph-e2e-design.md): prepare is done by run(),
        assess fans out per news, rank joins, the judges fan out per top news, the duplicate gate joins them,
        the writers fan out per remaining positive decision, collect joins."""
        if getattr(self, '_graph', None) is None:
            g = StateGraph(_NewsState)
            g.add_node('assess_one', self._assess_node)
            g.add_node('rank', self._rank_node)
            g.add_node('judge_one', self._judge_node)
            g.add_node('gate', self._gate_node)
            g.add_node('write_one', self._write_node)
            g.add_node('collect', self._collect_node)
            g.add_conditional_edges(START, self._fan_assess, ['assess_one', 'rank'])
            g.add_edge('assess_one', 'rank')
            g.add_conditional_edges('rank', self._fan_judge, ['judge_one', 'gate'])
            g.add_edge('judge_one', 'gate')
            g.add_conditional_edges('gate', self._fan_write, ['write_one', 'collect'])
            g.add_edge('write_one', 'collect')
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
                for p, (pr, n, s) in enumerate(state['top'])] or ['gate']

    @staticmethod
    def _fan_write(state: _NewsState) -> list:
        return [Send('write_one', {'pos': p, 'news': n, 'result': r, 'decision': d, 'ctx': contextvars.copy_context()})
                for p, n, r, d in state.get('to_write') or []] or ['collect']

    def _prepare(self, company_id: str, role: str, n_candidates: int, retrieval: str) -> list[dict]:
        if isinstance(self.related, RelatedNews):
            self.related.refresh()
        facts = build([company_id], use_profiles=True)[company_id]                     # stage 2
        pers = self._params.get('personalization')
        if pers is not None and pers.answers:
            facts = [*facts, *pers.answer_facts()]
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

    def _data_signature(self):
        # Payment scenarios read the case tables directly, beyond the facts printed in a card.
        from common.data import DATA as input_root
        paths=[input_root/name for name in ('df_Uid_data_hack.csv','df_Uid_payment_hack.csv','df_Fid_Uid_link_hack.csv')]
        paths.append(input_root/'companies'/'profiles.jsonl')
        def version(path):
            try:
                info=path.stat();return (str(path),info.st_size,info.st_mtime_ns,info.st_ctime_ns)
            except OSError:return (str(path),None)
        return digest([version(path) for path in paths])

    def _cached(self, stage: str, key: str):
        return self.stages[stage].get(key)

    def _stored(self, stage: str, key: str, value) -> None:
        self.stages[stage].put(key, value)  # objects built without __init__ (tests) have no cache: callers check _stage_cache_on

    def _relevance_key(self, news: dict, facts_text: str, role: str) -> str:
        return digest('relevance', self._version['relevance'], article_fingerprint(news), facts_text, role)

    async def _relevance(self, news: dict, facts_text: str, role: str) -> Relevance:
        if not getattr(self, '_stage_cache_on', False):
            return await self.relevance.ascore(news, facts_text, role)
        key = self._relevance_key(news, facts_text, role)
        if (hit := self._cached('relevance', key)) is not None:
            return Relevance(**hit)
        value = await self.relevance.ascore(news, facts_text, role)
        self._stored('relevance', key, value.model_dump())
        return value

    def _bank_sig(self) -> str:
        if self._bank_signature is None:
            self._bank_signature = digest(getattr(self.bank,'chunks',[]),getattr(self.bank,'rows',[]),getattr(self.bank,'segments',{})) if self.bank is not None else ''
        return self._bank_signature

    def _event_sig(self, news: dict) -> str:
        """The news and the rest of its event: a new report joining the event changes what the judge can read."""
        if not isinstance(self.related, RelatedNews):
            return ''
        members = self.related.members.get(self.related.event_of.get(news['id']), [])
        return digest(sorted(article_fingerprint(self.related.get(i)) for i in members if i in self.related.row_of))

    async def _judge(self, news: dict, stage4: dict) -> InsightDecision:
        """The judge's decision for what it reads: article, its event, company facts, role, bank library, stage-4 input."""
        p = self._params
        if not getattr(self, '_stage_cache_on', False):
            return await self.judge.adecide(news['id'], p['company_id'], p['facts'], p['role'], stage4)
        key = digest('judge', self._version['judge'], article_fingerprint(news), self._event_sig(news), p['facts'], p['role'],
                     p['company_id'], self._bank_sig(), date.today().isoformat(), self._data_signature(), self._relevance_key(news, '\n'.join(p['key_facts']), p['role']),
                     stage4)
        if (hit := self._cached('judge', key)) is not None:
            decision = InsightDecision.model_validate(hit['decision'])  # tool-collected fields are excluded from model_dump
            decision.calculation_facts, decision.calculation_bank, decision.bank_evidence = (
                hit['calculation_facts'], hit['calculation_bank'], hit['bank_evidence'])
            return decision
        decision = await self.judge.adecide(news['id'], p['company_id'], p['facts'], p['role'], stage4)
        self._stored('judge', key, dict(decision=decision.model_dump(), calculation_facts=decision.calculation_facts,
                                        calculation_bank=decision.calculation_bank, bank_evidence=decision.bank_evidence))
        return decision

    async def _written(self, company_id: str, role: str, decision: InsightDecision, evidence: Evidence) -> tuple[Insight, Review]:
        if not getattr(self, '_stage_cache_on', False):
            return await self.writer.awrite(company_id, role, decision.model_dump_json(), evidence)
        key = digest('writer', self._version['writer'], company_id, role, decision.model_dump_json(),
                     [article_fingerprint(n) for n in evidence.news], evidence.facts, evidence.bank, evidence.calculations, evidence.origin, sorted(evidence.full_bank_ids))
        if (hit := self._cached('writer', key)) is not None:
            return Insight.model_validate(hit['insight']), Review.model_validate(hit['review'])
        insight, review = await self.writer.awrite(company_id, role, decision.model_dump_json(), evidence)
        if review.ok:  # a rejected draft is retried next time
            self._stored('writer', key, dict(insight=insight.model_dump(), review=review.model_dump()))
        return insight, review

    def _record(self, news: dict, **fields) -> dict:
        result = {'news_id': news['id'], 'title': news['title'], **fields}
        result['matched_fragments'] = news.get('matched_fragments', [])
        if 'event_id' in news:
            result['retrieval'] = {k: news[k] for k in ('event_id', 'event_news_ids', 'generators')}
        if isinstance(self.related, RelatedNews):
            result['credibility']=for_result(news,self.related.same_event(news['id']).to_dict('records'),
                                             self.related.event_plausibility(news['id']))
            result = bind_result(result, self.related.news, allow_unresolved=True)
        return result

    @staticmethod
    def _model_error(error: StructuredOutputError, stage: str) -> dict:
        return {'stage': stage, 'schema': error.schema, 'attempts': error.attempts, 'message': str(error)}

    async def _assess_node(self, item: dict) -> dict:
        """Stage 4 for one news: (idx, priority, news, stage4) — priority None marks a model error."""
        idx, news = item['idx'], item['news']
        p = self._params

        async def body():
            plausibility = self.related.event_plausibility(news['id'])
            stage = 'relevance'
            try:
                relevance = await self._relevance(news, '\n'.join(p['key_facts']), p['role'])
                stage = 'consequences'
                stage4 = {'matched_fragments': news.get('matched_fragments', []), 'relevance': relevance.model_dump(),
                          'consequences': await self.consequences.aextract(news), 'plausibility': plausibility}
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

        # the task's own copy of the run context (Langfuse session, news snapshot), as the thread version had
        priority, news, out = await asyncio.get_running_loop().create_task(body(), context=item['ctx'])
        progress.tick('analyzing')
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
        progress.expect(len(top))
        progress.emit('ranked',rows=[((0,n),row) for n,row in enumerate(error_rows)])
        return {'news_results': error_rows, 'top': [(priority, news, stage4) for _, priority, news, stage4 in top]}

    async def _judge_node(self, item: dict) -> dict:
        """Stage 5 for one top news: the judge's decision; the writer waits for the duplicate gate."""
        result, news, decision = await asyncio.get_running_loop().create_task(self._decide(item), context=item['ctx'])
        return {'decided': [(item['pos'], news, result, decision)]}

    async def _decide(self, item: dict) -> tuple[dict, dict, object]:
        """(result, news, decision if it is an insight else None); a model error is the result row's status."""
        p = self._params
        news, result = item['news'], {'stage4': item['stage4']}
        try:
            decision = await self._judge(news,item['stage4'])
        except Exception as error:  # a dropped connection or a timeout loses this news, not the whole batch
            result.update(status='model_error', error=self._model_error(error, 'judge') if isinstance(error, StructuredOutputError)
                          else {'stage': 'judge', 'type': type(error).__name__, 'message': str(error)[:500]})
            return result, news, None
        result['decision'] = decision.model_dump()
        if decision.bank_evidence:
            result['judge_bank'] = decision.bank_evidence
        if decision.calculation_facts:
            result['calculations'] = decision.calculation_facts
            result['calculation_bank'] = decision.calculation_bank
        return result, news, decision if decision.is_insight else None

    async def _gate_node(self, state: _NewsState) -> dict:
        """Join of the judges: duplicate positive decisions (same event or bank forecast and the same consequence) are
        merged before any writer runs — the best-scored one stays and gets the others' news; the rest become
        `duplicate` rows. Rows without a positive decision are final here."""
        decided = sorted(state.get('decided') or [], key=lambda t: t[0])
        rows = [(pos, self._record(news, **result)) for pos, news, result, decision in decided if decision is None]
        positive = [t for t in decided if t[3] is not None]
        by_pos = {str(t[0]): t for t in positive}
        try:  # fewer than two positive decisions: nothing to merge, no call
            groups = await self.dedup.agroups([self._gate_item(*t) for t in positive], self._params['role']) \
                if len(positive) > 1 else []
        except Exception as error:  # no gate verdict: every positive decision is written, as before the gate
            from common.llm.log import log_tool
            log_tool('dedup_gate_error', {}, f'{type(error).__name__}: {error}')
            groups = []
        dropped = set()
        for group in groups:
            members = [by_pos[i] for i in group]
            pos, news, result, decision = min(members, key=lambda t: (-t[3].score, t[0]))
            losers = [t for t in members if t[0] != pos]
            merged = [n['id'] for _, n, _, _ in losers] + decision.related_news_ids + \
                     [i for _, _, _, d in losers for i in d.related_news_ids]
            decision.related_news_ids = [i for i in dict.fromkeys(merged) if i != news['id']]
            result.update(decision=decision.model_dump(), merged_duplicates=[n['id'] for _, n, _, _ in losers])
            for lpos, lnews, lresult, _ in losers:
                rows.append((lpos, self._record(lnews, **lresult, status='duplicate', duplicate_of=news['id'])))
                dropped.add(lpos)
        for pos, row in rows:
            if (p := progress.current()) is not None: p.phase(pos,'done')
            progress.emit('row',key=(1,pos),row=row)
        return {'judge_rows': rows, 'to_write': [t for t in positive if t[0] not in dropped]}

    def _gate_item(self, pos: int, news: dict, result: dict, decision) -> dict:
        related = [self.related.get(i)['title'] for i in decision.related_news_ids[:3] if i in self.related.row_of]
        return {'id': str(pos), 'kind': 'news', 'news_titles': [news['title'], *related],
                'company_link': decision.company_link, 'mechanism': decision.mechanism}

    async def _write_node(self, item: dict) -> dict:
        """Stage 6 for one positive decision: writer -> checker -> (one fix) -> audit, as one result row."""
        async def tracked():
            with progress.track(item['pos']):
                return await self._write(item)
        result, news = await asyncio.get_running_loop().create_task(tracked(), context=item['ctx'])
        row = self._record(news, **result)
        progress.emit('row',key=(1,item['pos']),row=row)
        return {'judge_rows': [(item['pos'], row)]}

    async def _write(self, item: dict) -> tuple[dict, dict]:
        p = self._params
        company_id, role, facts, all_facts = p['company_id'], p['role'], p['facts'], p['all_facts']
        news, result, decision = item['news'], dict(item['result']), item['decision']
        calculations, stage = decision.calculation_facts, 'writer'
        try:
            # Preserve the original numbering for discussion even if company tables later change.
            result['evidence_facts'] = list(all_facts)
            related = [self.related.get(i) for i in decision.related_news_ids[:3] if i in self.related.row_of]
            # Reuse the exact sources read by the judge (hits, neighbours, compared tables).
            # A fresh mechanism query is only a fallback when the judge read no bank material.
            bank = decision.bank_evidence or ((await asyncio.to_thread(self.bank.search, decision.mechanism)) if self.bank else [])
            bank = list({b.get('id') or f'bank:{b["report"]}:{b["page"]}': b
                         for b in [*bank, *decision.calculation_bank]}.values())
            evidence=Evidence(news=[news,*related],facts=all_facts,bank=bank,calculations=calculations,
                              full_bank_ids={b['id'] for b in [*decision.bank_evidence,*decision.calculation_bank]})
            insight,review=await self._written(company_id,role,decision,evidence)
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
                progress.phase('audit')
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
    if pers is not None and pers.answers:
        facts = [*facts, *pers.answer_facts()]
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


def analytics_rows(run, error: str, news_results: list[dict], share: float, cache_only: bool,
                   gate: DedupGate | None = None, role: str = '') -> list[dict]:
    """The rows of the analytics part of the result list: the insights within the quota next to the ready news insights + the run row.
    With a duplicate gate, analytics insights that repeat a news insight or each other are dropped first (the news one
    and the best-ranked analytics one stay). No run (error, or cache only without a cached run): no rows, the reason is
    printed; the news results stay as they are."""
    from stage5_insight_detection.analytics_insights.service import result_rows, select
    if run is None:
        print('analytics: ' + (error or ('cache only: no cached run for this company and role' if cache_only else 'no run')))
        return []
    ready = [r for r in news_results if r.get('status') == 'ready' and r.get('insight')]
    insights = analytics_dedup(run.insights, ready, gate, role) if gate else run.insights
    return [enrich(row,run.company_id) for row in result_rows(run, select(insights, len(ready), share))]


def analytics_dedup(insights: list, news_rows: list[dict], gate: DedupGate, role: str) -> list:
    """Analytics insights without the duplicates of news insights and of each other (one gate call); no verdict: all."""
    from stage5_insight_detection.analytics_insights.service import select
    items = [{'id': f'n{k}', 'kind': 'news', 'company_link': (r.get('decision') or {}).get('company_link', ''),
              'mechanism': (r.get('decision') or {}).get('mechanism', ''),
              'news_titles': [n.get('title', '') for n in (r.get('news_references') or {}).values()] or [r.get('title', '')]}
             for k, r in enumerate(news_rows)]
    items += [{'id': f'a{k}', 'kind': 'analytics', 'title': i.title,
               'thesis': i.why_it_matters[-1].text if i.why_it_matters else ''} for k, i in enumerate(insights)]
    try:
        groups = gate.groups(items, role)  # sync: the merge node runs outside the news graph's event loop
    except Exception as error:
        from common.llm.log import log_tool
        log_tool('dedup_gate_error', {'stage': 'analytics'}, f'{type(error).__name__}: {error}')
        return insights
    drop = set()
    for group in groups:
        mine = [insights[int(i[1:])] for i in group if i.startswith('a')]
        keep = [] if len(mine) < len(group) else select(mine, 0)[:1]  # a news insight in the group wins
        drop |= {id(i) for i in mine if all(i is not k for k in keep)}
    return [i for i in insights if id(i) not in drop]


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
        progress.enter('finalizing')
        results = [migrate_fn(r) for r in s['news_results']]  # contract v2 (a superset of the old one)
        results += analytics_rows_fn(s['analytics'], s['analytics_error'], results) if with_analytics else []
        progress.emit('merged',rows=results)
        return {'results': results}

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
    # 15 events with a quota per candidate generator (stage3_news_search/retrieval.py), all of them go to the judge
    ap.add_argument('--candidates', type=int, default=15)
    ap.add_argument('--judge', type=int, default=15)
    ap.add_argument('--workers', type=int, default=16, help='judge engine: news processed in parallel (LLM calls in flight); 16 = all 15 judged news at once')
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

            pipe = InsightPipeline(models.cheap(), models.agent(), bank=bank)
            graph = build_e2e_graph(
                pipe,
                with_analytics=a.source == 'both',
                run_analytics_fn=_run_analytics,
                analytics_rows_fn=lambda run, error, rows: analytics_rows(run, error, rows, a.analytics_share,
                                                                         a.analytics_cache_only, pipe.dedup, a.role),
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
