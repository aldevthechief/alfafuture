"""Application orchestration, independent pools and version binding for every analysis job.

The judge engine goes through the LangGraph E2E (pipeline.build_e2e_graph): the news engine and the
bank-analytics branch run in parallel in one code path with the CLI, and a SqliteSaver checkpointer
(thread_id = run id) lets a job resume after a crash instead of starting over."""
from pathlib import Path
from uuid import uuid4

from common.data import DATA
from common.llm import ChatModels, PromptLibrary
from common.news_corpus import save_results
from common.news_snapshot import news_snapshot
from common.personalization import load_personalization
from common.search import news_pool, corpus_fingerprint
from backend.catalog import Catalog
from backend.jobs import Jobs
from backend.progress import RunProgress
from backend.insight_previews import rows_of
from common import progress
from stage5_insight_detection.bank_analytics.ocr import PDF_DIR
from stage5_insight_detection.bank_analytics.prepare import PreparationOptions

PROMPTS = PromptLibrary(Path(__file__).with_name('prompts'))


class Service:
    def __init__(self,root=None,models_factory=ChatModels,processor=None,agent_workers=4,pdf_workers=2,
                 checkpointing=False,auto_compact=False):
        self.root=Path(root) if root is not None else DATA/'api'
        self.models_factory=models_factory
        self.catalog=Catalog(self.root/'bank',processor)
        self.agents=Jobs(self.root/'analyses',agent_workers)
        from backend.calendar_index import maybe_index_job
        self.agents.on_finish=maybe_index_job  # calendar SQLite index per finished analysis; never breaks the pool
        self.pdfs=Jobs(self.root/'pdf_jobs',pdf_workers)
        self.exports=Jobs(self.root/'exports',2)
        self._checkpointing=checkpointing
        self._checkpointer=None
        self._compact_locks={}; self.auto_compact=auto_compact

    @property
    def checkpointer(self):
        """SqliteSaver over api/checkpoints.sqlite (absent -> None: the graph runs without resume)."""
        if not self._checkpointing:
            return None
        if self._checkpointer is None:
            try:
                import sqlite3
                from langgraph.checkpoint.sqlite import SqliteSaver
                self.root.mkdir(parents=True, exist_ok=True)
                self._conn = sqlite3.connect(str(self.root / 'checkpoints.sqlite'), check_same_thread=False)
                self._checkpointer = SqliteSaver(self._conn)
            except Exception:
                self._checkpointer = False
        return self._checkpointer or None

    def prepare(self,path,options,seed_dir=None):
        path=Path(path)
        return self.pdfs.submit('report',dict(file=path.name,options=options.__dict__),
            lambda:self.catalog.prepare(path,options,self.models_factory(),seed_dir),start='preparing_reports')

    def analyze(self,request):
        job=self.agents.submit('analysis',request.model_dump(),lambda:self.run(request))
        if self.auto_compact:  # screen versions right after the analysis, in their own thread: the agent worker is free
            import threading
            self.agents.futures[job['id']].add_done_callback(
                lambda _, id=job['id']: threading.Thread(target=self.compact,args=(id,),daemon=True).start())
        return job

    def run(self,r):
        import pipeline
        progress.listen(RunProgress(progress.results,lambda row:pipeline.migrate_result(row,r.company_id)))
        prep=r.bank_prepare or r.bank_charts or r.bank_tables or r.bank_digest
        progress.plan([*(['preparing_reports'] if prep or r.ocr else []),
                       *(['analyzing','finalizing'] if r.source=='analytics' else progress.DEFAULT_PLAN)])
        personalization = load_personalization(r.user_id, r.company_id, r.role) if r.user_id else None
        models=self.models_factory()
        run_dir=self.root/'results'/uuid4().hex
        prep=r.bank_prepare or r.bank_charts or r.bank_tables or r.bank_digest
        if prep or r.ocr:
            progress.enter('preparing_reports')
            options=PreparationOptions(ocr=r.ocr or prep,charts=r.bank_prepare or r.bank_charts,tables=r.bank_prepare or r.bank_tables,
                digests=r.bank_prepare or r.bank_digest,chart_mode=r.chart_mode,
                ocr_budget=15 if r.ocr_budget is None else r.ocr_budget,
                charts_budget=8 if r.charts_budget is None else r.charts_budget,
                tables_budget=7 if r.tables_budget is None else r.tables_budget,
                digest_budget=3 if r.digest_budget is None else r.digest_budget)
            ids=[self.prepare(path,options,PDF_DIR)['id'] for path in sorted(PDF_DIR.glob('*.pdf'))]
            for id in ids:
                job=self.pdfs.wait(id)
                if job['status']!='succeeded': raise RuntimeError(job['error'])
        # Preparation finishes before pinning the search view used by both judge and analytics.
        version,_,bank=self.catalog.snapshot(models)
        components={'news':{'status':'skipped'},'analytics':{'status':'skipped'}}
        errors=[]
        with news_snapshot():
            news=news_pool() if r.source!='analytics' else None
            if news is not None:
                from common.news_corpus import archive_corpus
                archive_corpus(news,DATA/'news')

            def analytics_run_task():
                progress.branch('analytics','running')
                if r.source=='analytics': progress.enter('analyzing',label='Разбираем аналитику банка')
                # Snapshot-specific cache avoids serving an analysis from a different set of reports.
                from stage2_company_facts.profile.integration import load_company
                from stage5_insight_detection.analytics_insights.service import analytics_run
                try:
                    run=analytics_run(r.company_id,r.role,load_company(r.company_id).facts,models=models,bank=bank,
                        cache_only=r.analytics_cache_only,refresh=r.analytics_refresh,
                        cache_dir=self.root/'analytics_cache'/version)
                    progress.branch('analytics','failed' if getattr(run,'error','') else 'done',getattr(run,'error','') or None)
                    return run,''
                except Exception as exc:
                    progress.branch('analytics','failed',f'{type(exc).__name__}: {exc}')
                    return None,f'{type(exc).__name__}: {exc}'

            def note_analytics(run,error):
                error=error or (getattr(run,'error','') if run is not None else '')
                if error:
                    components['analytics']={'status':'failed','error':error}
                    errors.append({'component':'analytics','message':error})
                    if r.source=='analytics': raise RuntimeError(error)
                elif run is None:
                    components['analytics']={'status':'cache_miss' if r.analytics_cache_only else 'empty'}
                else:
                    components['analytics']={'status':'succeeded'}

            def judge_engine():
                pipe=pipeline.InsightPipeline(models.cheap(),models.agent(),bank=bank)
                graph=pipeline.build_e2e_graph(
                    pipe,
                    with_analytics=r.source=='both',
                    run_analytics_fn=analytics_run_task,
                    analytics_rows_fn=lambda run,error,rows:
                        [] if (run is None or error or getattr(run,'error',''))
                        else pipeline.analytics_rows(run,'',rows,r.analytics_share,r.analytics_cache_only,pipe.dedup,r.role),
                    migrate_fn=lambda row:pipeline.migrate_result(row,r.company_id),
                    save_fn=lambda rows:(save_results(run_dir/'results.json',rows,news,DATA/'news'),
                                         str(run_dir/'results.json'))[1],
                    checkpointer=self.checkpointer)
                state=graph.invoke({'company_id':r.company_id,'role':r.role,'n_candidates':r.candidates,
                    'n_judge':r.judge,'retrieval':r.retrieval,'workers':r.workers,
                    'personalization':personalization.model_dump() if personalization is not None else None},
                    config={'max_concurrency':max(1,r.workers),'configurable':{'thread_id':run_dir.name}})
                if r.source=='both':
                    note_analytics(state.get('analytics'),state.get('analytics_error',''))
                return state['results']

            if r.source=='analytics':
                run,error=analytics_run_task()
                note_analytics(run,error)
                results=pipeline.analytics_rows(run,'',[],r.analytics_share,r.analytics_cache_only) if run is not None else []
                progress.enter('finalizing')
                progress.emit('merged',rows=results)
            else:
                results=judge_engine()
                components['news']={'status':'succeeded'}
            if news is not None and (run_dir/'results.json').exists() is False:
                save_results(run_dir/'results.json',results,news,DATA/'news')
            return dict(status='partial' if errors else 'succeeded',components=components,errors=errors,
                        company_id=r.company_id,role=r.role,user_id=r.user_id,bank_version=version,
                        corpus_fingerprint=corpus_fingerprint(news) if news is not None else None,results=results)

    async def ask(self, company_id: str, role: str, question: str, insight_title: str | None = None,
                  user_id: str | None = None):
        """Streamed answer to a reader's question about an insight: the saved result rows are the context.
        Async end to end (model.astream): the event loop streams tokens while the call is in flight."""
        from langchain_core.messages import SystemMessage, HumanMessage
        from common.llm.cache import astream_cached
        from common.roles import role_context
        messages = [SystemMessage(content=PROMPTS.render('ask_system', **role_context(role))),
                    HumanMessage(content=self.ask_context(company_id, role, question, insight_title, user_id))]
        # low reasoning: at the default GLM reasons ~60 s before the first token of a 3-6 sentence answer (24.10)
        model = self.models_factory().agent(temperature=0.0).model_copy(update={'reasoning_effort': 'low'})
        async for chunk in astream_cached(model, messages):  # a repeated question comes from the cache at once
            yield chunk

    def ask_context(self, company_id: str, role: str, question: str, insight_title: str | None = None,
                    user_id: str | None = None) -> str:
        """The material for a question about one insight: the insight, the news it came from, the company's key facts,
        the bank analytics found for the question itself (so the answer can go beyond the insight text) and the user's
        own notes from feedback_api (unverified, marked as such: the block the pipeline uses)."""
        import json
        rows = self.latest_rows(company_id, role, user_id)
        ready = [r for r in rows if r.get('insight')]
        row = next((r for r in ready if r['insight'].get('title') == insight_title), ready[0] if ready else {})
        keep = ('title', 'what_happened', 'why_it_matters', 'risks', 'opportunities', 'actions', 'bank_view', 'ai_conclusion',
                'hypotheses', 'impact', 'trigger')
        parts = ['Инсайт:\n' + json.dumps({k: v for k, v in row.get('insight', {}).items() if k in keep}, ensure_ascii=False, default=str)[:12000]]
        if row.get('news_id') is not None:
            news = news_pool()
            hit = news[news['id'] == row['news_id']]
            if len(hit):
                n = hit.iloc[0]
                parts.append(f"Новость-источник ({n['date']}, {n['source']}): {n['title']}\n{str(n['text'])[:4000]}")
        try:
            from stage2_company_facts.profile.integration import load_company
            facts = [f for f in load_company(company_id).facts if f.get('depth') == 1][:25]
            parts.append('Факты о компании:\n' + '\n'.join(f"- {f['text']}" for f in facts))
        except Exception:
            pass
        try:
            from stage5_insight_detection.bank_analytics.tools import BankTools
            _, _, bank = self.catalog.snapshot(self.models_factory())
            parts.append('Аналитика Банка по вопросу:\n' + BankTools(bank).search_bank_analytics(question)[:8000])
        except Exception:
            pass
        if user_id:
            try:
                from pipeline import personalization_context
                from common.personalization import load_personalization
                if note := personalization_context(load_personalization(user_id, company_id, role)):
                    parts.append(note)
            except Exception:
                pass
        return '\n\n'.join(parts) + f'\n\nВопрос: {question}'

    def compact(self, job_id: str, job=None, cache_only=False) -> dict:
        """Screen versions of a finished job's insights, {insight id: compact or None (UI shows the full one)}.
        Computed once per insight (GLM, low reasoning, all in parallel) and kept in api/compact/<job>.json. Called
        right after an analysis finishes (auto_compact) and by the API; a per-job lock keeps it to one computation."""
        import asyncio, threading
        with self._compact_locks.setdefault(job_id, threading.Lock()):
            return asyncio.run(self._compact(job_id, job=job, cache_only=cache_only))

    async def _compact(self, job_id: str, job=None, cache_only=False) -> dict:
        import asyncio, json
        from stage6_insight_generation.insight_compact.compact import InsightCompactor
        path = self.root / 'compact' / f'{job_id}.json'
        results = rows_of(job if job is not None else self.agents.get(job_id))
        # the id the UI gives the card (backend/insight_previews.py): its own id, else the row index in the results
        ids = {id(r): r['insight'].get('id') or f"row-{n}-{r.get('news_id', 'bank')}"
               for n, r in enumerate(results) if r.get('status') == 'ready' and r.get('insight')}
        rows = [r for r in results if id(r) in ids]
        key = lambda r: ids[id(r)]
        done = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        if not cache_only and (todo := [r for r in rows if key(r) not in done]):
            model = self.models_factory().agent(temperature=0.0).model_copy(update={'reasoning_effort': 'low'})
            compactor = InsightCompactor(model)
            found = await asyncio.gather(*(compactor.acompact(r['insight']) for r in todo), return_exceptions=True)
            from common.llm.log import log_tool
            for r, c in zip(todo, found):
                if isinstance(c, BaseException):
                    log_tool('compact_error', {'title': r['insight'].get('title')}, f'{type(c).__name__}: {c}')
            done |= {key(r): None if isinstance(c, BaseException) else c for r, c in zip(todo, found)}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(done, ensure_ascii=False), encoding='utf-8')
        return {key(r): done[key(r)] for r in rows if key(r) in done}

    def latest_rows(self, company_id: str, role: str, user_id: str | None = None) -> list[dict]:
        """Result rows of the newest finished backend analysis of this pair that this user may see: their own first,
        then a shared one, as the UI picks it (api.js latestJobs); the CLI output file as a fallback."""
        import json
        done = [j for j in self.agents.items.values() if j['status'] in ('succeeded', 'partial')
                and j['request'].get('company_id') == company_id and j['request'].get('role') == role]
        mine = [j for j in done if user_id and j['request'].get('user_id') == user_id]
        shared = [j for j in done if not j['request'].get('user_id')]
        if mine or shared:
            return max(mine or shared, key=lambda j: j['created_at'])['result']['results']
        path = DATA / 'insights' / f'{company_id}_{role}.json'
        return json.loads(path.read_text(encoding='utf-8')) if path.exists() else []

    # --- saved-data exports; independent of the agent and PDF pipelines
    def case_tables(self):
        from common.data import load
        return load()

    def may_see(self,user_id,company_id,staff=False):
        """Optional demo access file; X-User-ID follows the existing development identity contract."""
        import json
        path=self.root/'access.json'
        if staff or not path.exists(): return True
        allowed=set(json.loads(path.read_text(encoding='utf-8')).get(user_id or '',[]))
        if company_id in allowed: return True
        companies=self.case_tables()[0]
        return company_id in companies.index and companies.at[company_id,'GROUP_ID'] in allowed

    def export_tables(self,q,user_id=None,staff=False):
        from backend import exports
        from backend.cabinet import cabinet_feed, cached_job
        if q.kind!='insights':
            U,P,S,F=self.case_tables()
            if q.company_id not in U.index or not self.may_see(user_id,q.company_id,staff): raise KeyError(q.company_id)
            ids=exports.group_members(U,q.company_id) if q.scope=='group' else [q.company_id]
            if any(not self.may_see(user_id,company,staff) for company in ids): raise KeyError(q.company_id)
            tables=exports.payment_tables(U,P,S,ids,q.side) if q.kind=='payments' else exports.counterparty_tables(U,P,F,ids,q.top)
            return tables,dict(company_id=q.company_id,scope=q.scope,companies=len(ids),period=None,
                               group_id=str(U.at[q.company_id,'GROUP_ID'])),\
                f'alfa-{q.kind}_{q.company_id}' + ('_group' if q.scope=='group' else '')

        entries=[]
        if q.job_id:
            job=cached_job(q.job_id,self.root) if q.job_id.startswith('demo-') else self.agents.get(q.job_id)
            request=job['request']
            if request.get('user_id') not in (None,user_id): raise KeyError(q.job_id)
            company=request['company_id']; role=request['role']
            if (q.company_id and q.company_id!=company) or (q.role and q.role!=role): raise KeyError(q.job_id)
            from backend.insight_previews import preview_feed
            cards=preview_feed(job)['insights']
            entries=[(job, rows_of(job)[card['result_index']], card['id']) for card in cards]
        else:
            company=q.company_id; role=q.role or 'cfo'
            with self.agents.lock:
                jobs=tuple(dict(job) for job in self.agents.items.values())
            feed=cabinet_feed(company,role,jobs,user_id,self.root)
            by_id={job['id']:job for job in jobs}
            for card in feed['insights']:
                job=by_id.get(card['job_id'])
                if job is None:
                    job=cached_job(card['job_id'],self.root)
                    by_id[card['job_id']]=job
                entries.append((job,rows_of(job)[card['result_index']],card['id']))
        if not self.may_see(user_id,company,staff): raise KeyError(company)
        merged={}; selected={}
        for job,row,card_id in entries:
            insight=row.get('insight') or {}
            bank=row.get('source')=='bank_analytics' or insight.get('origin')=='bank'
            ref=(row.get('news_references') or {}).get(f"news:{row.get('news_id')}") or {}
            when=str(insight.get('event_date') or ref.get('date') or '')[:10]
            if q.origin!='all' and (q.origin=='bank')!=bank: continue
            if q.date_from and (not when or when<q.date_from): continue
            if q.date_to and (not when or when>q.date_to): continue
            row=dict(row,insight=dict(insight,id=card_id))
            for name,(columns,rows) in exports.insight_tables([row],job=job,company_id=company,role=role).items():
                merged.setdefault(name,(columns,[]))[1].extend(rows)
            selected[job['id']]=job
        if not merged or not merged['insights'][1]: raise KeyError(company)
        versions={str((job.get('result') or {}).get('bank_version')) for job in selected.values()
                  if (job.get('result') or {}).get('bank_version')}
        fingerprints={str((job.get('result') or {}).get('corpus_fingerprint')) for job in selected.values()
                      if (job.get('result') or {}).get('corpus_fingerprint')}
        return merged,dict(company_id=company,role=role,origin=q.origin,
                           job_id=next(iter(selected)) if len(selected)==1 else None,job_ids=list(selected),
                           bank_version=next(iter(versions)) if len(versions)==1 else None,bank_versions=sorted(versions),
                           corpus_fingerprints=sorted(fingerprints)),f'alfa-insights_{company}_{role}'

    def export_batch(self,batch,user_id=None,staff=False):
        from backend import exports
        def work():
            merged={};metas=[]
            for q in batch.items:
                tables,meta,_=self.export_tables(q,user_id,staff)
                metas.append(meta)
                for name,(columns,rows) in tables.items():
                    merged.setdefault(name,(columns,[]))[1].extend(rows)
            kinds=sorted({q.kind for q in batch.items})
            data,media,ext=exports.render(merged,exports.meta_for(kinds[0],items=metas,kinds=kinds),batch.format,
                                          decimal=batch.decimal)
            folder=self.root/'exports'/'files';folder.mkdir(parents=True,exist_ok=True)
            from datetime import date
            path=folder/(uuid4().hex+'.'+ext);path.write_bytes(data)
            return dict(file=path.relative_to(self.root).as_posix(),name=f'alfa-export_{date.today().isoformat()}.{ext}',
                        media_type=media,bytes=len(data),rows={k:len(v[1]) for k,v in merged.items()})
        return self.exports.submit('export',dict(batch.model_dump(),user_id=user_id),work)

    def close(self):
        # Analysts may wait on the PDF pool; stop them first.
        self.agents.close(); self.pdfs.close(); self.exports.close()
        conn = getattr(self, '_conn', None)
        if conn is not None:
            conn.close()
