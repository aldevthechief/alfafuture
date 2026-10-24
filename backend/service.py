"""Application orchestration, independent pools and version binding for every analysis job.

The judge engine goes through the LangGraph E2E (pipeline.build_e2e_graph): the news engine and the
bank-analytics branch run in parallel in one code path with the CLI, and a SqliteSaver checkpointer
(thread_id = run id) lets a job resume after a crash instead of starting over."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

from common.data import DATA
from common.llm import ChatModels
from common.news_corpus import save_results
from common.news_snapshot import news_snapshot
from common.search import news_pool, corpus_fingerprint
from backend.catalog import Catalog
from backend.jobs import Jobs
from stage5_insight_detection.bank_analytics.ocr import PDF_DIR
from stage5_insight_detection.bank_analytics.prepare import PreparationOptions, PreparedBank

class Service:
    def __init__(self,root=None,models_factory=ChatModels,processor=None,agent_workers=4,pdf_workers=2,
                 checkpointing=False):
        self.root=Path(root) if root is not None else DATA/'api'
        self.models_factory=models_factory
        self.catalog=Catalog(self.root/'bank',processor)
        self.agents=Jobs(self.root/'analyses',agent_workers)
        self.pdfs=Jobs(self.root/'pdf_jobs',pdf_workers)
        self._checkpointing=checkpointing
        self._checkpointer=None

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
            lambda:self.catalog.prepare(path,options,self.models_factory(),seed_dir))

    def analyze(self,request):
        return self.agents.submit('analysis',request.model_dump(),lambda:self.run(request))

    def run(self,r):
        import pipeline
        models=self.models_factory()
        run_dir=self.root/'results'/uuid4().hex
        prep=r.bank_prepare or r.bank_charts or r.bank_tables or r.bank_digest
        if prep or r.ocr:
            options=PreparationOptions(ocr=r.ocr or prep,charts=r.bank_prepare or r.bank_charts,tables=r.bank_prepare or r.bank_tables,
                digests=r.bank_prepare or r.bank_digest,
                ocr_budget=15 if r.ocr_budget is None else r.ocr_budget,
                charts_budget=8 if r.charts_budget is None else r.charts_budget,
                tables_budget=7 if r.tables_budget is None else r.tables_budget,
                digest_budget=3 if r.digest_budget is None else r.digest_budget)
            ids=[self.prepare(path,options,PDF_DIR)['id'] for path in sorted(PDF_DIR.glob('*.pdf'))]
            for id in ids:
                job=self.pdfs.wait(id)
                if job['status']!='succeeded': raise RuntimeError(job['error'])
        version,library,bank=self.catalog.snapshot(models)
        components={'news':{'status':'skipped'},'analytics':{'status':'skipped'}}
        errors=[]
        with news_snapshot():
            news=news_pool() if r.source!='analytics' else None
            if news is not None:
                from common.news_corpus import archive_corpus
                archive_corpus(news,DATA/'news')
            prepared=PreparedBank(library,dict(status='complete',version=version,reports=len(bank.parts)))

            def analytics_run_task():
                # Snapshot-specific cache avoids serving an analysis from a different set of reports.
                from stage2_company_facts.profile.integration import load_company
                from stage5_insight_detection.analytics_insights.service import analytics_run
                try:
                    run=analytics_run(r.company_id,r.role,load_company(r.company_id).facts,models=models,bank=bank,
                        cache_only=r.analytics_cache_only,refresh=r.analytics_refresh,
                        cache_dir=self.root/'analytics_cache'/version)
                    return run,''
                except Exception as exc:
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

            def loop_engine():
                results=pipeline.run_loop(r.company_id,r.role,models,r.candidates,r.judge,retrieval=r.retrieval,
                    bank_review=r.bank_review,bank_budget=r.bank_budget,prepared_bank=prepared,
                    workers=r.workers,cards_dir=run_dir/'cards')
                if r.source!='news':  # loop + analytics join sequentially (the loop owns its own pool)
                    import contextvars
                    with ThreadPoolExecutor(1) as executor:
                        future=executor.submit(contextvars.copy_context().run,analytics_run_task)
                        results=[pipeline.migrate_result(row,r.company_id) for row in results]
                        run,error=future.result()
                        note_analytics(run,error)
                        if run is not None:
                            results+=pipeline.analytics_rows(run,'',results,r.analytics_share,r.analytics_cache_only)
                        return results
                return results

            def judge_engine():
                graph=pipeline.build_e2e_graph(
                    pipeline.InsightPipeline(models.cheap(),models.agent(),bank=bank),
                    with_analytics=r.source=='both',
                    run_analytics_fn=analytics_run_task,
                    analytics_rows_fn=lambda run,error,rows:
                        [] if (run is None or error or getattr(run,'error',''))
                        else pipeline.analytics_rows(run,'',rows,r.analytics_share,r.analytics_cache_only),
                    migrate_fn=lambda row:pipeline.migrate_result(row,r.company_id),
                    save_fn=lambda rows:(save_results(run_dir/'results.json',rows,news,DATA/'news'),
                                         str(run_dir/'results.json'))[1],
                    checkpointer=self.checkpointer)
                state=graph.invoke({'company_id':r.company_id,'role':r.role,'n_candidates':r.candidates,
                    'n_judge':r.judge,'retrieval':r.retrieval,'workers':r.workers},
                    config={'max_concurrency':max(1,r.workers),'configurable':{'thread_id':run_dir.name}})
                if r.source=='both':
                    note_analytics(state.get('analytics'),state.get('analytics_error',''))
                return state['results']

            if r.source=='analytics':
                run,error=analytics_run_task()
                note_analytics(run,error)
                results=pipeline.analytics_rows(run,'',[],r.analytics_share,r.analytics_cache_only) if run is not None else []
            else:
                results=(loop_engine() if r.engine=='loop' else judge_engine())
                components['news']={'status':'succeeded'}
                if r.engine=='loop' and r.source!='news':
                    pass  # analytics already joined inside loop_engine
            if r.engine=='loop' and r.source=='news':
                results=[pipeline.migrate_result(row,r.company_id) for row in results]
            if news is not None and (run_dir/'results.json').exists() is False:
                save_results(run_dir/'results.json',results,news,DATA/'news')
            return dict(status='partial' if errors else 'succeeded',components=components,errors=errors,
                        company_id=r.company_id,role=r.role,bank_version=version,
                        corpus_fingerprint=corpus_fingerprint(news) if news is not None else None,results=results)

    def ask(self, company_id: str, role: str, question: str, insight_title: str | None = None):
        """Streamed answer to a reader's question about an insight: the saved result rows are the context."""
        import json
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.messages import SystemMessage, HumanMessage
        path = DATA / 'insights' / f'{company_id}_{role}.json'
        rows = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
        insights = [row['insight'] for row in rows if row.get('insight')]
        if insight_title:
            insights = [i for i in insights if i.get('title') == insight_title] or insights
        context = json.dumps(insights[:3], ensure_ascii=False, default=str)[:24000]
        model = self.models_factory().agent(temperature=0.0)
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                'Ты — ассистент банковской аналитики в интернет-банке Альфа-Бизнес. Отвечай на вопрос читателя '
                'по инсайтам ниже: только по их материалам, просто и по делу, на русском. Чего нет в материалах — '
                'скажи прямо и предложи, какие данные клиента это уточнят.')),
            HumanMessage(content=f'Инсайты:\n{context}\n\nВопрос: {question}'),
        ])
        yield from (text for chunk in (prompt | model | StrOutputParser()).stream({})
                    if (text := chunk or ''))

    def close(self):
        # Analysts may wait on the PDF pool; stop them first.
        self.agents.close(); self.pdfs.close()
        conn = getattr(self, '_conn', None)
        if conn is not None:
            conn.close()
