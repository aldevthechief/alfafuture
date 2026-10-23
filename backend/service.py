"""Application orchestration, independent pools and version binding for every analysis job."""
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
    def __init__(self,root=None,models_factory=ChatModels,processor=None,agent_workers=4,pdf_workers=2):
        self.root=Path(root) if root is not None else DATA/'api'
        self.models_factory=models_factory
        self.catalog=Catalog(self.root/'bank',processor)
        self.agents=Jobs(self.root/'analyses',agent_workers)
        self.pdfs=Jobs(self.root/'pdf_jobs',pdf_workers)

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
        with news_snapshot():
            news=news_pool() if r.source!='analytics' else None
            if news is not None:
                from common.news_corpus import archive_corpus
                archive_corpus(news,DATA/'news')
            prepared=PreparedBank(library,dict(status='complete',version=version,reports=len(bank.parts)))
            def run_news():
                if r.source=='analytics': return []
                if r.engine=='loop':
                    return pipeline.run_loop(r.company_id,r.role,models,r.candidates,r.judge,retrieval=r.retrieval,
                        bank_review=r.bank_review,bank_budget=r.bank_budget,prepared_bank=prepared,
                        bank_insights_enabled=r.bank_insights,workers=r.workers,cards_dir=run_dir/'cards')
                return pipeline.InsightPipeline(models.cheap(),models.agent(),bank=bank).run(
                    r.company_id,r.role,r.candidates,r.judge,retrieval=r.retrieval,workers=r.workers)
            if r.source=='news':
                results=run_news()
            else:
                # Copy the pinned news context to the independent analytics task.
                import contextvars
                with ThreadPoolExecutor(1) as executor:
                    def analytics():
                        from stage2_company_facts.profile.integration import load_company
                        from stage5_insight_detection.analytics_insights.service import analytics_run
                        # Snapshot-specific cache avoids serving an analysis from a different set of reports.
                        try:
                            run=analytics_run(r.company_id,r.role,load_company(r.company_id).facts,models=models,bank=bank,
                                cache_only=r.analytics_cache_only,refresh=r.analytics_refresh,
                                cache_dir=self.root/'analytics_cache'/version)
                            return run,''
                        except Exception as exc:
                            return None,f'{type(exc).__name__}: {exc}'
                    future=executor.submit(contextvars.copy_context().run,analytics)
                    results=run_news()
                    run,error=future.result()
                    results+=pipeline.analytics_rows(run,error,results,r.analytics_share,r.analytics_cache_only)
                    if r.source=='analytics' and error: raise RuntimeError(error)
            results=[pipeline.migrate_result(row,r.company_id) for row in results]
            if news is not None:
                save_results(run_dir/'results.json',results,news,DATA/'news')
            return dict(company_id=r.company_id,role=r.role,bank_version=version,
                        corpus_fingerprint=corpus_fingerprint(news) if news is not None else None,results=results)

    def close(self):
        # Analysts may wait on the PDF pool; stop them first.
        self.agents.close(); self.pdfs.close()
