"""API, actual parallel jobs and immutable bank views; synthetic PDFs and processors, no model calls."""
import io
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pymupdf
from fastapi.testclient import TestClient
from pydantic import ValidationError
from backend.app import create_app
from backend.catalog import Catalog
from backend.jobs import Jobs, QueueFull
from backend.schemas import AnalysisRequest
from backend.service import Service
from stage5_insight_detection.bank_analytics.library import BankLibrary
from stage5_insight_detection.bank_analytics.prepare import PreparationOptions

def processor(folder,options,models):
    filename=next(folder.glob('*.pdf')).name
    bank=SimpleNamespace(rows=[],chunks=[],index={},available=[filename],missing=[],reports=[],pdf_paths={},twins={},
                         search=lambda *a,**kw:[],passages=lambda *a,**kw:[])
    return BankLibrary([]),bank

def pdf_bytes():
    with pymupdf.open() as doc:
        doc.new_page().insert_text((50,50),'Synthetic bank report')
        return doc.tobytes()

class JobTests(unittest.TestCase):
    def test_agents_and_reports_use_independent_pools_and_results_survive_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            agents=Jobs(Path(tmp)/'agents',workers=2,capacity=2)
            reports=Jobs(Path(tmp)/'reports',workers=1)
            started=threading.Barrier(3); release=threading.Event()
            def work():
                started.wait(5); release.wait(5); return {'done':True}
            try:
                first=agents.submit('analysis',{},work);second=agents.submit('analysis',{},work)
                started.wait(5)
                with self.assertRaises(QueueFull): agents.submit('analysis',{},work)
                uploaded=reports.submit('report',{},lambda:{'ready':True})
                self.assertEqual(reports.wait(uploaded['id'])['status'],'succeeded')
                self.assertEqual(agents.get(first['id'])['status'],'running')
                release.set()
                self.assertEqual(agents.wait(first['id'])['result'],{'done':True})
                self.assertEqual(agents.wait(second['id'])['status'],'succeeded')
            finally:
                release.set();agents.close();reports.close()
            restarted=Jobs(Path(tmp)/'agents')
            self.assertEqual(restarted.get(first['id'])['status'],'succeeded');restarted.close()

class CacheTests(unittest.TestCase):
    def test_different_keys_compute_concurrently_but_same_key_is_reused(self):
        from common.llm.cache import JsonlCache
        with tempfile.TemporaryDirectory() as tmp:
            cache=JsonlCache('parallel',Path(tmp));barrier=threading.Barrier(2)
            def compute(value):
                barrier.wait(5);return value
            with ThreadPoolExecutor(2) as pool:
                a=pool.submit(cache.get_or_compute,'a',lambda:compute(1))
                b=pool.submit(cache.get_or_compute,'b',lambda:compute(2))
                self.assertEqual((a.result(),b.result()),(1,2))
            other=JsonlCache('parallel',Path(tmp))
            self.assertEqual(other.get_or_compute('a',lambda:self.fail('must reuse disk cache')),1)

class CatalogTests(unittest.TestCase):
    def test_ready_report_does_not_mutate_existing_snapshot_and_cache_is_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); pdf=root/'old.pdf';pdf.write_bytes(pdf_bytes())
            make=Mock(side_effect=processor)
            catalog=Catalog(root/'catalog',make)
            catalog.prepare(pdf,PreparationOptions(),None)
            version,lib,view=catalog.snapshot(None)
            catalog.prepare(pdf,PreparationOptions(),None)
            self.assertEqual(make.call_count,1)
            new=root/'new.pdf';new.write_bytes(pdf_bytes())
            catalog.prepare(new,PreparationOptions(),None)
            newer,_,newview=catalog.snapshot(None)
            self.assertNotEqual(version,newer)
            self.assertEqual(view.available,['old.pdf'])
            self.assertEqual(set(newview.available),{'old.pdf','new.pdf'})
            restarted=Catalog(root/'catalog',processor)
            self.assertEqual(restarted.snapshot(None)[0],newer)

    def test_reports_prepare_concurrently_and_failed_report_is_not_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); barrier=threading.Barrier(2)
            def process(*args):
                barrier.wait(5);return processor(*args)
            catalog=Catalog(root/'catalog',process)
            files=[]
            for name in ['one.pdf','two.pdf']:
                p=root/name;p.write_bytes(pdf_bytes());files.append(p)
            with ThreadPoolExecutor(2) as ex:
                results=list(ex.map(lambda p:catalog.prepare(p,PreparationOptions(),None),files))
            self.assertTrue(all(r['published'] for r in results))
            catalog.processor=Mock(side_effect=RuntimeError('OCR failed'))
            p=root/'bad.pdf';p.write_bytes(pdf_bytes())
            with self.assertRaises(RuntimeError):catalog.prepare(p,PreparationOptions(),None)
            self.assertNotIn('bad.pdf',catalog.current)

class ApiTests(unittest.TestCase):
    def test_swagger_upload_health_status_and_rejected_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            service=Service(Path(tmp),models_factory=lambda:None,processor=processor)
            with TestClient(create_app(service)) as client:
                self.assertEqual(client.get('/docs').status_code,200)
                schema=client.get('/openapi.json').json()
                props=schema['components']['schemas']['AnalysisRequest']['properties']
                for name in ['engine','retrieval','source','workers','bank_prepare','ocr','bank_review',
                             'bank_insights','analytics_refresh','analytics_cache_only','analytics_share',
                             'ocr_budget','charts_budget','tables_budget','digest_budget','bank_budget']:
                    self.assertIn(name,props)
                response=client.post('/v1/reports/upload',files={'files':('report.pdf',pdf_bytes(),'application/pdf')},
                                     data={'options':'{"charts":false,"tables":false,"digests":false}'})
                self.assertEqual(response.status_code,202,response.text)
                job=response.json()['jobs'][0]
                self.assertEqual(service.pdfs.wait(job['id'])['status'],'succeeded')
                self.assertEqual(client.get('/v1/jobs/'+job['id']).json()['status'],'succeeded')
                self.assertIn('report.pdf',client.get('/v1/reports').json()['active'])
                self.assertEqual(client.get('/health').status_code,200)
                self.assertEqual(client.get('/v1/jobs/unknown').status_code,404)
                self.assertEqual(client.post('/v1/reports/upload',files={'files':('x.pdf',b'garbage','application/pdf')}).status_code,422)
                self.assertEqual(client.post('/v1/analyses',json={'company_id':'U_1','bank_prepare':True}).status_code,422)
                self.assertEqual(client.post('/v1/analyses',json={'company_id':'../unsafe'}).status_code,422)

    def test_upload_finishes_while_analysis_uses_older_bank_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            service=Service(Path(tmp),models_factory=lambda:None,processor=processor)
            path=Path(tmp)/'old.pdf';path.write_bytes(pdf_bytes());service.catalog.prepare(path,PreparationOptions(),None)
            started=threading.Event();release=threading.Event()
            def analyze(request):
                version,_,view=service.catalog.snapshot(None);started.set();release.wait(5)
                return {'version':version,'reports':view.available}
            service.run=analyze
            with TestClient(create_app(service)) as client:
                try:
                    analysis=client.post('/v1/analyses',json={'company_id':'U_1','source':'news'}).json()
                    self.assertTrue(started.wait(5))
                    upload=client.post('/v1/reports/upload',files={'files':('new.pdf',pdf_bytes(),'application/pdf')}).json()
                    service.pdfs.wait(upload['jobs'][0]['id'])
                    self.assertEqual(client.get('/v1/jobs/'+analysis['id']).json()['status'],'running')
                    release.set()
                    self.assertEqual(service.agents.wait(analysis['id'])['result']['reports'],['old.pdf'])
                    nextjob=client.post('/v1/analyses',json={'company_id':'U_1','source':'news'}).json()
                    self.assertEqual(set(service.agents.wait(nextjob['id'])['result']['reports']),{'old.pdf','new.pdf'})
                finally: release.set()

    def test_loop_parallelism_uses_separate_instances_and_pinned_contexts(self):
        import pipeline
        from test_merge_regressions import NEWS,profile_snapshot
        from common.news_snapshot import news_snapshot
        from common.search import news_pool
        barrier=threading.Barrier(2)
        instances=[]
        class Loop:
            def __init__(self,*a,**kw): instances.append(self)
            def run(self,row,*args):
                self.news_id=row['id'];barrier.wait(5)
                self.assert_snapshot=len(news_pool())
                return dict(news_id=row['id'],title=row['title'],status='reject')
        profile=profile_snapshot()
        scores=[dict(id=r['id'],importance=3,affected_roles=['cfo']) for r in NEWS.to_dict('records')]
        with tempfile.TemporaryDirectory() as tmp,patch.object(pipeline,'DATA',Path(tmp)), \
             patch.object(pipeline,'load_company',return_value=SimpleNamespace(card=profile['card'],facts=profile['facts'])), \
             patch.object(pipeline,'candidates',return_value=(NEWS,[])), \
             patch.object(pipeline,'InsightWriter'), \
             patch('stage5_insight_detection.analyst_loop.pool.RelatedNewsPool',return_value=SimpleNamespace(related=SimpleNamespace(news=NEWS))), \
             patch('stage4_news_relevance.news_classifier.classifier.NewsClassifier') as classifier, \
             patch('stage5_insight_detection.analyst_loop.loop.AnalystLoop',Loop), \
             patch('stage6_insight_generation.insight_card.render.render',return_value='<html/>'):
            classifier.return_value.classify.return_value=scores
            with news_snapshot(NEWS):
                result=pipeline.run_loop('synthetic','cfo',Mock(),n_candidates=2,n_judge=2,workers=2,
                    prepared_bank=SimpleNamespace(library=BankLibrary([]),report={'status':'complete'}))
            self.assertEqual(len(result),2)
            self.assertEqual(len([i for i in instances if hasattr(i,'news_id')]),2)
            self.assertTrue(all(i.assert_snapshot==2 for i in instances if hasattr(i,'news_id')))

    def test_invalid_flag_combinations_and_infinite_budgets(self):
        for kwargs in [dict(ocr_budget=1),dict(bank_budget=1),dict(analytics_refresh=True,analytics_cache_only=True),
                       dict(role='unknown'),dict(engine='loop',bank_insights=True,source='both'),
                       dict(engine='loop',bank_review=True,bank_budget=float('inf'))]:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValidationError):AnalysisRequest(company_id='U_1',**kwargs)

if __name__=='__main__': unittest.main()
