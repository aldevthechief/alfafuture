"""Regression checks for simultaneous cache fills, global ranking, partial API results and CORS."""
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from fastapi.testclient import TestClient
from backend.app import create_app
from backend.catalog import BankView
from backend.schemas import AnalysisRequest
from backend.service import Service
from test_hybrid_news_search import NEWS
from test_backend_api import processor

class CacheRaceTests(unittest.TestCase):
    def test_simultaneous_fills_leave_complete_union_and_reuse_it(self):
        from stage3_news_search.retrieval_v2 import hybrid
        with tempfile.TemporaryDirectory() as tmp,patch.object(hybrid,'CACHE',Path(tmp)):
            barrier=threading.Barrier(2)
            def encode(slug,texts):
                barrier.wait(5)
                return np.array([[len(text),1.] for text in texts],dtype=np.float32)
            with ThreadPoolExecutor(2) as pool:
                a=pool.submit(hybrid.doc_vectors,'test',['first','shared'],encode)
                b=pool.submit(hybrid.doc_vectors,'test',['second','shared'],encode)
                self.assertEqual(a.result().shape,(2,2))
                self.assertEqual(b.result().shape,(2,2))
            cache=hybrid._load_cache('test')
            self.assertEqual(set(cache),{hybrid.text_hash(x) for x in ['first','second','shared']})
            never=Mock(side_effect=AssertionError('must reuse persisted cache'))
            vectors=hybrid.doc_vectors('test',['second','first','shared'],never)
            np.testing.assert_array_equal(vectors[:,0],[6,5,6])
            self.assertEqual(list(Path(tmp).glob('*.tmp.npz')),[])

class GlobalRankingTests(unittest.TestCase):
    def test_all_reports_share_dense_and_lexical_ranks_and_passages(self):
        def part(name,vector,when):
            chunk=dict(id=name,report=name,date=when,stale=False,freshness='fresh',kind='text',page=1,pages=[1],
                       text=name+' text',own_view=True,seg=None,start=0,end=0)
            # Local per-PDF search would tie both at rank zero. It must never be used for fusion.
            return SimpleNamespace(rows=[chunk],chunks=[chunk],D=np.array([vector]),available=[name],reports=[],
                pdf_paths={},twins={},segments={},search=Mock(side_effect=AssertionError('local ranks are incomparable')))
        bad=part('irrelevant',[0.,1.],'2026-10-23')
        good=part('relevant',[1.,0.],'2026-10-20')
        view=BankView([bad,good])
        query_encoder=Mock(encode=Mock(return_value=np.array([[1.,0.]])))
        with patch('stage5_insight_detection.bank_analytics.search.encoder',return_value=query_encoder), \
             patch('stage5_insight_detection.bank_analytics.search.bm25_index',return_value=lambda query:np.array([0.,10.])) as bm25:
            hit=view.search('loan',k=1)[0]
            self.assertEqual(hit['id'],'relevant')
            self.assertEqual(hit['pos'],1)
            self.assertEqual(query_encoder.encode.call_count,1)
            self.assertEqual(bm25.call_args.args[0],['irrelevant text','relevant text'])
            passages=view.passages('loan',k=1,context=0)
            self.assertEqual(passages[0]['ids'],['relevant'])
            self.assertEqual(passages[0]['report'],'relevant')
            self.assertIn('relevant text',passages[0]['text'])
            self.assertEqual(bm25.call_count,1)

class ApiFailureTests(unittest.TestCase):
    def test_bank_failure_is_partial_with_news_preserved_and_exposed_error(self):
        import pipeline
        with tempfile.TemporaryDirectory() as tmp:
            service=Service(Path(tmp),models_factory=Mock(),processor=processor)
            news=[dict(news_id=int(NEWS.iloc[0].id),title='test',status='ready')]
            with patch.object(pipeline,'InsightPipeline') as runner, \
                 patch.object(pipeline,'migrate_result',side_effect=lambda row,company:row), \
                 patch('backend.service.DATA',Path(tmp)), \
                 patch('backend.service.news_pool',return_value=NEWS), \
                 patch('backend.service.save_results'), \
                 patch('stage2_company_facts.profile.integration.load_company',return_value=SimpleNamespace(facts=[])), \
                 patch('stage5_insight_detection.analytics_insights.service.analytics_run',side_effect=RuntimeError('bank unavailable')):
                runner.return_value.run.return_value=news
                with TestClient(create_app(service)) as client:
                    submitted=client.post('/v1/analyses',json={'company_id':'U_1','source':'both'}).json()
                    service.agents.wait(submitted['id'])
                    job=client.get('/v1/jobs/'+submitted['id']).json()
                    self.assertEqual(job['status'],'partial')
                    self.assertIn('bank unavailable',job['error'])
                    self.assertEqual(job['result']['results'],news)
                    self.assertEqual(job['result']['components']['analytics']['status'],'failed')
                    self.assertEqual(job['result']['components']['news']['status'],'succeeded')
                    only=client.post('/v1/analyses',json={'company_id':'U_1','source':'analytics'}).json()
                    failed=service.agents.wait(only['id'])
                    self.assertEqual(failed['status'],'failed')
                    self.assertIn('bank unavailable',failed['error'])

    def test_run_error_field_is_not_mistaken_for_success(self):
        import pipeline
        with tempfile.TemporaryDirectory() as tmp:
            service=Service(Path(tmp),models_factory=Mock(),processor=processor)
            try:
                with patch.object(pipeline,'InsightPipeline') as runner, \
                     patch.object(pipeline,'migrate_result',side_effect=lambda row,company:row), \
                     patch('backend.service.DATA',Path(tmp)),patch('backend.service.news_pool',return_value=NEWS), \
                     patch('backend.service.save_results'), \
                     patch('stage2_company_facts.profile.integration.load_company',return_value=SimpleNamespace(facts=[])), \
                     patch('stage5_insight_detection.analytics_insights.service.analytics_run',return_value=SimpleNamespace(error='agent stopped')):
                    runner.return_value.run.return_value=[]
                    result=service.run(AnalysisRequest(company_id='U_1',source='both'))
                    self.assertEqual(result['status'],'partial')
                    self.assertIn('agent stopped',result['errors'][0]['message'])
            finally:service.close()

class CorsTests(unittest.TestCase):
    def test_configured_frontend_can_preflight_and_poll_but_other_origins_cannot(self):
        with tempfile.TemporaryDirectory() as tmp:
            service=Service(Path(tmp),models_factory=lambda:None,processor=processor)
            origin='http://frontend.test:5173'
            with TestClient(create_app(service,cors_origins=[origin])) as client:
                response=client.options('/v1/analyses',headers={'Origin':origin,'Access-Control-Request-Method':'POST',
                    'Access-Control-Request-Headers':'content-type,authorization'})
                self.assertEqual(response.status_code,200)
                self.assertEqual(response.headers['access-control-allow-origin'],origin)
                self.assertEqual(client.get('/health',headers={'Origin':origin}).headers['access-control-allow-origin'],origin)
                denied=client.options('/v1/analyses',headers={'Origin':'http://other.test','Access-Control-Request-Method':'POST'})
                self.assertEqual(denied.status_code,400)
                self.assertNotIn('access-control-allow-origin',denied.headers)

if __name__=='__main__':unittest.main()
