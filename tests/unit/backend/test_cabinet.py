"""Prepared accounts work without models; only the full variant can queue inference."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.cabinet import search_companies
from backend.service import Service
from common.evidence import EvidenceSnapshot


class CabinetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.models = Mock(side_effect=AssertionError('Cached entry must not initialize models'))
        self.service = Service(Path(self.tmp.name), models_factory=self.models)

    def tearDown(self):
        self.service.close()
        self.tmp.cleanup()

    def test_cached_accounts_have_real_evidence_and_lazy_card_routes(self):
        with patch.dict('os.environ', CABINET_MODE='server'), TestClient(create_app(self.service)) as client:
            accounts = client.get('/v1/cabinet/accounts').json()['accounts']
            self.assertEqual(len(accounts), 3)
            self.assertEqual({a['role'] for a in accounts}, {'ceo', 'cfo', 'treasury'})
            for account in accounts:
                url = f"/v1/jobs/{account['job_id']}/insights"
                preview = client.get(url)
                self.assertEqual(preview.status_code, 200)
                cards = preview.json()['insights']
                self.assertEqual(len(cards), account['insight_count'])
                self.assertNotIn('evidence', preview.text)
                for card in cards:
                    opened = client.get(f"{url}/{card['result_index']}").json()
                    self.assertEqual(len(opened['result']['results']), 1)
                    row = opened['result']['results'][0]
                    snapshot = EvidenceSnapshot.model_validate(row['evidence'])
                    self.assertEqual(snapshot.company_id, account['company_id'])
                    known = {f['id'] for f in row['evidence']['facts']} | {b['id'] for b in row['evidence']['bank']}
                    for block in ('what_happened', 'why_it_matters', 'risks', 'opportunities', 'hypotheses', 'bank_view'):
                        for claim in row['insight'].get(block, []):
                            self.assertTrue(set(claim.get('sources', [])) <= known)
                self.assertEqual(client.get(url + '/-1').status_code, 404)
            self.assertEqual(client.get('/v1/jobs/demo-unknown/insights').status_code, 404)
            self.assertEqual(self.service.agents.items, {})
            self.models.assert_not_called()

    def test_server_variant_rejects_all_inference_entry_points(self):
        with patch.dict('os.environ', CABINET_MODE='server'), TestClient(create_app(self.service)) as client:
            requests = [('/v1/analyses', {'company_id':'U_1'}),
                        ('/v1/analyses/batch', {'items':[{'company_id':'U_1'}]}),
                        ('/v1/reports/prepare', {}),
                        ('/v1/insights/ask', {'company_id':'U_1', 'question':'Что делать?'}),
                        ('/v1/search/news', {'query':'test'}), ('/v1/search/bank', {'query':'test'})]
            for path, payload in requests:
                self.assertEqual(client.post(path, json=payload).status_code, 403, path)
            self.assertEqual(client.post('/v1/reports/upload', files={'files':('a.pdf', b'x')}).status_code, 403)
            self.assertEqual(self.service.agents.items, {})
            self.models.assert_not_called()

    def test_full_variant_queues_the_requested_company_role_and_agent_options(self):
        run = Mock(return_value={'results':[]})
        self.service.run = run
        with patch.dict('os.environ', CABINET_MODE='full'), TestClient(create_app(self.service)) as client:
            response = client.post('/v1/analyses', headers={'X-User-ID':'reader'}, json={
                'company_id':'U_14366', 'role':'marketing', 'user_id':'reader', 'engine':'judge',
                'source':'both', 'analytics_cache_only':False, 'analytics_refresh':True})
            self.assertEqual(response.status_code, 202, response.text)
            job = self.service.agents.wait(response.json()['id'])
            self.assertEqual(job['status'], 'succeeded')
            request = run.call_args.args[0]
            self.assertEqual((request.company_id, request.role, request.source), ('U_14366', 'marketing', 'both'))
            self.assertTrue(request.analytics_refresh)
            self.assertFalse(request.analytics_cache_only)

    def test_company_search_uses_case_ids_or_literal_industry_without_models(self):
        table = pd.DataFrame([{'SYNTHETIC_U':'U_14366','MAINOKVED_NAME':'Финансовый лизинг','CITY':'Москва'},
                              {'SYNTHETIC_U':'U_777','MAINOKVED_NAME':'Строительство','CITY':'Казань'}])
        with patch('backend.cabinet._company_table', return_value=table):
            self.assertEqual(search_companies('u_14366')[0]['id'], 'U_14366')
            self.assertEqual(search_companies('лизинг')[0]['id'], 'U_14366')
            self.assertEqual(search_companies('.*'), [])
            self.assertEqual(search_companies('u'), [])
        self.models.assert_not_called()

    def test_server_startup_does_not_prepare_reports(self):
        with patch.dict('os.environ', CABINET_MODE='server'), \
             patch('backend.app.Service', return_value=self.service), \
             patch.object(self.service, 'prepare') as prepare, TestClient(create_app()):
            prepare.assert_not_called()

    def test_server_variant_disables_the_separate_discussion_service(self):
        from feedback_api.app import create_app as feedback_app
        store, analyzer = Mock(), Mock()
        with patch.dict('os.environ', CABINET_MODE='server'), \
             TestClient(feedback_app(store=store, analyzer=analyzer)) as client:
            base = '/v1/insights/00000000-0000-0000-0000-000000000001'
            headers = {'X-User-ID':'reader'}
            self.assertEqual(client.post(base + '/ask', headers=headers, json={'question':'Что делать?', 'request_id':'ask-1'}).status_code, 403)
            self.assertEqual(client.post(base + '/feedback', headers=headers, json={'kind':'useful', 'request_id':'like-1'}).status_code, 403)
            analyzer.analyze.assert_not_called()
            store.add.assert_not_called()

