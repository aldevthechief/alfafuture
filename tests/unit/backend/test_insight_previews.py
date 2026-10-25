"""Feed requests must not serialize full evidence or another insight's payload."""
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.service import Service


class InsightPreviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.service = Service(Path(self.tmp.name))
        self.row = dict(status='ready', news_id=41, insight=dict(
            id='card-41', title='Кредит', role='cfo', why_it_matters=[dict(text='Стоимость долга')],
            risks=[dict(text='Риск')], opportunities=[], actions=['Проверить ставку'], bank_view=[],
            impact=dict(level='high', criteria='Existing assessment', estimate='1%')),
            evidence=dict(role='cfo', facts=[dict(id='fact:1', text='original evidence ' * 10000)]))
        self.service.agents.items['job-1'] = dict(id='job-1', kind='analysis', status='succeeded',
            request=dict(company_id='U_1', role='cfo', user_id='alice'), created_at='2026-10-25',
            finished_at='2026-10-25', error=None,
            result=dict(results=[self.row, dict(status='ready', insight=dict(id='other'),
                                               evidence=dict(role='cfo', unvisited=object()))]))
        self.client = TestClient(create_app(self.service))
        self.client.__enter__()
        self.headers = {'X-User-ID': 'alice'}

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.tmp.cleanup()

    def test_metadata_and_previews_do_not_copy_or_serialize_evidence(self):
        response = self.client.get('/v1/jobs?include_results=false', headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertNotIn('result', response.json()['analyses'][0])
        single = self.client.get('/v1/jobs/job-1?include_result=false', headers=self.headers)
        self.assertEqual(single.status_code, 200)
        self.assertIsNone(single.json()['result'])
        response = self.client.get('/v1/jobs/job-1/insights', headers=self.headers)
        self.assertEqual(response.status_code, 200)
        card = response.json()['insights'][0]
        self.assertEqual(card['id'], 'card-41')
        self.assertEqual(card['counts']['risks'], 1)
        self.assertEqual(card['impact_level'], 'high')
        self.assertIsNone(response.json()['insights'][1]['impact_level'])
        self.assertEqual(card['result_index'], 0)
        self.assertTrue(card['preview'])
        self.assertNotIn('evidence', response.text)
        self.assertLess(len(response.content), 3000)

    def test_opening_one_card_returns_only_its_original_result(self):
        response = self.client.get('/v1/jobs/job-1/insights/0', headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['result']['results'], [self.row])
        for index in (-1, 99):
            self.assertEqual(self.client.get(f'/v1/jobs/job-1/insights/{index}', headers=self.headers).status_code, 404)

    def test_previews_and_details_keep_user_visibility(self):
        for headers in ({}, {'X-User-ID': 'bob'}):
            self.assertEqual(self.client.get('/v1/jobs?include_results=false', headers=headers).json()['analyses'], [])
            for path in ('/v1/jobs/job-1/insights', '/v1/jobs/job-1/insights/0'):
                self.assertEqual(self.client.get(path, headers=headers).status_code, 404)
