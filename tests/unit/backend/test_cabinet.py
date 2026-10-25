"""Prepared accounts work without models; only the full variant can queue inference."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.cabinet import CACHE, search_companies
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
            self.assertEqual(len(accounts), 4)
            self.assertEqual({a['role'] for a in accounts}, {'ceo', 'cfo', 'treasury'})
            self.assertTrue(all(a['name'] == f"{a['sector']} №{a['company_id'].removeprefix('U_')}" for a in accounts))
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
                    known = {f['id'] for f in row['evidence']['facts']} | {b['id'] for b in row['evidence']['bank']} | {
                        f"news:{n['id']}" for n in row['evidence']['news']}
                    for block in ('what_happened', 'why_it_matters', 'risks', 'opportunities', 'hypotheses', 'bank_view'):
                        for claim in row['insight'].get(block, []):
                            self.assertTrue(set(claim.get('sources', [])) <= known)
                self.assertEqual(client.get(url + '/-1').status_code, 404)
            self.assertEqual(client.get('/v1/jobs/demo-unknown/insights').status_code, 404)
            self.assertEqual(self.service.agents.items, {})
            self.models.assert_not_called()

    def test_cached_news_preserves_company_role_facts_and_article_links_without_models(self):
        with patch.dict('os.environ', CABINET_MODE='server'), TestClient(create_app(self.service)) as client:
            accounts = client.get('/v1/cabinet/accounts').json()['accounts']
            account = accounts[0]
            self.assertEqual((account['company_id'], account['role']), ('U_14366', 'ceo'))
            url = f"/v1/jobs/{account['job_id']}/insights"
            cards = client.get(url).json()['insights']
            self.assertEqual(len(cards), 3)
            origins = []
            for card in cards:
                row = client.get(f"{url}/{card['result_index']}").json()['result']['results'][0]
                self.assertTrue(row['review']['ok'])
                self.assertEqual(row['evidence']['role'], 'ceo')
                self.assertEqual(row['evidence']['company_id'], 'U_14366')
                origins.append(row['insight']['origin'])
                if row['insight']['origin'] == 'bank':
                    self.assertTrue(row['insight']['bank_view'])
                    self.assertTrue(row['evidence']['bank'])
                    continue
                self.assertTrue(row['insight']['what_happened'])
                news = {f"news:{n['id']}": n for n in row['evidence']['news']}
                cited = {source for claim in row['insight']['what_happened'] for source in claim['sources']
                         if source.startswith('news:')}
                self.assertTrue(cited)
                for source in cited:
                    self.assertTrue(news[source]['text'])
                    self.assertTrue(news[source]['url'].startswith(('http://', 'https://')))
            self.assertEqual(origins, ['news', 'news', 'bank'])
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

    def test_every_cabinet_merges_cached_analytics_and_completed_news_without_duplicates(self):
        def job(id, company='U_14366', role='cfo', user='reader', status='succeeded', date='2026-10-26'):
            row = dict(status='ready', news_id=123, insight=dict(id='saved-news', origin='news', role=role,
                title='Saved news', what_happened=[dict(text='Source fact', sources=['news:123'])]),
                evidence=dict(company_id=company, role=role, news=[dict(id=123, title='Original article',
                    text='Original source text', source='News', date=date, url='https://example.org/article')]))
            return dict(id=id, status=status, created_at=date+'T00:00:00Z', finished_at=date+'T01:00:00Z',
                error=None, request=dict(company_id=company, role=role, user_id=user),
                result=dict(results=[dict(status='model_error'), row]))

        old, new = job('old', date='2026-10-25'), job('new')
        new['status']='partial'
        new['result']['errors']=[dict(component='analytics', message='Analytics unavailable')]
        foreign = job('foreign', user='other')
        foreign['result']['results'][1]['insight']['id']='private-news'
        other_role = job('other-role', role='ceo')
        other_company = job('other-company', company='U_999')
        self.service.agents.items.update({j['id']:j for j in (old,new,foreign,other_role,other_company)})
        with patch.dict('os.environ', CABINET_MODE='server'), TestClient(create_app(self.service)) as client:
            headers={'X-User-ID':'reader'}
            feed=client.get('/v1/cabinet/feed?company_id=U_14366&role=cfo', headers=headers).json()
            self.assertEqual(len(feed['insights']), 6)  # five cached bank cards plus this role's news
            live=[card for card in feed['insights'] if card['id']=='saved-news']
            self.assertEqual(len(live), 1)
            self.assertEqual((live[0]['job_id'], live[0]['result_index']), ('new',1))
            self.assertNotIn('private-news', str(feed))
            self.assertNotIn('Original source text', str(feed))  # previews only
            opened=client.get('/v1/jobs/new/insights/1',headers=headers).json()['result']['results'][0]
            self.assertEqual(opened['evidence']['news'][0]['url'], 'https://example.org/article')
            account=next(a for a in client.get('/v1/cabinet/accounts',headers=headers).json()['accounts']
                         if a['company_id']=='U_14366' and a['role']=='cfo')
            self.assertEqual(account['insight_count'], 6)
            # A later inference is picked up without choosing another account or overwriting the bank cards.
            later=job('later',date='2026-10-27')
            later['result']['results'][1]['insight']['id']='second-news'
            self.service.agents.items['later']=later
            updated=client.get('/v1/cabinet/feed?company_id=U_14366&role=cfo',headers=headers).json()
            self.assertEqual(len(updated['insights']), 7)
            self.models.assert_not_called()

    def test_local_analytics_cache_is_combined_with_saved_news_and_running_job_metadata(self):
        # A cache outside the committed demo set must also be included, without a model call.
        run=json.loads((CACHE/'U_14366_cfo.json').read_text(encoding='utf-8'))
        run=copy.deepcopy(run)
        run['insights']=run['insights'][:1]
        run['insights'][0]['id']='local-bank'
        path=self.service.root/'analytics_cache'/'snapshot'/'U_14366_cfo.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(run),encoding='utf-8')
        pending=dict(id='pending',status='running',created_at='2026-10-27T00:00:00Z',result=None,error=None,
                     request=dict(company_id='U_14366',role='cfo',user_id='reader'))
        self.service.agents.items['pending']=pending
        with TestClient(create_app(self.service)) as client:
            feed=client.get('/v1/cabinet/feed?company_id=U_14366&role=cfo',
                            headers={'X-User-ID':'reader'}).json()
            self.assertEqual(len(feed['insights']),6)
            self.assertEqual(feed['running_job'],{'id':'pending','status':'running'})
            cached=next(c for c in feed['insights'] if c['id']=='local-bank')
            row=client.get(f"/v1/jobs/{cached['job_id']}/insights/{cached['result_index']}").json()['result']['results'][0]
            self.assertEqual(row['insight']['id'],'local-bank')
            self.assertEqual(row['evidence']['role'],'cfo')
            # Re-resolve by stable card ID if a cache refresh has changed its row index.
            identity_url=f"/v1/jobs/{cached['job_id']}/insights/999?insight_id=local-bank"
            self.assertEqual(client.get(identity_url).json()['result']['results'][0]['insight']['id'],'local-bank')
            self.assertEqual(client.get('/v1/cabinet/feed?company_id=../bad&role=cfo').status_code,422)
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

    def test_demo_compact_versions_use_preview_ids_and_can_be_read_without_models_in_server_mode(self):
        async def shortened(self, insight):
            return {'title': 'Кратко: ' + insight['title'], 'lead': 'Краткое описание', 'facts': [],
                    'why': [], 'important_if': ['Есть плавающая ставка'], 'risks': [],
                    'opportunities': [], 'actions': [], 'bank_view': []}

        class Model:
            def model_copy(self, update): return self

        self.service.models_factory = Mock(return_value=Mock(agent=Mock(return_value=Model())))
        with TestClient(create_app(self.service)) as client:
            job_id = 'demo-U_14366-ceo'
            path = f'/v1/jobs/{job_id}/compact'
            with patch.dict('os.environ', CABINET_MODE='server'):
                self.assertEqual(client.get(path).json(), {})
                self.service.models_factory.assert_not_called()
            with patch.dict('os.environ', CABINET_MODE='full'), \
                 patch('stage6_insight_generation.insight_compact.compact.InsightCompactor.__init__', lambda self, model: None), \
                 patch('stage6_insight_generation.insight_compact.compact.InsightCompactor.acompact', shortened):
                response = client.get(path)
                self.assertEqual(response.status_code, 200, response.text)
                versions = response.json()
                cards = client.get(f'/v1/jobs/{job_id}/insights').json()['insights']
                self.assertEqual(set(versions), {card['id'] for card in cards})
            before = self.service.models_factory.call_count
            with patch.dict('os.environ', CABINET_MODE='server'):
                self.assertEqual(client.get(path).json(), versions)
                self.assertEqual(self.service.models_factory.call_count, before)

    def test_server_startup_does_not_prepare_reports(self):
        with patch.dict('os.environ', CABINET_MODE='server'), \
             patch('backend.app.Service', return_value=self.service), \
             patch.object(self.service, 'prepare') as prepare, TestClient(create_app()):
            prepare.assert_not_called()

    def test_company_profile_projects_real_products_scale_and_channels_without_models(self):
        profile = dict(id='U_1', card=dict(okved_name='Финансовый лизинг', revenue=20e9, products=['loan','leasing'], region='Москва'),
            scale=dict(class_ru='крупный бизнес', class_basis='выручка', case_large=True, group_case_large=True),
            facts=[dict(cat='group', text='Группа из трёх компаний', depth=1)], sensitivity={},
            external_factors=[dict(label='Ключевая ставка', mechanism='Влияет на стоимость фондирования', weight=1)])
        with patch('stage2_company_facts.profile.load.load_profile', return_value=profile), \
             TestClient(create_app(self.service)) as client:
            response = client.get('/v1/companies/U_1')
            self.assertEqual(response.status_code, 200, response.text)
            data = response.json()
            self.assertEqual(data['profile']['name'], 'Финансовый лизинг №1')
            self.assertEqual(data['profile']['bank_products'], ['Кредитные продукты','Лизинг'])
            self.assertIn('Крупный бизнес', data['profile']['scale'])
            self.assertEqual(data['profile']['group'], 'Группа из трёх компаний')
            self.assertEqual(data['profile']['channels'][0]['name'], 'Ключевая ставка')
            self.assertEqual(data['card'], profile['card'])
            self.assertTrue(data['facts'])
        self.models.assert_not_called()

    def test_server_variant_disables_the_separate_discussion_service(self):
        from feedback_api.app import create_app as feedback_app
        from feedback_api.store import Store
        store, analyzer = Mock(wraps=Store(Path(self.tmp.name)/'feedback.sqlite3')), Mock()
        with patch.dict('os.environ', CABINET_MODE='server'), \
             TestClient(feedback_app(store=store, analyzer=analyzer)) as client:
            base = '/v1/insights/00000000-0000-0000-0000-000000000001'
            headers = {'X-User-ID':'reader'}
            self.assertEqual(client.post(base + '/ask', headers=headers, json={'question':'Что делать?', 'request_id':'ask-1'}).status_code, 403)
            self.assertEqual(client.post(base + '/feedback', headers=headers, json={'kind':'useful', 'request_id':'like-1'}).status_code, 403)
            analyzer.analyze.assert_not_called()
            store.add.assert_not_called()

