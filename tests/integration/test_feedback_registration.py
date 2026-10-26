"""Real UI registration requests retain original evidence and the same discussion across reloads."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from feedback_api.app import create_app
from feedback_api.store import Store


def result():
    return dict(status='ready', news_id=41, insight={
        'id':'insight-41', 'title':'Кредит', 'role':'cfo', 'why_it_matters':[{'text':'Кредитные продукты', 'sources':['fact:1']}],
        'what_happened':[{'text':'Ставка сохранена', 'sources':['news:41']}]},
        evidence=dict(version=2, company_id='U_1', role='cfo', facts=[{'id':'fact:1','text':'Исходный факт'}],
            news=[dict(id=41, date='2026-10-20', source='Test', title='Ставка сохранена', text='Исходная новость', url='https://example.com/41')],
            bank=[]))


class RegistrationTests(unittest.TestCase):
    def test_reassessment_keeps_discussion_and_original_evidence_across_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(Path(tmp)/'feedback.sqlite3')
            headers={'X-User-ID':'alice'}
            body={'company_id':'U_1','results':[result()],'registration_key':'job-1:insight-41'}
            with TestClient(create_app(store=store,api_token='')) as client:
                original=client.post('/v1/insights/import',json=body,headers=headers).json()[0]
                response=client.post(f'/v1/insights/{original["id"]}/feedback',headers=headers,
                                     json={'kind':'like','request_id':'like-before-reassessment'})
                self.assertEqual(response.status_code,201)
            changed=copy.deepcopy(body)
            changed['results'][0]['insight'].update(confidence='high',bank_product='Обсудить фиксацию ставки')
            with TestClient(create_app(store=Store(store.path),api_token='')) as client:
                response=client.post('/v1/insights/import',json=changed,headers=headers)
                self.assertEqual(response.status_code,201,response.text)
                updated=response.json()[0]
                self.assertEqual(updated['id'],original['id'])
                self.assertEqual(updated['evidence'],original['evidence'])
                self.assertEqual(updated['created_at'],original['created_at'])
                self.assertEqual(updated['insight']['confidence'],'high')
                self.assertEqual(updated['insight']['bank_product'],'Обсудить фиксацию ставки')
                history=client.get(f'/v1/insights/{updated["id"]}/feedback',headers=headers).json()
                self.assertEqual([item['request_id'] for item in history],['like-before-reassessment'])
                for field in ('title','evidence'):
                    collision=copy.deepcopy(changed)
                    if field=='title': collision['results'][0]['insight']['title']='Другой инсайт'
                    else: collision['results'][0]['evidence']['news'][0]['text']='Другой источник'
                    rejected=client.post('/v1/insights/import',json=collision,headers=headers)
                    self.assertEqual(rejected.status_code,409,rejected.text)
                stored=client.get(f'/v1/insights/{updated["id"]}',headers=headers).json()
                self.assertEqual(stored,updated)

    def test_legacy_display_role_uses_same_identity_and_original_evidence(self):
        original = result()
        original['insight']['role'] = 'Финансовый директор (CFO)'
        with tempfile.TemporaryDirectory() as tmp, \
             patch('common.search.news_pool', side_effect=AssertionError('must use original snapshot')), \
             patch('stage2_company_facts.deterministic_facts.facts.build', side_effect=AssertionError('must not rebuild facts')), \
             TestClient(create_app(store=Store(Path(tmp)/'feedback.sqlite3'), api_token='')) as client:
            body = {'company_id':'U_1', 'results':[original], 'registration_key':'legacy-title:insight-41'}
            headers = {'X-User-ID':'alice'}
            response = client.post('/v1/insights/import', json=body, headers=headers)
            self.assertEqual(response.status_code, 201, response.text)
            self.assertEqual(len(response.json()), 1)
            card = response.json()[0]
            self.assertEqual(card['insight']['role'], 'cfo')
            self.assertEqual(card['evidence']['news'][0]['text'], 'Исходная новость')
            self.assertEqual(original['insight']['role'], 'Финансовый директор (CFO)')
            body['results'][0] = result()
            self.assertEqual(client.post('/v1/insights/import', json=body, headers=headers).json()[0]['id'], card['id'])
            history = client.get(f'/v1/insights/{card["id"]}/feedback', headers=headers)
            self.assertEqual(history.status_code, 200)
            self.assertEqual(history.json(), [])

    def test_role_alias_does_not_mask_actual_company_or_role_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp, \
             TestClient(create_app(store=Store(Path(tmp)/'feedback.sqlite3'), api_token='')) as client:
            for role, company in [('Генеральный директор (CEO)', 'U_1'),
                                  ('Финансовый директор (CFO)', 'U_2')]:
                with self.subTest(role=role, company=company):
                    row = result()
                    row['insight']['role'] = role
                    response = client.post('/v1/insights/import', headers={'X-User-ID':'alice'},
                        json={'company_id':company, 'results':[row]})
                    self.assertEqual(response.status_code, 201)
                    self.assertEqual(response.json(), [])

    def test_legacy_rows_keep_source_lookup_and_stable_registration_keys(self):
        import pandas as pd
        from feedback_api.importer import register_results
        from common.search import corpus_fingerprint

        row = result()
        del row['evidence']
        row['evidence_facts'] = ['Исходный факт']
        pool = pd.DataFrame([dict(id=41, date='2026-10-20', source='Test', title='Ставка',
                                  text='Исходная новость', url='https://example.com/41')])
        fingerprinted = copy.deepcopy(row)
        fingerprinted['corpus_fingerprint'] = corpus_fingerprint(pool)
        with tempfile.TemporaryDirectory() as tmp, patch('common.search.news_pool', return_value=pool), \
             patch('common.news_corpus.result_references', return_value={'ok': True}):
            store = Store(Path(tmp)/'feedback.sqlite3')
            store.initialize()
            records = register_results(store, 'alice', 'U_1', [fingerprinted, row], registration_key='legacy-job')
            self.assertEqual(len(records), 2)
            self.assertNotEqual(records[0].id, records[1].id)
            self.assertTrue(all(r.evidence.news[0].text == 'Исходная новость' for r in records))
            again = register_results(store, 'alice', 'U_1', [fingerprinted, row], registration_key='legacy-job')
            self.assertEqual([r.id for r in records], [r.id for r in again])

    def test_original_snapshot_and_history_survive_reimport_and_service_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp)/'feedback.sqlite3')
            headers = {'X-User-ID':'alice'}
            body = {'company_id':'U_1', 'results':[result()], 'registration_key':'job-1:insight-41'}
            with patch('common.search.news_pool', side_effect=AssertionError('must use original snapshot')), \
                 patch('stage2_company_facts.deterministic_facts.facts.build', side_effect=AssertionError('must not rebuild facts')), \
                 TestClient(create_app(store=store, api_token='')) as client:
                response = client.post('/v1/insights/import', json=body, headers=headers)
                self.assertEqual(response.status_code, 201, response.text)
                record = response.json()[0]
                self.assertEqual(record['insight']['role'], 'cfo')
                self.assertEqual(record['evidence']['facts'][0]['text'], 'Исходный факт')
                self.assertEqual(record['evidence']['news'][0]['text'], 'Исходная новость')
                reaction = client.post(f'/v1/insights/{record["id"]}/feedback',
                    json={'kind':'already_known','request_id':'r1'}, headers=headers)
                self.assertEqual(reaction.status_code, 201)
                again = client.post('/v1/insights/import', json=body, headers=headers).json()[0]
                self.assertEqual(again['id'], record['id'])
                other = client.post('/v1/insights/import', json=body, headers={'X-User-ID':'bob'}).json()[0]
                self.assertNotEqual(other['id'], record['id'])
                changed = copy.deepcopy(body)
                changed['results'][0]['insight']['title'] = 'Different content'
                self.assertEqual(client.post('/v1/insights/import', json=changed, headers=headers).status_code, 409)
            with TestClient(create_app(store=Store(store.path), api_token='')) as client:
                again = client.post('/v1/insights/import', json=body, headers=headers).json()[0]
                self.assertEqual(again['id'], record['id'])
                self.assertEqual(len(client.get(f'/v1/insights/{record["id"]}/feedback', headers=headers).json()), 1)
                personalization = client.get('/v1/companies/U_1/personalization?role=cfo', headers=headers).json()
                self.assertEqual(personalization['excluded_news_ids'], [41])


if __name__ == '__main__':
    unittest.main()
