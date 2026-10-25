"""Both HTTP APIs + real LangGraph + shared SQLite: analysis -> card -> question -> reaction -> new selection.

Only model-dependent stages and retrieval inputs are scripted; jobs, graph, source snapshots, import,
history, ownership and personalization are production code. No network or paid model calls.
"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
from fastapi.testclient import TestClient

import pipeline
from backend.app import create_app as backend_app
from backend.service import Service
from common.llm.testing import ScriptedChatModel, calls, tool_call
from feedback_api.analyzer import FeedbackAnalyzer
from feedback_api.app import create_app as feedback_app
from feedback_api.store import Store
from stage5_insight_detection.bank_analytics.library import BankLibrary
from stage6_insight_generation.insight_writer.insight import Insight, Review


NEWS = pd.DataFrame([
    dict(id=1, title='Бензин дорожает', text='Бензин дорожает', date='2026-10-24', source='Test', url='https://example.com/1'),
    dict(id=2, title='Ключевая ставка сохранена', text='Ключевая ставка сохранена', date='2026-10-24', source='Test', url='https://example.com/2'),
])
FACTS = [dict(cat='bank', text='Кредитные продукты', depth=1)]


class FeedbackE2ETests(unittest.TestCase):
    def test_http_analysis_discussion_and_next_analysis_apply_reaction_only_to_its_user(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assessed = []
            model = ScriptedChatModel(responses=[calls(tool_call('Analysis', intent='question',
                response='Соотнесите расходы на топливо с оборотным капиталом компании.',
                requested_action='answer', source_ids=['news:1', 'fact:1']))])

            def init(pipe, *args, **kwargs):
                pipe.related = SimpleNamespace(row_of={}, event_plausibility=lambda i:1.0)
                # the news graph awaits stage 4, the judge and the writer (async LLM path)
                async def score(news, *args):
                    assessed.append(news['id'])
                    return SimpleNamespace(score=2, channel='', model_dump=lambda:{'score':2})
                async def extract(news):
                    return {}
                async def decide(*args):
                    return SimpleNamespace(is_insight=True, score=2, company_link='', bank_evidence=[], calculation_facts=[],
                        calculation_bank=[], related_news_ids=[], mechanism='', model_dump=lambda:{'is_insight':True},
                        model_dump_json=lambda:'{}')
                pipe.relevance = SimpleNamespace(ascore=score)
                pipe.consequences = SimpleNamespace(aextract=extract)
                pipe.judge = SimpleNamespace(adecide=decide)
                pipe.bank = None
                async def write(company, role, decision, evidence):
                    news = evidence.news[0]
                    return Insight(title=news['title'], role=role,
                        what_happened=[dict(text=news['text'], sources=[f'news:{news["id"]}'])],
                        why_it_matters=[dict(text='У компании есть кредитные продукты', sources=['fact:1'])],
                        actions=['Проверить план расходов'], ai_conclusion='Проверить влияние на компанию',
                        confidence='low'), Review(ok=True)
                pipe.writer = SimpleNamespace(awrite=write)

            service = Service(root/'api', models_factory=Mock())
            store = Store(root/'feedback'/'feedback.sqlite3')
            with patch.dict('os.environ', {'ALFA_DATA':str(root)}), \
                 patch.object(pipeline.InsightPipeline, '__init__', init), \
                 patch.object(pipeline, 'build', return_value={'U_1':FACTS}), \
                 patch.object(pipeline, 'candidates', side_effect=lambda *a, **kw:(NEWS.copy(), [])), \
                 patch('backend.service.DATA', root), \
                 patch('backend.service.news_pool', return_value=NEWS), \
                 patch('common.search.news_pool', return_value=NEWS), \
                 patch.object(service.catalog, 'snapshot', return_value=('v',BankLibrary([]),SimpleNamespace(parts=[]))), \
                 TestClient(backend_app(service)) as api, \
                 TestClient(feedback_app(store=store, analyzer=FeedbackAnalyzer(model=model), api_token='')) as feedback:

                def analyze(user):
                    response = api.post('/v1/analyses', headers={'X-User-ID':user}, json={
                        'company_id':'U_1', 'role':'cfo', 'engine':'judge', 'source':'news',
                        'retrieval':'lexical', 'candidates':2, 'judge':1, 'workers':1})
                    self.assertEqual(response.status_code, 202, response.text)
                    service.agents.wait(response.json()['id'])
                    job = api.get('/v1/jobs/'+response.json()['id'], headers={'X-User-ID':user}).json()
                    self.assertEqual(job['status'], 'succeeded', job)
                    self.assertEqual(job['result']['user_id'], user)
                    self.assertEqual(job['result']['results'][0]['status'], 'ready', job)
                    return job

                first = analyze('alice')
                row = first['result']['results'][0]
                self.assertEqual(row['news_id'], 1)
                self.assertEqual(row['evidence']['facts'][0]['text'], '[bank] Кредитные продукты')
                headers = {'X-User-ID':'alice'}
                registration = dict(company_id='U_1', results=[row], registration_key=first['id'])
                response = feedback.post('/v1/insights/import', headers=headers, json=registration)
                self.assertEqual(response.status_code, 201, response.text)
                card = response.json()[0]
                self.assertEqual(card['insight']['role'], 'cfo')
                path = '/v1/insights/'+card['id']
                response = feedback.post(path+'/ask', headers=headers,
                    json={'question':'Что делать?', 'request_id':'question-1'})
                self.assertEqual(response.status_code, 201, response.text)
                self.assertEqual(response.json()['analysis']['source_ids'], ['news:1','fact:1'])
                response = feedback.post(path+'/feedback', headers=headers,
                    json={'kind':'already_known', 'request_id':'reaction-1'})
                self.assertEqual(response.status_code, 201, response.text)
                self.assertEqual(response.json()['scope'], 'future')
                personalization = feedback.get('/v1/companies/U_1/personalization?role=cfo', headers=headers).json()
                self.assertEqual(personalization['excluded_news_ids'], [1])
                repeated = feedback.post('/v1/insights/import', headers=headers, json=registration).json()[0]
                self.assertEqual(repeated['id'], card['id'])
                self.assertEqual(len(feedback.get(path+'/feedback', headers=headers).json()), 2)
                assessed.clear()
                second = analyze('alice')
                self.assertEqual(second['result']['results'][0]['news_id'], 2)
                self.assertEqual(assessed, [2])  # excluded before assessment, no model work on known news
                self.assertEqual(analyze('bob')['result']['results'][0]['news_id'], 1)
                self.assertEqual(feedback.get(path+'/feedback', headers={'X-User-ID':'bob'}).status_code, 404)


if __name__ == '__main__':
    unittest.main()
