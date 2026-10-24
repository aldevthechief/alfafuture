"""Validated answer reuse, context isolation, live lookup freshness and Redis failure/expiry behavior."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from common.llm.testing import ScriptedChatModel, calls, tool_call
from feedback_api.analyzer import FeedbackAnalyzer, InvalidAnalysis
from feedback_api.app import create_app
from feedback_api.cache import PREFIX, RedisAnswerCache
from feedback_api.research import ResearchTools
from common.evidence import BankEvidence
from feedback_api.schemas import AnalysisDetails, FeedbackRecord, InsightRecord, Scope
from feedback_api.store import Store


class FakeRedis:
    def __init__(self):
        self.values, self.expires, self.now, self.broken = {}, {}, 0, False

    def get(self, key):
        if self.broken:
            raise ConnectionError('offline')
        if self.expires.get(key, 0) <= self.now:
            return None
        return self.values.get(key)

    def setex(self, key, ttl, value):
        if self.broken:
            raise ConnectionError('offline')
        self.values[key], self.expires[key] = value, self.now + ttl


class CountingModel(ScriptedChatModel):
    invocations: int = 0
    variant: str = 'a'

    @property
    def _identifying_params(self):
        return {'variant': self.variant}

    def _generate(self, *args, **kwargs):
        self.invocations += 1
        return super()._generate(*args, **kwargs)


def answer(*sources):
    return calls(tool_call('Analysis', intent='question', response='Со слов пользователя user:current.',
                           requested_action='answer', source_ids=list(sources or ['fact:1', 'user:current'])))


def insight():
    return InsightRecord.model_validate(dict(id='insight-1', user_id='alice', company_id='U_1', news_id=41,
        created_at='2026-10-24', insight={'title':'Кредит', 'role':'cfo'}, evidence={
            'company_id':'U_1', 'role':'cfo', 'captured_at':'2026-10-24',
            'facts':[{'id':'fact:1', 'text':'У компании есть кредит'}],
            'news':[dict(id=41, date='2026-10-20', source='Test', title='Ставка', text='Ставка сохранена')]}))


def question(identifier='q-1', text='Что делать?'):
    return FeedbackRecord.model_validate(dict(id=identifier, insight_id='insight-1', kind='question',
        comment=text, scope='insight', request_id=identifier, created_at='2026-10-24'))


class DiscussionCacheTests(unittest.TestCase):
    def setUp(self):
        self.redis = FakeRedis()
        self.cache = RedisAnswerCache(self.redis, ttl=60)
        self.model = CountingModel(responses=[answer()])
        self.processor = FeedbackAnalyzer(model=self.model, cache=self.cache)

    def test_reuse_across_processors_and_transport_ids_restores_real_user_references(self):
        first = self.processor.analyze(insight(), question(), [])
        second = FeedbackAnalyzer(model=self.model, cache=self.cache).analyze(insight(), question('q-2'), [])
        self.assertEqual(self.model.invocations, 1)
        self.assertIn('user:q-1', first.source_ids)
        self.assertIn('user:q-2', second.source_ids)
        self.assertIn('user:q-2', second.response)
        self.assertNotIn('user:q-1', second.response)
        key = next(iter(self.redis.values))
        self.assertTrue(key.startswith(PREFIX))
        self.assertNotIn('alice', key)
        self.assertEqual(len(key.removeprefix(PREFIX)), 64)

    def test_history_aliases_are_restored_without_reusing_old_message_ids(self):
        self.model.responses = [answer('user:history:0', 'user:current')]
        first = self.processor.analyze(insight(), question(), [question('old-1', 'У нас фиксированная ставка')])
        second = self.processor.analyze(insight(), question('q-2'), [question('old-2', 'У нас фиксированная ставка')])
        self.assertEqual(self.model.invocations, 1)
        self.assertEqual(first.source_ids, ['user:old-1', 'user:q-1'])
        self.assertEqual(second.source_ids, ['user:old-2', 'user:q-2'])

    def test_all_semantic_context_changes_miss_the_cache(self):
        self.processor.analyze(insight(), question(), [])
        changes = []
        for field, value in [('user_id','bob'), ('company_id','U_2'), ('news_id',42)]:
            rec = insight().model_copy(update={field:value})
            changes.append((rec, question(), []))
        rec = insight().model_copy(deep=True)
        rec.insight.role = 'ceo'
        changes.append((rec, question(), []))
        rec = insight().model_copy(deep=True)
        rec.evidence.facts[0].text = 'Кредит погашен'
        changes.append((rec, question(), []))
        rec = insight().model_copy(deep=True)
        rec.evidence.news[0].text = 'Ставка повышена'
        changes.append((rec, question(), []))
        rec = insight().model_copy(deep=True)
        rec.evidence.bank = [BankEvidence(id='bank:1', report='Ставка', text='Прогноз банка')]
        changes.append((rec, question(), []))
        changes += [(insight(), question(text='Что с курсом?'), []),
                    (insight(), question().model_copy(update={'scope':Scope.future}), []),
                    (insight(), question().model_copy(update={'created_at':'2026-10-25'}), []),
                    (insight(), question(), [question('old', 'Кредит погашен')]),
                    (insight(), question(), [question('old', 'Кредит ещё есть')])]
        for index, (rec, feedback, history) in enumerate(changes, 2):
            self.processor.analyze(rec, feedback, history)
            self.assertEqual(self.model.invocations, index)
        self.model.variant = 'b'
        self.processor.analyze(insight(), question(), [])
        self.assertEqual(self.model.invocations, len(changes) + 2)
        FeedbackAnalyzer(model=self.model, max_steps=2, cache=self.cache).analyze(insight(), question(), [])
        self.assertEqual(self.model.invocations, len(changes) + 3)

    def test_history_answer_changes_also_miss(self):
        prior = question('old', 'Почему?')
        prior.analysis = AnalysisDetails(intent='question', response='Первая версия', requested_action='answer')
        self.processor.analyze(insight(), question(), [prior])
        prior.analysis.response = 'Исправленный ответ'
        self.processor.analyze(insight(), question(), [prior])
        self.assertEqual(self.model.invocations, 2)

    def test_expiry_corrupt_entries_and_redis_outage_fall_back_to_the_model(self):
        self.processor.analyze(insight(), question(), [])
        self.redis.now = 61
        self.processor.analyze(insight(), question(), [])
        key = next(iter(self.redis.values))
        self.redis.values[key] = 'invalid json'
        self.processor.analyze(insight(), question(), [])
        self.redis.broken = True
        self.processor.analyze(insight(), question(), [])
        self.assertEqual(self.model.invocations, 4)

    def test_invalid_cached_citations_are_revalidated(self):
        self.processor.analyze(insight(), question(), [])
        key = next(iter(self.redis.values))
        value = json.loads(self.redis.values[key])
        value['source_ids'] = ['news:999']
        self.redis.values[key] = json.dumps(value)
        result = self.processor.analyze(insight(), question('q-2'), [])
        self.assertEqual(self.model.invocations, 2)
        self.assertNotIn('news:999', result.source_ids)

    def test_failed_and_truncated_model_answers_are_never_cached(self):
        self.model.responses = [answer('news:999')]
        with self.assertRaises(InvalidAnalysis):
            self.processor.analyze(insight(), question(), [])
        self.assertEqual(self.redis.values, {})
        self.model.responses = [answer().model_copy(update={'response_metadata': {'finish_reason':'length'}})]
        with self.assertRaises(InvalidAnalysis):
            self.processor.analyze(insight(), question(), [])
        self.assertEqual(self.redis.values, {})

    def test_research_is_replayed_and_changed_data_triggers_new_inference(self):
        current = [{'cat':'bank', 'text':'Кредит ещё есть'}]
        loads = []

        def factory(rec):
            def load(company):
                loads.append(company)
                return current
            return ResearchTools(rec, facts_loader=load, max_calls=1)

        self.model.responses = [calls(tool_call('company_facts', category='bank')), answer('fact:1')]
        processor = FeedbackAnalyzer(model=self.model, cache=self.cache, research_factory=factory)
        first = processor.analyze(insight(), question(), [])
        processor.analyze(insight(), question('q-2'), [])
        self.assertEqual(self.model.invocations, 2)
        self.assertEqual(len(loads), 2)  # the second answer checks current data, without invoking a model
        current[0]['text'] = 'Кредит погашен'
        updated = processor.analyze(insight(), question('q-3'), [])
        self.assertEqual(self.model.invocations, 4)
        self.assertNotEqual(first.retrieved_sources[0].text, updated.retrieved_sources[0].text)
        self.assertEqual(updated.tool_calls[0].status, 'ok')  # replay didn't consume the new loop's budget

    def test_unavailable_lookup_answers_are_not_cached(self):
        def factory(rec):
            def unavailable(company):
                raise RuntimeError('offline')
            return ResearchTools(rec, facts_loader=unavailable)
        self.model.responses = [calls(tool_call('company_facts')), answer('fact:1')]
        FeedbackAnalyzer(model=self.model, cache=self.cache, research_factory=factory).analyze(insight(), question(), [])
        self.assertEqual(self.redis.values, {})

    def test_configuration_uses_existing_redis_url_and_can_be_disabled(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(RedisAnswerCache.from_env())
        with patch.dict(os.environ, {'ALFA_CACHE_REDIS_URL':'redis://cache:6379/0'}, clear=True), \
             patch('redis.Redis.from_url', return_value=self.redis) as factory:
            self.assertEqual(RedisAnswerCache.from_env().ttl, 3600)
            factory.assert_called_once_with('redis://cache:6379/0', decode_responses=True,
                                            socket_connect_timeout=0.3, socket_timeout=0.5)
            os.environ['FEEDBACK_LLM_CACHE_TTL'] = '0'
            self.assertIsNone(RedisAnswerCache.from_env())

    def test_api_keeps_sqlite_history_and_request_id_idempotency(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = insight()
            body = {'company_id':rec.company_id, 'insight':rec.insight.model_dump(mode='json'),
                    'evidence':rec.evidence.model_dump(mode='json')}
            with TestClient(create_app(store=Store(Path(tmp)/'feedback.sqlite3'), analyzer=self.processor,
                                       api_token='')) as client:
                headers = {'X-User-ID':'alice'}
                registered = client.post('/v1/insights', headers=headers, json=body).json()
                path = f'/v1/insights/{registered["id"]}'
                first = client.post(path+'/ask', headers=headers, json={'question':'Что делать?', 'request_id':'one'})
                self.assertEqual(first.status_code, 201, first.text)
                same = client.post(path+'/ask', headers=headers, json={'question':'Что делать?', 'request_id':'one'})
                self.assertEqual(first.json(), same.json())
                self.assertEqual(self.model.invocations, 1)
                client.delete(f'/v1/feedback/{first.json()["id"]}', headers=headers)
                cached = client.post(path+'/ask', headers=headers, json={'question':'Что делать?', 'request_id':'two'})
                self.assertEqual(cached.status_code, 201, cached.text)
                self.assertEqual(self.model.invocations, 1)
                self.assertIn('user:'+cached.json()['id'], cached.json()['analysis']['source_ids'])
                self.assertEqual(len(client.get(path+'/feedback', headers=headers).json()), 1)
                # Same text after a previous turn is a different conversation context.
                client.post(path+'/ask', headers=headers, json={'question':'Что делать?', 'request_id':'three'})
                self.assertEqual(self.model.invocations, 2)


if __name__ == '__main__':
    unittest.main()
