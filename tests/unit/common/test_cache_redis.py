"""Unit cover for the Redis branches of the caches (a fake client, no server needed): the same
answer replays from the shared store, cut answers are never stored, the named JsonlCache maps
keys into its own hash. ALFA_CACHE_REDIS_URL switches the branch; the fake stands in for redis."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration

from common.llm import cache as cache_mod


class FakeRedis:
    def __init__(self):
        self.hashes: dict[str, dict[str, str]] = {}

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    def hget(self, key, field):
        return self.hashes.get(key, {}).get(field)

    def hset(self, key, field, value):
        self.hashes.setdefault(key, {})[field] = value

    def delete(self, key):
        self.hashes.pop(key, None)


class RedisCacheTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeRedis()
        patcher = patch.object(cache_mod, '_redis', lambda prefix: (self.fake, f'alfa:{prefix}'))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_llm_answer_replays_and_cut_answers_are_not_stored(self):
        store = cache_mod._redis_llm_cache()
        self.assertIsNotNone(store)
        good = [ChatGeneration(message=AIMessage(content='ответ'))]
        store.update('prompt-1', 'model-a', good)
        self.assertEqual(len(self.fake.hashes['alfa:llm']), 1)
        replayed = store.lookup('prompt-1', 'model-a')
        self.assertEqual(replayed[0].text, 'ответ')
        self.assertIsNone(store.lookup('prompt-2', 'model-a'))

        cut = [ChatGeneration(message=AIMessage(content=''), generation_info={'finish_reason': 'length'})]
        store.update('prompt-3', 'model-a', cut)
        self.assertNotIn(cache_mod.FileLLMCache._key('prompt-3', 'model-a'), self.fake.hashes['alfa:llm'])

    def test_seeded_llm_answers_replay(self):
        # the JSONL cache value is langchain's dumps string: seeded as is, or (old seed script) JSON-encoded once more
        from langchain_core.load import dumps
        raw = dumps([ChatGeneration(message=AIMessage(content='из сида'))])
        store = cache_mod._redis_llm_cache()
        self.fake.hset('alfa:llm', cache_mod.FileLLMCache._key('p-raw', 'm'), raw)
        self.fake.hset('alfa:llm', cache_mod.FileLLMCache._key('p-old', 'm'), json.dumps(raw))
        self.assertEqual(store.lookup('p-raw', 'm')[0].text, 'из сида')
        self.assertEqual(store.lookup('p-old', 'm')[0].text, 'из сида')

    def test_named_cache_keeps_its_own_hash_and_round_trips_json(self):
        named = cache_mod.JsonlCache('company_brief')
        named.put('U_1', {'industry': 'leasing'})
        other = cache_mod.JsonlCache('consequences')
        other.put('U_1', ['x'])
        self.assertEqual(named.get('U_1'), {'industry': 'leasing'})
        self.assertEqual(other.get('U_1'), ['x'])
        self.assertEqual(len(self.fake.hashes['alfa:company_brief']), 1)
        # the stored value is JSON: a fresh cache instance loads the same hash
        self.assertEqual(cache_mod.JsonlCache('company_brief').get('U_1'), {'industry': 'leasing'})


if __name__ == '__main__':
    unittest.main()
