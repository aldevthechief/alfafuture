"""Async execution must preserve verdicts, cache identities, source context and the five-news limit."""
import asyncio
import contextvars
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
from langchain_core.load import dumps
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.tools import tool
from pydantic import BaseModel

import pipeline
from backend.schemas import AnalysisRequest
from common.llm import FileLLMCache, JsonlCache, StructuredOutputError, structured
from common.llm.cache import stable_prompt
from common.llm.testing import ScriptedChatModel, calls, tool_call
from common.llm.tool_loop import ToolLoop
from common.model_load import load_once
from stage4_news_relevance.relevance_scoring.relevance import Relevance
from stage5_insight_detection.insight_judge.judge import InsightDecision
from stage6_insight_generation.insight_writer.insight import Insight, Review
from stage6_insight_generation.insight_writer.writer import Evidence, InsightWriter


class CacheAccelerationTests(unittest.TestCase):
    def test_every_tool_loop_step_replays_in_sync_and_async_execution(self):
        @tool
        def lookup(q: str) -> str:
            """Look up evidence."""
            return 'evidence ' + q

        class Done(BaseModel):
            answer: str

        for asynchronous in (False, True):
            with self.subTest(asynchronous=asynchronous), tempfile.TemporaryDirectory() as tmp:
                def run(script):
                    model = ScriptedChatModel(responses=script, cache=FileLLMCache(Path(tmp)))
                    loop = ToolLoop(model, [lookup], Done, 'system')
                    result = asyncio.run(loop.arun('question')) if asynchronous else loop.run('question')
                    return result.answer, model.i
                self.assertEqual(run([calls(tool_call('lookup', q='x')), calls(tool_call('Done', answer='ok'))])[0], 'ok')
                self.assertEqual(run([calls(tool_call('Done', answer='cache miss'))]), ('ok', 0))

    def test_only_ai_transport_metadata_is_ignored_and_legacy_keys_still_work(self):
        def prompt(cost, text='original evidence', arg='x'):
            reply = calls(tool_call('lookup', q=arg))
            reply.tool_calls[0]['id'] = 'tool-1'
            reply.id = f'reply-{cost}'
            reply.response_metadata = {'cost':cost}
            return dumps([HumanMessage('question'), reply, ToolMessage(text, tool_call_id='tool-1')])

        with tempfile.TemporaryDirectory() as tmp:
            cache = FileLLMCache(Path(tmp))
            answer = [ChatGeneration(message=AIMessage(content='answer'))]
            cache.update(prompt(7), 'model', answer)
            self.assertEqual(cache.lookup(prompt(0), 'model')[0].text, 'answer')
            self.assertIsNone(cache.lookup(prompt(0, text='new evidence'), 'model'))
            self.assertIsNone(cache.lookup(prompt(0, arg='new argument'), 'model'))
            self.assertIsNone(cache.lookup(prompt(0), 'other model parameters'))
            cache.store.put(cache._key(prompt(8), 'legacy'), dumps(answer))
            self.assertEqual(cache.lookup(prompt(8), 'legacy')[0].text, 'answer')
            for value in ('plain old prompt', '{"not":"messages"}'):
                self.assertEqual(stable_prompt(value), value)

    def test_sync_and_async_consequence_cache_misses_compute_once(self):
        async def scenario(cache):
            computing, release = asyncio.Event(), asyncio.Event()
            count = 0
            async def compute():
                nonlocal count
                count += 1
                computing.set()
                await release.wait()
                return {'value':42}
            first = asyncio.create_task(cache.aget_or_compute('key', compute))
            await computing.wait()
            sync = asyncio.create_task(asyncio.to_thread(cache.get_or_compute, 'key', lambda: self.fail('duplicate sync compute')))
            others = [asyncio.create_task(cache.aget_or_compute('key', compute)) for _ in range(16)]
            release.set()
            self.assertTrue(all(v == {'value':42} for v in await asyncio.gather(first, sync, *others)))
            self.assertEqual(count, 1)
        with tempfile.TemporaryDirectory() as tmp:
            asyncio.run(scenario(JsonlCache('consequences', Path(tmp))))

    def test_cancelled_async_compute_releases_the_shared_lock(self):
        async def scenario(cache):
            started = asyncio.Event()
            async def cancelled():
                started.set()
                await asyncio.Event().wait()
            task = asyncio.create_task(cache.aget_or_compute('key', cancelled))
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            async def retry(): return 42
            self.assertEqual(await asyncio.wait_for(cache.aget_or_compute('key', retry), 2), 42)
        with tempfile.TemporaryDirectory() as tmp:
            asyncio.run(scenario(JsonlCache('test', Path(tmp))))


class AsyncCompatibilityTests(unittest.TestCase):
    def test_structured_async_retains_retries_and_rejects_missing_verdicts(self):
        class Number(BaseModel):
            value: int
        model = ScriptedChatModel(responses=[AIMessage(content='bad'), calls(tool_call('Number', value=7))])
        self.assertEqual(asyncio.run(structured(model, Number).ainvoke('question')).value, 7)
        with self.assertRaises(StructuredOutputError):
            asyncio.run(structured(ScriptedChatModel(responses=[AIMessage(content='bad')]), Number).ainvoke('question'))

    def test_async_writer_has_the_same_rechecks_and_final_rejections_as_sync(self):
        insight = Insight(title='Test', role='generated label', what_happened=[], why_it_matters=[],
                          actions=[], ai_conclusion='', confidence='low')
        for final_ok in (False, True):
            script = []
            for ok in (False, False, final_ok):
                script += [calls(tool_call('Insight', **insight.model_dump())), calls(tool_call('Review', ok=ok, issues=[] if ok else ['fix']))]
            outputs = []
            for asynchronous in (False, True):
                model = ScriptedChatModel(responses=script)
                writer = InsightWriter(model)
                with patch.object(writer, '_remind', return_value=[]):
                    evidence = Evidence(news=[], facts=[])
                    result = asyncio.run(writer.awrite('U_1', 'cfo', '{}', evidence)) if asynchronous else writer.write('U_1', 'cfo', '{}', evidence)
                self.assertEqual(model.i, 0)  # all six responses used, including the final recheck
                self.assertEqual(result[1].ok, final_ok)
                self.assertEqual(result[0].role, 'cfo')
                outputs.append([v.model_dump() for v in result])
            self.assertEqual(*outputs)

    def test_sixteen_tasks_share_one_event_loop_but_only_five_news_reach_judge(self):
        rows = pd.DataFrame([dict(id=i, title=f'News {i}', text='text', date='2026-10-01', source='test') for i in range(20)])
        session = contextvars.ContextVar('session', default=None)
        session.set('analysis-1')
        class Scorer:
            active = 0
            peak = 0
            loops = set()
            gate = None
            async def ascore(inner, *args):
                self.assertEqual(session.get(), 'analysis-1')
                inner.loops.add(id(asyncio.get_running_loop()))
                inner.active += 1
                inner.peak = max(inner.peak, inner.active)
                inner.gate = inner.gate or asyncio.Event()
                if inner.active == 16: inner.gate.set()
                await asyncio.wait_for(inner.gate.wait(), 2)
                await asyncio.sleep(0)
                inner.active -= 1
                return Relevance(score=2, role_fit=2)
        class Judge:
            ids = []
            async def adecide(inner, news_id, *args):
                inner.ids.append(news_id)
                return InsightDecision(is_insight=False, score=0, company_link='', mechanism='')
        async def extract(news): return {}
        runner = object.__new__(pipeline.InsightPipeline)
        runner.related = SimpleNamespace(event_plausibility=lambda i:1, row_of={})
        runner.relevance, runner.judge = Scorer(), Judge()
        runner.consequences = SimpleNamespace(aextract=extract)
        with patch.object(pipeline, 'build', return_value={'U_1':[]}), patch.object(pipeline, 'candidates', return_value=(rows, [])):
            result = runner.run.__wrapped__(runner, 'U_1', 'cfo', 30, 5, workers=16)
        self.assertEqual(runner.relevance.peak, 16)
        self.assertEqual(len(runner.relevance.loops), 1)
        self.assertEqual(len(runner.judge.ids), 5)
        self.assertEqual([r['news_id'] for r in result], list(range(5)))
        self.assertEqual(AnalysisRequest(company_id='U_1').model_dump()['workers'], 16)
        self.assertEqual((AnalysisRequest(company_id='U_1').candidates, AnalysisRequest(company_id='U_1').judge), (30, 5))

    def test_sync_pipeline_entry_still_works_inside_an_existing_event_loop(self):
        from test_graph_personalization import runner, run
        async def scenario():
            return run(runner())
        self.assertEqual(asyncio.run(scenario())[0]['news_id'], 1)

    def test_cross_encoder_loads_once_and_uses_the_existing_encoder_lock(self):
        from common.encoders import _LOCK
        from common.model_load import _LOCK as shared
        self.assertIs(shared, _LOCK)
        key = ('synthetic-model', object())
        factory = Mock(return_value=object())
        barrier = threading.Barrier(16)
        def load():
            barrier.wait(timeout=5)
            return load_once(key, factory)
        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(lambda _:load(), range(16)))
        factory.assert_called_once()
        self.assertTrue(all(result is results[0] for result in results))
