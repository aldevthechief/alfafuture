"""Offline publication/retry/isolation regressions; all news, companies and model responses are synthetic.

python -X utf8 -m unittest test_pipeline_failures
"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
from langchain_core.messages import AIMessage
from pydantic import BaseModel

import pipeline
from test_merge_regressions import profile_snapshot
from common.llm import FileLLMCache, JsonlCache, StructuredOutputError, structured
from common.llm.testing import ScriptedChatModel, calls, tool_call as call
from common.trace import AgentTrace
from stage4_news_relevance.consequence_extraction.consequences import ConsequenceExtractor
from stage5_insight_detection.analyst_loop.loop import AnalystLoop
from stage5_insight_detection.analyst_loop.pool import ListNewsPool
from stage5_insight_detection.analyst_loop.profile import CompanyProfile
from stage5_insight_detection.bank_analytics.library import BankLibrary
from stage5_insight_detection.insight_judge.judge import InsightJudge
from stage6_insight_generation.insight_card.render import render
from stage6_insight_generation.insight_writer.insight import Claim, Insight, Review
from stage6_insight_generation.insight_writer.writer import InsightWriter


NEWS = pd.DataFrame([
    dict(id=41, title='ЦБ повысил ставку до 15%', text='Ключевая ставка повышена на 1 п.п.',
         date='2026-10-17', source='Источник А', url='https://example.com/a', tier=1),
    dict(id=42, title='Банк России повысил ключевую ставку до 15%', text='Ставка повышена на 1 п.п.',
         date='2026-10-17', source='Источник Б', url='https://example.com/b', tier=1)])
FACTS = [dict(cat='activity', text='Строительство зданий', depth=1)]


def insight():
    return Insight(title='Синтетический инсайт', role='cfo', what_happened=[], why_it_matters=[],
                   bank_view=[], actions=['Проверить условия кредита'], ai_conclusion='Проверить данные', confidence='low')


def relevance():
    return calls(call('Relevance', score=2, role_fit=2))


def consequences():
    return calls(call('NewsConsequences', economic=True, event='Ставка выросла', status='fact', significance=2))


def decision():
    return calls(call('decide', is_insight=True, score=2, company_link='Кредит', mechanism='Ставка'))


def written():
    return calls(call('Insight', **insight().model_dump()))


def reviewed():
    return calls(call('Review', ok=True))


def related(plausibility=1):
    rows = {int(r['id']): r for r in NEWS.to_dict('records')}
    return SimpleNamespace(event_plausibility=Mock(return_value=plausibility), row_of=rows,
                           get=lambda i: rows[int(i)])


def profile():
    return CompanyProfile('synthetic', '41.20', 'Строительство', 'Москва', 1e9, products=['loan'])


class Number(BaseModel):
    value: int


class StructuredTests(unittest.TestCase):
    def test_retry_uses_changed_request_to_escape_bad_cached_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = FileLLMCache(Path(tmp))
            model = ScriptedChatModel(cache=cache, responses=[AIMessage(content='wrong'),
                                      calls(call('Number', value=7))])
            chain = structured(model, Number)
            self.assertEqual(chain.invoke('synthetic').value, 7)
            self.assertEqual(chain.invoke('synthetic').value, 7)
            self.assertEqual(len(cache.store._load()), 2)

    def test_exhaustion_raises_explicit_error_instead_of_none(self):
        model = ScriptedChatModel(responses=[AIMessage(content=''), AIMessage(content='invalid')])
        with self.assertRaises(StructuredOutputError) as caught:
            structured(model, Number).invoke('synthetic')
        self.assertEqual(caught.exception.schema, 'Number')
        self.assertEqual(caught.exception.attempts, 2)

    def test_invalid_tool_arguments_are_retried(self):
        model = ScriptedChatModel(responses=[calls(call('Number', value='bad')),
                                            calls(call('Number', value=7))])
        self.assertEqual(structured(model, Number).invoke('synthetic').value, 7)

    def test_json_content_fallback_still_works_without_retry(self):
        model = ScriptedChatModel(responses=[AIMessage(content='```json\n{"value":7}\n```'),
                                            AIMessage(content='unused')])
        self.assertEqual(structured(model, Number).invoke('synthetic').value, 7)
        self.assertEqual(model.i, 1)


class JudgePipelineTests(unittest.TestCase):
    def run_pipeline(self, cheap_responses, agent_responses, plausibility=1):
        with tempfile.TemporaryDirectory() as tmp, patch.object(pipeline, 'build', return_value={'synthetic': FACTS}), \
                patch.object(pipeline, 'candidates', return_value=(NEWS, [])):
            cheap = ScriptedChatModel(responses=cheap_responses)
            agent = ScriptedChatModel(responses=agent_responses)
            runner = pipeline.InsightPipeline(cheap, agent, related=related(plausibility),
                        consequences=ConsequenceExtractor(cheap, JsonlCache('consequences', Path(tmp))))
            return runner.run('synthetic', 'cfo', 2, 2)

    def test_bad_relevance_does_not_lose_next_news(self):
        out = self.run_pipeline([AIMessage(content='bad'), AIMessage(content='bad'), relevance(), consequences()],
                                [decision(), written(), reviewed()])
        self.assertEqual(out[0]['error']['stage'], 'relevance')
        self.assertIsNone(out[0].get('insight'))
        self.assertEqual(out[1]['status'], 'ready')
        self.assertEqual(out[1]['news_id'], 42)

    def test_bad_consequences_does_not_lose_next_news(self):
        out = self.run_pipeline([relevance(), AIMessage(content='bad'), AIMessage(content='bad'),
                                 relevance(), consequences()], [decision(), written(), reviewed()])
        self.assertEqual(out[0]['error']['stage'], 'consequences')
        self.assertEqual(out[1]['status'], 'ready')

    def test_bad_decision_writer_and_review_do_not_lose_next_news(self):
        bad = AIMessage(content='bad')
        cases = [
            ('judge', 'InsightDecision', [calls(call('decide', is_insight='invalid')),
                                         calls(call('decide', is_insight='invalid'))]),
            ('writer', 'Insight', [decision(), bad, bad]),
            ('writer', 'Review', [decision(), written(), bad, bad])]
        for stage, schema, script in cases:
            with self.subTest(stage=stage, schema=schema):
                out = self.run_pipeline([relevance(), consequences(), relevance(), consequences()],
                                       script + [decision(), written(), reviewed()])
                self.assertEqual(out[0]['status'], 'model_error')
                self.assertEqual(out[0]['error']['stage'], stage)
                self.assertEqual(out[0]['error']['schema'], schema)
                self.assertNotIn('insight', out[0])
                self.assertEqual(out[1]['status'], 'ready')

    def test_judge_recovers_from_invalid_final_call(self):
        out = self.run_pipeline([relevance(), consequences(), relevance(), consequences()],
                               [calls(call('decide', is_insight='invalid')), decision(), written(), reviewed(),
                                decision(), written(), reviewed()])
        self.assertEqual([r['status'] for r in out], ['ready', 'ready'])

    def test_judge_missing_final_answer_is_retried_then_reported(self):
        agent = ScriptedChatModel(responses=[AIMessage(content='no decision')])
        judge = InsightJudge(agent, related(), max_steps=1)
        with self.assertRaises(StructuredOutputError):
            judge.decide(41, 'synthetic', FACTS, 'cfo')

    def test_zero_plausibility_blocks_judge_but_none_uses_default(self):
        for value in (0, None):
            with self.subTest(plausibility=value):
                out = self.run_pipeline([relevance(), consequences(), relevance(), consequences()],
                                       [decision(), written(), reviewed()], plausibility=value)
                self.assertEqual(len(out), 0 if value == 0 else 2)
                if value is None:
                    self.assertTrue(all(r['status'] == 'ready' for r in out))


class AnalystLoopTests(unittest.TestCase):
    def loop(self, writer, model=None, explain=False):
        return AnalystLoop(BankLibrary([]), ListNewsPool(NEWS.to_dict('records')), model=model,
                           writer=writer, explain=explain, trace=AgentTrace(folder=None))

    def test_audit_errors_preserve_draft_and_block_ready(self):
        for text, refs, code in [('Ставка 15%', [], 'no_source'),
                                 ('Ставка 15%', ['news:999'], 'unknown_ref')]:
            with self.subTest(code=code):
                draft = insight()
                draft.what_happened = [Claim(text=text, sources=refs)]
                writer = SimpleNamespace(write=Mock(return_value=(draft, Review(ok=True))))
                loop = self.loop(writer)
                result = loop.run(loop.pool.get(41), profile(), 'cfo', {'importance': 4})
                self.assertEqual(result['status'], 'audit_rejected')
                self.assertEqual(result['verdict']['status'], 'reject')
                self.assertNotIn('insight', result)
                self.assertEqual(result['draft']['what_happened'][0]['text'], text)
                self.assertIn(code, [w['code'] for w in result['audit']])
                self.assertIn('Финальный аудит', render(result))
                self.assertNotIn('Что сделать', render(result))

    def test_non_blocking_audit_warning_keeps_valid_insight_ready(self):
        draft = insight()
        draft.what_happened = [Claim(text='Ставка 15%', sources=['news:41'])]
        draft.why_it_matters = [Claim(text='Стоимость кредита выросла', sources=['news:41'])]
        loop = self.loop(SimpleNamespace(write=Mock(return_value=(draft, Review(ok=True)))))
        out = loop.run(loop.pool.get(41), profile(), 'cfo', {'importance': 4})
        self.assertEqual(out['status'], 'ready')
        self.assertIn('no_number', [w['code'] for w in out['audit']])

    def test_assessor_failure_does_not_prevent_next_news(self):
        model = ScriptedChatModel(responses=[AIMessage(content='bad'), AIMessage(content='bad'),
                                            calls(call('assess', explanations=[], missing=[]))])
        loop = self.loop(SimpleNamespace(write=Mock(return_value=(insight(), Review(ok=True)))), model, True)
        out = [loop.run(loop.pool.get(i), profile(), 'cfo', {'importance': 4}) for i in (41, 42)]
        self.assertEqual(out[0]['status'], 'model_error')
        self.assertEqual(out[0]['error']['schema'], 'Assessment')
        self.assertEqual(out[1]['status'], 'ready')

    def test_writer_and_checker_failure_do_not_prevent_next_news(self):
        for schema, script in [('Insight', [AIMessage(content='bad'), AIMessage(content='bad')]),
                                ('Review', [written(), AIMessage(content='bad'), AIMessage(content='bad')])]:
            with self.subTest(schema=schema):
                model = ScriptedChatModel(responses=script + [written(), reviewed()])
                loop = self.loop(InsightWriter(model))
                out = [loop.run(loop.pool.get(i), profile(), 'cfo', {'importance': 4}) for i in (41, 42)]
                self.assertEqual(out[0]['status'], 'model_error')
                self.assertEqual(out[0]['error']['schema'], schema)
                self.assertNotIn('insight', out[0])
                self.assertEqual(out[1]['status'], 'ready')

    def test_console_loop_continues_after_bad_assessment_and_renders_results(self):
        model = ScriptedChatModel(responses=[AIMessage(content='bad'), AIMessage(content='bad'),
                    calls(call('assess', explanations=[], missing=[])), written(), reviewed()])
        models = SimpleNamespace(cheap=lambda **kw: model, agent=lambda **kw: model)
        scores = [dict(id=i, importance=4, affected_roles=['cfo']) for i in (41, 42)]
        with tempfile.TemporaryDirectory() as tmp, patch.object(pipeline, 'DATA', Path(tmp)), \
                patch.object(pipeline, 'build', return_value={'synthetic': FACTS}), \
                patch.object(pipeline, 'candidates', return_value=(NEWS, [])), \
                patch('common.trace.AgentTrace', return_value=AgentTrace(folder=None)), \
                patch('stage5_insight_detection.analyst_loop.pool.RelatedNewsPool',
                      return_value=ListNewsPool(NEWS.to_dict('records'))), \
                patch('stage2_company_facts.profile.integration.load_profile', return_value=profile_snapshot(profile(), FACTS)), \
                patch('stage4_news_relevance.news_classifier.classifier.NewsClassifier') as classifier, \
                patch('stage5_insight_detection.bank_analytics.library.FRAGMENTS') as fragments, \
                patch.object(BankLibrary, 'load', return_value=BankLibrary([])):
            fragments.exists.return_value = True
            classifier.return_value.classify.return_value = scores
            out = pipeline.run_loop('synthetic', 'cfo', models, 2, 2)
            self.assertEqual([r['status'] for r in out], ['model_error', 'ready'])
            cards = list((Path(tmp) / 'insights/cards').glob('*.html'))
            self.assertEqual(len(cards), 2)


class ParallelJudgeTests(unittest.TestCase):
    def test_workers_overlap_slow_calls_and_keep_order(self):
        import contextvars, time
        session = contextvars.ContextVar('session')  # Langfuse keeps the run's session in a context variable
        session.set('run-1')
        seen = []
        from stage4_news_relevance.relevance_scoring.relevance import Relevance
        from stage5_insight_detection.insight_judge.judge import InsightDecision
        rows = pd.DataFrame([{'id': i, 'title': f't{i}', 'text': 'x', 'date': '2026-10-01', 'source': 's'} for i in range(4)])
        runner = pipeline.InsightPipeline.__new__(pipeline.InsightPipeline)
        runner.related, runner.bank = SimpleNamespace(event_plausibility=lambda i: 1, row_of={}, get=lambda i: {}), None
        slow = lambda value: (lambda *a, **k: (time.sleep(0.5), seen.append(session.get(None)), value)[2])
        runner.relevance = Mock(score=slow(Relevance(score=2, role_fit=2)))
        runner.consequences = Mock(extract=lambda news: {})
        runner.judge = Mock(decide=slow(InsightDecision(is_insight=False, score=0, company_link='', mechanism='')))
        with patch.object(pipeline, 'build', return_value={'x': []}), patch.object(pipeline, 'candidates', return_value=(rows, [])):
            start = time.time()
            out = runner.run('x', 'cfo', 4, 4, workers=4)
        self.assertEqual([r['news_id'] for r in out], [0, 1, 2, 3])
        self.assertLess(time.time() - start, 1.8)  # 8 half-second calls: ~1 s in parallel, 4 s one by one
        self.assertEqual(seen, ['run-1'] * 8)  # worker threads keep the session


if __name__ == '__main__':
    unittest.main()
