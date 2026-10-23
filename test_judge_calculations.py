"""Judge calculator integration: real arithmetic, scripted judge/writer, no network or encoder."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pipeline
from common.llm import JsonlCache
from common.llm.testing import ScriptedChatModel, calls, tool_call as c
from stage4_news_relevance.consequence_extraction.consequences import ConsequenceExtractor
from stage5_insight_detection.insight_judge.calculations import JudgeCalculations
from stage5_insight_detection.insight_judge.judge import InsightDecision, InsightJudge, JudgeTools
from stage5_insight_detection.bank_analytics.chunks import build_chunks, passages
from stage6_insight_generation.insight_writer.writer import InsightWriter
from stage6_insight_generation.insight_writer.insight import Claim
from test_pipeline_failures import NEWS, FACTS, related, relevance, consequences, insight


ROW = NEWS.iloc[0].to_dict()
ROW.update(title='ЦБ повысил ключевую ставку до 15%', text='Ключевая ставка составляет 15%.', date='2026-10-20')
CARD = {'revenue': 300_000_000, 'group_revenue': 1_000_000_000}


def bank_source(indicator='ставка', value=14, kind='actual', **changes):
    row = {'id': 'bank:report:p1.1', 'report': 'Отчёт', 'date': '2026-10-01', 'page': 1,
           'kind': 'text', 'text': f'Наш прогноз: {indicator} {value}%.', 'own_view': True,
           'figures': [dict(indicator=indicator, value=value, unit='%', kind=kind, period='',
                            measure='level', verified=True, status='verified')]}
    row.update(changes)
    return row


def bank(row):
    chunk = dict(row, id=row['id'] + '+c1', frag_ids=[row['id']], stale=False, freshness='fresh')
    passage = dict(report=row['report'], date=row['date'], pages=[1], stale=False, ids=[chunk['id']], text=row['text'])
    return SimpleNamespace(rows=[row], chunks=[chunk], index={chunk['id']: 0}, twins={},
                           passages=Mock(return_value=[passage]), search=Mock(side_effect=AssertionError('extra search')))


class CalculatorTests(unittest.TestCase):
    def test_units_need_no_company_loading_search_or_new_model(self):
        calc = JudgeCalculations(ROW, 'company', None)
        with patch('stage5_insight_detection.insight_judge.calculations.load_company', side_effect=AssertionError('profile loaded')), \
             patch('common.llm.ChatModels', side_effect=AssertionError('new model')):
            reply = json.loads(calc.run('Переведи 1.5 п.п. в б.п.'))
        self.assertTrue(reply['ok'])
        self.assertEqual(reply['facts'][0]['value'], 150)
        self.assertEqual(reply['facts'][0]['kind'], 'hypothesis')

    def test_share_uses_exact_profile_numbers_and_repeated_request_keeps_ids(self):
        card = dict(revenue=20_828_575_458, group_revenue=30_100_000_000)
        calc = JudgeCalculations(ROW, 'company', None)
        with patch('stage5_insight_detection.insight_judge.calculations.load_company', return_value=SimpleNamespace(card=card)) as load:
            first = calc.run('Доля выручки компании в группе')
            self.assertEqual(first, calc.run('Доля выручки компании в группе'))
        load.assert_called_once_with('company')
        result = json.loads(first)['facts'][-1]
        self.assertEqual(result['value'], round(100 * card['revenue'] / card['group_revenue'], 1))
        self.assertEqual(result['id'], 'fact:calc_1_share')
        self.assertEqual(calc.facts['fact:revenue']['value'], card['revenue'])

    def test_bank_arithmetic_reuses_one_search_and_preserves_the_actual_source(self):
        source = bank_source()
        library = bank(source)
        tools = JudgeTools(41, FACTS, related(), library, ROW, 'company')
        self.assertFalse(json.loads(tools.compute('Разница ключевой ставки с показателем Банка'))['ok'])
        tools.search_bank_analytics('ключевая ставка')
        reply = json.loads(tools.compute('Разница ключевой ставки с показателем Банка'))
        self.assertTrue(reply['ok'], reply)
        self.assertEqual(reply['facts'][0]['value'], 1)
        self.assertEqual(reply['facts'][0]['unit'], 'п.п.')
        self.assertIn(source['id'], reply['facts'][0]['source'])
        self.assertEqual(tools.calculations.bank_sources[source['id']]['text'], source['text'])
        library.passages.assert_called_once()
        library.search.assert_not_called()

    def test_stale_future_and_unverified_bank_figures_are_not_used(self):
        cases = [bank_source(date='2021-01-01'), bank_source(date='2026-10-21'), bank_source()]
        cases[-1]['figures'][0].update(status='approx', verified=False)
        for source in cases:
            with self.subTest(source=source):
                tools = JudgeTools(41, FACTS, related(), bank(source), ROW, 'company')
                tools.search_bank_analytics('ставка')
                self.assertFalse(json.loads(tools.compute('Разница ключевой ставки с показателем Банка'))['ok'])
                self.assertEqual(tools.calculations.facts, {})

    def test_forecast_on_revenue_stays_a_scenario(self):
        tools = JudgeTools(41, FACTS, related(), bank(bank_source('инфляция', 6, 'forecast')), ROW, 'company')
        tools.search_bank_analytics('инфляция')
        with patch('stage5_insight_detection.insight_judge.calculations.load_company', return_value=SimpleNamespace(card=CARD)):
            reply = json.loads(tools.compute('Прогноз инфляции на выручку компании'))
        self.assertTrue(reply['ok'], reply)
        result = reply['facts'][-1]
        self.assertEqual((result['value'], result['unit']), (18.0, 'млн ₽'))
        self.assertEqual(result['kind'], 'hypothesis')
        self.assertIn('допущение', result['formula'])

    def test_missing_inputs_and_unsupported_formulas_return_a_reason(self):
        calc = JudgeCalculations(ROW, 'company', None)
        with patch('stage5_insight_detection.insight_judge.calculations.load_company', return_value=SimpleNamespace(card={})):
            for query in ('Доля выручки компании в группе', 'Посчитай EBITDA'):
                reply = json.loads(calc.run(query))
                self.assertFalse(reply['ok'])
                self.assertTrue(reply['reason'])
        self.assertEqual(calc.facts, {})

    def test_company_and_event_state_is_not_shared_between_judge_workers(self):
        first, second = JudgeCalculations(ROW, 'a', None), JudgeCalculations(ROW, 'b', None)
        with patch('stage5_insight_detection.insight_judge.calculations.load_company', side_effect=[
                SimpleNamespace(card=CARD), SimpleNamespace(card=dict(CARD, revenue=100_000_000))]):
            a = json.loads(first.run('Доля выручки компании в группе'))['facts'][-1]['value']
            b = json.loads(second.run('Доля выручки компании в группе'))['facts'][-1]['value']
        self.assertEqual((a, b), (30, 10))


class JudgeRouteTests(unittest.TestCase):
    def test_scripted_judge_can_call_compute_and_returns_code_owned_evidence(self):
        model = ScriptedChatModel(responses=[
            calls(c('compute', query='Доля выручки компании в группе')),
            calls(c('decide', is_insight=True, score=2, company_link='Выручка группы', mechanism='Доля'))])
        with patch('stage5_insight_detection.insight_judge.calculations.load_company', return_value=SimpleNamespace(card=CARD)), \
             patch('common.llm.ChatModels', side_effect=AssertionError('a second agent was created')):
            decision = InsightJudge(model=model, related=related()).decide(41, 'company', FACTS, 'cfo')
        self.assertTrue(decision.is_insight)
        self.assertEqual(decision.calculation_facts[-1]['value'], 30)
        self.assertNotIn('calculation_facts', InsightDecision.model_json_schema()['properties'])
        self.assertNotIn('calculation_facts', decision.model_dump())
        self.assertNotIn('bank_evidence', InsightDecision.model_json_schema()['properties'])
        self.assertNotIn('bank_evidence', decision.model_dump())
        self.assertEqual(model.i, 0)

    def test_result_reaches_the_writer_and_audit_as_an_allowed_source(self):
        card = insight()
        card.why_it_matters = [Claim(text='Доля компании в выручке группы — 30%.', sources=['fact:calc_1_share'])]
        model = ScriptedChatModel(responses=[calls(c('compute', query='Доля выручки компании в группе')),
            calls(c('decide', is_insight=True, score=2, company_link='Группа', mechanism='Доля выручки')),
            calls(c('Insight', **card.model_dump())), calls(c('Review', ok=True))])
        cheap = ScriptedChatModel(responses=[relevance(), consequences()])
        rendered = []
        original_write = InsightWriter.write
        def write(writer, company, role, decision, evidence):
            rendered.append(evidence.render())
            return original_write(writer, company, role, decision, evidence)
        with TemporaryDirectory() as tmp, patch.object(pipeline, 'build', return_value={'company': FACTS}), \
             patch.object(pipeline, 'candidates', return_value=(NEWS.head(1), [])), \
             patch('stage5_insight_detection.insight_judge.calculations.load_company', return_value=SimpleNamespace(card=CARD)), \
             patch.object(InsightWriter, 'write', write):
            runner = pipeline.InsightPipeline(cheap, model, related=related(),
                consequences=ConsequenceExtractor(cheap, JsonlCache('consequences', Path(tmp))))
            result = runner.run('company', 'cfo', 1, 1)[0]
        self.assertEqual(result['status'], 'ready', result)
        self.assertEqual(result['calculations'][-1]['value'], 30)
        self.assertIn('fact:calc_1_share', rendered[0])
        self.assertIn('fact:revenue / fact:group_revenue', rendered[0])
        self.assertFalse(any(w['code'] == 'unknown_ref' for w in result['audit']))

    def test_used_bank_sources_reach_the_writer_even_if_the_final_search_returns_nothing(self):
        source = bank_source()
        library = bank(source)
        library.search = Mock(return_value=[])
        card = insight()
        card.why_it_matters = [Claim(text='Ставка выше показателя Банка на 1 п.п.', sources=['fact:calc_1_change'])]
        card.bank_view = [Claim(text='Наш прогноз ставки 14%.', sources=[source['id']], kind='bank')]
        model = ScriptedChatModel(responses=[
            calls(c('search_bank_analytics', query='ставка'), c('compute', query='Разница ключевой ставки с показателем Банка')),
            calls(c('decide', is_insight=True, score=2, company_link='Кредит', mechanism='Ставка')),
            calls(c('Insight', **card.model_dump())), calls(c('Review', ok=True))])
        cheap = ScriptedChatModel(responses=[relevance(), consequences()])
        captured = []
        original_write = InsightWriter.write
        def write(writer, company, role, decision, evidence):
            captured.append(evidence)
            return original_write(writer, company, role, decision, evidence)
        with TemporaryDirectory() as tmp, patch.object(pipeline, 'build', return_value={'company': FACTS}), \
             patch.object(pipeline, 'candidates', return_value=(NEWS.head(1), [])), patch.object(InsightWriter, 'write', write):
            runner = pipeline.InsightPipeline(cheap, model, related=related(), bank=library,
                consequences=ConsequenceExtractor(cheap, JsonlCache('consequences', Path(tmp))))
            result = runner.run('company', 'cfo', 1, 1)[0]
        self.assertEqual(result['status'], 'ready', result)
        self.assertIn(source['id'], [b['id'] for b in captured[0].bank])
        self.assertEqual(result['insight']['bank_view'][0]['sources'], [source['id']])
        self.assertFalse(result['insight']['not_bank_position'])
        self.assertEqual(library.passages.call_count, 1)
        library.search.assert_not_called()

    def test_search_evidence_without_compute_reaches_writer_and_audit_in_full(self):
        rows = [bank_source(own_view=False, text='Ставка в сентябре составила 14%.'),
                bank_source(id='bank:report:p2.1', page=2,
                            text='Мы ожидаем снижения ставки. ' + 'Объяснение прогноза. ' * 70 + 'Конец прогноза.')]
        chunks, streams = build_chunks(rows, len, chunk_tokens=0, keep=lambda f: True)
        window = passages(chunks, streams, [0], context=1)
        library = SimpleNamespace(rows=rows, chunks=chunks, index={c['id']: i for i, c in enumerate(chunks)},
                                  twins={}, passages=Mock(return_value=window),
                                  search=Mock(side_effect=AssertionError('writer repeated the search')))
        card = insight()
        card.bank_view = [Claim(text='Банк ожидает снижения ставки.', sources=[chunks[1]['id']], kind='bank'),
                          Claim(text='Ставка в сентябре составила 14%.', sources=[chunks[0]['id']], kind='bank')]
        model = ScriptedChatModel(responses=[calls(c('search_bank_analytics', query='ставка')),
            calls(c('decide', is_insight=True, score=2, company_link='Кредит', mechanism='Ставка')),
            calls(c('Insight', **card.model_dump())), calls(c('Review', ok=True))])
        cheap = ScriptedChatModel(responses=[relevance(), consequences()])
        captured = []
        original_write = InsightWriter.write
        def write(writer, company, role, decision, evidence):
            captured.append(evidence.render())
            return original_write(writer, company, role, decision, evidence)
        with TemporaryDirectory() as tmp, patch.object(pipeline, 'build', return_value={'company': FACTS}), \
             patch.object(pipeline, 'candidates', return_value=(NEWS.head(1), [])), patch.object(InsightWriter, 'write', write):
            result = pipeline.InsightPipeline(cheap, model, related=related(), bank=library,
                consequences=ConsequenceExtractor(cheap, JsonlCache('consequences', Path(tmp)))).run('company', 'cfo', 1, 1)[0]
        self.assertEqual(result['status'], 'ready', result)
        self.assertNotIn('calculations', result)
        self.assertEqual([b['id'] for b in result['judge_bank']], [c['id'] for c in chunks])
        self.assertIn('Конец прогноза.', captured[0])  # beyond the old 900-character writer cutoff
        self.assertEqual([c['sources'] for c in result['insight']['bank_view']], [[chunks[1]['id']]])
        self.assertTrue(any(chunks[0]['id'] in c['sources'] for c in result['insight']['why_it_matters']))
        self.assertFalse(any(w['code'] in {'unknown_ref', 'no_source'} for w in result['audit']))
        library.passages.assert_called_once()
        library.search.assert_not_called()

    def test_only_displayed_twins_are_carried_and_repeated_searches_deduplicate(self):
        rows = [bank_source()] + [bank_source(id=f'bank:twin{i}:p1.1', report=f'Отчёт {i}',
                                             text=f'Похожая таблица {i}') for i in range(3)]
        chunks, streams = build_chunks(rows, len, chunk_tokens=0, keep=lambda f: True)
        library = SimpleNamespace(rows=rows, chunks=chunks, index={c['id']: i for i, c in enumerate(chunks)},
                                  twins={0: [1, 2, 3]}, passages=Mock(return_value=passages(chunks, streams, [0], context=0)))
        first = JudgeTools(41, FACTS, related(), library, ROW, 'company')
        second = JudgeTools(42, FACTS, related(), library, ROW, 'company')
        for _ in range(2):
            shown = first.search_bank_analytics('ставка')
        self.assertIn('Похожая таблица 1', shown)
        self.assertNotIn('Похожая таблица 2', shown)
        self.assertEqual(list(first.bank_tools.evidence), [c['id'] for c in chunks[:3]])
        self.assertEqual(second.bank_tools.evidence, {})

    def test_search_still_runs_as_fallback_when_judge_did_not_read_bank_material(self):
        library = bank(bank_source())
        library.search = Mock(return_value=[])
        card = insight()
        model = ScriptedChatModel(responses=[
            calls(c('decide', is_insight=True, score=2, company_link='Кредит', mechanism='Ставка')),
            calls(c('Insight', **card.model_dump())), calls(c('Review', ok=True))])
        cheap = ScriptedChatModel(responses=[relevance(), consequences()])
        with TemporaryDirectory() as tmp, patch.object(pipeline, 'build', return_value={'company': FACTS}), \
             patch.object(pipeline, 'candidates', return_value=(NEWS.head(1), [])):
            result = pipeline.InsightPipeline(cheap, model, related=related(), bank=library,
                consequences=ConsequenceExtractor(cheap, JsonlCache('consequences', Path(tmp)))).run('company', 'cfo', 1, 1)[0]
        self.assertEqual(result['status'], 'ready', result)
        self.assertNotIn('judge_bank', result)
        library.search.assert_called_once_with('Ставка')
        library.passages.assert_not_called()

    def test_unit_conversion_remains_an_explicit_hypothesis_after_the_final_audit(self):
        card = insight()
        model = ScriptedChatModel(responses=[calls(c('compute', query='Переведи 1 п.п. в б.п.')),
            calls(c('decide', is_insight=True, score=2, company_link='Кредит', mechanism='Ставка')),
            calls(c('Insight', **card.model_dump())), calls(c('Review', ok=True))])
        cheap = ScriptedChatModel(responses=[relevance(), consequences()])
        with TemporaryDirectory() as tmp, patch.object(pipeline, 'build', return_value={'company': FACTS}), \
             patch.object(pipeline, 'candidates', return_value=(NEWS.head(1), [])):
            runner = pipeline.InsightPipeline(cheap, model, related=related(),
                consequences=ConsequenceExtractor(cheap, JsonlCache('consequences', Path(tmp))))
            result = runner.run('company', 'cfo', 1, 1)[0]
        self.assertEqual(result['status'], 'ready', result)
        hypothesis = result['insight']['hypotheses'][0]
        self.assertEqual(hypothesis['kind'], 'hypothesis')
        self.assertEqual(hypothesis['sources'], ['fact:calc_1_units'])


if __name__ == '__main__':
    unittest.main()
