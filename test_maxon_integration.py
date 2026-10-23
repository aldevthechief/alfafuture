"""Offline regressions for the Maxon/Kirill/Vladimir merge. No encoders, downloads or paid model calls."""
import argparse
from contextlib import redirect_stderr
from copy import deepcopy
from datetime import date
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pipeline
from common.news_corpus import load_results, save_results
from stage2_company_facts.profile.context import CONTEXT_CATEGORY
from stage5_insight_detection.analytics_insights.agent import AnalyticsCandidate, AnalyticsInsightAgent, AnalyticsTools, Finish
from stage5_insight_detection.analytics_insights.service import cache_key
from stage5_insight_detection.analytics_insights.to_insight import candidate_to_insight
from stage5_insight_detection.analyst_loop.audit import audit
from stage5_insight_detection.analyst_loop.verdict import InsightBrief
from stage5_insight_detection.bank_analytics.chunks import build_chunks, passages
from stage6_insight_generation.insight_writer import insight_v2 as v2
from stage6_insight_generation.insight_writer.insight import Claim
from test_merge_regressions import NEWS, insight


CONTEXT = {'cat': CONTEXT_CATEGORY, 'depth': 1, 'text': 'Гипотеза по правилам профиля: [sensitivity] Ставка 0.8'}


class AnalyticsProfileTests(unittest.TestCase):
    def test_analytics_compacts_only_against_facts_in_its_actual_prompt(self):
        from stage2_company_facts.profile.test_context import LINKS, CONCENTRATION
        facts = [{'cat': 'activity', 'depth': 1, 'text': 'Производство стали'},
                 dict(CONCENTRATION, depth=1), LINKS, CONTEXT]
        for limit in (1, 2):
            with self.subTest(limit=limit), patch('stage5_insight_detection.analytics_insights.agent.ToolLoop') as loop:
                loop.return_value.run.return_value = Finish(summary='готово')
                agent = AnalyticsInsightAgent(model=Mock(), bank=Mock(available=['Отчёт']), vision=Mock(), key_facts=limit)
                agent.run('company', facts, 'cfo', '2026-10-23')
                task = loop.return_value.run.call_args.args[0]
            self.assertEqual('крупнейший 50%, топ-5 90%' in task, limit == 1)
            self.assertIn('Ставка 0.8', task)
        self.assertIn('крупнейший 50%, топ-5 90%', facts[2]['text'])

    def test_pipeline_loads_the_profile_for_the_new_analytics_agent(self):
        facts = [{'cat': 'bank', 'text': 'Лизинг', 'depth': 1}, CONTEXT]
        bank, models, result = Mock(), Mock(), object()
        with patch.object(pipeline, 'load_company', return_value=SimpleNamespace(facts=facts)) as load, \
             patch('stage5_insight_detection.analytics_insights.service.analytics_run', return_value=result) as run:
            self.assertEqual(pipeline.run_analytics('company', 'cfo', models, bank=bank), (result, ''))
        load.assert_called_once_with('company')
        self.assertIs(run.call_args.args[2], facts)

    def test_context_is_in_the_prompt_beyond_25_facts_but_not_in_the_fact_tool(self):
        facts = [{'cat': 'activity', 'depth': 1, 'text': f'исходный факт {i}'} for i in range(30)]
        facts.insert(1, CONTEXT)
        bank = Mock(available=['Отчёт'])
        with patch('stage5_insight_detection.analytics_insights.agent.ToolLoop') as loop:
            loop.return_value.run.return_value = Finish(summary='готово')
            agent = AnalyticsInsightAgent(model=Mock(), bank=bank, vision=Mock())
            self.assertEqual(agent.run('company', facts, 'cfo', '2026-10-23'), [])
        task = loop.return_value.run.call_args.args[0]
        self.assertIn('[fact:25][activity] исходный факт 24', task)
        self.assertNotIn('[fact:26]', task)
        self.assertIn('## Пояснения из профиля компании', task)
        self.assertIn('Ставка 0.8', task)
        self.assertIn('не вероятность', task)
        tools = AnalyticsTools(facts, bank, Mock(), 5)
        self.assertNotIn('profile_context', tools.company_facts())
        self.assertIn('[fact:26][activity] исходный факт 25', tools.company_facts())

        cand = AnalyticsCandidate(kind='forecast', title='Проверка', conclusion='Вывод', channel='ставка',
                                  effect='risk', bank_claims=[], company_facts=[26],
                                  link_inference='Связь', mechanism='Механизм')
        mapped = candidate_to_insight(cand, 'company', 'cfo', [], facts)
        self.assertEqual(mapped.why_it_matters[0].text, 'исходный факт 25')
        self.assertEqual(mapped.why_it_matters[0].sources, ['fact:26'])

    def test_profile_context_changes_invalidate_the_analytics_cache(self):
        with TemporaryDirectory() as tmp:
            first = cache_key('company', 'cfo', [CONTEXT], Path(tmp))
            changed = cache_key('company', 'cfo', [dict(CONTEXT, text='Другая чувствительность')], Path(tmp))
        self.assertNotEqual(first, changed)


class BankMetadataTests(unittest.TestCase):
    def row(self, n, text, own=None, when='2026-10-01', **extra):
        row = dict(id=f'bank:report:p1.{n}', report='Отчёт', date=when, page=1,
                   kind='text', text=text, freshness='fresh', **extra)
        if own is not None:
            row['own_view'] = own
        return row

    def chunks(self, rows, **kw):
        return build_chunks(rows, len, keep=lambda f: True, as_of=date(2026, 10, 23), **kw)

    def test_original_attribution_survives_and_missing_ocr_metadata_is_inferred(self):
        rows = [self.row(1, 'Инфляция в сентябре составила 8%.', False),
                self.row(2, 'Мы ожидаем инфляцию 6% в следующем году.', True),
                self.row(3, 'По прогнозу ЦБ инфляция снизится.', False),
                self.row(4, 'Мы ожидаем снижения ключевой ставки до 12%.')]
        chunks, streams = self.chunks(rows, chunk_tokens=0)
        self.assertEqual([c['own_view'] for c in chunks], [False, True, False, True])
        self.assertEqual(chunks[1]['view_frag_ids'], [rows[1]['id']])
        self.assertEqual(chunks[0]['view_frag_ids'], [])
        window = passages(chunks, streams, [0], context=0)[0]
        self.assertFalse(window['own_view'])
        self.assertEqual(window['freshness'], 'fresh')

    def test_attribution_is_preserved_through_real_recursive_chunking(self):
        rows = [self.row(1, 'Мы ожидаем снижения ставки. ' * 12, True)]
        chunks, _ = self.chunks(rows, chunk_tokens=100, overlap_tokens=10)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c['own_view'] and c['view_frag_ids'] == [rows[0]['id']] for c in chunks))

    def test_stale_undated_and_future_reports_cannot_look_fresh(self):
        for when, expected in [('2021-04-15', 'stale'), ('', 'unknown'), ('2027-01-01', 'unknown')]:
            with self.subTest(when=when):
                chunks, _ = self.chunks([self.row(1, 'Наш прогноз — 6%.', True, when)], chunk_tokens=0)
                self.assertEqual(chunks[0]['freshness'], expected)
                self.assertTrue(chunks[0]['stale'])

    def test_audit_keeps_the_banks_own_view_but_not_statistics_or_stale_views(self):
        for when, kept in [('2026-10-01', 1), ('2021-04-15', 0)]:
            with self.subTest(when=when):
                rows = [self.row(1, 'Фактическая инфляция 8%.', False, when),
                        self.row(2, 'Наш прогноз инфляции 6%.', True, when)]
                chunks, _ = self.chunks(rows, chunk_tokens=0)
                card = insight()
                card.bank_view = [Claim(text=c['text'], sources=[c['id']], kind='bank') for c in chunks]
                brief = InsightBrief(refs=[c['id'] for c in chunks], facts=[], consequences=[], risks=[], bank=chunks,
                                     hypotheses=[], impact={'level': 'low', 'criteria': 'тест'},
                                     not_bank_position=False, warnings=[])
                card, _ = audit(card, brief, {c['id']: c['text'] for c in chunks})
                self.assertEqual(len(card.bank_view), kept)
                if kept:
                    self.assertEqual(card.bank_view[0].sources, [chunks[1]['id']])
                self.assertTrue(any(c.text == rows[0]['text'] for c in card.why_it_matters))


class ContractMigrationTests(unittest.TestCase):
    def bank_result(self, key='insight'):
        card = insight()
        card.origin = 'bank'
        card.what_happened = [Claim(text='Банк ожидает ставку 12%.', sources=['bank:report:p1.1'], kind='bank',
                                   evidence=[dict(source='bank:report:p1.1', quote='Банк ожидает ставку 12%.', start=5, end=30)])]
        return {'origin': 'bank', 'news_id': None, 'status': 'ready' if key == 'insight' else 'review_rejected',
                key: card.model_dump(), 'review': {'ok': key == 'insight', 'issues': ['Оговорка']},
                'audit': [{'code': 'no_number', 'text': 'Замечание аудита'}],
                'bank': [{'id': 'bank:report:p1.1', 'report': 'Отчёт', 'date': '2026-10-01', 'page': 1}],
                'verdict': {'status': 'insight', 'reasons': ['Обоснован']}, 'open_requests': [{'query': 'Уточнить'}]}

    def test_bank_ready_and_rejected_drafts_migrate_without_losing_provenance(self):
        for key in ('insight', 'draft'):
            with self.subTest(key=key):
                original = self.bank_result(key)
                snapshot = deepcopy(original)
                migrated = v2.migrate_result(original, 'company')
                card = v2.Insight.model_validate(migrated[key])
                self.assertEqual(card.origin, 'bank')
                self.assertEqual(card.trigger.reports, [{'title': 'Отчёт', 'date': '2026-10-01'}])
                self.assertEqual(card.trigger.news_ids, [])
                self.assertEqual(v2.to_v1(card).model_dump(), original[key])
                self.assertEqual(card.review.warnings, ['Оговорка', 'Замечание аудита'])
                self.assertEqual(original, snapshot)
                self.assertEqual(migrated['open_requests'], original['open_requests'])
                self.assertEqual(migrated['audit'], original['audit'])
                self.assertEqual(v2.migrate_result(migrated, 'company'), migrated)

    def test_bank_cards_can_be_saved_and_loaded_with_a_news_batch(self):
        bank = v2.migrate_result(self.bank_result(), 'company')
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path = folder / 'result.json'
            save_results(path, [bank], NEWS, folder / 'news')
            loaded = load_results(path, NEWS)
        self.assertEqual(loaded[0]['insight'], bank['insight'])
        self.assertEqual(loaded[0]['news_references'], {})

    def test_direct_v1_conversion_preserves_bank_origin(self):
        old = insight()
        old.origin = 'bank'
        self.assertEqual(v2.from_v1(old, 'company').origin, 'bank')


class SourceChoiceTests(unittest.TestCase):
    def test_alternative_bank_theses_require_an_explicit_news_only_loop(self):
        for engine, source in [('judge', 'news'), ('loop', 'both'), ('loop', 'analytics')]:
            with self.subTest(engine=engine, source=source), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    pipeline.validate_sources(argparse.ArgumentParser(), SimpleNamespace(
                        engine=engine, source=source, bank_insights=True))
        pipeline.validate_sources(argparse.ArgumentParser(), SimpleNamespace(
            engine='loop', source='news', bank_insights=True))

    def test_the_main_analytics_agent_remains_available_with_either_engine(self):
        for engine in ('judge', 'loop'):
            for source in ('news', 'analytics', 'both'):
                pipeline.validate_sources(argparse.ArgumentParser(), SimpleNamespace(
                    engine=engine, source=source, bank_insights=False))


if __name__ == '__main__':
    unittest.main()
