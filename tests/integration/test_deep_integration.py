"""Integration boundaries missed by the original merge tests. Synthetic inputs, no API or model downloads."""
import hashlib
import json
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import numpy as np
import pymupdf
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration
from pydantic import BaseModel

import pipeline
from common import news_corpus as corpus
from common.llm import FileLLMCache, structured
from common.llm.testing import ScriptedChatModel, calls, tool_call
from common.trace import AgentTrace
from stage5_insight_detection.analyst_loop.audit import audit, known
from stage5_insight_detection.analyst_loop.data_requests import Answer, DataRequest, RequestBroker
from stage5_insight_detection.analyst_loop.dossier import Dossier
from stage5_insight_detection.analyst_loop.enrich import Enricher, Enrichment
from stage5_insight_detection.analyst_loop.event import EventCard
from stage5_insight_detection.analyst_loop.loop import AnalystLoop
from stage5_insight_detection.analyst_loop.impact import indicator_delta
from stage5_insight_detection.analyst_loop.pool import ListNewsPool
from stage5_insight_detection.analyst_loop.profile import CompanyProfile
from stage5_insight_detection.analyst_loop.risks import bank_stance
from stage5_insight_detection.bank_analytics import assets, library, tables
from stage5_insight_detection.bank_analytics.digest import KeyFigure, ReportDigest, verify
from stage5_insight_detection.bank_analytics.digest import BudgetExceeded, ReportDigester
from stage5_insight_detection.bank_analytics.charts import ChartReader, ChartReadError, valid_answer
from stage5_insight_detection.bank_analytics.digitize import page_records
from stage5_insight_detection.bank_analytics.library import BankLibrary, best_pages, figures_of, page_texts, stance_of, table_parts
from stage5_insight_detection.bank_analytics.match import BankMatch, BankMatcher
from stage5_insight_detection.bank_analytics.ocr import AnalyticsOCR
from stage5_insight_detection.bank_analytics.review import BankReviewer, bank_positions
from stage5_insight_detection.bank_analytics.search import BankAnalytics
from stage6_insight_generation.insight_writer.insight import Claim, Review
from test_merge_regressions import NEWS, brief, insight, profile_snapshot

TODAY = date(2026, 10, 20)


def profile():
    return CompanyProfile('synthetic', '41.20', 'Строительство', 'Москва', 1e9, products=['loan'])


def pdf(path, text='', scans=0):
    with pymupdf.open() as doc:
        if text:
            page = doc.new_page()
            page.insert_text((50, 50), text)
        for _ in range(scans):
            doc.new_page()
        doc.save(path)
    return path


def table_record():
    rows = [['Показатель', '2025', '2026П'], ['Выручка', '1 710', '1 852'], ['Расходы', '100', '120']]
    return {'tables': [{'title': 'Финансы, млн ₽', 'bbox':[0,0,1000,1000], 'readings': {k: rows for k in ('gemma', 'qwen', 'tesseract')}}]}


class PdfIntegrationTests(unittest.TestCase):
    def test_mixed_pdf_keeps_text_and_recovers_scan_from_cache(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = pdf(root / 'mixed.pdf', 'The text page has enough readable words. ' * 5, 1)
            cache = root / 'ocr'
            cache.mkdir()
            (cache / 'mixed.jsonl').write_text(json.dumps({'key':'2', 'value':'Скан: ставка 14%.'}), encoding='utf-8')
            pages = page_texts(report, cache)
            self.assertIn('readable words', pages[0])
            self.assertEqual(pages[1], 'Скан: ставка 14%.')
            ocr = AnalyticsOCR(ScriptedChatModel(responses=[AIMessage(content='Новый скан')]), out_dir=root/'other')
            pages = page_texts(report, root/'other', ocr)
            self.assertIn('readable words', pages[0])
            self.assertEqual(pages[1], 'Новый скан')
            self.assertEqual(ocr.model.i, 0)  # exactly one response consumed, fake list cycles after one
            self.assertEqual(list(json.loads(ln)['key'] for ln in (root/'other/mixed.jsonl').read_text(encoding='utf-8').splitlines()), ['2'])

    def test_reconciled_tables_keep_chart_values_and_unmatched_tables(self):
        chart = '[График] Ставка\nОцифровано кодом\n| дата | ставка |\n|---|---|\n| 2026 | 14 |\n[/График]'
        text = chart + '\n\n| Выручка | I7I0 | I852 |\n| Расходы | 100 | 120 |\n\n| Тоннаж | 42 | 43 |'
        replaced = tables.replace_tables(text, table_record())
        self.assertIn(chart, replaced)
        self.assertIn('| Тоннаж | 42 | 43 |', replaced)
        self.assertNotIn('I7I0', replaced)
        self.assertIn('таблица сверена', replaced)

    def test_text_layer_gets_table_derivatives_in_both_search_engines(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = pdf(root/'synthetic.pdf', 'Revenue and cost table in a policy report. ' * 4)
            (root/'tables').mkdir()
            (root/'tables/synthetic_p1.json').write_text(json.dumps(table_record(), ensure_ascii=False), encoding='utf-8')
            prepared = best_pages(report, root/'ocr_full', root/'ocr')
            self.assertIn('таблица сверена', prepared[0])
            # This boundary checks prepared table text, not model embeddings. Keep the test offline.
            search = 'stage5_insight_detection.bank_analytics.search'
            tokenizer = SimpleNamespace(encode=lambda text, **kw:text.split())
            with patch(search+'.encoder',return_value=SimpleNamespace(tokenizer=tokenizer)), \
                 patch(search+'.embed',side_effect=lambda texts,folder:np.ones((len(texts),2))), \
                 patch(search+'.bm25_index',return_value=lambda query:[]):
                bank = BankAnalytics(root/'ocr', reports=[])
            self.assertTrue(any('таблица сверена' in c['text'] for c in bank.chunks))
            self.assertTrue(all(c['id'].startswith('bank:synthetic:p1.') for c in bank.chunks))

    def test_library_refresh_tracks_derivatives_and_deleted_reports(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = pdf(root/'synthetic.pdf', 'A readable policy report discussing revenue. ' * 4)
            def refresh():
                return BankLibrary.refresh(root, root/'ocr', TODAY, out=root/'fragments.jsonl',
                                           registry=root/'registry.json', full_dir=root/'ocr_full')
            initial = refresh()
            with patch.object(library, 'best_pages', wraps=best_pages) as parse:
                refresh()
                parse.assert_not_called()
            (root/'tables').mkdir()
            (root/'tables/synthetic_p1.json').write_text(json.dumps(table_record(), ensure_ascii=False), encoding='utf-8')
            changed = refresh()
            self.assertGreater(len(changed.fragments), len(initial.fragments))
            report.unlink()
            self.assertEqual(refresh().fragments, [])
            self.assertEqual(json.loads((root/'registry.json').read_text()), {})

    def test_replaced_pdf_invalidates_ocr_without_deleting_previous_output(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = pdf(root/'synthetic.pdf', scans=1)
            ocr = AnalyticsOCR(ScriptedChatModel(responses=[AIMessage(content='Старый текст'), AIMessage(content='Новый текст')]), out_dir=root/'ocr')
            self.assertEqual(ocr.transcribe(report), ['Старый текст'])
            pdf(root/'replacement.pdf', scans=2).replace(report)
            self.assertEqual(page_texts(report, root/'ocr'), ['', ''])
            self.assertEqual(ocr.transcribe(report, only_pages=[1]), ['Новый текст', ''])
            archived = list((root/'ocr').glob('*.previous.jsonl'))
            self.assertEqual(len(archived), 1)
            self.assertIn('Старый текст', archived[0].read_text(encoding='utf-8'))

    def test_pdf_bound_chart_assets_are_invalidated_and_preserved(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = pdf(root/'synthetic.pdf', scans=1)
            folder = root/'charts'
            assets.prepare(report, folder, 'recipe')
            (folder/'synthetic_p1.qwen.json').write_text('[{"title":"old"}]')
            self.assertEqual(page_records('synthetic', 1, folder)[0]['title'], 'old')
            pdf(root/'replacement.pdf', scans=2).replace(report)
            self.assertFalse(assets.current(report, folder))
            assets.prepare(report, folder, 'recipe')
            self.assertEqual(page_records('synthetic', 1, folder), [])
            self.assertTrue(list((folder/'previous').glob('*/synthetic_p1.qwen.json')))

    def test_table_thousands_currency_and_numeric_year_headers(self):
        parts = table_parts('Выручка, US$ млн', '| Показатель | 2025 | 2026П |\n|---|---|---|\n| Выручка | 1 710 | 1 852 |')
        figures = parts[0][1]
        self.assertEqual([(f.value, f.unit, f.period, f.kind) for f in figures],
                         [(1710, 'млн $', '2025', 'actual'), (1852, 'млн $', '2026П', 'forecast')])
        self.assertEqual(next(f for f in figures_of('Выручка 1 710 млн ₽.') if f.unit).value, 1710)


class TrustTests(unittest.TestCase):
    def test_own_view_wordings_and_first_person_cb_quote_stay_distinct(self):
        for text in ('Мы ждём снижения ставки.', 'По нашим оценкам, инфляция замедлится.', 'Наше мнение: ставка снизится.'):
            with self.subTest(text=text):
                self.assertTrue(stance_of(text)[0])
        for text in ('ЦБ: «Мы ожидаем снижения ставки».', 'Глава ЦБ заявила: «Мы ждём снижения ставки».'):
            with self.subTest(text=text):
                self.assertEqual(stance_of(text), (False, {}))

    def test_chart_description_and_axis_ticks_cannot_confirm_a_value(self):
        checked = verify(ReportDigest(summary='Тест', key_figures=[dict(indicator='ставка', value='9', unit='%', page=1)]),
                         {1:'[График] Ставка\nОсь Y: 2%; 9%.\nЧто видно: значение 9%.\n[/График]'})
        self.assertTrue(checked.key_figures[0].unverified)

    def test_bank_change_or_future_report_cannot_supply_a_rate_level(self):
        event = EventCard.from_news(dict(NEWS.iloc[0], title='Ключевая ставка выросла до 15%', text=''))
        for text, when in [('Ставка выросла на 1%.', '2026-10-10'), ('Ставка составила 1%.', '2026-10-19')]:
            with self.subTest(text=text, date=when):
                lib = BankLibrary.from_pages({'synthetic.pdf':[text]}, TODAY,
                                            [dict(file='synthetic.pdf', title='Synthetic', date=when)])
                delta = indicator_delta('ставка', event, [BankMatch(lib.fragments[0],1,{},'')])
                self.assertEqual(delta[:2], (1, 'п.п.'))
                self.assertEqual(delta[3], ['чувствительность на 1 п.п.'])

    def test_historical_year_does_not_turn_an_actual_into_a_forecast(self):
        actual = next(f for f in figures_of('В 2025 году инфляция составила 8%.') if f.unit == '%')
        self.assertEqual(actual.kind, 'actual')

    def test_bank_reference_prefix_does_not_match_other_fragment(self):
        self.assertFalse(known('bank:x:p1.1', {'bank:x:p1.10'}))
        self.assertTrue(known('bank:x:p1', {'bank:x:p1.10'}))
        self.assertFalse(known('news:4', {'news:41'}))

    def test_quoted_cb_position_does_not_become_bank_stance(self):
        own, stance = stance_of('ЦБ ожидает ускорения инфляции.\nМы ожидаем снижения ключевой ставки.')
        self.assertTrue(own)
        self.assertNotIn('инфляция', stance)
        self.assertEqual(stance['ставка'], 'down')
        lib = BankLibrary.from_pages({'synthetic.pdf': ['ЦБ ожидает ускорения инфляции.']}, TODAY,
                                     [dict(file='synthetic.pdf', date='2026-10-19', title='Synthetic')])
        fragment = lib.fragments[0]
        fragment.stance = {'инфляция': 'up'}  # old saved metadata
        self.assertEqual(bank_stance('инфляция', [BankMatch(fragment, 1, {}, '')]), (None, []))
        b = brief()
        b.refs = [fragment.id]
        b.bank = [BankMatch(fragment, 1, {}, '').dict()]
        ins = insight()
        ins.bank_view = [Claim(text='ЦБ ожидает ускорения инфляции.', sources=[fragment.id])]
        audited, _ = audit(ins, b, {})
        self.assertEqual(audited.bank_view, [])
        self.assertTrue(audited.not_bank_position)

    def test_digest_checks_currency_and_own_view(self):
        digest = ReportDigest(summary='Тест', key_figures=[KeyFigure(indicator='Выручка', value='1', unit='млрд ₽', page=1)],
                              bank_views=[dict(indicator='ставка', value='14', unit='%', page=1)])
        checked = verify(digest, {1: 'Выручка 1 млн ₽. ЦБ ожидает ставку 14%.'})
        self.assertTrue(checked.key_figures[0].unverified)
        self.assertTrue(checked.bank_views[0].unverified)

    def test_stale_digest_is_not_passed_to_writer(self):
        lib = BankLibrary.from_pages({'synthetic.pdf': ['Мы ожидаем ключевую ставку 14%.']}, TODAY,
                                    [dict(file='synthetic.pdf', date='2026-10-19', title='Synthetic')])
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            md = '# Synthetic\n\n## Страница 1\nМы ожидаем ключевую ставку 14%.'
            (root/'synthetic.md').write_text(md, encoding='utf-8')
            payload = ReportDigest(summary='Тест', bank_views=[dict(indicator='ставка', value='14', unit='%', page=1)]).model_dump()
            payload['source_sha256'] = hashlib.sha256(md.encode()).hexdigest()
            (root/'synthetic.digest.json').write_text(json.dumps(payload), encoding='utf-8')
            matches = [BankMatch(lib.fragments[0], 1, {}, '')]
            self.assertEqual(len(bank_positions(matches, root)), 1)
            (root/'synthetic.md').write_text(md.replace('14%', '15%'), encoding='utf-8')
            self.assertEqual(bank_positions(matches, root), [])

    def test_unknown_news_in_rejected_draft_does_not_abort_snapshot_save(self):
        rejected = dict(news_id=41, status='audit_rejected', draft={'sources':['news:999']}, verdict={'status':'reject'})
        bound = corpus.bind_result(rejected, NEWS, allow_unresolved=True)
        self.assertEqual(bound['unresolved_news_ids'], ['999'])
        self.assertNotIn('news:999', bound['news_references'])
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'result.json'
            corpus.save_results(path, [bound, dict(news_id=42, status='model_error')], NEWS, root/'news')
            self.assertEqual(len(corpus.load_results(path, NEWS)), 2)
            self.assertEqual(len(corpus.load_history(path, root/'news')), 2)
        spoofed = dict(bound, status='ready', insight={'sources':['news:999']})
        self.assertFalse(corpus.result_references(spoofed, NEWS)['ok'])
        with self.assertRaises(corpus.StaleNewsReferences):
            corpus.bind_result(dict(rejected, news_id=999), NEWS, allow_unresolved=True)

    def test_unknown_citations_on_ready_result_are_demoted_before_binding(self):
        bad = dict(news_id=41, status='ready', insight={'sources':['news:999']})
        bound = corpus.bind_result(bad, NEWS, allow_unresolved=True)
        self.assertNotIn('insight', bound)
        self.assertIn('draft', bound)
        self.assertEqual(bound['status'], 'audit_rejected')
        self.assertIn('insight', bad)  # caller input is not mutated


class LoopBoundaryTests(unittest.TestCase):
    def test_real_batch_binding_keeps_processing_after_unknown_draft_reference(self):
        pool = SimpleNamespace(related=SimpleNamespace(news=NEWS))
        models = SimpleNamespace(cheap=lambda **kw: None, agent=lambda **kw: SimpleNamespace(model_copy=lambda **kw: None))
        fake_loop = Mock(run=Mock(side_effect=[
            dict(news_id=41, status='audit_rejected', draft={'sources':['news:999']}, verdict={'status':'reject'}),
            dict(news_id=42, status='ready', insight={'sources':['news:42']})]))
        scores = [dict(id=n, importance=3, affected_roles=['cfo']) for n in (41,42)]
        with TemporaryDirectory() as tmp, patch.object(pipeline, 'DATA', Path(tmp)), \
             patch.object(pipeline, 'build', return_value={'synthetic':[]}), \
             patch.object(pipeline, 'candidates', return_value=(NEWS, [])), \
             patch.object(pipeline, 'InsightWriter', return_value=Mock()), \
             patch('stage5_insight_detection.analyst_loop.pool.RelatedNewsPool', return_value=pool), \
             patch('stage2_company_facts.profile.integration.load_profile', return_value=profile_snapshot(profile(), [])), \
             patch('stage4_news_relevance.news_classifier.classifier.NewsClassifier', return_value=SimpleNamespace(classify=lambda *a:scores)), \
             patch.object(BankLibrary, 'refresh', return_value=BankLibrary([])), \
             patch('stage5_insight_detection.analyst_loop.loop.AnalystLoop', return_value=fake_loop), \
             patch('stage6_insight_generation.insight_card.render.render', return_value='<html></html>'), \
             patch('common.trace.AgentTrace', return_value=AgentTrace(folder=None)):
            results = pipeline.run_loop('synthetic', 'cfo', models, n_candidates=2, n_judge=2)
            self.assertEqual([r['status'] for r in results], ['audit_rejected','ready'])
            corpus.save_results(Path(tmp)/'results.json', results, NEWS, Path(tmp)/'news')
            self.assertEqual(len(corpus.load_results(Path(tmp)/'results.json', NEWS)), 2)

    def test_extra_bank_search_obeys_event_filters(self):
        lib = BankLibrary.from_pages({'foreign.pdf':['В США ФРС повысила ставку. Мы ожидаем инфляцию в США.'],
                                     'domestic.pdf':['Банк России повысил ключевую ставку. Мы ожидаем снижения ставки.']}, TODAY)
        pool = ListNewsPool(NEWS.to_dict('records'))
        event = EventCard.from_news(NEWS.iloc[0].to_dict())
        matcher = BankMatcher(lib)
        hits = matcher.select(event, profile(), query='США ФРС инфляция')
        self.assertTrue(hits)
        self.assertFalse(any(m.fragment.file == 'foreign.pdf' for m in hits))
        with patch.object(matcher, 'match', side_effect=AssertionError('legacy filter bypass')):
            Enricher(pool, matcher).enrich(event, [], profile())
            loop = AnalystLoop(lib, pool, explain=False, trace=AgentTrace(folder=None))
            d = Dossier(event, profile(), 'cfo')
            loop.matcher = matcher
            loop._handlers(d)['bank_analytics'](DataRequest(kind='bank_analytics', query='США ФРС инфляция'))
            self.assertFalse(any(m.fragment.file == 'foreign.pdf' for m in d.bank))

    def test_followups_and_reprints_are_not_independent_confirmations(self):
        event = EventCard.from_news(NEWS.iloc[0].to_dict())
        enrichment = Enrichment(confirmations=[dict(NEWS.iloc[0], id=43), NEWS.iloc[1].to_dict()], same_event_ids={'43'})
        self.assertEqual(enrichment.independent_confirmations(event), 0)
        enrichment.same_event_ids.add('42')
        self.assertEqual(enrichment.independent_confirmations(event), 1)

    def test_duplicate_request_can_upgrade_an_unanswered_need_to_blocking(self):
        broker = RequestBroker({}, AgentTrace(folder=None))
        broker.dispatch([DataRequest(kind='computation', query='Расчёт')])
        broker.dispatch([DataRequest(kind='computation', query='расчёт', blocking=True)])
        self.assertEqual(len(broker.open), 1)
        self.assertEqual(len(broker.blocking_open()), 1)

    def test_last_round_success_is_recomputed_before_verdict(self):
        pool = ListNewsPool(NEWS.to_dict('records'))
        loop = AnalystLoop(BankLibrary([]), pool, explain=False, max_rounds=1, trace=AgentTrace(folder=None))
        with patch.object(loop, '_handlers', return_value={'bank_analytics': lambda r: Answer(r, True, 'Получены данные')}), \
             patch.object(loop, '_compute', wraps=loop._compute) as compute:
            result = loop.run(NEWS.iloc[0].to_dict(), profile(), 'cfo')
        self.assertEqual(compute.call_count, 2)
        self.assertIn('grounds', result)

    def test_payment_denominator_includes_unclassified_counterparties(self):
        uid = pd.DataFrame([dict(id='synthetic', REVENUE_COMPANY=1e9, GROUP_ID='g', okdiv='41', okved='41.20',
                                MAINOKVED_NAME='Строительство', REGION='Москва', FTS_COUNTRY_NAME=None, FTS_FLAG=0),
                            dict(id='known', REVENUE_COMPANY=1e8, GROUP_ID='other', okdiv='24')]).set_index('id')
        payments = pd.DataFrame([dict(PAYER='synthetic', BENEFICIARY='known', PAY_AMT=5),
                                 dict(PAYER='synthetic', BENEFICIARY='unknown', PAY_AMT=95)])
        with patch('common.data.load', return_value=(uid, payments, None, None)):
            p = CompanyProfile.from_tables('synthetic')
        self.assertEqual(p.pay_out, {'24': .05})
        self.assertEqual(p.pay_out_total, 100)

    def test_every_payment_exposure_has_a_fact_even_beyond_top_five(self):
        p = CompanyProfile('synthetic', '00', '', '', 1e9, pay_out={d:1/6 for d in ('01','07','24','35','41','62')})
        self.assertTrue(all(ref in p.facts for refs in p.reasons.values() for ref in refs))


class OutputBoundaryTests(unittest.TestCase):
    def test_invalid_research_verdict_leaves_a_weak_hypothesis(self):
        from stage2_company_facts.research_agent.agent import ResearchAgent
        from stage2_company_facts.research_agent.memory import Memory
        agent = ResearchAgent.__new__(ResearchAgent)
        agent.model = ScriptedChatModel(responses=[calls(tool_call('report_verdict', status='invalid', reason='test'))])
        agent.data_tools, agent.verify_steps = {}, 1
        memory = Memory()
        memory.propose_hypothesis('Синтетическая гипотеза', '', '', '')
        reply = agent._verify('synthetic', 'H1', memory)
        self.assertIn('weak', reply)
        self.assertEqual(memory.hypothesis('H1')['status'], 'weak')

    def test_tool_loop_cannot_accept_a_truncated_final_decision(self):
        from common.llm import ToolLoop
        class Result(BaseModel):
            value: int
        response = calls(tool_call('Result', value=7))
        response.response_metadata['finish_reason'] = 'length'
        loop = ToolLoop(ScriptedChatModel(responses=[response]), [], Result, 'Synthetic', max_steps=1)
        self.assertIsNone(loop.run('Synthetic'))

    def test_classifier_retry_escapes_invalid_cached_answer(self):
        from stage4_news_relevance.news_classifier.classifier import NewsClassifier
        with TemporaryDirectory() as tmp:
            cache = FileLLMCache(Path(tmp))
            response = calls(tool_call('scores', items=[dict(index=0, category='ставка_и_кредит', sentiment='negative', importance=3)]))
            model = ScriptedChatModel(cache=cache, responses=[AIMessage(content='broken'), response])
            result = NewsClassifier(model).classify(NEWS.head(1).to_dict('records'), 'Синтетическая компания')
            self.assertEqual([r['id'] for r in result], [41])
            self.assertEqual(len(cache.store._load()), 2)

    def chart_reader(self, responses):
        reader = ChartReader.__new__(ChartReader)
        reader.model = ScriptedChatModel(responses=responses)
        reader.budget, reader.max_call, reader.calls = None, 0, []
        reader.meter = SimpleNamespace(cost=0)
        return reader

    def test_chart_grounding_retries_invalid_answer_instead_of_caching_empty_page(self):
        reader = self.chart_reader([AIMessage(content='{}'), AIMessage(content='[]')])
        data, _ = reader.ask('chart_boxes', b'synthetic image', 'synthetic')
        self.assertEqual(data, [])
        self.assertEqual(len(reader.calls), 2)
        bad = self.chart_reader([AIMessage(content='broken')])
        with self.assertRaises(ChartReadError):
            bad.ask('chart_boxes', b'synthetic image', 'synthetic')

    def test_chart_geometry_and_truncation_are_validated(self):
        self.assertFalse(valid_answer('chart_boxes', [{'bbox':[900,0,100,800]}]))
        self.assertFalse(valid_answer('chart_boxes', [{'bbox':[0,0,1001,800]}]))
        reader = self.chart_reader([AIMessage(content='[]', response_metadata={'finish_reason':'length'}), AIMessage(content='[]')])
        self.assertEqual(reader.ask('chart_boxes', b'synthetic image', 'synthetic')[0], [])
        self.assertEqual(len(reader.calls), 2)

    def metered_model(self, responses, cost):
        class MeteredModel(ScriptedChatModel):
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
                result.llm_output = {'token_usage': {'cost': cost}}
                return result
        return MeteredModel(responses=responses)

    def test_bank_reviewer_meters_injected_models_and_checks_budget_before_retry(self):
        lib = BankLibrary.from_pages({'synthetic.pdf':['Мы ожидаем снижения ключевой ставки.']}, TODAY)
        matches = [BankMatch(lib.fragments[0], 1, {}, '')]
        model = self.metered_model([AIMessage(content='broken')], .1)
        reviewer = BankReviewer(model, budget=.14)
        with self.assertRaises(BudgetExceeded):
            reviewer.review(EventCard.from_news(NEWS.iloc[0].to_dict()), profile(), matches)
        self.assertAlmostEqual(reviewer.meter.cost, .1)

    def test_digester_meters_injected_models(self):
        model = self.metered_model([calls(tool_call('ReportDigest', summary='Синтетическая сводка'))], .2)
        digester = ReportDigester(model, budget=.25)
        digester.digest('# Report\n\n## Страница 1\nТекст страницы.')
        self.assertAlmostEqual(digester.meter.cost, .2)
        with self.assertRaises(BudgetExceeded):
            digester.digest('# Report\n\n## Страница 1\nТекст страницы.')

    def test_valid_but_truncated_response_is_retried_and_not_cached(self):
        class Result(BaseModel):
            value: int
        cut = calls(tool_call('Result', value=7))
        cut.response_metadata['finish_reason'] = 'length'
        self.assertFalse(FileLLMCache.complete([ChatGeneration(message=cut)]))
        model = ScriptedChatModel(responses=[cut, calls(tool_call('Result', value=8))])
        self.assertEqual(structured(model, Result).invoke('Synthetic').value, 8)

    def test_default_judge_publishes_despite_a_number_absent_from_sources(self):
        # the number-vs-source audit was removed: it rejected checker-approved insights over page numbers and signs
        row = NEWS.iloc[0].to_dict()
        related = SimpleNamespace(event_plausibility=lambda _:1, row_of={41:row}, get=lambda _:row)
        runner = pipeline.InsightPipeline.__new__(pipeline.InsightPipeline)
        runner.related, runner.bank = related, None
        runner.relevance = Mock(score=Mock(return_value=SimpleNamespace(score=2, model_dump=lambda:{})))
        runner.consequences = Mock(extract=Mock(return_value={}))
        from stage5_insight_detection.insight_judge.judge import InsightDecision
        runner.judge = Mock(decide=Mock(return_value=InsightDecision(is_insight=True, score=2, company_link='Кредит', mechanism='Ставка')))
        ins = insight()
        ins.what_happened = [Claim(text='Ключевая ставка 999%', sources=['news:41'])]
        runner.writer = Mock(write=Mock(return_value=(ins, Review(ok=True))))
        with patch.object(pipeline, 'build', return_value={'synthetic':[]}), patch.object(pipeline, 'candidates', return_value=(NEWS.head(1), [])):
            result = runner.run('synthetic', 'cfo')[0]
        self.assertEqual(result['status'], 'ready')
        self.assertIn('insight', result)


if __name__ == '__main__':
    unittest.main()
