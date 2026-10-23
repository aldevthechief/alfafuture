"""Regression checks for the combined backup pipeline and Vladimir's PDF/chart functionality. No LLM API calls."""
from contextlib import redirect_stdout
from datetime import date
from io import StringIO
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pymupdf
from langchain_core.messages import AIMessage

from common.llm.testing import ScriptedChatModel, calls, tool_call
from common.trace import AgentTrace
from stage5_insight_detection.analyst_loop.audit import audit
from stage5_insight_detection.analyst_loop.event import EventCard
from stage5_insight_detection.analyst_loop.impact import indicator_delta
from stage5_insight_detection.analyst_loop.profile import CompanyProfile
from stage5_insight_detection.bank_analytics.library import BankLibrary, Fragment, Figure, figures_of
from stage5_insight_detection.bank_analytics.match import BankMatch
from stage5_insight_detection.bank_analytics.ocr import AnalyticsOCR, BudgetExceeded
from stage5_insight_detection.bank_analytics.review import BankReviewer
from stage6_insight_generation.insight_writer.insight import Claim
from test_merge_regressions import NEWS, brief, insight, profile_snapshot


class Related:
    def __init__(self):
        self.news = NEWS
    def refresh(self):
        pass


class CliTests(unittest.TestCase):
    def test_actual_main_accepts_retrieval_ocr_and_saves_a_bound_empty_batch(self):
        # Exercise argparse and the actual __main__, not just InsightPipeline.run().
        for engine, bank_review in (('judge', False), ('loop', False), ('loop', True)):
            with self.subTest(engine=engine, bank_review=bank_review), tempfile.TemporaryDirectory() as tmp:
                model = ScriptedChatModel(responses=[AIMessage(content='unused')])
                factory = SimpleNamespace(cheap=lambda **kw: model, agent=lambda **kw: model)
                args = ['pipeline.py', 'synthetic', '--engine', engine, '--source', 'news', '--retrieval', 'lexical', '--ocr']
                if bank_review:
                    args += ['--bank-review', '--bank-budget', '1.5']
                with patch.object(sys, 'argv', args), patch('common.data.DATA', Path(tmp)), \
                     patch('common.search.news_pool', return_value=NEWS), \
                     patch('common.llm.ChatModels', return_value=factory), \
                     patch('stage2_company_facts.deterministic_facts.facts.build', return_value={'synthetic': []}), \
                     patch('stage3_news_search.retrieval.candidates', return_value=(NEWS.head(0), [])), \
                     patch('stage5_insight_detection.news_enrichment.related_news.RelatedNews', Related), \
                     patch('stage5_insight_detection.bank_analytics.search.BankAnalytics'), \
                     patch('stage2_company_facts.profile.integration.load_profile', return_value=profile_snapshot()), \
                     patch('common.trace.AgentTrace', return_value=AgentTrace(folder=None)), \
                     patch('stage5_insight_detection.bank_analytics.ocr.AnalyticsOCR') as ocr, \
                     patch('stage5_insight_detection.bank_analytics.review.BankReviewer') as reviewer, \
                     patch.object(BankLibrary, 'refresh', return_value=BankLibrary([])) as refresh, \
                     patch.object(BankLibrary, 'load', return_value=BankLibrary([])), \
                     patch('stage5_insight_detection.bank_analytics.library.FRAGMENTS') as fragments, \
                     redirect_stdout(StringIO()):
                    fragments.exists.return_value = True
                    runpy.run_path('pipeline.py', run_name='__main__')
                    refresh.assert_any_call(ocr=ocr.return_value, rebuild=True)
                    self.assertEqual(refresh.call_count, 2 if engine == 'loop' else 1)
                    if bank_review:
                        reviewer.assert_called_once_with(budget=1.5)
                    else:
                        reviewer.assert_not_called()
                self.assertEqual(json.loads((Path(tmp) / 'insights/synthetic_cfo.json').read_text()), [])
                self.assertTrue(list((Path(tmp) / 'news/versions').glob('*/news.parquet')))


class OcrTests(unittest.TestCase):
    def pdf(self, folder, pages=2):
        path = Path(folder) / 'synthetic.pdf'
        with pymupdf.open() as doc:
            for _ in range(pages): doc.new_page()
            doc.save(path)
        return path

    def test_real_pdf_rendering_and_cache_need_no_api_on_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf = self.pdf(tmp)
            model = ScriptedChatModel(responses=[AIMessage(content='Первая страница'), AIMessage(content='Вторая страница')])
            ocr = AnalyticsOCR(model=model, out_dir=Path(tmp)/'ocr')
            self.assertEqual(ocr.transcribe(pdf), ['Первая страница', 'Вторая страница'])
            with patch.object(ScriptedChatModel, 'invoke', side_effect=AssertionError('cached OCR called the model')):
                self.assertEqual(ocr.transcribe(pdf), ['Первая страница', 'Вторая страница'])

    def test_progress_distinguishes_pages_and_failure_reasons(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = ScriptedChatModel(responses=[AIMessage(content=''), AIMessage(content='Первая страница'),
                AIMessage(content='Повтор строки\n' * 12), AIMessage(content='Вторая страница')])
            ocr = AnalyticsOCR(model=model, out_dir=Path(tmp)/'ocr')
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(ocr.transcribe(self.pdf(tmp)), ['Первая страница','Вторая страница'])
            text = output.getvalue()
            self.assertIn('synthetic.pdf p1/2: attempt 1/2 failed', text)
            self.assertIn('empty OCR response', text)
            self.assertIn('synthetic.pdf p1/2: attempt 2/2 OK', text)
            self.assertIn('synthetic.pdf p2/2: attempt 1/2 failed', text)
            self.assertIn('one line repeated 12 times', text)
            self.assertIn('synthetic.pdf p2/2: attempt 2/2 OK', text)
            with redirect_stdout(output), patch.object(ScriptedChatModel,'invoke',side_effect=AssertionError('cache called LLM')):
                ocr.transcribe(Path(tmp)/'synthetic.pdf')
            self.assertIn('2 pages selected, 2 cached', output.getvalue())

    def test_budget_stops_before_next_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = ScriptedChatModel(responses=[AIMessage(content='Страница', response_metadata={'token_usage': {'cost': .1}})])
            ocr = AnalyticsOCR(model=model, budget=.15, out_dir=Path(tmp)/'ocr')
            with self.assertRaises(BudgetExceeded): ocr.transcribe(self.pdf(tmp))
            self.assertAlmostEqual(ocr.cost, .1)

    def test_failed_page_is_skipped_and_not_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            ocr = AnalyticsOCR(model=ScriptedChatModel(responses=[]), out_dir=Path(tmp)/'ocr')
            with patch.object(ScriptedChatModel, 'invoke', side_effect=TimeoutError('synthetic timeout')) as invoke, \
                    redirect_stdout(StringIO()) as output:
                self.assertEqual(ocr.transcribe(self.pdf(tmp, 1)), [''])
            self.assertEqual(invoke.call_count, 2)
            self.assertEqual(len(ocr.skipped), 1)
            self.assertIn('synthetic.pdf p1/1: skipped after 2 failed attempts; not cached',output.getvalue())
            self.assertFalse((Path(tmp)/'ocr/synthetic.jsonl').exists())

    def test_empty_repeating_and_truncated_responses_are_retried(self):
        for bad in (AIMessage(content=''), AIMessage(content='Строка повторяется\n' * 12),
                    AIMessage(content='Обрезанный текст', response_metadata={'finish_reason':'length'})):
            with self.subTest(response=bad.content[:20]), tempfile.TemporaryDirectory() as tmp:
                model = ScriptedChatModel(responses=[bad, AIMessage(content='Корректный текст')])
                ocr = AnalyticsOCR(model=model, out_dir=Path(tmp)/'ocr')
                with redirect_stdout(StringIO()):
                    self.assertEqual(ocr.transcribe(self.pdf(tmp, 1)), ['Корректный текст'])
                self.assertEqual(ocr.skipped, [])


class CombinedNumericTests(unittest.TestCase):
    def test_measure_and_verification_are_independent(self):
        level = next(f for f in figures_of('Инфляция снизилась до 8%') if f.unit == '%')
        self.assertEqual(level.measure, 'level')
        self.assertIs(level.verified, True)
        self.assertEqual(Figure(2, '%', 'инфляция', 'actual', '', '', False, 'approx').measure, 'unknown')

    def test_unverified_bank_chart_cannot_supply_rate_baseline(self):
        fragment = Fragment('bank:synthetic:p1.1', 'synthetic.pdf', 'Synthetic', '2026-10-20', 1, 1, '',
                            '', ['ставка'], [{'value':14,'unit':'%','indicator':'ставка','kind':'actual',
                             'period':'','verified':False,'status':'approx'}], False, freshness='fresh')
        event = EventCard.from_news(dict(NEWS.iloc[0], title='Ставка повысилась до 15%', text=''))
        delta = indicator_delta('ставка', event, [BankMatch(fragment,1,{},'')])
        self.assertEqual(delta[3], ['чувствительность на 1 п.п.'])

class BankReviewTests(unittest.TestCase):
    def candidates(self):
        lib = BankLibrary.from_pages({'SYNTH-Policy_10Oct2026.pdf':['Мы ожидаем сохранения ключевой ставки на 14%.']},
                                     date(2026,10,20))
        return [BankMatch(lib.fragments[0],1,{},'synthetic')]

    def test_invalid_responses_fall_back_to_code(self):
        reviewer = BankReviewer(ScriptedChatModel(responses=[AIMessage(content='bad')]))
        event = EventCard.from_news(NEWS.iloc[0].to_dict())
        matches = self.candidates()
        picked, result = reviewer.review(event, CompanyProfile('synthetic','41.20','Строительство','Москва',1e9),matches)
        self.assertEqual(picked, matches)
        self.assertEqual(result['by'],'code')

    def test_larger_retry_recovers_after_invalid_responses(self):
        matches = self.candidates()
        model = ScriptedChatModel(responses=[AIMessage(content='bad'),
            calls(tool_call('BankReview', items=[dict(id=matches[0].fragment.id, fits=True, channel='ставка')]))])
        picked, result = BankReviewer(model).review(EventCard.from_news(NEWS.iloc[0].to_dict()),
                           CompanyProfile('synthetic','41.20','Строительство','Москва',1e9),matches)
        self.assertEqual(picked, matches)
        self.assertEqual(result['by'],'agent')


if __name__ == '__main__':
    unittest.main()
