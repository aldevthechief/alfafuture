"""Flag-driven bank preparation and evidence delivery. Synthetic PDFs/models only, no network."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import pymupdf
from langchain_core.messages import AIMessage

import pipeline
from common.llm.testing import ScriptedChatModel, calls, tool_call
from common.trace import AgentTrace
from stage5_insight_detection.analyst_loop.loop import AnalystLoop
from stage5_insight_detection.analyst_loop.pool import ListNewsPool
from stage5_insight_detection.bank_analytics import charts, prepare, tables
from stage5_insight_detection.bank_analytics.digest import BudgetExceeded
from stage5_insight_detection.bank_analytics.library import BankLibrary
from stage5_insight_detection.bank_analytics.prepare import PreparedBank, PreparationOptions, prepare_bank
from stage6_insight_generation.insight_writer.insight import Claim, Review
from test_merge_regressions import NEWS, insight, profile_snapshot
from test_deep_integration import profile

ROWS = [['Показатель', '2025', '2026'], ['Ставка', '14', '15'], ['Кредит', '16', '17']]
PAGE = ('Мы ожидаем снижения ключевой ставки.\n\n'
        '[График] Ключевая ставка, %\nЗначения: ставка: 14%\nЧто видно: пауза\n[/График]\n\n'
        'Ключевая ставка, %\n| Показатель | 2025 | 2026 |\n|---|---|---|\n| Ставка | 14 | 15 |\n| Кредит | 16 | 17 |')
CHART = {'title':'Ключевая ставка, %', 'x_axis':{'first':'2026','last':'2026'}, 'y_axis':{'unit':'%'},
         'series':[{'name':'Ставка','printed_values':[{'x':'2026','value':'14'}]}]}


class FakeReader:
    requests = []
    def __init__(self, *args, **kwargs):
        self.meter = SimpleNamespace(cost=0)
        self.max_call, self.calls = 0, []
        self.model = ScriptedChatModel(responses=[])
    def boxes(self, pdf, page):
        self.requests.append('chart_boxes')
        return [{'title':CHART['title'], 'bbox':[100,100,900,500]}]
    def read(self, png, what):
        self.requests.append('ocr_chart')
        self.calls.append(dict(what=what, cost=0, sec=0))
        return CHART
    def ask(self, prompt, png, what):
        self.requests.append(prompt)
        if prompt == 'table_boxes':
            return [{'title':'Ключевая ставка, %', 'kind':'table', 'bbox':[100,550,900,950]}], ''
        return {'header':ROWS[0], 'rows':ROWS[1:]}, ''


class PreparationTests(unittest.TestCase):
    def test_chart_mode_freezes_environment_and_explicit_value_wins(self):
        with patch.dict(os.environ, {'BANK_CHARTS':'selective'}):
            options = PreparationOptions()
            self.assertEqual(options.chart_mode, 'selective')
            self.assertEqual(PreparationOptions(chart_mode='all').chart_mode, 'all')
        with patch.dict(os.environ, {'BANK_CHARTS':'off'}):
            self.assertEqual(options.chart_mode, 'selective')
        with self.assertRaises(ValueError):
            PreparationOptions(chart_mode='unknown')

    def test_chart_mode_reaches_reading_library_and_digest_export(self):
        for mode in ('all', 'selective', 'off'):
            with self.subTest(mode=mode), TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.make_pdf(root)
                models = SimpleNamespace(callbacks=[],
                    vision=lambda: ScriptedChatModel(responses=[AIMessage(content=PAGE)]),
                    agent=lambda: ScriptedChatModel(responses=[calls(tool_call('ReportDigest', summary='Обзор ставки'))]))
                with patch.dict(os.environ, {'BANK_CHARTS':'off' if mode != 'off' else 'all'}), \
                     patch.object(prepare, 'check_local_ocr') as local_ocr, \
                     patch.object(prepare, 'read_charts', return_value={}) as read, \
                     patch.object(BankLibrary, 'refresh', wraps=BankLibrary.refresh) as refresh, \
                     patch.object(prepare, 'export', wraps=prepare.export) as export, redirect_stdout(StringIO()):
                    result = prepare_bank(PreparationOptions(charts=True, digests=True, chart_mode=mode), models, pdf_dir=root)
                self.assertEqual(result.report['status'], 'complete')
                self.assertEqual(result.report['options']['chart_mode'], mode)
                self.assertEqual(read.call_args.kwargs['charts'], mode)
                self.assertTrue(all(call.kwargs['charts'] == mode for call in refresh.call_args_list))
                self.assertEqual(export.call_args.kwargs['charts'], mode)
                self.assertEqual(local_ocr.call_count, int(mode != 'off'))
                if mode == 'off':
                    self.assertFalse(any(f.kind == 'chart' for f in result.library.fragments))

    def test_malformed_chart_series_is_not_accepted_as_verified_input(self):
        self.assertTrue(charts.valid_answer('ocr_chart', CHART))
        self.assertFalse(charts.valid_answer('ocr_chart', dict(CHART,series=['broken'])))
        invalid = dict(CHART, series=[{'printed_values':['broken']}])
        self.assertFalse(charts.valid_answer('ocr_chart', invalid))

    def make_pdf(self, folder, pages=1):
        path = Path(folder)/'Synthetic_10Oct2026.pdf'
        with pymupdf.open() as doc:
            for _ in range(pages): doc.new_page()
            doc.save(path)
        return path

    def test_full_preparation_reaches_loop_evidence_and_rerun_reuses_every_cache(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_pdf(root)
            vision = ScriptedChatModel(responses=[AIMessage(content=PAGE)])
            agent = ScriptedChatModel(responses=[calls(tool_call('ReportDigest', summary='Обзор ставки'))])
            models = SimpleNamespace(vision=lambda:vision, agent=lambda **kw:agent, callbacks=[])
            FakeReader.requests = []
            trace = AgentTrace(folder=None)
            with patch.object(prepare, 'check_local_ocr'), patch.object(charts, 'ChartReader', FakeReader), \
                 patch.object(charts, 'measure', return_value={'code':{'numbers':['14']}}), \
                 patch.object(tables, 'tesseract_cells', return_value=ROWS), redirect_stdout(StringIO()):
                result = prepare_bank(PreparationOptions(charts=True,tables=True,digests=True),models,pdf_dir=root,trace=trace)
                self.assertEqual(result.report['status'], 'complete')
                self.assertEqual([s['name'] for s in result.report['stages']], ['ocr','charts','tables','digests'])
                self.assertEqual(FakeReader.requests.count('ocr_chart'), 1)
                self.assertEqual(FakeReader.requests.count('ocr_table'), 2)
                self.assertEqual({f.kind for f in result.library.fragments}, {'text','chart','table'})
                chart = next(f for f in result.library.fragments if f.kind == 'chart')
                table = next(f for f in result.library.fragments if f.kind == 'table')
                self.assertTrue(any(f['value']==14 and f['verified'] for f in chart.figures))
                self.assertTrue(any(f['value']==15 and f['verified'] for f in table.figures))
                self.assertIn('таблица сверена', table.text)
                self.assertTrue(list((root/'text').glob('*.digest.json')))
                with patch.object(ScriptedChatModel, 'invoke', side_effect=AssertionError('cached stage invoked LLM')):
                    repeated = prepare_bank(PreparationOptions(charts=True,tables=True,digests=True),models,pdf_dir=root)
                self.assertEqual(FakeReader.requests.count('ocr_chart'), 1)
                self.assertEqual(FakeReader.requests.count('ocr_table'), 2)
                self.assertEqual(repeated.report['stages'][-1]['cached_reports'], 1)
            pool = ListNewsPool(NEWS.to_dict('records'))
            writer = Mock()
            out = insight()
            out.what_happened = [Claim(text='Ключевая ставка 15%', sources=['news:41'])]
            writer.write.return_value = (out, Review(ok=True))
            loop = AnalystLoop(result.library, pool, writer=writer, explain=False, trace=AgentTrace(folder=None))
            generated = loop.run(NEWS.iloc[0].to_dict(), profile(), 'cfo', {'importance':4})
            self.assertEqual(generated['status'], 'ready')
            evidence = writer.write.call_args.args[3].render()
            self.assertIn('таблица сверена', evidence)
            self.assertIn('Подписи на графике (сверены', evidence)
            self.assertTrue(trace.steps('bank_preparation'))

    def test_budget_exhaustion_is_partial_and_other_stages_continue(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_pdf(root)
            models = SimpleNamespace(vision=lambda:ScriptedChatModel(responses=[AIMessage(content=PAGE)]), callbacks=[])
            with patch.object(prepare, 'check_local_ocr'), \
                 patch.object(prepare, 'read_charts', side_effect=BudgetExceeded('synthetic budget')), \
                 patch.object(prepare, 'read_tables', return_value=[dict(stem='Synthetic_10Oct2026', page=1)]) as table_stage, \
                 redirect_stdout(StringIO()):
                result = prepare_bank(PreparationOptions(charts=True,tables=True),models,pdf_dir=root)
            self.assertEqual(result.report['status'], 'partial')
            self.assertEqual(result.report['stages'][1]['reason'], 'budget')
            table_stage.assert_called_once()
            self.assertEqual(json.loads((root/'preparation.json').read_text())['status'],'partial')

    def test_individual_stage_does_not_enable_other_paid_stages(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.make_pdf(root)
            models=SimpleNamespace(vision=lambda:ScriptedChatModel(responses=[AIMessage(content=PAGE)]),callbacks=[])
            with patch.object(prepare,'check_local_ocr'), patch.object(prepare,'read_charts',return_value={}) as graph, \
                 patch.object(prepare,'read_tables',side_effect=AssertionError('tables were not requested')), \
                 patch.object(prepare,'ReportDigester',side_effect=AssertionError('digests were not requested')), redirect_stdout(StringIO()):
                result=prepare_bank(PreparationOptions(charts=True),models,pdf_dir=root)
            graph.assert_called_once()
            self.assertEqual([s['name'] for s in result.report['stages']],['ocr','charts'])

    def test_empty_folder_and_disabled_options_do_not_construct_models(self):
        for options,status in ((PreparationOptions(charts=True),'empty'),(PreparationOptions(),'disabled')):
            with self.subTest(status=status), TemporaryDirectory() as tmp:
                models=Mock()
                result=prepare_bank(options,models,pdf_dir=Path(tmp))
                self.assertEqual(result.report['status'],status)
                self.assertEqual(result.library.fragments,[])
                models.vision.assert_not_called()

    def test_local_ocr_check_fails_before_paid_calls(self):
        with TemporaryDirectory() as tmp:
            self.make_pdf(tmp)
            models=Mock()
            with patch.object(prepare,'check_local_ocr',side_effect=RuntimeError('missing local OCR')):
                with self.assertRaisesRegex(RuntimeError,'missing local OCR'):
                    prepare_bank(PreparationOptions(tables=True),models,pdf_dir=Path(tmp))
            models.vision.assert_not_called()


class FlagTests(unittest.TestCase):
    def test_actual_cli_flags_prepare_bank_before_this_judge(self):
        variants=[(['--bank-charts','--chart-mode','selective'],(True,False,False)),
                  (['--bank-tables'],(False,True,False)), (['--bank-digest'],(False,False,True)),
                  (['--ocr'],(False,False,False)), (['--ocr','--chart-mode','off'],(False,False,False)),
                  (['--bank-prepare','--ocr','--ocr-budget','0','--charts-budget','0','--tables-budget','0','--digest-budget','0'],(True,True,True))]
        class Related:
            news = NEWS
            def __init__(self,*args,**kwargs): pass
            def refresh(self): pass
            def event_plausibility(self,news_id): return 1.0
            def same_event(self,news_id): return NEWS.iloc[:0]
        for flags,enabled in variants:
            with self.subTest(flags=flags), TemporaryDirectory() as tmp, ExitStack() as stack:
                report=dict(status='complete',stages=[])
                order=[]
                lib=BankLibrary([])
                bank=Mock()
                model=ScriptedChatModel(responses=[])
                models=SimpleNamespace(cheap=lambda **kw:model,agent=lambda **kw:model)
                # the news graph awaits stage 4 and the judge (async LLM path)
                scorer=SimpleNamespace(ascore=AsyncMock(return_value=SimpleNamespace(score=2,channel='',model_dump=lambda:{'score':2})))
                judge=Mock(adecide=AsyncMock(return_value=SimpleNamespace(is_insight=False,bank_evidence=[],
                    calculation_facts=[],calculation_bank=[],model_dump=lambda:{'is_insight':False})))
                args=['pipeline.py','synthetic','--engine','judge','--source','news','--retrieval','lexical',*flags]
                patches=[patch.object(sys,'argv',args), patch('common.data.DATA',Path(tmp)),
                    patch('common.search.news_pool',return_value=NEWS), patch('common.llm.ChatModels',return_value=models),
                    patch('stage2_company_facts.deterministic_facts.facts.build',return_value={'synthetic':[]}),
                    patch('stage3_news_search.retrieval.candidates',return_value=(NEWS.head(1),[])),
                    patch('stage5_insight_detection.bank_analytics.search.BankAnalytics',side_effect=lambda:(order.append('search'),bank)[1]),
                    patch('stage5_insight_detection.news_enrichment.related_news.RelatedNews',Related),
                    patch('stage4_news_relevance.relevance_scoring.relevance.RelevanceScorer',return_value=scorer),
                    patch('stage4_news_relevance.consequence_extraction.consequences.ConsequenceExtractor',return_value=SimpleNamespace(aextract=AsyncMock(return_value={})))]
                for p in patches: stack.enter_context(p)
                preparation=stack.enter_context(patch.object(prepare,'prepare_bank',
                    side_effect=lambda *a,**kw:(order.append('prepare'),PreparedBank(lib,report))[1]))
                judge_factory=stack.enter_context(patch('stage5_insight_detection.insight_judge.judge.InsightJudge',return_value=judge))
                loop=stack.enter_context(patch('stage5_insight_detection.analyst_loop.loop.AnalystLoop'))
                stack.enter_context(redirect_stdout(StringIO()))
                runpy.run_path('pipeline.py',run_name='__main__')
                options=preparation.call_args.args[0]
                self.assertEqual(preparation.call_count,1)
                self.assertTrue(options.ocr)
                self.assertEqual(order,['prepare','search'])
                self.assertEqual((options.charts,options.tables,options.digests),enabled)
                if '--chart-mode' in flags: self.assertEqual(options.chart_mode,flags[flags.index('--chart-mode')+1])
                if '--bank-prepare' in flags: self.assertEqual(options.ocr_budget,0)
                self.assertIs(judge_factory.call_args.args[2],bank)
                judge.adecide.assert_called_once()
                loop.assert_not_called()
                saved=json.loads((Path(tmp)/'insights/synthetic_cfo.json').read_text())
                self.assertEqual(saved[0]['bank_preparation'],report)

    def test_invalid_flags_fail_before_models_or_preparation(self):
        cases=[['--engine','loop'], ['--loop'], ['--bank-review'], ['--bank-budget','1'], ['--chart-mode','selective'],
               ['--bank-charts','--chart-mode','unknown'], ['--charts-budget','2'],
               ['--bank-charts','--charts-budget','-1'], ['--bank-tables','--tables-budget','nan']]
        for flags in cases:
            with self.subTest(flags=flags),patch.object(sys,'argv',['pipeline.py','synthetic',*flags]), \
                 patch('common.llm.ChatModels') as models,patch.object(prepare,'prepare_bank') as preparation, \
                 redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as error: runpy.run_path('pipeline.py',run_name='__main__')
                self.assertEqual(error.exception.code,2)
                models.assert_not_called()
                preparation.assert_not_called()


if __name__=='__main__':
    unittest.main()
