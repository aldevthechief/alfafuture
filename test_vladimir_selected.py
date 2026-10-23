"""Selected Vladimir integration: synthetic inputs and scripted models, no network or case CSVs."""
import json
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pymupdf
import pipeline

from common.llm.testing import ScriptedChatModel, calls, tool_call
from common.news_corpus import bind_result, referenced_news_ids, result_references
from common.news_evidence import verify_claim_evidence
from common.trace import AgentTrace
from stage2_company_facts.profile.integration import CompanyInputs
from stage5_insight_detection.analyst_loop import bank_run, grounds, impact
from stage5_insight_detection.analyst_loop.compute import compute
from stage5_insight_detection.analyst_loop.data_requests import DataRequest
from stage5_insight_detection.analyst_loop.dossier import Dossier
from stage5_insight_detection.analyst_loop.event import EventCard
from stage5_insight_detection.analyst_loop.loop import AnalystLoop, bank_insights
from stage5_insight_detection.analyst_loop.pool import EmptyNewsPool
from stage5_insight_detection.analyst_loop.profile import CompanyProfile
from stage5_insight_detection.analyst_loop.verdict import decide
from stage5_insight_detection.bank_analytics import charts, digitize
from stage5_insight_detection.bank_analytics.library import BankLibrary, Figure
from stage5_insight_detection.bank_analytics.match import BankMatch
from stage6_insight_generation.insight_card.render import render
from stage6_insight_generation.insight_writer.insight import Claim, Insight
from stage6_insight_generation.insight_writer.writer import InsightWriter

AS_OF = date(2026, 10, 20)
NOW = datetime(2026, 10, 20, 9)
RATE = 'SYNTH-Policy_Rate.pdf'
QUOTE = 'Мы ожидаем снижения ключевой ставки до 12% к концу 2027 года.'


def library():
    return BankLibrary.from_pages({RATE: ['Ключевая ставка ЦБ сохранена на уровне 14%.\n\n' + QUOTE]}, AS_OF,
                                  meta=[{'file': RATE, 'date': '2026-10-10', 'title': 'Прогноз ключевой ставки'}])


def profile(**overrides):
    fields = dict(id='U_TEST', okved='41.20', okved_name='Строительство', region='Москва',
                  revenue=10e9, group_revenue=50e9, products=['loan'])
    return CompanyProfile(**(fields | overrides))


def event(title='ЦБ повысил ключевую ставку до 18%'):
    return EventCard.from_news(dict(id=1, title=title, text='', date='2026-10-20', source='test'))


def bank_answer(ref):
    return dict(title='Прогноз ставки для заёмщика', role='cfo',
                what_happened=[dict(text=QUOTE, sources=[ref], evidence=[dict(source=ref, quote=QUOTE)])],
                why_it_matters=[dict(text='Компания использует кредитные продукты', sources=['fact:product_loan'])],
                risks=[], opportunities=[], actions=['Проверить график рефинансирования'],
                bank_view=[dict(text=QUOTE, sources=[ref])], ai_conclusion='Возможен пересмотр стоимости долга.',
                confidence='medium')


class ImpactTests(unittest.TestCase):
    def fuel(self, title):
        truck = profile(okved='49.41', exposure={'энергия_и_сырьё': 1.0})
        return impact.assess(event(title), truck, [])[0][0]

    def test_petrol_does_not_become_a_diesel_cost_estimate(self):
        c = self.fuel('Средние цены бензина Аи-92 упали на 0,6% к сентябрю')
        self.assertIsNone(c.estimate)
        self.assertIn('дизель', c.level_why)

    def test_weekly_change_is_not_annualized_but_yearly_scenario_still_works(self):
        self.assertIsNone(self.fuel('Биржевые цены на дизель упали за неделю на 0,9%').estimate)
        annual = self.fuel('Цены на дизельное топливо выросли на 5% с начала года')
        self.assertIsNotNone(annual.estimate)
        self.assertEqual(annual.estimate.kind, 'hypothesis')

    def test_payroll_share_is_an_explicit_hypothesis(self):
        p = profile(okved='47.11', exposure={'рынок_труда': 1.0})
        c = impact.assess(event('Номинальные зарплаты выросли на 10% за год'), p, [])[0][0]
        self.assertEqual(c.estimate.kind, 'hypothesis')
        self.assertTrue(any('ФОТ' in f.label and f.kind == 'hypothesis' for f in c.assumptions))
        self.assertIn('score', c.level_why)

    def test_future_report_and_a_rate_change_cannot_supply_the_current_rate(self):
        lib = library()
        fragment = next(f for f in lib.fragments if any(x['kind'] == 'actual' for x in f.figures))
        future = replace(fragment, date='2026-10-21')
        delta = impact.indicator_delta('ставка', event(), [BankMatch(future, 1, {}, '')])
        self.assertEqual(delta[0], 1.0)
        self.assertTrue(delta[3][0].startswith('чувств'))
        changed = replace(fragment, figures=[dict(value=14, indicator='ставка', kind='actual', unit='%',
                                                  period='', measure='change', verified=True)])
        self.assertEqual(impact.indicator_delta('ставка', event(), [BankMatch(changed, 1, {}, '')])[0], 1.0)


class CalculationTests(unittest.TestCase):
    def dossier(self):
        lib = library()
        return Dossier(event(), profile(), 'cfo', matches=[BankMatch(f, 1, {}, '') for f in lib.fragments])

    def test_group_share(self):
        facts, why = compute('доля выручки компании в группе', self.dossier())
        self.assertFalse(why)
        self.assertEqual((facts[0].value, facts[0].unit, facts[0].kind), (20.0, '%', 'fact'))

    def test_unit_conversion_handles_thousands_without_promoting_the_input_to_a_source_fact(self):
        facts, why = compute('переведи 1 000 млн в млрд', self.dossier())
        self.assertFalse(why)
        self.assertEqual(facts[0].value, 1.0)
        self.assertEqual(facts[0].kind, 'hypothesis')

    def test_change_uses_comparable_levels_and_refuses_different_periods(self):
        d = self.dossier()
        facts, why = compute('разница ключевой ставки', d)
        self.assertFalse(why)
        self.assertEqual((facts[0].value, facts[0].unit), (4.0, 'п.п.'))
        d.event.figures = [Figure(18, '%', 'ставка', 'actual', '2025 год', '', measure='level')]
        d.matches = [BankMatch(replace(d.matches[0].fragment, figures=[dict(value=14, unit='%', indicator='ставка',
                            kind='actual', period='2024 год', measure='level')]), 1, {}, '')]
        self.assertEqual(compute('разница ключевой ставки', d)[0], [])

    def test_forecast_times_revenue_is_only_a_scenario(self):
        facts, why = compute('выручка по прогнозу ключевой ставки', self.dossier())
        self.assertFalse(why)
        self.assertEqual(facts[0].kind, 'hypothesis')
        self.assertIn('допущение', facts[0].formula)

    def comparison(self, figures, news=None):
        d = self.dossier()
        d.event.figures = [news or Figure(18, '%', 'ставка', 'actual', '', '', measure='level')]
        d.matches = [BankMatch(replace(d.matches[0].fragment, figures=figures), 1, {}, '')]
        return d

    def test_unknown_news_period_does_not_select_a_historical_bank_value(self):
        d = self.comparison([dict(value=14, unit='%', indicator='ставка', kind='actual', period='2019')])
        self.assertEqual(compute('разница ставки', d)[0], [])

    def test_requested_forecast_year_wins_over_history_and_other_forecasts(self):
        figures = [dict(value=value, unit='%', indicator='ставка', kind=kind, period=year)
                   for value, kind, year in [(10, 'actual', '2019'), (12, 'forecast', '2026'), (11, 'forecast', '2027')]]
        facts, why = compute('разница ставки с прогнозом на 2026 год', self.comparison(figures))
        self.assertFalse(why)
        self.assertEqual(facts[0].value, 6)
        self.assertIn('forecast, 2026', facts[0].label)

    def test_ambiguous_pairs_refuse_but_explicit_cross_year_comparison_works(self):
        figures = [dict(value=v, unit='%', indicator='ставка', kind='actual', period='') for v in (13, 14)]
        self.assertIn('несколько', compute('разница ставки', self.comparison(figures))[1])
        d = self.comparison([dict(value=14, unit='%', indicator='ставка', kind='actual', period='2024 год')],
                            Figure(18, '%', 'ставка', 'actual', '2025 год', '', measure='level'))
        self.assertEqual(compute('разница ставки за 2025 и 2024 годы', d)[0][0].value, 4)

    def test_currencies_are_not_mixed_and_multi_currency_requests_are_refused(self):
        figures = [dict(value=v, unit='руб', indicator='курс', kind='forecast', period='2026', sentence=s)
                   for v, s in [(87, 'Курс доллара'), (13, 'Курс юаня')]]
        d = self.comparison(figures, Figure(12.76, 'руб', 'курс', 'actual', '', 'Курс юаня', measure='level'))
        facts, why = compute('разница курса юаня с прогнозом на 2026', d)
        self.assertFalse(why)
        self.assertEqual(facts[0].value, -0.24)
        self.assertEqual(compute('разница курсов доллара и юаня с прогнозами на 2026', d)[0], [])

    def test_requested_facts_survive_recomputation_and_reach_the_brief(self):
        d = self.dossier()
        loop = AnalystLoop(library(), EmptyNewsPool(), trace=AgentTrace(folder=None), explain=False)
        answer = loop._handlers(d)['computation'](DataRequest(kind='computation', query='доля выручки в группе'))
        self.assertTrue(answer.ok)
        ref = d.requested[0].id
        loop._compute(d)
        self.assertIn(ref, d.refs())
        brief = decide(d, grounds.check(d), []).brief
        self.assertIn(ref, {f['id'] for f in brief.facts})

    def test_unsupported_computation_returns_a_reason(self):
        facts, why = compute('посчитай неизвестный долг компании', self.dossier())
        self.assertEqual(facts, [])
        self.assertTrue(why)


class GraphTests(unittest.TestCase):
    def test_monthly_knots_are_recovered_without_inventing_a_daily_series(self):
        xs = np.arange(20, 141)
        knots = np.arange(20, 141, 20)
        values = [30, 50, 10, 40, 70, 20, 60]
        got = digitize.monthly_points(xs, np.interp(xs, knots, values), ['янв.26', 'апр.26', 'июл.26'],
                                     [], [20, 80, 140])
        self.assertNotIsInstance(got, str)
        self.assertEqual(len(got[0]), 7)
        np.testing.assert_allclose(got[1], values, atol=0.01)
        self.assertIsInstance(digitize.monthly_points(xs, xs, ['a', 'b', 'c'], [], [20, 80, 140]), str)

    def test_forecast_points_use_the_report_date_and_explicit_labels(self):
        points = [('ряд', 'дек.25', '10'), ('ряд', '2027', '12'), ('ряд', '2026П', '11')]
        with patch.object(digitize, 'report_date', return_value=date(2026, 10, 1)):
            _, ahead = digitize.forecast_points({'crop': 'SYNTH_p1_1.png'}, points)
        self.assertEqual([p[1] for p in ahead], ['2027', '2026П'])

    def test_future_chart_periods_keep_their_verification_status(self):
        pages = {'chart.pdf': ['[График] Ключевая ставка, %\nПодписи на графике (сверены: пиксели): '
                              'ставка — 2027П: 12\n[/График]']}
        lib = BankLibrary.from_pages(pages, AS_OF, meta=[dict(file='chart.pdf', date='2026-10-01', title='Обзор')])
        f = lib.fragments[0].figures[0]
        self.assertEqual((f['kind'], f['status'], f['verified']), ('forecast', 'verified', True))

    def test_crop_grows_to_the_chart_end_without_taking_the_neighbour(self):
        with TemporaryDirectory() as tmp:
            pdf = Path(tmp) / 'charts.pdf'
            doc = pymupdf.open()
            page = doc.new_page(width=240, height=180)
            page.draw_polyline([(20, 100), (50, 65), (80, 90), (115, 70)], color=(1, 0, 0), width=2)
            page.draw_polyline([(145, 100), (180, 65), (220, 90)], color=(0, 0, 1), width=2)
            doc.save(pdf)
            doc.close()
            boxes = [dict(bbox=[145, 300, 400, 650]), dict(bbox=[600, 300, 920, 650])]
            sides = charts.chart_sides(pdf, 1, boxes)
            old = digitize.to_rgb(charts.render(pdf, 1, 180, boxes[0]['bbox']))
            new = digitize.to_rgb(charts.render(pdf, 1, 180, boxes[0]['bbox'], sides=sides[0]))
            red = lambda a: ((a[:, :, 0] > 180) & (a[:, :, 1] < 80) & (a[:, :, 2] < 80)).any(0).sum()
            self.assertGreater(red(new), red(old))
            self.assertFalse(((new[:, :, 2] > 180) & (new[:, :, 0] < 80)).any())

    def test_legacy_crop_is_retried_and_a_failed_read_keeps_the_previous_assets(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / 'SYNTH.pdf'
            with pymupdf.open() as doc:
                doc.new_page(width=240, height=180)
                doc.save(pdf)
            folder = root / 'charts'
            folder.mkdir()
            box = dict(title='График', bbox=[100, 100, 900, 800])
            (folder / 'SYNTH_p1.boxes.json').write_text(json.dumps([box]), encoding='utf-8')
            name = 'SYNTH_p1_1.png'
            (folder / name).write_bytes(b'old crop')
            read = dict(x_axis={}, y_axis={}, series=[])
            old = [{'crop': name, 'read': read}]
            output = folder / 'SYNTH_p1.gemma.json'
            output.write_text(json.dumps(old), encoding='utf-8')
            reader = SimpleNamespace(read=Mock(side_effect=charts.ChartReadError('synthetic failure')),
                                     calls=[dict(cost=0, sec=0)])
            result = charts.read_pages([('SYNTH', 1)], pdf_dir=root, out_dir=folder, reader=reader)
            self.assertEqual(len(result['errors']), 1)
            self.assertEqual((folder / name).read_bytes(), b'old crop')
            self.assertEqual(json.loads(output.read_text()), old)
            reader.read = Mock(return_value=read)
            with patch.object(charts, 'measure', return_value={'code': {}, 'digitized': {}}):
                self.assertEqual(charts.read_pages([('SYNTH', 1)], pdf_dir=root, out_dir=folder,
                                                  reader=reader)['charts'], 1)
                self.assertEqual(charts.read_pages([('SYNTH', 1)], pdf_dir=root, out_dir=folder,
                                                  reader=reader)['cached_charts'], 1)
            reader.read.assert_called_once()
            self.assertEqual(json.loads(output.read_text())[0]['crop_version'], charts.CROP_VERSION)


if __name__ == '__main__':
    unittest.main()
