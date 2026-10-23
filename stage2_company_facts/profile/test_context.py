"""Offline prompt-route checks; models and tool loops are replaced by in-memory doubles."""
import json
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from common.trace import AgentTrace
from stage2_company_facts.profile.context import compact_context, evidence_facts, judge_facts, render_context
from stage5_insight_detection.analyst_loop.data_requests import DataRequest
from stage5_insight_detection.analyst_loop.dossier import Dossier
from stage5_insight_detection.analyst_loop.loop import AnalystLoop, Assessment, BriefEvidence
from stage5_insight_detection.analyst_loop.pool import ListNewsPool
from stage5_insight_detection.analyst_loop.profile import CompanyProfile
from stage5_insight_detection.bank_analytics.library import BankLibrary
from stage5_insight_detection.insight_judge.judge import InsightDecision, InsightJudge
from stage6_insight_generation.insight_writer.writer import Evidence


CONTEXT = {'cat': 'profile_context', 'depth': 1,
           'text': 'Гипотеза по правилам профиля: [sensitivity] Ставка 0.8', 'kind': 'hypothesis'}

LINKS = {'cat': 'profile_context', 'depth': 1,
         'text': '[links] Высокая зависимость от поставщиков: крупнейший 50%, топ-5 90% исходящих платежей'}
CONCENTRATION = {'cat': 'payments_out', 'depth': 2,
                 'text': 'Концентрация (исходящие платежи): крупнейший контрагент 50.0%, топ-5 90%, топ-10 100%'}


class CompactContextTests(unittest.TestCase):
    def test_writer_omits_repeated_percentages_but_keeps_advice_and_fact_numbers(self):
        rendered = evidence_facts(['[profile_context] ' + LINKS['text'],
                                  '[payments_out] ' + CONCENTRATION['text'],
                                  '[profile_context] ' + CONTEXT['text']])
        text = '\n'.join(rendered)
        self.assertEqual(text.count('топ-5'), 1)
        self.assertIn('fact:2 [payments_out]', text)
        self.assertNotIn('fact:1 ', text)
        self.assertIn('[links] Высокая зависимость', text)
        self.assertIn('Ставка 0.8', text)
        self.assertIn('не вероятность', text)

    def test_judge_retains_percentages_when_the_fact_is_outside_its_prompt(self):
        for source in (CONCENTRATION, dict(CONCENTRATION, depth=1)):
            facts = [dict(CONCENTRATION, text='Первый факт', depth=1), source, LINKS]
            text = '\n'.join(judge_facts(facts, 1))
            self.assertIn('крупнейший 50%, топ-5 90%', text)
        text = '\n'.join(judge_facts([dict(CONCENTRATION, depth=1), LINKS], 25))
        self.assertNotIn('крупнейший 50%, топ-5 90%', text)

    def test_other_direction_different_values_or_sector_shares_are_not_duplicates(self):
        for text in [CONCENTRATION['text'].replace('исходящие', 'входящие'),
                     CONCENTRATION['text'].replace('50.0%', '49.6%'),
                     'Поставщикам с ОКВЭД 24: 50%, топ-5 отраслей 90%']:
            rendered = render_context([LINKS], visible_facts=[{'text': text}])
            self.assertIn('крупнейший 50%, топ-5 90%', rendered)

    def test_scale_is_kept_when_only_revenue_is_visible(self):
        scale = dict(CONTEXT, text='[scale] Масштаб: крупный бизнес (основание: выручка)')
        self.assertIn(scale['text'], render_context([scale], visible_facts=[{'text': 'Выручка за 2025: 20 млрд ₽'}]))
        self.assertNotIn(scale['text'], render_context([scale], visible_facts=[{'text': scale['text']}]))

    def test_scale_group_clause_is_removed_without_dropping_the_company_class(self):
        context = '[scale] Масштаб: малый бизнес (основание: штат); группа — крупный бизнес по критерию кейса'
        text = compact_context(context, ['Группа — крупный бизнес по критерию кейса'])
        self.assertEqual(text, '[scale] Масштаб: малый бизнес (основание: штат)')

    def test_rendering_does_not_mutate_facts_or_drop_factor_mechanisms(self):
        from copy import deepcopy
        factor = dict(CONTEXT, text='Гипотеза по правилам профиля: [factor 0.8] Ставка: стоимость кредита')
        facts = [CONCENTRATION, LINKS, factor]
        before = deepcopy(facts)
        self.assertIn(factor['text'], render_context(facts, visible_facts=[CONCENTRATION]))
        self.assertEqual(facts, before)


class JudgeContextTests(unittest.TestCase):
    def test_context_reaches_the_actual_judge_task_beyond_the_fact_limit(self):
        facts = [{'cat': 'activity', 'depth': 1, 'text': f'исходный факт {i}'} for i in range(30)] + [CONTEXT]
        related = Mock()
        related.get.return_value = {'date': '2026-10-21', 'source': 'test', 'title': 'Новость', 'text': 'Текст'}
        related.event_plausibility.return_value = 1.0
        decision = InsightDecision(is_insight=False, score=0, company_link='', mechanism='')
        with patch('stage5_insight_detection.insight_judge.judge.ToolLoop') as tool_loop:
            tool_loop.return_value.run.return_value = decision
            result = InsightJudge(model=Mock(), related=related).decide(1, 'company', facts, 'cfo')
        task = tool_loop.return_value.run.call_args.args[0]
        self.assertIs(result, decision)
        self.assertIn('исходный факт 24\n', task)
        self.assertNotIn('исходный факт 25\n', task)
        self.assertIn('## Пояснения из профиля компании', task)
        self.assertIn('Ставка 0.8', task)
        self.assertIn('не вероятность', task)

    def test_ordinary_facts_keep_the_previous_limit_and_format(self):
        facts = [{'cat': 'activity', 'depth': 1, 'text': 'сталь'},
                 {'cat': 'group', 'depth': 2, 'text': 'группа'}]
        self.assertEqual(judge_facts(facts, 1), ['[activity] сталь'])
        self.assertEqual(render_context(facts), '')


class WriterContextTests(unittest.TestCase):
    def test_context_is_separate_and_source_fact_numbers_stay_stable(self):
        evidence = Evidence(news=[], facts=['[activity] сталь', '[profile_context] ' + CONTEXT['text'],
                                            '[group] выручка группы'])
        rendered = evidence.render()
        self.assertIn('fact:1 [activity] сталь', rendered)
        self.assertIn('fact:3 [group] выручка группы', rendered)
        self.assertNotIn('fact:2', rendered)
        self.assertIn('## Пояснения из профиля компании', rendered)
        self.assertIn('не являются самостоятельным доказательством', rendered)

    def test_without_context_evidence_fact_rendering_is_unchanged(self):
        self.assertEqual(evidence_facts(['первый', 'второй']), ['fact:1 первый', 'fact:2 второй'])


class LoopContextTests(unittest.TestCase):
    def dossier(self):
        event = SimpleNamespace(id='news:1', news_id='1', title='Новость', text='Текст',
                                date='2026-10-21', source='test', url='', matched_fragments=[])
        profile = SimpleNamespace(facts={}, card=lambda: 'Карточка компании')
        return Dossier(event, profile, 'cfo', profile_context=render_context([CONTEXT]))

    def loop(self, facts=None):
        # Exercise composition and the handler without running the pipeline or constructing any model.
        loop = object.__new__(AnalystLoop)
        loop.company_facts = facts or [CONTEXT]
        loop.trace = AgentTrace(folder=None)
        loop.assessor = Mock()
        loop.assessor.invoke.return_value = Assessment()
        return loop

    def test_assessor_receives_context_without_a_company_data_request(self):
        dossier, loop = self.dossier(), self.loop()
        self.assertEqual(loop._explain(dossier), [])
        inputs = loop.assessor.invoke.call_args.args[0]
        self.assertIn('Карточка компании', inputs['card'])
        self.assertIn('Ставка 0.8', inputs['card'])
        self.assertIn('не доказывает отсутствие связи', inputs['card'])
        self.assertEqual(inputs['facts'], '')
        self.assertEqual(dossier.extra_facts, [])

    def test_assessor_compacts_only_against_its_visible_card(self):
        dossier, loop = self.dossier(), self.loop()
        dossier.profile.card = lambda: CONCENTRATION['text']
        dossier.profile_context = render_context([LINKS, CONTEXT])
        loop._explain(dossier)
        card = loop.assessor.invoke.call_args.args[0]['card']
        self.assertEqual(card.count('топ-5'), 1)
        self.assertIn('Высокая зависимость от поставщиков', card)
        self.assertIn('Ставка 0.8', card)
        self.assertIn('крупнейший 50%, топ-5 90%', dossier.profile_context)

    def test_loop_writer_compacts_only_against_facts_it_actually_receives(self):
        dossier = self.dossier()
        dossier.profile_context = render_context([LINKS, CONTEXT])
        dossier.extra_facts = [CONCENTRATION['text']]
        evidence = BriefEvidence(dossier, SimpleNamespace(facts=[], hypotheses=[]))
        text = evidence.render()
        self.assertEqual(text.count('топ-5'), 1)
        self.assertIn('fact:stage2_1', text)
        self.assertIn('Высокая зависимость от поставщиков', text)
        self.assertIn('Ставка 0.8', text)
        self.assertEqual(dossier.extra_facts, [CONCENTRATION['text']])

    def test_writer_receives_context_without_making_it_a_citeable_fact(self):
        dossier = self.dossier()
        evidence = BriefEvidence(dossier, SimpleNamespace(facts=[], hypotheses=[]))
        self.assertIn('Ставка 0.8', evidence.render())
        self.assertEqual(dossier.facts(), {})
        self.assertEqual(dossier.refs(), {'news:1'})

    def test_company_requests_do_not_promote_profile_hypotheses_to_facts(self):
        dossier, loop = self.dossier(), self.loop()
        handler = loop._handlers(dossier)['company_facts']
        answer = handler(DataRequest(kind='company_facts', query='ставка'))
        self.assertFalse(answer.ok)
        self.assertEqual(dossier.extra_facts, [])

    def test_company_requests_still_return_original_facts(self):
        dossier = self.dossier()
        loop = self.loop([CONTEXT, {'cat': 'bank', 'text': 'Кредит компании'}])
        answer = loop._handlers(dossier)['company_facts'](DataRequest(kind='company_facts', query='кредит'))
        self.assertTrue(answer.ok)
        self.assertEqual(dossier.extra_facts, ['Кредит компании'])

    def test_synthetic_loop_delivers_context_without_changing_computed_verdict(self):
        fixtures = Path(__file__).resolve().parents[2] / 'stage5_insight_detection/analyst_loop/synthetic'
        news = json.loads((fixtures / 'news.json').read_text(encoding='utf-8'))['news']
        pages = json.loads((fixtures / 'bank_pages.json').read_text(encoding='utf-8'))['reports']
        company = {k: v for k, v in json.loads((fixtures / 'company.json').read_text(encoding='utf-8')).items()
                   if not k.startswith('_')}
        library = BankLibrary.from_pages(pages, date(2026, 10, 20))
        results, inputs = [], []
        for context in ([], [CONTEXT]):
            pool = ListNewsPool(news)
            loop = AnalystLoop(library, pool, trace=AgentTrace(folder=None), company_facts=context, explain=False)
            loop.assessor = Mock()
            loop.assessor.invoke.return_value = Assessment()
            results.append(loop.run(pool.get(101), CompanyProfile(**company), 'cfo',
                                    scores={'category': 'ставка_и_кредит', 'importance': 4}))
            loop.assessor.invoke.assert_called_once()
            inputs.append(loop.assessor.invoke.call_args.args[0])
        self.assertNotIn('Пояснения из профиля', inputs[0]['card'])
        self.assertIn('Ставка 0.8', inputs[1]['card'])
        for key in ('consequences', 'grounds', 'verdict'):
            self.assertEqual(results[0][key], results[1][key], key)


if __name__ == '__main__':
    unittest.main()
