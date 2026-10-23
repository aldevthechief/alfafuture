"""Profile regressions on synthetic payments and temporary JSONL files; no case data or LLM calls."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from stage2_company_facts.profile import load as profile_load
from stage2_company_facts.profile import integration
from stage2_company_facts.deterministic_facts.facts import build
from stage2_company_facts.profile.profile import check, payment_shares


class PaymentShareTests(unittest.TestCase):
    def setUp(self):
        self.companies = pd.DataFrame({'okdiv': ['47', '49', None]}, index=['shop', 'carrier', 'unclassified'])

    def payments(self, rows):
        return pd.DataFrame(rows, columns=['PAYER', 'BENEFICIARY', 'PAY_AMT'])

    def test_outgoing_denominator_includes_missing_and_unclassified_counterparties(self):
        payments = self.payments([
            ('company', 'shop', 10), ('company', 'carrier', 10),
            ('company', 'missing', 60), ('company', 'unclassified', 20),
            ('other', 'shop', 900),
        ])
        shares, total = payment_shares(payments, self.companies, 'company', 'PAYER', 'BENEFICIARY')
        self.assertEqual(total, 100)
        self.assertEqual(shares, {'47': 0.1, '49': 0.1})

    def test_incoming_denominator_includes_unknown_buyers(self):
        payments = self.payments([('shop', 'company', 25), ('missing', 'company', 75), ('shop', 'other', 100)])
        shares, total = payment_shares(payments, self.companies, 'company', 'BENEFICIARY', 'PAYER')
        self.assertEqual((shares, total), ({'47': 0.25}, 100))

    def test_no_known_sectors_keeps_the_total(self):
        for rows, total in [([], 0), ([('company', 'missing', 100)], 100)]:
            with self.subTest(rows=rows):
                self.assertEqual(payment_shares(self.payments(rows), self.companies, 'company', 'PAYER', 'BENEFICIARY'),
                                 ({}, total))

    def test_fully_classified_payments_keep_the_previous_shares(self):
        payments = self.payments([('company', 'shop', 75), ('company', 'carrier', 25)])
        self.assertEqual(payment_shares(payments, self.companies, 'company', 'PAYER', 'BENEFICIARY'),
                         ({'47': 0.75, '49': 0.25}, 100))


class ProfileValidationTests(unittest.TestCase):
    def profile(self, shares):
        return {'id': 'company', 'facts': [], 'card': {'pay_out': shares, 'pay_in': {}},
                'scale': {'class': 'small', 'revenue_percentile': {}, 'case_large': False},
                'external_factors': [], 'sensitivity': {}}

    def test_partial_sector_coverage_is_valid(self):
        check(self.profile({'47': 0.1, '49': 0.1}))
        check(self.profile({}))

    def test_invalid_shares_are_still_rejected(self):
        for shares in [{'47': 0.7, '49': 0.5}, {'47': -0.1}, {'47': float('nan')}]:
            with self.subTest(shares=shares), self.assertRaises(AssertionError):
                check(self.profile(shares))

    def test_rounding_does_not_reject_full_coverage(self):
        check(self.profile({str(i): 0.1667 for i in range(6)}))


class ProfileCacheTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / 'profiles.jsonl'
        profile_load._read_file.cache_clear()
        self.addCleanup(profile_load._read_file.cache_clear)
        active = patch.object(profile_load, 'PROFILES', self.path)
        active.start()
        self.addCleanup(active.stop)

    def write(self, marker, path=None):
        (path or self.path).write_text(json.dumps({'id': 'company', 'marker': marker}) + '\n', encoding='utf-8')

    def test_unchanged_file_is_not_read_again(self):
        self.write('old')
        with patch('builtins.open', wraps=open) as reader:
            first = profile_load.load_profile('company')
            self.assertIs(profile_load.load_profile('company'), first)
        self.assertEqual(reader.call_count, 1)

    def test_changed_file_updates_the_next_lookup_and_keeps_the_loaded_profile(self):
        self.write('old')
        first = profile_load.load_profile('company')
        self.write('updated profile')
        self.assertEqual(profile_load.load_profile('company')['marker'], 'updated profile')
        self.assertEqual(first['marker'], 'old')

    def test_same_size_atomic_replacement_with_preserved_mtime_is_detected(self):
        self.write('old')
        self.assertEqual(profile_load.load_profile('company')['marker'], 'old')
        info = self.path.stat()
        replacement = self.path.with_suffix('.next')
        self.write('new', replacement)
        os.utime(replacement, ns=(info.st_atime_ns, info.st_mtime_ns))
        replacement.replace(self.path)
        self.assertEqual(profile_load.load_profile('company')['marker'], 'new')

    def test_file_created_after_a_missing_lookup_is_used(self):
        fallback = {'id': 'company', 'marker': 'built'}
        with patch('stage2_company_facts.profile.profile.profiles', return_value={'company': fallback}) as build:
            self.assertIs(profile_load.load_profile('company'), fallback)
            self.write('from file')
            self.assertEqual(profile_load.load_profile('company')['marker'], 'from file')
        build.assert_called_once_with(['company'])

    def test_deleted_file_uses_the_existing_fallback(self):
        self.write('old')
        profile_load.load_profile('company')
        self.path.unlink()
        fallback = {'id': 'company', 'marker': 'built'}
        with patch('stage2_company_facts.profile.profile.profiles', return_value={'company': fallback}):
            self.assertIs(profile_load.load_profile('company'), fallback)


class ProfileIntegrationTests(unittest.TestCase):
    def profile(self):
        return {'id': 'company', 'card': {'id': 'company', 'revenue': 2e9},
                'facts': [
                    {'id': 'fact:activity_1', 'cat': 'activity', 'text': 'Основной вид деятельности: производство стали',
                     'depth': 1, 'entities': ['24.10', 'производство стали'], 'kind': 'fact'},
                    {'id': 'fact:group_1', 'cat': 'group', 'text': 'Выручка группы 10 млрд рублей',
                     'depth': 2, 'entities': [], 'kind': 'fact'},
                ],
                'scale': {'class_ru': 'крупный бизнес', 'class_basis': 'выручка',
                          'case_large': False, 'group_case_large': False},
                'sensitivity': {'D1': {'lever': 'D1', 'name': 'Ставка', 'weight': 0.8}},
                'external_factors': [{'weight': 0.8, 'label': 'Ставка', 'mechanism': 'стоимость кредита'}]}

    def test_card_and_facts_come_from_one_profile_lookup(self):
        profile = self.profile()
        with patch.object(integration, 'load_profile', return_value=profile) as load:
            result = integration.load_company('company')
        load.assert_called_once_with('company')
        self.assertEqual(result.card, profile['card'])
        self.assertEqual(result.facts[:2], profile['facts'])

    def test_context_preserves_deeper_facts_without_modifying_the_cached_profile(self):
        profile = self.profile()
        with patch.object(integration, 'load_profile', return_value=profile):
            result = integration.load_company('company')
        self.assertEqual(len(profile['facts']), 2)
        self.assertTrue(any(f['depth'] == 2 and '10 млрд' in f['text'] for f in result.facts))
        result.facts[0]['text'] = 'changed by caller'
        self.assertIn('производство стали', profile['facts'][0]['text'])

    def test_rule_based_context_is_explicitly_a_hypothesis(self):
        with patch.object(integration, 'load_profile', return_value=self.profile()):
            result = integration.load_company('company')
        context = [f for f in result.facts if f['cat'] == 'profile_context']
        hypotheses = [f for f in context if f['kind'] == 'hypothesis']
        self.assertEqual(len(hypotheses), 2)
        self.assertTrue(all(f['text'].startswith('Гипотеза по правилам профиля:') for f in hypotheses))
        self.assertTrue(all(f['depth'] == 1 for f in context))
        self.assertEqual(len({f['id'] for f in result.facts}), len(result.facts))

    def test_pipeline_fact_builder_uses_profiles_without_loading_case_tables(self):
        with patch.object(integration, 'load_profile', return_value=self.profile()), patch(
                'stage2_company_facts.deterministic_facts.facts.load', side_effect=AssertionError('unexpected CSV read')):
            facts = build(['company'], use_profiles=True)['company']
        self.assertTrue(any(f['cat'] == 'group' for f in facts))
        self.assertTrue(any(f['cat'] == 'profile_context' for f in facts))

    def test_existing_candidate_generators_ignore_explanatory_context(self):
        from stage3_news_search.candidate_generators.generators import generators
        profile = self.profile()
        with patch.object(integration, 'load_profile', return_value=profile):
            result = integration.load_company('company')
        self.assertEqual(generators(result.facts, role='cfo'), generators(profile['facts'], role='cfo'))


if __name__ == '__main__':
    unittest.main()
