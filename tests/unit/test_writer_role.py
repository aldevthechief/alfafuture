"""Model-produced role labels must never replace the caller's role identity."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from stage6_insight_generation.insight_writer.insight import Insight, Review
from stage6_insight_generation.insight_writer.writer import InsightWriter


class WriterRoleTests(unittest.TestCase):
    def test_initial_reminder_and_review_rewrites_all_keep_caller_role(self):
        for remind in [False, True]:
            with self.subTest(remind=remind):
                draft = Insight(title='Ставка', role='Финансовый директор (CFO)', what_happened=[],
                    why_it_matters=[], actions=[], ai_conclusion='Вывод', confidence='low')
                writer = object.__new__(InsightWriter)
                writer.writer = SimpleNamespace(invoke=Mock(return_value=draft))
                writer._remind = Mock(side_effect=[['Стиль'] if remind else [], []])
                writer._check = Mock(side_effect=[Review(ok=False, issues=['Число']), Review(ok=True)])
                writer.revisions = 1
                insight, review = writer.write('U_1', 'cfo', '{}', SimpleNamespace(render=lambda: 'Доказательства'))
                self.assertEqual(insight.role, 'cfo')
                self.assertTrue(review.ok)
                self.assertEqual(draft.role, 'Финансовый директор (CFO)')
                self.assertEqual(writer.writer.invoke.call_count, 3 if remind else 2)
                self.assertTrue(all(c.args[0].role == 'cfo' for c in writer._check.call_args_list))
                self.assertTrue(all(c.args[0].role == 'cfo' for c in writer._remind.call_args_list))


if __name__ == '__main__':
    unittest.main()
