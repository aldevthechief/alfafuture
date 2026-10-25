"""Model-produced role labels must never replace the caller's role identity (draft, checked draft and the one fix)."""
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from stage6_insight_generation.insight_writer.insight import Insight, Review
from stage6_insight_generation.insight_writer.writer import InsightWriter


def draft():
    return Insight(title='Ставка', role='Финансовый директор (CFO)', what_happened=[], why_it_matters=[], actions=[],
                   ai_conclusion='Вывод', confidence='low')


class WriterRoleTests(unittest.TestCase):
    def writer(self):
        writer = object.__new__(InsightWriter)
        writer._dialog = Mock(return_value=({}, []))
        writer._fix = Mock(return_value=[])
        return writer

    def test_draft_and_fix_keep_caller_role(self):
        writer = self.writer()
        writer.writer = SimpleNamespace(invoke=Mock(side_effect=[draft(), draft()]))
        writer._check = Mock(return_value=Review(ok=False, issues=['Число']))
        insight, review = writer.write('U_1', 'cfo', '{}', SimpleNamespace())
        self.assertEqual(insight.role, 'cfo')
        self.assertTrue(review.ok)  # fixed once, final
        self.assertEqual(writer._check.call_args.args[0].role, 'cfo')

    def test_async_draft_and_fix_keep_caller_role(self):
        writer = self.writer()
        writer.writer = SimpleNamespace(ainvoke=AsyncMock(side_effect=[draft(), draft()]))
        writer.checker = SimpleNamespace(ainvoke=AsyncMock(return_value=Review(ok=False, issues=['Число'])))
        writer._check_inputs = Mock(return_value={})
        evidence = SimpleNamespace(news=[], bank=[], origin='news')
        insight, _ = asyncio.run(writer.awrite('U_1', 'cfo', '{}', evidence))
        self.assertEqual(insight.role, 'cfo')
        self.assertEqual(writer._check_inputs.call_args.args[0].role, 'cfo')


if __name__ == '__main__':
    unittest.main()
