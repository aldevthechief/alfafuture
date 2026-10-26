"""Offline test of the suggested-questions suggester with scripted models (no network): schema shape,
code checks (exactly 3 unique short questions), one fix attempt in the same dialog, and the trimmed render.
Run: python -X utf8 tests/unit/backend/test_suggested_questions.py"""
import asyncio
import sys
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import common.llm.models as llm_models
llm_models.LANGFUSE = []
from common.llm.testing import ScriptedChatModel, tool_call
from stage6_insight_generation.insight_suggestions.suggest import (QuestionSuggester, problems, profile_lines,
                                                                   render)

INSIGHT = {'title': 'Ставка 14% до конца 2026 года — бюджет процентных расходов под давлением',
           'role': 'cfo',
           'what_happened': [{'id': 'w1', 'text': 'Альфа-Банк ждёт ключевую ставку 14,0% до конца 2026 года'},
                             {'id': 'w2', 'text': 'К концу 2027-го прогноз 12,0%'}],
           'why_it_matters': [{'id': 'h1', 'text': 'Фондирование лизингового портфеля остаётся дорогим весь 2026 год'}],
           'risks': [{'id': 'r1', 'text': 'Маржа новых договоров сжимается при плавающей ставке'}],
           'actions': ['Пересчитать бюджет процентных расходов на 2027 год'],
           'ai_conclusion': 'Закладывать удешевление фондирования в 2026 году нельзя.'}
FACTS = [{'cat': 'activity', 'depth': 1, 'text': 'Лизинговая компания, ОКВЭД 64.91.2'},
         {'cat': 'profile_context', 'depth': 1, 'text': '[factor 0.6] Импорт автомобилей'}]
GOOD = ['Когда ставка снизится до 12,0%?',
        'Что будет с маржой, если ставку не снизят?',
        'Стоит ли фиксировать ставку до конца года?']


class SuggestTests(unittest.TestCase):
    def test_problems_catches_shape_violations(self):
        self.assertEqual(problems(GOOD), [])
        dup = [GOOD[0], GOOD[0].upper(), GOOD[1]]
        self.assertTrue(any('повторяют' in p for p in problems(dup)))
        self.assertTrue(any('не оканчивается' in p for p in problems([*GOOD[:2], 'делать'])))
        self.assertTrue(any('нужно ровно 3' in p for p in problems(GOOD[:2])))

    def test_render_trims_blocks_and_profile_skips_context(self):
        text = render(INSIGHT)
        self.assertIn('Ставка 14%', text)
        self.assertIn('Альфа-Банк ждёт ключевую ставку', text)
        self.assertNotIn('w1', text)  # ids and sources stay out of the prompt
        lines = profile_lines(FACTS)
        self.assertIn('Лизинговая компания', lines)
        self.assertNotIn('Импорт автомобилей', lines)

    def test_suggester_accepts_good_answer_and_fixes_once(self):
        model = ScriptedChatModel(responses=[
            AIMessage(content='', tool_calls=[tool_call('SuggestedQuestions', questions=GOOD)]),
        ])
        self.assertEqual(asyncio.run(QuestionSuggester(model).asuggest(INSIGHT, profile_lines(FACTS))), GOOD)

    def test_suggester_fixes_duplicates_in_one_more_call(self):
        model = ScriptedChatModel(responses=[
            AIMessage(content='', tool_calls=[tool_call('SuggestedQuestions', questions=[
                GOOD[0], GOOD[0].upper(), 'делать'])]),  # duplicates and no question mark
            AIMessage(content='', tool_calls=[tool_call('SuggestedQuestions', questions=GOOD)]),
        ])
        self.assertEqual(asyncio.run(QuestionSuggester(model).asuggest(INSIGHT, profile_lines(FACTS))), GOOD)


if __name__ == '__main__':
    unittest.main()
