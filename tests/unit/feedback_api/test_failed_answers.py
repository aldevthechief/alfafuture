import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / 'feedback_api'
RULE = 'отсутствие факта не значит, что компания этим не занимается'


class FailedAnswersTests(unittest.TestCase):
    def test_every_example_is_well_formed(self):
        data = json.loads((ROOT / 'golden_set' / 'failed_answers.json').read_text(encoding='utf-8'))
        ids = [x['id'] for x in data['examples']]
        self.assertEqual(len(ids), len(set(ids)))
        for x in data['examples']:
            self.assertIn(x['failure_type'], data['failure_types'])
            for key in ('question', 'answer', 'expected', 'company_id', 'insight_id'):
                self.assertTrue(x[key].strip(), key)
            for span in x['bad_spans']:
                self.assertIn(span['text'], x['answer'])

    def test_both_agent_prompts_forbid_unsupported_company_claims(self):
        for name in ('analyze.md', 'company_chat.md'):
            self.assertIn(RULE, (ROOT / 'prompts' / name).read_text(encoding='utf-8'), name)


if __name__ == '__main__':
    unittest.main()
