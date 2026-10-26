import datetime as dt
import unittest

from backend.cache_check import check_row

TODAY = dt.date(2026, 10, 26)


def row(sources, evidence=None, text='Факт', **extra):
    return dict(status='ready', insight=dict(what_happened=[dict(text=text, sources=sources)], actions=['Сделать']),
                evidence=evidence or {}, **extra)


class CacheCheckTests(unittest.TestCase):
    def test_resolvable_news_with_url_passes(self):
        news = dict(news=[dict(id=1, url='https://example.org/a')])
        self.assertEqual(check_row(row(['news:1'], news), TODAY), [])

    def test_unknown_source_fails_and_facts_are_resolved_by_the_ui(self):
        self.assertEqual([lvl for lvl, _ in check_row(row(['news:2']), TODAY)], ['fail'])
        self.assertEqual(check_row(row(['fact:3']), TODAY), [])

    def test_claim_without_sources_and_not_ready_row_fail(self):
        self.assertEqual(check_row(row([]), TODAY)[0][0], 'fail')
        self.assertEqual(check_row(dict(status='model_error'), TODAY)[0][0], 'fail')

    def test_past_deadline_is_only_a_warning(self):
        got = check_row(row(['fact:1'], text='Решить до 23 октября'), TODAY)
        self.assertEqual([lvl for lvl, _ in got], ['warn'])


if __name__ == '__main__':
    unittest.main()
