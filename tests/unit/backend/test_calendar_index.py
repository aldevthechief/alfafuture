"""Calendar index: only structural dates, latest run per company+role wins, range queries."""
import tempfile
import unittest
from pathlib import Path

from backend.calendar_index import _iso, events_between, index_job, maybe_index_job


def job(rows, company_id='U_1', role='cfo', job_id='job-1', status='succeeded'):
    return dict(id=job_id, kind='analysis', status=status, created_at='2026-10-25',
                request=dict(company_id=company_id, role=role, user_id='alice'),
                result=dict(results=rows))


class CalendarIndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / 'events.db'

    def tearDown(self):
        self.tmp.cleanup()

    def test_structural_dates_only(self):
        rows = [
            dict(status='ready', news_id=1, insight=dict(title='Точная дата', event_date='2026-10-07')),
            dict(status='ready', news_id=2, insight=dict(title='Дата новости'),
                 news_references={'news:2': dict(date='2026-10-12')}),
            dict(status='ready', news_id=3, insight=dict(title='Только срок из текста')),  # без даты
            dict(status='ready', news_id=4, insight=dict(title='Кривая дата', event_date='конец квартала')),
            dict(status='failed', news_id=5, insight=dict(title='Неготовая строка', event_date='2026-10-01')),
        ]
        self.assertEqual(index_job(job(rows), db_path=self.db), 2)
        dates = [e['event_date'] for e in events_between('2026-10-01', '2026-10-31', db_path=self.db)]
        self.assertEqual(dates, ['2026-10-07', '2026-10-12'])

    def test_latest_run_replaces_company_role(self):
        self.assertEqual(index_job(job([dict(status='ready', news_id=1,
            insight=dict(title='Старый прогон', event_date='2026-10-01'))]), db_path=self.db), 1)
        self.assertEqual(index_job(job([dict(status='ready', news_id=2,
            insight=dict(title='Новый прогон', event_date='2026-10-02'))]), db_path=self.db), 1)
        events = events_between('2026-01-01', '2026-12-31', db_path=self.db)
        self.assertEqual([e['title'] for e in events], ['Новый прогон'])
        self.assertEqual(events[0]['kind'], 'calendar')

    def test_maybe_index_job_never_raises_and_skips_unfinished(self):
        maybe_index_job(job([], status='running'), db_path=self.db)  # не упало и не проиндексировало
        maybe_index_job(dict(kind='report', status='succeeded'), db_path=self.db)
        self.assertEqual(events_between('2000-01-01', '2100-01-01', db_path=self.db), [])

    def test_range_validation(self):
        with self.assertRaises(ValueError):
            events_between('до конца квартала', '2026-10-31', db_path=self.db)
        self.assertFalse(_iso('2026-13-40'))
        self.assertEqual(_iso(' 2026-10-07 '), '2026-10-07')


if __name__ == '__main__':
    unittest.main()
