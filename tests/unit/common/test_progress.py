"""Job stage machine: forward-only transitions, per-insight phases, parallel context propagation, job publishing."""
import contextvars
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from backend.jobs import Jobs
from common import progress
from common.news_corpus import write_json
from common.progress import Progress


def codes(p):
    return [h['code'] for h in p.snapshot()['history']]


class ProgressTests(unittest.TestCase):
    def test_stages_only_move_forward_and_terminal_is_final(self):
        p = Progress()
        p.enter('analyzing', total=3)
        self.assertIsNone(p.enter('collecting'))
        p.tick('analyzing'); p.tick('analyzing')
        snap = p.snapshot()
        self.assertEqual((snap['code'], snap['done'], snap['total'], snap['step'], snap['steps']), ('analyzing', 2, 3, 2, 5))
        p.finish('failed', error='boom')
        self.assertIsNone(p.enter('finalizing'))
        self.assertIsNone(p.finish('done'))
        self.assertEqual(codes(p), ['queued', 'analyzing', 'failed'])
        self.assertEqual(p.snapshot()['label'], progress.TERMINAL['failed'])

    def test_insights_move_from_generating_to_verifying_and_a_rewrite_does_not_go_back(self):
        p = Progress()
        p.enter('analyzing', total=2)
        p.expect(2)
        self.assertEqual(p.snapshot()['code'], 'generating')
        p.phase(0, 'check')
        self.assertEqual(p.snapshot()['code'], 'generating')  # insight 1 is still being judged
        p.phase(1, 'write'); p.phase(1, 'check')
        self.assertEqual(p.snapshot()['code'], 'verifying')
        p.phase(1, 'write')  # the checker sent the draft back
        p.phase(0, 'done')
        snap = p.snapshot()
        self.assertEqual((snap['code'], snap['done'], snap['total']), ('verifying', 1, 2))
        self.assertEqual(snap['label'], 'Проверяем инсайты')
        p.enter('finalizing')
        self.assertEqual(codes(p), ['queued', 'analyzing', 'generating', 'verifying', 'finalizing'])

    def test_labels_mention_the_parallel_analytics_branch(self):
        p = Progress()
        p.branch('analytics', 'running')
        p.enter('analyzing', total=1)
        self.assertEqual(p.snapshot()['label'], 'Разбираем новости и аналитику банка')
        p.expect(1); p.phase(0, 'done')
        self.assertEqual(p.snapshot()['label'], 'Ещё чуть-чуть: дожидаемся аналитики банка')
        p.branch('analytics', 'done')
        self.assertEqual(p.snapshot()['label'], 'Формируем инсайты')
        self.assertEqual(p.snapshot()['branches']['analytics']['status'], 'done')

    def test_module_calls_are_noops_without_a_job_and_reach_copied_contexts(self):
        progress.enter('analyzing'); progress.phase('write')  # CLI and tests: nothing bound
        p = Progress()

        def job():
            progress.bind(p)
            progress.enter('analyzing', total=4)
            with ThreadPoolExecutor(4) as pool:  # like LangGraph Send tasks: each runs in its own context copy
                for f in [pool.submit(contextvars.copy_context().run, progress.tick, 'analyzing') for _ in range(4)]:
                    f.result()
            progress.expect(2)
            for pos in range(2):
                ctx = contextvars.copy_context()
                def judge(pos=pos):
                    with progress.track(pos):
                        progress.phase('write'); progress.phase('check')
                threading.Thread(target=ctx.run, args=(judge,)).start()

        contextvars.copy_context().run(job)
        for t in threading.enumerate():
            if t is not threading.current_thread() and not t.daemon:
                t.join(5)
        snap = p.snapshot()
        self.assertEqual((snap['code'], snap['done'], snap['total']), ('verifying', 2, 2))
        self.assertIn('analyzing', codes(p))
        self.assertIsNone(progress.current())


class JobStageTests(unittest.TestCase):
    def test_job_publishes_stages_and_persists_the_final_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs = Jobs(Path(tmp), workers=1)
            seen = threading.Event(); go = threading.Event()

            def work():
                progress.enter('collecting')
                seen.set(); go.wait(5)
                progress.enter('finalizing')
                return {'status': 'succeeded'}

            id = jobs.submit('analysis', {}, work)['id']
            self.assertTrue(seen.wait(5))
            self.assertEqual(jobs.get(id)['stage']['code'], 'collecting')
            self.assertEqual(jobs.get(id)['stage']['label'], 'Собираем факты о компании и подбираем новости')
            go.set()
            job = jobs.wait(id)
            self.assertEqual(job['stage']['code'], 'done')
            self.assertEqual([h['code'] for h in job['stage']['history']], ['queued', 'collecting', 'finalizing', 'done'])
            failed = jobs.wait(jobs.submit('report', {}, lambda: 1 / 0, start='preparing_reports')['id'])
            self.assertEqual([h['code'] for h in failed['stage']['history']], ['queued', 'preparing_reports', 'failed'])
            self.assertEqual(failed['stage']['steps'], 1)
            jobs.close()
            self.assertEqual(Jobs(Path(tmp)).get(id)['stage']['code'], 'done')

    def test_restart_marks_a_running_stage_interrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage = Progress(); stage.enter('analyzing')
            write_json(Path(tmp) / 'a.json', dict(id='a', kind='analysis', request={}, status='running',
                                                  created_at='', result=None, error=None, stage=stage.snapshot()))
            job = Jobs(Path(tmp)).get('a')
            self.assertEqual((job['status'], job['stage']['code']), ('interrupted', 'interrupted'))
            self.assertEqual(job['stage']['label'], progress.TERMINAL['interrupted'])


if __name__ == '__main__':
    unittest.main()
