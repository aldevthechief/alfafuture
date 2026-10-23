"""Independent periodic collector; never imported or started by the agent pipeline.

python -m stage1_news_pool.updater --once --only cbr
python -m stage1_news_pool.updater --interval 900 --days 7
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Event

import pandas as pd

from common.data import DATA
from common.news_corpus import normalize_url, publish_news, stable_id, write_json
from common.search import corpus_fingerprint

LOG = logging.getLogger(__name__)
SCRIPTS = Path(__file__).parent / 'web_scraping'


@contextmanager
def update_lock(folder: Path):
    """OS lock is released on process exit, including crashes; lock file may stay."""
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'updater.lock').open('a+b') as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            acquire = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            release = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            acquire = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as exc:
            raise RuntimeError('Another news updater is already running') from exc
        try:
            yield
        finally:
            handle.seek(0)
            release()


def merge_news(previous: pd.DataFrame | None, incoming: pd.DataFrame) -> pd.DataFrame:
    """Keep existing history; normalized URL identifies updates. Repeat runs are idempotent."""
    frames = [x for x in (previous, incoming) if x is not None and not x.empty]
    if not frames:
        return incoming.copy()
    result = pd.concat(frames, ignore_index=True)
    result['url'] = result.url.map(normalize_url)
    result = result.drop_duplicates('url', keep='last').copy()
    result['id'] = result.url.map(stable_id)
    if result.id.duplicated().any():
        raise ValueError('Stable news ID collision')
    return result.sort_values('id').reset_index(drop=True)


def collect(staging: Path, *, days: int, only: list[str], timeout: float, limit: int) -> pd.DataFrame | None:
    """Existing scraper scripts run in isolation; failures never overwrite the live corpus."""
    def run(script, *args):
        subprocess.run([sys.executable, '-X', 'utf8', str(SCRIPTS / script), *map(str, args)],
                       cwd=SCRIPTS.parents[1], check=True, timeout=timeout)
    index, articles = staging / 'index.jsonl', staging / 'articles.jsonl'
    run('discover.py', '--days', days, '--out', index, *(['--only', *only] if only else []))
    if not index.exists() or not index.read_text(encoding='utf-8').strip():
        return None
    from stage1_news_pool.incremental import collect_changed
    records = [json.loads(line) for line in index.read_text(encoding='utf-8').splitlines() if line.strip()]
    return collect_changed(staging, records, lambda: run('build.py', '--index', index, '--articles', articles,
                           '--out-dir', staging, '--days', days), limit=limit)


def refresh_once(folder: Path = DATA / 'news', *, days: int = 7, only: list[str] = (),
                 timeout: float = 1800, limit: int = 0, collector=collect,
                 enrich_facts: bool = False, fact_limit: int | None = None, extractor=None) -> dict:
    if days <= 0 or timeout <= 0 or limit < 0:
        raise ValueError('days/timeout must be positive and limit nonnegative')
    folder = Path(folder)
    with update_lock(folder):
        try:
            with tempfile.TemporaryDirectory(prefix='news-refresh-', dir=folder) as tmp:
                incoming = collector(Path(tmp), days=days, only=list(only), timeout=timeout, limit=limit)
                live = folder / 'news.parquet'
                previous = pd.read_parquet(live) if live.exists() else None
                report = dict(status='unchanged', received=0 if incoming is None else len(incoming))
                if incoming is not None and not incoming.empty:
                    merged = merge_news(previous, incoming)
                    if previous is None or corpus_fingerprint(merged) != corpus_fingerprint(previous):
                        report.update(publish_news(merged, folder), status='updated')
                if enrich_facts and live.exists():
                    from common.news_fact_index import FactStore
                    report['facts'] = FactStore(folder / 'facts.sqlite3').refresh(pd.read_parquet(live), extractor, limit=fact_limit)
                report['finished_at'] = datetime.now(timezone.utc).isoformat()
                write_json(folder / 'updater_status.json', report)
                return report
        except Exception as exc:
            write_json(folder / 'updater_status.json', dict(status='failed', error=str(exc),
                       finished_at=datetime.now(timezone.utc).isoformat()))
            raise


def serve(interval: float, stop: Event | None = None, **kwargs):
    if interval <= 0:
        raise ValueError('interval must be positive')
    stop = stop or Event()
    while not stop.is_set():
        try:
            LOG.info('%s', refresh_once(**kwargs))
        except Exception:
            LOG.exception('Refresh failed; retrying on the next interval')
        stop.wait(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--enrich-facts', action='store_true', help='incrementally extract supported facts after collection (uses LLM)')
    parser.add_argument('--fact-limit', type=int, default=None, help='maximum new/changed articles to extract per cycle')
    parser.add_argument('--interval', type=float, default=900)
    parser.add_argument('--days', type=int, default=7)
    parser.add_argument('--only', nargs='+', default=[])
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--folder', type=Path, default=DATA / 'news')
    args = vars(parser.parse_args())
    once, interval = args.pop('once'), args.pop('interval')
    if interval <= 0 or args['days'] <= 0 or args['timeout'] <= 0 or args['limit'] < 0:
        parser.error('interval/days/timeout must be positive and limit nonnegative')
    if args['fact_limit'] is not None and (args['fact_limit'] < 0 or not args['enrich_facts']):
        parser.error('--fact-limit must be nonnegative and requires --enrich-facts')
    from stage1_news_pool.web_scraping.sources import SOURCES
    unknown = set(args['only']) - SOURCES.keys()
    if unknown:
        parser.error('Unknown sources: ' + ', '.join(sorted(unknown)))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    try:
        if once:
            print(json.dumps(refresh_once(**args), ensure_ascii=False, default=str))
        else:
            serve(interval, **args)
    except KeyboardInterrupt:
        LOG.info('Updater stopped')


if __name__ == '__main__':
    main()
