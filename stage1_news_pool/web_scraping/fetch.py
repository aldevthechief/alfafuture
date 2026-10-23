"""Stage 2: download articles from data/index.jsonl, extract text -> data/articles.jsonl (resumable)"""
import argparse, json, random, threading, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests
import trafilatura

try:
    from .discover import DATA, H
except ImportError:
    from discover import DATA, H

PER_HOST = 3  # concurrent requests per host
MIN_TEXT = 200
sems = defaultdict(lambda: threading.Semaphore(PER_HOST))
lock = threading.Lock()
local = threading.local()


def norm(u):
    return u.split('#')[0].rstrip('/')


def get(url):
    if not hasattr(local, 's'):
        local.s = requests.Session()
    last_error = None
    for attempt in range(3):
        try:
            # Do not hold a host slot during retry backoff.
            with sems[urlparse(url).netloc]:
                r = local.s.get(url, headers=H, timeout=(10, 30))
            if r.status_code in (429, 500, 502, 503, 504):
                last_error = requests.HTTPError(
                    f"HTTP {r.status_code} for {url}", response=r
                )
                if attempt == 2:
                    raise last_error
                retry_after = r.headers.get('Retry-After')
                delay = min(30.0, 2 ** attempt + random.uniform(0, 1))
                if retry_after and retry_after.isdigit():
                    delay = min(60.0, float(retry_after))
                time.sleep(delay)
                continue
            r.raise_for_status()
            if not r.encoding or r.encoding.lower() == 'iso-8859-1':
                r.encoding = r.apparent_encoding
            return r.text
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = exc
            if attempt == 2:
                raise
            time.sleep(min(30.0, 2 ** attempt + random.uniform(0, 1)))
    raise last_error or RuntimeError(f"Failed to fetch {url}")


def extract(html, url):
    # runs in a worker process: lxml parsing in threads segfaults on Windows
    kw = dict(url=url, output_format='json', with_metadata=True, include_comments=False, include_tables=False)
    j = json.loads(trafilatura.extract(html, favor_precision=True, **kw) or '{}')
    if len(j.get('text') or '') < 500:  # precision mode over-trims some layouts (dp.ru)
        j2 = json.loads(trafilatura.extract(html, **kw) or '{}')
        j = j2 if len(j2.get('text') or '') > len(j.get('text') or '') else j
    return j


def work(rec, out, procs):
    try:
        j = procs.submit(extract, get(rec['url']), rec['url']).result()
    except Exception as e:
        j = {'error': str(e)[:200]}
    row = {
        'id': None, 'source': rec['source'], 'url': rec['url'],
        'title': j.get('title') or rec.get('title'),
        'published': rec.get('date'), 'published_page': j.get('date'), 'date_src': rec.get('date_src'),
        'summary': rec.get('summary') or j.get('excerpt') or j.get('description'),
        'text': j.get('text'), 'author': j.get('author'), 'categories': j.get('categories'), 'tags': j.get('tags'),
        'rss_category': rec.get('rss_category'), 'error': j.get('error'), 'fetched_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }
    with lock:
        out.write(json.dumps(row, ensure_ascii=False) + '\n')
        out.flush()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', default=str(DATA / 'index.jsonl'))
    ap.add_argument('--out', default=str(DATA / 'articles.jsonl'))
    ap.add_argument('--tier', type=int, default=2, help='fetch sources with tier <= this')
    ap.add_argument('--per-source', type=int, default=0, help='random sample size per source (0 = all)')
    ap.add_argument('--workers', type=int, default=48)
    ap.add_argument('--limit', type=int, default=0, help='max articles for a safe test (0 = all)')
    a = ap.parse_args()

    done = set()
    try:
        with open(a.out, encoding='utf-8') as f:
            # short/failed rows are retried on the next run; build.py keeps the best row per url
            done = {norm(r['url']) for r in map(json.loads, f) if len(r.get('text') or '') >= MIN_TEXT}
    except FileNotFoundError:
        pass
    by_src, seen = defaultdict(list), set(done)
    for l in open(a.index, encoding='utf-8'):
        r = json.loads(l)
        k = norm(r['url'])
        if k not in seen and r['tier'] <= a.tier:
            seen.add(k)
            by_src[r['source']].append(r)
    todo = []
    for src, rs in by_src.items():
        random.Random(0).shuffle(rs)
        todo += rs[:a.per_source] if a.per_source else rs
    random.Random(1).shuffle(todo)  # interleave hosts so per-host limits don't serialize the pool
    todo.sort(key=lambda r: r['tier'])  # stable: business sources first, still interleaved
    if a.limit > 0:
        todo = todo[:a.limit]
    print(f'already {len(done)}, todo {len(todo)}', flush=True)
    with open(a.out, 'a', encoding='utf-8') as out, ProcessPoolExecutor(6) as procs, ThreadPoolExecutor(a.workers) as ex:
        for i, _ in enumerate(ex.map(lambda r: work(r, out, procs), todo), 1):
            if i % 200 == 0:
                print(f'{i}/{len(todo)}', flush=True)
    print('done')
