"""Conditional requests and persisted page hashes avoid parsing/cleaning unchanged articles."""
import hashlib
import json
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import closing
from datetime import datetime, timezone

import pandas as pd
import requests

from common.news_corpus import normalize_url, article_fingerprint
from stage1_news_pool.web_scraping.fetch import extract
from stage1_news_pool.web_scraping.discover import H

LOG = logging.getLogger(__name__)

def fetch_change(record, previous, session=None, parser=extract):
    headers = dict(H)
    if previous:
        if previous.get('etag'):
            headers['If-None-Match'] = previous['etag']
        if previous.get('modified'):
            headers['If-Modified-Since'] = previous['modified']
    request = session.get if session is not None else requests.get
    response = request(record['url'], headers=headers, timeout=(10, 30))
    if response.status_code == 304:
        if not previous:
            raise ValueError('304 without a cached article')
        return None, previous
    response.raise_for_status()
    digest = hashlib.sha256(response.content).hexdigest()
    state = dict(etag=response.headers.get('ETag'), modified=response.headers.get('Last-Modified'), digest=digest)
    if previous and digest == previous['digest']:
        return None, state
    if not response.encoding or response.encoding.lower() == 'iso-8859-1':
        response.encoding = response.apparent_encoding
    parsed = parser(response.text, record['url'])
    raw = dict(source=record['source'], url=record['url'], title=parsed.get('title') or record.get('title'),
               text=parsed.get('text'), published=record.get('date'), published_page=parsed.get('date'),
               summary=record.get('summary') or parsed.get('excerpt'), date_src=record.get('date_src'),
               rss_category=record.get('rss_category'), categories=parsed.get('categories'),
               author=parsed.get('author'), tags=parsed.get('tags'))
    return raw, state

def collect_changed(staging, records, run_build, *, limit=0):
    folder = staging.parent
    with closing(sqlite3.connect(folder / 'collector.sqlite3', timeout=60)) as db:
        db.execute('CREATE TABLE IF NOT EXISTS pages (url TEXT PRIMARY KEY, state TEXT NOT NULL)')
        known = {url: json.loads(state) for url, state in db.execute('SELECT url,state FROM pages')}
        live = pd.read_parquet(folder / 'news.parquet') if (folder / 'news.parquet').exists() else None
        live_rows = {normalize_url(r['url']): r for r in live.to_dict('records')} if live is not None else {}
        records = list({normalize_url(r['url']): dict(r, url=normalize_url(r['url'])) for r in records}.values())
        if limit:
            # New URLs first; known ones rotate by last check, so a limit does not starve updates forever.
            records.sort(key=lambda r: known.get(r['url'], {}).get('checked', ''))
            records = records[:limit]
        changed, states = [], {}
        with ProcessPoolExecutor(4) as parsers, ThreadPoolExecutor(8) as workers:
            def work(record):
                try:
                    previous = known.get(record['url'])
                    active = live_rows.get(record['url'])
                    if active is None or previous is None or previous.get('cleaned_fingerprint') != article_fingerprint(active):
                        previous = None
                    raw, state = fetch_change(record, previous,
                        parser=lambda html, url: parsers.submit(extract, html, url).result())
                    return record, raw, state
                except Exception as exc:
                    LOG.warning('Retry next cycle: %s: %s', record['url'], exc)
                    return record, None, None
            for record, raw, state in workers.map(work, records):
                if state is None:
                    continue
                state['checked'] = datetime.now(timezone.utc).isoformat()
                states[record['url']] = state
                if raw is not None:
                    changed.append(raw)
        if changed:
            (staging / 'articles.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in changed), encoding='utf-8')
            run_build()
        path = staging / 'news.parquet'
        incoming = pd.read_parquet(path) if path.exists() else None
        accepted_rows = {normalize_url(r['url']): r for r in incoming.to_dict('records')} if incoming is not None else {}
        accepted = set(accepted_rows)
        changed_urls = {r['url'] for r in changed}
        # Failed/filtered new content must be retried, never acknowledged as successfully published.
        for url, state in states.items():
            if url not in changed_urls or url in accepted:
                state['cleaned_fingerprint'] = article_fingerprint(accepted_rows[url] if url in accepted_rows else live_rows[url])
                db.execute('INSERT OR REPLACE INTO pages VALUES (?,?)', (url, json.dumps(state)))
        db.commit()
        return incoming
