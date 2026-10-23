"""Cross-source news deduplication in two passes, keeping every source of a merged story.

1. exact: normalized url, then a hash of the normalized title;
2. reprints: cosine of stem TF-IDF over title + lead >= `threshold` (0.45, the stage-4 event rule; embeddings would
   allow ~0.9) within `window_hours` -> one story.
The canonical row is the earliest of the best tier; `sources` / `urls` / `n_sources` list all outlets, so a card can say
"пишут 5 источников" (a confirmation and a significance signal). Stage 1 already drops identical urls and texts; this
pass catches retitled reprints and works for any list of dicts (live scraping, tests).
"""
import hashlib
import re
from datetime import datetime
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from common.search import StemIndex

TRACKING = re.compile(r'^(utm_|yclid|gclid|fbclid|from|ref)', re.I)


def norm_url(url: str) -> str:
    if not url:
        return ''
    p = urlsplit(url.strip())
    query = '&'.join(q for q in p.query.split('&') if q and not TRACKING.match(q))
    host = p.netloc.lower().removeprefix('www.').removeprefix('m.')
    return urlunsplit(('', host, p.path.rstrip('/'), query, ''))


def title_key(title: str) -> str:
    t = re.sub(r'[^\w\s]', ' ', str(title or '').lower().replace('ё', 'е'))
    return hashlib.sha1(' '.join(t.split()).encode()).hexdigest()[:16]


def _ts(value) -> float | None:
    if value in (None, ''):
        return None
    if isinstance(value, datetime):
        return value.timestamp()
    try:
        return datetime.fromisoformat(str(value)[:19]).timestamp()
    except ValueError:
        return None  # an unparsable date is not "now": the row just never matches by time window


def dedup_news(items: list[dict], threshold: float = 0.45, window_hours: float = 72) -> list[dict]:
    """items: {'id', 'title', 'text', 'date', 'source', 'url', optional 'tier'} -> canonical items with sources merged."""
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    seen: dict[str, int] = {}
    for i, it in enumerate(items):
        for key in (('u', norm_url(it.get('url', ''))), ('t', title_key(it.get('title', '')))):
            if not key[1] or key == ('t', title_key('')):
                continue
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i
    if len(items) > 1:
        docs = [f'{it.get("title", "")}. {str(it.get("text", ""))[:400]}' for it in items]
        try:
            index = StemIndex(docs, min_df=1, max_df=1.0)
            sims = (index.matrix @ index.matrix.T).toarray()
        except ValueError:  # empty vocabulary
            sims = np.zeros((len(items), len(items)))
        ts = [_ts(it.get('date')) for it in items]
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if sims[i, j] >= threshold and ts[i] is not None and ts[j] is not None \
                        and abs(ts[i] - ts[j]) <= window_hours * 3600:
                    union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(len(items)):
        groups.setdefault(find(i), []).append(i)
    out = []
    for members in groups.values():
        rows = [items[i] for i in members]
        best = min(rows, key=lambda r: (r.get('tier', 9), _ts(r.get('date')) or float('inf')))
        sources = list(dict.fromkeys(r.get('source', '') for r in rows if r.get('source')))
        urls = list(dict.fromkeys(r.get('url', '') for r in rows if r.get('url')))
        out.append(dict(best, sources=sources, urls=urls, n_sources=len(sources), duplicate_ids=[r.get('id') for r in rows]))
    return out
