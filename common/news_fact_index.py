"""Versioned, incremental fact storage and an additional lexical retrieval channel.

Reading/searching never calls an LLM. Only explicit refresh invokes ConsequenceExtractor.
SQLite transactions retain completed articles across interruption and isolate concurrent readers.
"""
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from common.data import DATA
from common.news_corpus import article_fingerprint
from common.search import StemIndex

FIELDS = ('participants', 'industries', 'countries', 'commodities')


def current_version():
    from stage4_news_relevance.consequence_extraction.consequences import extraction_version
    return extraction_version()


def validated_facts(row: dict, facts: list[dict]) -> list[dict]:
    """Only exact quotes from the current source survive, including when reading persisted data."""
    text = row.get('text') or ''
    result, seen = [], set()
    for fact in facts:
        evidence = fact.get('evidence') or {}
        quote, start, end = evidence.get('quote'), evidence.get('start'), evidence.get('end')
        if (not isinstance(quote, str) or not quote.strip() or not isinstance(start, int)
                or not isinstance(end, int) or not 0 <= start < end <= len(text) or text[start:end] != quote):
            continue
        if (fact.get('text'), quote) in seen:
            continue
        seen.add((fact.get('text'), quote))
        clean = {key: [str(v) for v in fact.get(key, []) if isinstance(v, str) and v.strip()] for key in FIELDS}
        clean.update(text=str(fact.get('text', '')), change=str(fact.get('change', '')),
                     evidence=dict(source=f"news:{row['id']}", quote=quote, start=start, end=end, url=row.get('url') or ''))
        result.append(clean)
    return result


class FactStore:
    def __init__(self, path: Path | None = None, *, version: str | None = None):
        self.path = Path(path) if path is not None else DATA / 'news' / 'facts.sqlite3'
        self.version = version or current_version()

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=30)
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('CREATE TABLE IF NOT EXISTS facts (news_id TEXT NOT NULL, fingerprint TEXT NOT NULL, '
                   'version TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(news_id, fingerprint, version))')
        return db

    def records(self, news: pd.DataFrame) -> dict[str, list[dict]]:
        if not self.path.exists() or news.empty:
            return {}
        wanted = {str(r['id']): r for r in news.to_dict('records')}
        fingerprints = {nid: article_fingerprint(row) for nid, row in wanted.items()}
        out = {}
        # Read-only mode must not create files or mutate a database while the search runs.
        with closing(sqlite3.connect(self.path.resolve().as_uri() + '?mode=ro', uri=True, timeout=30)) as db:
            if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='facts'").fetchone():
                return {}
            for nid, fingerprint, payload in db.execute('SELECT news_id, fingerprint, payload FROM facts WHERE version=?', (self.version,)):
                row = wanted.get(nid)
                if row is not None and fingerprint == fingerprints[nid]:
                    out[nid] = validated_facts(row, json.loads(payload))
        return out

    def refresh(self, news: pd.DataFrame, extractor=None, *, limit: int | None = None) -> dict:
        if limit is not None and limit < 0:
            raise ValueError('limit must be nonnegative')
        report = dict(processed=0, reused=0, failed=[], pending=0)
        with closing(self._connect()) as db:
            for row in news.to_dict('records'):
                key = (str(row['id']), article_fingerprint(row), self.version)
                if db.execute('SELECT 1 FROM facts WHERE news_id=? AND fingerprint=? AND version=?', key).fetchone():
                    report['reused'] += 1
                    continue
                if limit is not None and report['processed'] + len(report['failed']) >= limit:
                    report['pending'] += 1
                    continue
                try:
                    if extractor is None:
                        from stage4_news_relevance.consequence_extraction.consequences import ConsequenceExtractor
                        extractor = ConsequenceExtractor()
                    payload = extractor.extract(row)
                    facts = validated_facts(row, payload.get('facts', []))
                    # A result containing only rejected quotes is not a completed extraction; retry next refresh.
                    if not facts and (payload.get('facts') or payload.get('rejected_facts')):
                        raise ValueError('No extracted fact has a valid source quote')
                    db.execute('INSERT OR REPLACE INTO facts VALUES (?, ?, ?, ?)',
                               (*key, json.dumps(facts, ensure_ascii=False)))
                    db.commit()  # checkpoint each article; never lose earlier work on a later failure
                    report['processed'] += 1
                except Exception as exc:
                    db.rollback()
                    report['failed'].append(dict(news_id=str(row['id']), error=str(exc)))
        return report


class FactIndex:
    def __init__(self, news: pd.DataFrame, records: dict[str, list[dict]]):
        self.news, self.items, positions, docs = news, [], [], []
        for position, row in enumerate(news.to_dict('records')):
            for fact in records.get(str(row['id']), []):
                self.items.append(fact)
                positions.append(position)
                docs.append(' '.join([fact['text'], fact['change'], *[v for key in FIELDS for v in fact[key]]]))
        self.positions = np.array(positions, dtype=int)
        self.index = StemIndex(docs, min_df=1, max_df=1.0) if docs else None
        self._query, self._scores = None, None

    def _fact_scores(self, query):
        if query != self._query:
            self._query = query
            self._scores = self.index.scores(query) if self.index is not None else np.zeros(0)
        return self._scores

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.news))
        if self.index is not None:
            np.maximum.at(scores, self.positions, self._fact_scores(query))
        return scores

    def matches(self, query: str, position: int, k: int = 3) -> list[dict]:
        if self.index is None:
            return []
        scores = self._fact_scores(query)
        ids = np.flatnonzero((self.positions == position) & (scores > 0))
        ids = ids[np.argsort(-scores[ids], kind='stable')[:k]]
        return [dict(self.items[i], score=float(scores[i])) for i in ids]


def fact_fragments(row: dict, facts: list[dict]) -> list[dict]:
    return [dict(news_id=str(row['id']), chunk_index=None, text=f['evidence']['quote'],
                 start=f['evidence']['start'], end=f['evidence']['end'], score=f.get('score', 0), origin='fact')
            for f in facts]
