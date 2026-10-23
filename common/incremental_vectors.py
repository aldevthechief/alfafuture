"""Persistent vectors keyed by model, encoder input format and complete article content."""
import hashlib
import json
import sqlite3
from contextlib import closing

import numpy as np

def keys(news, model):
    return [hashlib.sha256(json.dumps([model, 'title-twice-lead1500-max512-v1',
            str(r.get('title') or ''), str(r.get('text') or '')], ensure_ascii=False).encode()).hexdigest()
            for r in news.to_dict('records')]

def vectors(news, model, folder, encode, documents, seed=None):
    path = folder / 'vectors.sqlite3'
    path.parent.mkdir(parents=True, exist_ok=True)
    identities = keys(news, model)
    with closing(sqlite3.connect(path, timeout=60)) as db:
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, value BLOB NOT NULL)')
        cached = {}
        for key in set(identities):
            row = db.execute('SELECT value FROM vectors WHERE key=?', (key,)).fetchone()
            if row:
                value = np.frombuffer(row[0], dtype=np.float32).copy()
                if value.size and np.isfinite(value).all():
                    cached[key] = value
        if seed is not None:
            cached.update(zip(identities, np.asarray(seed, dtype=np.float32)))
        missing = list(dict.fromkeys(k for k in identities if k not in cached))
        if missing:
            positions = {key: i for i, key in enumerate(identities)}
            batch = encode([documents[positions[key]] for key in missing], 'document', model, batch_size=4)
            batch = np.asarray(batch, dtype=np.float32)
            if batch.ndim != 2 or len(batch) != len(missing) or not np.isfinite(batch).all():
                raise ValueError('Invalid encoder output')
            cached.update(zip(missing, batch))
        if len({len(cached[k]) for k in identities}) > 1:
            raise ValueError('Encoder dimensions changed; change model name or clear vector cache')
        db.executemany('INSERT OR REPLACE INTO vectors VALUES (?,?)',
                       [(k, cached[k].astype(np.float32).tobytes()) for k in set(identities)])
        db.commit()
        return np.stack([cached[key] for key in identities]) if identities else np.empty((0, 0), dtype=np.float32)
