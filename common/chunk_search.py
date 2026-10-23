"""Full-article lexical retrieval with original offsets; no model or network required."""
from dataclasses import asdict

import numpy as np
import pandas as pd

from common.news_chunks import build_news_chunks
from common.search import StemIndex, corpus_fingerprint


class ChunkIndex:
    def __init__(self, news: pd.DataFrame):
        self.news = news
        self.chunks = build_news_chunks(news)
        positions = {str(n): i for i, n in enumerate(news['id'])}
        self.article_rows = np.array([positions[c.news_id] for c in self.chunks], dtype=int)
        self.index = None
        self._scores = {}
        if self.chunks:
            try:
                self.index = StemIndex([c.text for c in self.chunks], min_df=1, max_df=1.0)
            except ValueError as exc:
                if 'empty vocabulary' not in str(exc):
                    raise

    def scores(self, query: str) -> np.ndarray:
        if query not in self._scores:
            if len(self._scores) >= 128:
                self._scores.pop(next(iter(self._scores)))
            self._scores[query] = self.index.scores(query) if self.index else np.zeros(len(self.chunks))
        return self._scores[query]

    def article_scores(self, queries: list[str]) -> np.ndarray:
        scores = np.zeros((len(self.news), len(queries)))
        for j, query in enumerate(queries):
            np.maximum.at(scores[:, j], self.article_rows, self.scores(query))
        return scores

    def fragments(self, news_id, queries: list[str], k: int = 3) -> list[dict]:
        ids = [i for i, c in enumerate(self.chunks) if c.news_id == str(news_id)]
        if not ids or not queries or k <= 0:
            return []
        scores = np.max([self.scores(q)[ids] for q in queries], axis=0)
        ranked = sorted(zip(ids, scores), key=lambda x: (-x[1], x[0]))
        chosen = []
        for i, score in ranked:
            c = self.chunks[i]
            if score <= 0:
                break
            # Overlap is context, not another independent hit.
            if any(max(0, min(c.end, r['end']) - max(c.start, r['start'])) >
                   .5 * min(c.end - c.start, r['end'] - r['start']) for r in chosen):
                continue
            chosen.append(dict(asdict(c), score=float(score)))
            if len(chosen) == k:
                break
        return chosen


_cached = None


def chunk_index(news: pd.DataFrame) -> ChunkIndex:
    global _cached
    fingerprint = corpus_fingerprint(news)
    if _cached is None or _cached[0] != fingerprint:
        _cached = fingerprint, ChunkIndex(news)
    return _cached[1]


def attach_fragments(news: pd.DataFrame, queries: list[str]) -> pd.DataFrame:
    result = news.copy()
    index = chunk_index(news)
    result['matched_fragments'] = [index.fragments(i, queries) for i in news['id']]
    return result


def article_context(news: dict, limit: int = 3500) -> str:
    """Relevant original excerpts first, then lead. Never substitute a summary for the source."""
    text = news.get('text')
    text = text if isinstance(text, str) else ''
    pieces = []
    for fragment in news.get('matched_fragments', []):
        start, end = fragment.get('start'), fragment.get('end')
        if (str(fragment.get('news_id')) == str(news.get('id')) and isinstance(start, int)
                and isinstance(end, int) and 0 <= start < end <= len(text)
                and text[start:end] == fragment.get('text')):
            pieces.append(f'[фрагмент {start}:{end}] ' + text[start:end])
    if pieces:
        pieces.append('[начало статьи] ' + text[:600])
    else:
        pieces.append(text)
    return '\n\n'.join(pieces)[:limit]
