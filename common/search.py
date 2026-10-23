"""Lexical search shared by stages 3-5: TF-IDF over 6-letter stems with cosine similarity.

`StemIndex` indexes any list of texts (news, bank analytics pages); `news_index()` is the cached index of the news pool.
"""
import hashlib
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from common.data import DATA
from common.text import stems


class StemIndex:
    def __init__(self, docs: list[str], min_df: int = 2, max_df: float = 0.3):
        self.vectorizer = TfidfVectorizer(analyzer=stems, min_df=min_df, max_df=max_df, sublinear_tf=True)
        try:
            self.matrix = self.vectorizer.fit_transform([d.lower() for d in docs])
        except ValueError as exc:
            if 'empty vocabulary' not in str(exc):
                raise
            from scipy.sparse import csr_matrix
            self.vectorizer = TfidfVectorizer(analyzer=stems, vocabulary={'__empty_corpus__': 0})
            self.vectorizer.fit([''])
            self.matrix = csr_matrix((len(docs), 1))

    def transform(self, texts: list[str]):
        return self.vectorizer.transform([t.lower() for t in texts])

    def scores(self, query: str) -> np.ndarray:
        return (self.matrix @ self.transform([query]).T).toarray().ravel()

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Top-k (row, cosine) with a positive score."""
        s = self.scores(query)
        top = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in top if s[i] > 0]


def news_document(df: pd.DataFrame) -> pd.Series:
    """Title twice (it carries the event) + the first 1500 chars of the text."""
    return df.title.fillna('') + '. ' + df.title.fillna('') + '. ' + df.text.fillna('').str[:1500]


def news_version() -> tuple[str, int, int]:
    path = DATA / 'news' / 'news.parquet'
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


def corpus_fingerprint(df: pd.DataFrame) -> str:
    """Include content, dates and source quality: unchanged ids do not imply unchanged events or embeddings."""
    columns = ['id', 'url', 'title', 'text', 'date', 'source', 'tier']
    values = df.reindex(columns=columns).astype('string').fillna('')
    return hashlib.sha256(pd.util.hash_pandas_object(values, index=False).values.tobytes()).hexdigest()


@lru_cache(1)
def _news_pool(version: tuple[str, int, int]) -> pd.DataFrame:
    df = pd.read_parquet(version[0])
    df = df[~df.title.str.contains('пошлина на ввоз по коду', na=False)].reset_index(drop=True)
    df.attrs['corpus_fingerprint'] = corpus_fingerprint(df)
    return df  # lesprom tariff tables are not news


def news_pool() -> pd.DataFrame:
    from common.news_snapshot import ACTIVE
    state = ACTIVE.get()
    if state is None:
        return _news_pool(news_version())
    if state['news'] is None:
        state['news'] = _news_pool(news_version())
    return state['news']


@lru_cache(1)
def _news_index(version: tuple[str, int, int]) -> tuple[pd.DataFrame, StemIndex]:
    df = news_pool()
    docs = news_document(df).tolist()
    try:
        index = StemIndex(docs)
    except ValueError:
        # A freshly started collector may have only one article or no repeated vocabulary.
        index = StemIndex(docs, min_df=1, max_df=1.0)
    return df, index


def news_index() -> tuple[pd.DataFrame, StemIndex]:
    from common.news_snapshot import ACTIVE
    state = ACTIVE.get()
    if state is None:
        return _news_index(news_version())
    if state['index'] is None:
        df = news_pool()
        docs = news_document(df).tolist()
        try:
            index = StemIndex(docs)
        except ValueError:
            index = StemIndex(docs, min_df=1, max_df=1.0)
        state['index'] = df, index
    return state['index']


# Preserve the explicit invalidation hooks used by development scripts.
news_pool.cache_clear = _news_pool.cache_clear
news_index.cache_clear = _news_index.cache_clear
