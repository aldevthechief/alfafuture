"""Dense encoder for retrieval. News embeddings are computed once per model
and cached in data/news/emb_<model>.npz next to the ids and corpus fingerprint they belong to.
Default intfloat/multilingual-e5-large-instruct (the retrieval-v2 encoder family); override with ENCODER_MODEL."""
import os
from functools import lru_cache

import numpy as np
import pandas as pd

from common.data import DATA
from common.search import corpus_fingerprint, news_document, news_pool
from common.news_corpus import _atomic, archive_corpus

MODEL = os.environ.get('ENCODER_MODEL', 'intfloat/multilingual-e5-large-instruct')  # was FRIDA: replaced by the e5 tools (23.10)
PREFIX = {'ai-forever/FRIDA': ('search_query: ', 'search_document: '),
          'intfloat/multilingual-e5-base': ('query: ', 'passage: '), 'intfloat/multilingual-e5-large': ('query: ', 'passage: '),
          # the instruct model queries carry the task; its passages are plain text (model card)
          'intfloat/multilingual-e5-large-instruct':
              ('Instruct: Given a web search query, retrieve relevant passages that answer the query
Query: ', '')}


@lru_cache(None)
def _model(name: str = MODEL):
    from sentence_transformers import SentenceTransformer  # heavy import only when dense search is used
    import torch
    m = SentenceTransformer(name, device='cuda' if torch.cuda.is_available() else 'cpu')
    if m.device.type == 'cuda':
        m.half()
    m.max_seq_length = 512
    return m


def encode(texts: list[str], kind: str = 'query', name: str = MODEL, batch_size: int = 16) -> np.ndarray:
    """L2-normalized embeddings; kind is 'query' or 'document' (models trained with prefixes need the right one)."""
    q, d = PREFIX.get(name, ('', ''))
    prefix = q if kind == 'query' else d
    return _model(name).encode([prefix + t for t in texts], batch_size=batch_size, normalize_embeddings=True,
                               show_progress_bar=len(texts) > 500, convert_to_numpy=True).astype(np.float32)


_EMBEDDINGS = {}


def news_embeddings(name: str = MODEL, *, news: pd.DataFrame | None = None) -> np.ndarray:
    """Vectors for this exact snapshot, even if the active corpus changes while an encoder is running."""
    df = news_pool() if news is None else news
    fingerprint = corpus_fingerprint(df)
    key = (str(DATA), name, fingerprint)
    cached = _EMBEDDINGS.get(key)
    if cached is not None:
        return cached
    emb = _news_embeddings(name, fingerprint, df)
    _EMBEDDINGS.clear()
    _EMBEDDINGS[key] = emb
    return emb


def _news_embeddings(name: str, fingerprint: str, df: pd.DataFrame) -> np.ndarray:
    from common.incremental_vectors import vectors
    documents = news_document(df).str[:2000].tolist()
    snapshot = archive_corpus(df, DATA / 'news')
    filename = f'emb_{name.replace("/", "__")}.npz'
    path = snapshot.parent / filename
    for candidate in (path, DATA / 'news' / filename):  # accept a legacy cache only after full validation
        if not candidate.exists():
            continue
        try:
            with np.load(candidate, allow_pickle=False) as z:
                if ('fingerprint' in z and str(z['fingerprint']) == fingerprint
                        and np.array_equal(z['ids'], df['id'].to_numpy(dtype=str))):
                    emb = z['emb']
                    if emb.ndim == 2 and emb.shape[0] == len(df) and np.isfinite(emb).all():
                        vectors(df, name, DATA / 'news', encode, documents, seed=emb)
                        return emb
        except (OSError, ValueError, KeyError):
            pass
    emb = vectors(df, name, DATA / 'news', encode, documents)  # 4 GB GTX 1650
    _atomic(path, lambda p: np.savez(p, ids=df['id'].to_numpy(dtype=str), emb=emb, fingerprint=fingerprint))
    return emb


news_embeddings.cache_clear = _EMBEDDINGS.clear
