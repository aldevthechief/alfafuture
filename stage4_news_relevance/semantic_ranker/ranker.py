"""Cross-encoder reranking of an already retrieved shortlist, with lazy model loading."""
import os
from functools import lru_cache

import numpy as np
import pandas as pd

from common.chunk_search import article_context

DEFAULT_MODEL = 'BAAI/bge-reranker-v2-m3'


@lru_cache(2)
def _cross_encoder(name: str, device: str):
    from sentence_transformers import CrossEncoder
    from common.model_load import load_once
    return load_once(('cross-encoder', name, device), lambda: CrossEncoder(name, device=device, max_length=512))


class SemanticRanker:
    def __init__(self, model=None, *, model_name: str | None = None, device: str | None = None,
                 batch_size: int = 8, max_candidates: int = 40):
        if batch_size < 1 or max_candidates < 1:
            raise ValueError('batch_size and max_candidates must be positive')
        self.model = model
        self.model_name = model_name or os.environ.get('RERANKER_MODEL', DEFAULT_MODEL)
        # FRIDA may already occupy the GPU. Explicitly opt in to GPU reranking.
        self.device = device or os.environ.get('RERANKER_DEVICE', 'cpu')
        self.batch_size, self.max_candidates = batch_size, max_candidates

    def rank(self, query: str, candidates: pd.DataFrame, top: int | None = None) -> pd.DataFrame:
        if top is not None and (top < 0 or top > self.max_candidates):
            raise ValueError('top must be between zero and max_candidates')
        rows = candidates.head(self.max_candidates).copy()
        if rows.empty or not query.strip() or top == 0:
            return rows.head(0).assign(rerank_score=pd.Series(dtype=float))
        # Excerpts found in the article/fact channels precede the lead, so middle-of-article evidence survives truncation.
        pairs = [(query, f"{r.get('title', '')}\n{article_context(r)}") for r in rows.to_dict('records')]
        model = self.model if self.model is not None else _cross_encoder(self.model_name, self.device)
        scores = np.asarray(model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False), dtype=float)
        if scores.size != len(rows) or scores.ndim > 2 or (scores.ndim == 2 and scores.shape[1] != 1):
            raise ValueError('Cross-encoder must return one score per query/document pair')
        scores = scores.reshape(-1)
        if not np.isfinite(scores).all():
            raise ValueError('Cross-encoder returned non-finite scores')
        rows['rerank_score'] = scores
        result = rows.sort_values('rerank_score', ascending=False, kind='stable')
        result.attrs.update(candidates.attrs, reranker=self.model_name)
        return result.head(top) if top is not None else result
