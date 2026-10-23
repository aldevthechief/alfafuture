"""Lexical + FRIDA + extracted facts, fused before a bounded cross-encoder pass."""
import warnings

import numpy as np
import pandas as pd

from common.chunk_search import chunk_index
from common.fusion import reciprocal_rank_fusion, top_channel
from common.news_fact_index import FactIndex, FactStore, fact_fragments
from common.search import corpus_fingerprint


def hybrid_search(news: pd.DataFrame, lexical_index, query: str, k: int = 8, *, dense: bool = True,
                  rerank: bool = True, ranker=None, fact_store=None, candidate_limit: int = 40) -> pd.DataFrame:
    if k < 0 or candidate_limit < 1:
        raise ValueError('k must be nonnegative; candidate_limit must be positive')
    if not query.strip() or k == 0 or news.empty:
        return news.head(0).copy().assign(similarity=pd.Series(dtype=float), matched_fragments=[], matched_facts=[])
    limit = max(k, candidate_limit)
    chunks = chunk_index(news)
    lexical = np.maximum(lexical_index.scores(query), chunks.article_scores([query])[:, 0])
    channels = {'lexical': top_channel(lexical, limit)}
    diagnostics = {}
    if dense:
        try:
            from common.embed import encode, news_embeddings
            embeddings = news_embeddings(news=news)
            vector = encode([query])
            scores = (embeddings @ vector.T).reshape(-1)
            if len(scores) != len(news) or not np.isfinite(scores).all():
                raise ValueError('Invalid semantic scores')
            channels['semantic'] = top_channel(scores, limit)
            diagnostics['semantic'] = 'ok'
        except (OSError, RuntimeError, ImportError, ValueError) as exc:
            diagnostics['semantic'] = f'unavailable: {exc}'
            warnings.warn(f'FRIDA search unavailable; using remaining channels: {exc}', RuntimeWarning)
    store = fact_store if fact_store is not None else FactStore()
    facts = FactIndex(news, store.records(news))
    channels['facts'] = top_channel(facts.scores(query), limit)
    fused = reciprocal_rank_fusion(*channels.values())
    order = [int(i) for i in np.argsort(-fused, kind='stable') if fused[i] > 0][:limit]
    result = news.iloc[order].copy()
    result['similarity'] = [float(fused[i]) for i in order]  # legacy column: now RRF, not cosine
    result['retrieval_score'] = result['similarity']
    result['search_channels'] = [[name for name, values in channels.items() if np.isfinite(values[i])] for i in order]
    result['channel_scores'] = [{name: float(values[i]) for name, values in channels.items() if np.isfinite(values[i])} for i in order]
    matches = [facts.matches(query, i) for i in order]
    result['matched_facts'] = matches
    result['matched_fragments'] = [fact_fragments(news.iloc[i].to_dict(), found) + chunks.fragments(news.iloc[i]['id'], [query])
                                   for i, found in zip(order, matches)]
    result.attrs.update(news.attrs, corpus_fingerprint=corpus_fingerprint(news), search_diagnostics=diagnostics)
    if rerank and not result.empty:
        try:
            if ranker is None:
                from stage4_news_relevance.semantic_ranker.ranker import SemanticRanker
                ranker = SemanticRanker(max_candidates=limit)
            result = ranker.rank(query, result, top=min(k, ranker.max_candidates))
            diagnostics['rerank'] = 'ok'
        except (OSError, RuntimeError, ImportError, ValueError) as exc:
            diagnostics['rerank'] = f'unavailable: {exc}'
            warnings.warn(f'Cross-encoder unavailable; retaining fused order: {exc}', RuntimeWarning)
    result = result.head(k)
    result.attrs['search_diagnostics'] = diagnostics
    return result
