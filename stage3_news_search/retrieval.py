"""Stage-3 entry point: event retrieval adapted to the news-row contract of stages 4-6.
v2 (default) = standard generators + generator factory over e5 + qwen + BM25 (retrieval_v2/README.md);
v3 = v2 + news-first channel generators (retrieval_v3/README.md);
hybrid = standard generators over TF-IDF + FRIDA; lexical = facet search."""
import pandas as pd

from common.search import corpus_fingerprint, news_pool
from common.news_corpus import StaleNewsReferences
from stage3_news_search.candidate_generators.generators import candidates as hybrid_candidates
from stage3_news_search.facet_retrieval.candidates import candidates as lexical_candidates
from stage3_news_search.retrieval_v2.candidates import candidates as v2_candidates
from stage3_news_search.retrieval_v3.candidates import candidates as v3_candidates


def event_news(events: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    """One original representative per event, retaining membership, sources and retrieval explanations.

    Event ids never masquerade as news ids. Members are stored tier/date first by the event builder.
    """
    if events.attrs.get('corpus_fingerprint', corpus_fingerprint(news)) != corpus_fingerprint(news):
        raise StaleNewsReferences('News changed during retrieval; retry the batch against the new corpus')
    from common.news_fact_index import FactIndex, FactStore, fact_fragments
    fact_records = FactStore().records(news)
    by_id = {str(row['id']): row for row in news.to_dict('records')}
    rows = []
    for event in events.to_dict('records'):
        if any(str(i) not in by_id for i in event['news_ids']):
            raise StaleNewsReferences(f'Event {event["event"]} contains IDs absent from the current corpus')
        members = [by_id[str(i)] for i in event['news_ids']]
        if not members:
            raise ValueError(f'Event {event["event"]} has no members in the current news corpus')
        from common.chunk_search import ChunkIndex
        queries = [g['query'] for g in event['generators']]
        chunks = ChunkIndex(pd.DataFrame(members))
        scores = chunks.article_scores(queries)
        fact_index = FactIndex(pd.DataFrame(members), fact_records)
        if queries:
            import numpy as np
            scores = np.maximum(scores, np.column_stack([fact_index.scores(q) for q in queries]))
        best = int(scores.max(axis=1).argmax()) if queries else 0
        representative = dict(members[best])
        matched = [f for query in queries for f in fact_index.matches(query, best)]
        matched = list({(f['text'], f['evidence']['start'], f['evidence']['end']): f for f in matched}.values())[:3]
        representative['matched_facts'] = matched
        representative['matched_fragments'] = fact_fragments(representative, matched) + chunks.fragments(representative['id'], queries)
        rows.append(dict(representative, event_id=int(event['event']),
                         event_news_ids=[r['id'] for r in members],
                         sources=list(dict.fromkeys(r['source'] for r in members if r.get('source'))),
                         urls=list(dict.fromkeys(r['url'] for r in members if r.get('url'))),
                         n_sources=int(event['n_sources']), generators=event['generators'],
                         best_rank=int(event['best_rank'])))
    result = pd.DataFrame(rows) if rows else news.head(0).copy()
    result.attrs['corpus_fingerprint'] = corpus_fingerprint(news)
    return result


def candidates(company_id: str, role: str, top: int = 30, *, facts: list[dict], method: str = 'v2'):
    if top < 1:
        raise ValueError('Candidate count must be positive')
    if method == 'lexical':
        return lexical_candidates(company_id, role, top=top)
    if method in ('v2', 'v3'):
        news = news_pool()
        events, generators = (v2_candidates if method == 'v2' else v3_candidates)(company_id, facts, role, news=news)
        events = events.head(top)  # the pool is ordered by priority: its head mixes the best events of every generator
        events.attrs['corpus_fingerprint'] = corpus_fingerprint(news)
        return event_news(events, news), generators
    if method != 'hybrid':
        raise ValueError(f'Unknown retrieval method: {method}')
    # Keep the CLI budget in events; rounding individual generator quotas can make their union larger.
    events, generators = hybrid_candidates(facts, role=role, n=top, l=top)
    return event_news(events.head(top), news_pool()), generators
