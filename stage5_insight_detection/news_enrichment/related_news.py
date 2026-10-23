"""Step 5a: go back to the news pool for reports that complement a news item.

- same event: other reports of the same event (clusters from stage4_news_relevance/plausibility/events.py);
- follow-ups: later news on the same topic that are not the same event (developments, official confirmation);
- free search: any query over the pool (the insight judge uses it to check a hypothesis).
Deterministic, runs now: python -X utf8 stage5_insight_detection/news_enrichment/related_news.py <news_id>
"""
import argparse, os, sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.data import DATA
from common.search import news_index
from stage4_news_relevance.plausibility.events import ensure_events

COLUMNS = ['id', 'date', 'source', 'tier', 'title']


class RelatedNews:
    def __init__(self, events_path: Path = DATA / 'news' / 'events.parquet', *, dense: bool | None = None,
                 rerank: bool | None = None, ranker=None, fact_store=None, candidate_limit: int = 40):
        self.dense = os.environ.get('NEWS_SEARCH_DENSE', '1') != '0' if dense is None else dense
        self.rerank = os.environ.get('NEWS_SEARCH_RERANK', '1') != '0' if rerank is None else rerank
        self.ranker, self.fact_store, self.candidate_limit = ranker, fact_store, candidate_limit
        self.events_path, self.news = events_path, None
        self.refresh()

    def refresh(self) -> None:
        """Refresh between batches when the corpus file has changed; keep news and event membership in sync."""
        news, index = news_index()
        if news is self.news:
            return
        events = ensure_events(news, self.events_path)
        self.news, self.index = news, index
        self.row_of = {nid: i for i, nid in enumerate(self.news['id'])}
        self.members = dict(zip(events.event, events.news_ids.map(list)))
        self.event_of = {nid: e for e, ids in self.members.items() for nid in ids}
        self.plausibility = dict(zip(events.event, events.plausibility))

    def get(self, news_id: int) -> dict:
        return self.news.iloc[self.row_of[news_id]].to_dict()

    def same_event(self, news_id: int) -> pd.DataFrame:
        ids = [i for i in self.members.get(self.event_of.get(news_id), []) if i != news_id]
        return self.news[self.news['id'].isin(ids)][COLUMNS]

    def follow_ups(self, news_id: int, k: int = 5, min_sim: float = 0.25) -> pd.DataFrame:
        me = self.news.iloc[self.row_of[news_id]]
        sims = (self.index.matrix @ self.index.matrix[self.row_of[news_id]].T).toarray().ravel()
        same = set(self.members.get(self.event_of.get(news_id), [news_id]))
        df = self.news.assign(similarity=sims)
        df = df[(df.similarity >= min_sim) & (df.date >= me.date) & ~df['id'].isin(same)]
        return df.nlargest(k, 'similarity')[COLUMNS + ['similarity']]

    def search(self, query: str, k: int = 8) -> pd.DataFrame:
        from common.news_retrieval import hybrid_search
        return hybrid_search(self.news, self.index, query, k, dense=self.dense, rerank=self.rerank,
                             ranker=self.ranker, fact_store=self.fact_store, candidate_limit=self.candidate_limit)

    def event_plausibility(self, news_id: int) -> float | None:
        return self.plausibility.get(self.event_of.get(news_id))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('news_id', type=int)
    a = ap.parse_args()
    r = RelatedNews()
    n = r.get(a.news_id)
    print(f'{n["date"]} {n["source"]}: {n["title"]}\nplausibility of its event: {r.event_plausibility(a.news_id)}')
    print('same event:\n', r.same_event(a.news_id).to_string(index=False))
    print('follow-ups:\n', r.follow_ups(a.news_id).to_string(index=False))
