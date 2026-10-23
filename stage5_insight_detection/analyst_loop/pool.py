"""News pool behind one small interface, so the loop works on the real pool (RelatedNews over news.parquet and the
stage-4 events) and on an in-memory list (tests, live scraping before the parquet is rebuilt)."""
from typing import Protocol

from common.dedup import dedup_news
from common.search import StemIndex


class NewsPool(Protocol):
    def get(self, news_id) -> dict: ...
    def same_event(self, news_id) -> list[dict]: ...
    def follow_ups(self, news_id) -> list[dict]: ...
    def search(self, query: str, k: int = 8) -> list[dict]: ...
    def plausibility(self, news_id) -> float | None: ...


class EmptyNewsPool:
    """A bank-only run needs no news.parquet or news index."""

    def get(self, news_id):
        raise KeyError(news_id)

    def same_event(self, news_id):
        return []

    def follow_ups(self, news_id):
        return []

    def search(self, query, k=8):
        return []

    def plausibility(self, news_id):
        return None


class ListNewsPool:
    """Rows {'id', 'title', 'text', 'date', 'source', 'url'}; events are the reprint groups of common/dedup.py."""

    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.by_id = {str(r['id']): r for r in rows}
        self.index = StemIndex([f'{r["title"]}. {r["title"]}. {str(r.get("text", ""))[:1500]}' for r in rows], min_df=1, max_df=1.0)
        self.event_of = {}
        for story in dedup_news(rows, threshold=0.45, window_hours=72):  # same rule as stage 4 events
            for i in story['duplicate_ids']:
                self.event_of[str(i)] = [str(x) for x in story['duplicate_ids']]

    def get(self, news_id) -> dict:
        return self.by_id[str(news_id)]

    def same_event(self, news_id) -> list[dict]:
        return [self.by_id[i] for i in self.event_of.get(str(news_id), []) if i != str(news_id)]

    def follow_ups(self, news_id, k: int = 5, min_sim: float = 0.2) -> list[dict]:
        me = self.get(news_id)
        same = set(self.event_of.get(str(news_id), [str(news_id)]))
        hits = [(self.rows[i], s) for i, s in self.index.search(me['title'] + '. ' + str(me.get('text', ''))[:600], 20)]
        return [r for r, s in hits if s >= min_sim and str(r['id']) not in same and str(r['date']) >= str(me['date'])][:k]

    def search(self, query: str, k: int = 8) -> list[dict]:
        return [self.rows[i] for i, _ in self.index.search(query, k)]

    def plausibility(self, news_id) -> float | None:
        n = len(self.event_of.get(str(news_id), [str(news_id)]))
        return min(1.0, 0.4 + 0.2 * n)  # shortcut for lists: more outlets -> more plausible


class RelatedNewsPool:
    """Adapter over stage5 news_enrichment.RelatedNews (DataFrames -> dicts with text)."""

    def __init__(self, related=None):
        if related is None:
            from stage5_insight_detection.news_enrichment.related_news import RelatedNews
            related = RelatedNews()
        self.related = related

    def _rows(self, df) -> list[dict]:
        return [self.get(i) for i in df['id']] if len(df) else []

    def get(self, news_id) -> dict:
        return self.related.get(int(news_id))

    def same_event(self, news_id) -> list[dict]:
        return self._rows(self.related.same_event(int(news_id)))

    def follow_ups(self, news_id) -> list[dict]:
        return self._rows(self.related.follow_ups(int(news_id)))

    def search(self, query: str, k: int = 8) -> list[dict]:
        return self.related.search(query, k).to_dict('records')

    def plausibility(self, news_id) -> float | None:
        return self.related.event_plausibility(int(news_id))
