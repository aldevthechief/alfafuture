"""Profile enrichment from the news base: for each strong external factor of a company, 2-3 recent news that show the
factor is live right now (with date, source and url), so stages 5-6 and the UI can say «why it matters» with evidence.

- query = the factor's keywords; a news is accepted only if one keyword phrase matches whole (all its stems are in the
  news), not a single loose word;
- score = TF-IDF cosine x recency (half-life `half_life` days from the newest news of the pool);
- one news per title start (same event in several outlets counts once; the number of outlets is kept as `sources`);
- base factors (rate / inflation / taxes for everybody) are skipped: they say nothing company-specific.
The pool is any DataFrame with id, title, text, date, source, url (default: common.search.news_pool, stage 1 output),
so the corpus of stage 1 can be swapped without changes here. No LLM.
"""
import numpy as np
import pandas as pd

from common.search import StemIndex, news_document
from common.text import stems


class NewsContext:
    def __init__(self, pool: pd.DataFrame | None = None, half_life: float = 14.0):
        if pool is None:
            from common.search import news_pool
            pool = news_pool()
        self.pool = pool.reset_index(drop=True)
        docs = news_document(self.pool)
        self.index = StemIndex(docs.tolist())
        self.stems = [set(stems(d)) for d in docs]
        lead = self.pool.title.fillna('') + '. ' + self.pool.get('summary', pd.Series('', index=self.pool.index)).fillna('') \
            + '. ' + self.pool.text.fillna('').str[:300]
        self.lead = [set(stems(d)) for d in lead]
        dates = pd.to_datetime(self.pool.date)
        age = (dates.max() - dates).dt.days.to_numpy()
        self.recency = 0.5 ** (age / half_life)
        self.title_key = self.pool.title.fillna('').str.lower().str.replace(r'[^а-яёa-z0-9 ]', '', regex=True) \
            .str.split().str[:6].str.join(' ').to_numpy()

    def evidence(self, keywords: list[str], k: int = 3, candidates: int = 60, shared: frozenset = frozenset()) -> list[dict]:
        """Accept a news if a specific (not `shared`) phrase is in its lead (title, summary, first 300 chars) and the
        match is not one loose word: a multi-stem phrase, or two phrases."""
        kw = [w for w in keywords if stems(w)]
        own = [w for w in kw if w not in shared]
        if not own:
            return []
        s = self.index.scores(' '.join(kw))
        out, seen = [], {}
        for i in np.argsort(-s)[:candidates]:
            if s[i] <= 0:
                break
            hit = [w for w in kw if set(stems(w)) <= self.stems[i]]
            lead = [w for w in own if set(stems(w)) <= self.lead[i]]
            if not lead or not (len(hit) >= 2 or any(len(set(stems(w))) >= 2 for w in lead)):
                continue
            key = self.title_key[i]
            if key in seen:
                seen[key]['sources'] += 1
                continue
            r = self.pool.iloc[i]
            item = {'news_id': int(r.id), 'title': r.title, 'date': str(r.date), 'source': r.source, 'url': r.url,
                    'score': round(float(s[i] * self.recency[i]), 4), 'matched': hit, 'sources': 1}
            seen[key] = item
            out.append(item)
        return sorted(out, key=lambda x: -x['score'])[:k]

    def for_profile(self, profile: dict, n_factors: int = 8, k: int = 3) -> list[dict]:
        """Factors in weight order, skipping base and region ones; keywords used by 3+ factors are `shared`."""
        fs = [f for f in profile['external_factors'] if f['rule'] not in ('base', 'geo:region')]
        count = {}
        for f in fs:
            for w in set(f['keywords']):
                count[w] = count.get(w, 0) + 1
        shared = frozenset(w for w, n in count.items() if n >= 3)
        out = []
        for f in fs:
            news = self.evidence(f['keywords'], k, shared=shared)
            out.append({'factor': f['id'], 'label': f['label'], 'weight': f['weight'], 'news': news})
            if len(out) >= n_factors:
                break
        return out
