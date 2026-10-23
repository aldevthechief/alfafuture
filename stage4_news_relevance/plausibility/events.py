"""News -> events (clusters of reports about the same thing) + plausibility score.
Deterministic; an LLM contradiction check is proposed in stage4_news_relevance/README.md.
python -X utf8 stage4_news_relevance/plausibility/events.py  -> data/news/events.parquet"""
import re, sys, tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.data import DATA
from common.search import corpus_fingerprint, news_pool
from common.text import stems

OFFICIAL = {'cbr', 'consultant', 'garant'}  # regulator / legal texts in the corpus
NUM = re.compile(r'(\d+[,.]?\d*)\s?%')
EVENTS_VERSION = 1
EVENT_COLUMNS = ['event', 'title', 'first_date', 'n_reports', 'n_sources', 'sources', 'news_ids',
                 'plausibility', 'numbers_disagree']


def _components(X, d, sim_min, days):
    S = (X @ X.T).tocsr()
    S.data[S.data < sim_min] = 0
    S.eliminate_zeros()
    rows, cols = S.nonzero()
    ok = np.abs((d[rows] - d[cols]).astype('timedelta64[D]').astype(int)) <= days
    S = S.__class__((np.ones(ok.sum()), (rows[ok], cols[ok])), shape=S.shape)
    return connected_components(S, directed=False)[1]


def cluster(df, sim_min=0.45, days=3, max_size=10, step=0.1, sim_cap=0.85):
    """Same-event clusters: cosine of title+lead stems >= sim_min and dates within `days` (connected components).
    Components chain (A~B, B~C: «all airport closures of a week» became one event of 51 reports), so a component
    bigger than max_size is re-clustered inside with a threshold raised by `step`, until it splits or reaches sim_cap."""
    if df.empty:
        return np.array([], dtype=int)
    vec = TfidfVectorizer(analyzer=stems, min_df=1, sublinear_tf=True)
    X = vec.fit_transform((df.title.fillna('') + ' ' + df.text.fillna('').str[:400]).str.lower())
    d = pd.to_datetime(df.date).to_numpy()
    labels, stack = np.full(len(df), -1), [(np.arange(len(df)), sim_min)]
    while stack:
        idx, s = stack.pop()
        comp = _components(X[idx], d[idx], s, days)
        for c in np.unique(comp):
            sub = idx[comp == c]
            if len(sub) > max_size and s + step <= sim_cap + 1e-9:
                stack.append((sub, s + step))
            else:
                labels[sub] = labels.max() + 1
    return labels


def plausibility(g):
    """0..1 heuristic: corroboration by independent outlets, source tier, an official source,
    and a numeric-consistency check (same event reported with very different percentages)."""
    srcs = g.source.nunique()
    nums = [float(x.replace(',', '.')) for x in NUM.findall(' '.join(g.title.fillna('')))]
    spread = (max(nums) / max(min(nums), 1e-9)) if len(nums) >= 2 and min(nums) > 0 else 1
    s = 0.35 + 0.1 * min(srcs - 1, 3) + 0.15 * (g.tier == 1).mean() + 0.15 * bool(set(g.source) & OFFICIAL)
    s -= 0.15 * (spread > 1.5)  # e.g. "+12%" vs "+30%" for the same event: flag for an LLM contradiction check
    return round(float(np.clip(s, 0, 1)), 2), spread > 1.5


def build_events(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['event'] = cluster(df)
    ev = []
    for e, g in df.groupby('event'):
        p, contra = plausibility(g)
        g = g.sort_values(['tier', 'date'])
        ev.append({'event': e, 'title': g.title.iloc[0], 'first_date': g.date.min(), 'n_reports': len(g), 'n_sources': g.source.nunique(),
                   'sources': sorted(g.source.unique()), 'news_ids': g['id'].tolist(), 'plausibility': p, 'numbers_disagree': contra})
    return pd.DataFrame(ev, columns=EVENT_COLUMNS)


def ensure_events(df: pd.DataFrame | None = None, path: Path | None = None, *, force: bool = False) -> pd.DataFrame:
    """Build on first use; rebuild when any corpus input changes. Metadata travels atomically with the parquet."""
    df = news_pool() if df is None else df
    path = Path(path) if path is not None else DATA / 'news' / 'events.parquet'
    fingerprint = corpus_fingerprint(df)
    if path.exists() and not force:
        try:
            cached = pd.read_parquet(path)
            if (cached.attrs.get('corpus_fingerprint') == fingerprint
                    and cached.attrs.get('events_version') == EVENTS_VERSION
                    and set(EVENT_COLUMNS) <= set(cached.columns)):
                return cached
        except (OSError, ValueError):
            pass  # a partial/legacy artifact is rebuilt from the source corpus
    ev = build_events(df)
    ev.attrs.update(corpus_fingerprint=fingerprint, events_version=EVENTS_VERSION)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='events-', suffix='.parquet', delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        ev.to_parquet(tmp_path, index=False)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return ev


if __name__ == '__main__':
    df = news_pool()
    ev = ensure_events(df, force=True)
    multi = ev[ev.n_reports > 1]
    print(f'{len(df)} news -> {len(ev)} events; multi-report events {len(multi)} covering {multi.n_reports.sum()} news; '
          f'numbers disagree in {int(ev.numbers_disagree.sum())}')
    print(multi.sort_values('n_sources', ascending=False).head(12)[['n_reports', 'n_sources', 'plausibility', 'title']].to_string(index=False))
    assert ev.news_ids.map(len).sum() == len(df)
