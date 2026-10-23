"""Versioned candidate-generator sets: which generator families feed the pool, each with its own limit; the pool is the
union of their events (an event keeps every source that found it, with the rank there).
- v1 = standard + factory: standard = stage-3 generators (stage3_news_search/candidate_generators.generators(facts):
  reasons and weights) ranked by the experiments/retrieval hybrid (e5-large-instruct + qwen3-emb-0.6b + BM25, RRF),
  quota union ∝ weights at the smallest budget giving >= 200 events; factory = the saved factory v2-B run of the
  company (experiments/generator_factory/out/factory_<u>_B_tops.json), top-13 of every selected generator.
- v2 = v1 + news-first channel generator (experiments/news_channels/channel_gen.py: channel × company exposure,
  alpha 1, ensemble of 5 rankers), quota union at 200.
Evaluation vs judge v2 labels: python -X utf8 experiments/candidate_sets/sets.py [--companies …]"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / 'experiments/news_channels', ROOT / 'experiments/generator_factory'):
    sys.path.insert(0, str(p))
import channel_gen as cg  # also puts retrieval / briefs / judge_v2 on the path
from stage3_news_search.candidate_generators.generators import generators

OUT = HERE / 'out'
GF = ROOT / 'experiments/generator_factory/out'
VERSIONS = {'v1': {'standard': 200, 'factory': 13}, 'v2': {'standard': 200, 'factory': 13, 'channels': 200}}


_HYBRID = []


def standard(u: str, budget: int) -> dict[int, int]:
    """event -> best rank within its generator; cached per company (the hybrid needs both encoders on the GPU)."""
    f = OUT / f'standard_{u}.json'
    if f.exists():
        tops = json.loads(f.read_text())
    else:
        if not _HYBRID:  # loaded once: both encoders on a 4 GB GPU
            from factory import Hybrid
            _HYBRID.append(Hybrid())
        gens = generators(cg.facts_for(u))
        tops = {f'{g.reason}|{g.query[:60]}': {'weight': g.weight, 'top': _HYBRID[0].events(cg.REASON_INSTRUCTIONS[g.reason], [g.query], 300)}
                for g in gens}
        OUT.mkdir(exist_ok=True)
        f.write_text(json.dumps(tops, ensure_ascii=False))
    w = np.array([t['weight'] for t in tops.values()])
    for b in range(budget, 20 * budget, 5):
        chosen = {}
        for t, h in zip(tops.values(), np.maximum(1, np.round(b * w / w.sum())).astype(int)):
            for r, e in enumerate(t['top'][:h]):
                chosen[e] = min(r, chosen.get(e, r))
        if len(chosen) >= budget:
            return chosen


def factory(u: str, quota: int) -> dict[int, int]:
    tops = json.loads((GF / f'factory_{u}_B_tops.json').read_text())
    chosen = {}
    for k, top in tops.items():
        if k.startswith('sel:'):
            for r, e in enumerate(top[:quota]):
                chosen[e] = min(r, chosen.get(e, r))
    return chosen


class Channels:
    """channel × exposure pools for several companies with one encoder load."""

    def __init__(self, companies):
        self.P = cg.event_pct()
        self.FS = {u: cg.facets(cg.facts_for(u)) for u in companies}
        qk = [(u, k) for u in companies for k, f in enumerate(self.FS[u]) if f['query']]
        A = cg.Affinity(self.P.index)([self.FS[u][k]['query'] for u, k in qk])
        A.columns = pd.MultiIndex.from_tuples(qk)
        self.A = A

    def __call__(self, u: str, budget: int) -> dict[int, int]:
        fs = self.FS[u]
        S = cg.generator_scores(fs, self.P, self.A[u], 1)
        pool = cg.quota_union(S, np.array([f['weight'] for f in fs]), budget)
        R = S.rank(ascending=False).loc[pool].min(axis=1)
        return {int(e): int(R[e]) - 1 for e in pool}


def pool(u: str, version: str, channels: Channels | None = None) -> pd.DataFrame:
    """event × source → rank (NaN = the source did not pick it)."""
    src = {'standard': lambda n: standard(u, n), 'factory': lambda n: factory(u, n), 'channels': lambda n: channels(u, n)}
    return pd.DataFrame({s: pd.Series(src[s](n)) for s, n in VERSIONS[version].items()})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--companies', nargs='+', default=cg.COMPANIES)
    a = ap.parse_args()
    ch = Channels(a.companies)
    useful = {x['event'] for x in json.loads((ROOT / 'experiments/judge_v2/useful_negatives.json').read_text(encoding='utf-8'))['items']}
    rows = []
    for u in a.companies:
        L = cg.labels(u)
        for v in VERSIONS:
            P = pool(u, v, ch)
            lab = P.index.intersection(L.index)
            only_new = P.index[P.drop(columns=['channels'], errors='ignore').isna().all(axis=1)]
            rows.append({'company': u, 'version': v, 'events': len(P), **{f'from_{s}': int(P[s].notna().sum()) for s in P},
                         'labelled': len(lab), 'positives': int(L.loc[lab].pos.sum()), 'of_known': f"{int(L.loc[lab].pos.sum())}/{int(L.pos.sum())}",
                         'pos_per_labelled': round(L.loc[lab].pos.mean(), 3), 'only_channels': len(only_new),
                         'only_channels_labelled': len(only_new.intersection(L.index)),
                         'only_channels_pos': int(L.loc[only_new.intersection(L.index)].pos.sum()),
                         'manual_useful': f'{len(useful & set(P.index))}/{len(useful)}' if u == 'U_106783' else ''})
    R = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    R.to_csv(OUT / 'versions_eval.csv', index=False)
    pd.set_option('display.width', 250)
    print(R.to_string(index=False))
