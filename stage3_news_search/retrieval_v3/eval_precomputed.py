"""v3 on precomputed scores (precompute.py), CPU only: the channel share of the ordering weight chosen on the tuning
companies and reported on the held-out ones (judge-v2 labels), overlap of the channel-only pools between companies.
python -X utf8 stage3_news_search/retrieval_v3/eval_precomputed.py [--tune U_106783 U_9250] [--holdout U_150736 U_212593] [--new U_57706 …]"""
import argparse, itertools, sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.data import DATA
from stage2_company_facts.deterministic_facts.facts import build
from stage3_news_search.retrieval_v2 import candidates as v2
from stage3_news_search.retrieval_v3 import channels, precompute
from stage3_news_search.retrieval_v3 import candidates as v3
from stage3_news_search.retrieval_v3.eval_labels import labels

GATES = ['-']  # an affinity gate on the facets was tried (0.5–0.9): no effect on the pools, removed
SHARE_GRID = [0.0, 0.1, 0.2, 0.3]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tune', nargs='+', default=['U_106783', 'U_9250'])
    ap.add_argument('--holdout', nargs='+', default=['U_150736', 'U_212593'])
    ap.add_argument('--new', nargs='+', default=['U_57706', 'U_3263', 'U_86035', 'U_63139', 'U_44792', 'U_14366'])
    ap.add_argument('--labels', type=Path, default=ROOT.parent / 'alfafuture-hack/experiments/judge_v2/out')
    a = ap.parse_args()
    news, events = pd.read_parquet(DATA / 'news' / 'news.parquet'), pd.read_parquet(DATA / 'news' / 'events.parquet')
    search = precompute.load(news, events)
    v2.hybrid_search = lambda *_: search
    cos = a.tune + a.holdout + a.new
    facts = build(cos)
    base = {u: v2.candidates(u, facts[u], events=events, news=news, use_factory=u in a.tune + a.holdout, frozen_factory=True)
            for u in cos}
    rows, jac = [], {}
    for gate in GATES:
        only = {}
        for u in cos:
            res, gens = base[u]
            picked = dict(zip(res.event, res.generators))
            cg = channels.generators(facts[u], search)
            for g, k in zip(cg, v2.quotas([g['weight'] for g in cg], [g['top'] for g in cg], v3.CHANNEL_BUDGET)):
                for r, e in enumerate(g['top'][:k]):
                    picked.setdefault(e, []).append({'source': 'channel', 'reason': g['reason'], 'channel': g['channel'],
                                                     'query': g['query'], 'rank': r})
            rows3 = search.events.loc[list(picked)]
            only[u] = {e for e in picked if e not in set(res.event)}
            if u in a.new:
                continue
            L = labels(a.labels, u)
            for s in SHARE_GRID:
                sh = {k: w * (1 - s) for k, w in v2.SHARES.items()} | {'channel': s}
                order = list(v2.order(search.events, picked, gens + cg, sh).event)
                row = {'gate': gate, 'share': s, 'company': u, 'split': 'tune' if u in a.tune else 'holdout', 'pool': len(order),
                       'only_ch': len(only[u]), 'known': int(L.pos.sum())}
                for k in (30, 100, 200, len(order)):
                    row[f'pos@{k if k < 1000 and k != len(order) else "pool"}'] = int(L.loc[[e for e in order[:k] if e in L.index]].pos.sum())
                lab = [e for e in only[u] if e in L.index]
                row |= {'only_ch_lab': len(lab), 'only_ch_pos': int(L.loc[lab].pos.sum())}
                rows.append(row)
        pairs = {n: [len(only[x] & only[y]) / max(1, len(only[x] | only[y])) for x, y in itertools.combinations(g, 2)]
                 for n, g in (('all', cos), ('new', a.new))}
        jac[gate] = {n: sum(v) / len(v) for n, v in pairs.items()} | {'only_ch_per_company': sum(len(v) for v in only.values()) / len(only)}
    print(pd.DataFrame(jac).T.round(2).to_string(), '\n')
    R = pd.DataFrame(rows)
    cols = ['pos@30', 'pos@100', 'pos@200', 'pos@pool', 'known', 'only_ch', 'only_ch_lab', 'only_ch_pos']
    print(R.groupby(['split', 'gate', 'share'])[cols].sum().to_string())
    R.to_csv(DATA / 'retrieval_v2' / 'eval_gate.csv', index=False)
