"""v2 vs v3 on the judge-v2 labels (positive = insight potential, significance and mechanism >= 2), no LLM calls (factory
specs must be cached). v3 heads are shown for several channel shares of the ordering weight; «only channels» = events of
the v3 pool no v2 generator found. Labels' event ids belong to the events.parquet of the 20.10 snapshot (--events).
python -X utf8 stage3_news_search/retrieval_v3/eval_labels.py [--labels experiments/judge_v2/out] [--k 30 100 200]"""
import argparse, json, sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.data import DATA
from stage2_company_facts.deterministic_facts.facts import build
from stage3_news_search.retrieval_v2 import candidates as v2
from stage3_news_search.retrieval_v2.eval_labels import COMPANIES, USEFUL
from stage3_news_search.retrieval_v3 import candidates as v3

SHARE_GRID = [0.0, 0.1, 0.2, 0.3]


def labels(folder: Path, u: str) -> pd.DataFrame:
    L = pd.DataFrame(json.loads(l)['value'] for l in open(folder / f'labels_v2_{u}.jsonl', encoding='utf-8'))
    L = L[(L.input == 'gemma_summary') & (L.refusal == 'short')].drop_duplicates('event', keep='last').set_index('event')
    L['pos'] = (L.insight_potential >= 2) & (L.significance >= 2) & (L.mechanism_strength >= 2)
    return L


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--companies', nargs='+', default=COMPANIES)
    ap.add_argument('--k', nargs='+', type=int, default=[30, 100, 200])
    ap.add_argument('--labels', type=Path, default=ROOT / 'experiments/judge_v2/out')
    ap.add_argument('--events', type=Path, default=DATA / 'news' / 'events.parquet', help='events the labels refer to')
    a = ap.parse_args()
    events = pd.read_parquet(a.events)
    useful = {x['event'] for x in json.loads(USEFUL.read_text(encoding='utf-8'))['items']} if USEFUL.exists() else set()
    sets, only = {}, {}
    for u in a.companies:
        f = build([u])[u]
        r2, _ = v2.candidates(u, f, events=events, frozen_factory=True)
        r3, gens = v3.candidates(u, f, events=events, frozen_factory=True)
        sets[(u, 'v2', 'pool')] = list(r2.event)
        sets[(u, 'v3', 'pool')] = list(r3.event)
        only[u] = [e for e, xs in zip(r3.event, r3.generators) if all(x['source'] == 'channel' for x in xs)]
        picked = dict(zip(r3.event, r3.generators))
        rows3 = r3.drop(columns=['generators', 'best_rank', 'priority']).set_index('event')
        for k in a.k:
            sets[(u, 'v2', f'@{k}')] = list(r2.event.head(k))
            for s in SHARE_GRID:
                shares = {src: w * (1 - s) for src, w in v2.SHARES.items()} | {'channel': s}
                sets[(u, f'v3 share {s}', f'@{k}')] = list(v2.order(rows3, picked, gens, shares).event.head(k))
    rows = []
    for (u, name, k), s in sets.items():
        L = labels(a.labels, u)
        lab = [e for e in s if e in L.index]
        rows.append({'company': u, 'set': name, 'k': k, 'events': len(s), 'positives': f'{int(L.loc[lab].pos.sum())}/{int(L.pos.sum())}',
                     'pos/labelled': round(L.loc[lab].pos.mean(), 2) if lab else None, 'unlabelled': len(s) - len(lab),
                     'manual useful': f'{len(useful & set(s))}/{len(useful)}' if u == 'U_106783' else ''})
    for u, s in only.items():
        L = labels(a.labels, u)
        lab = [e for e in s if e in L.index]
        rows.append({'company': u, 'set': 'only channels', 'k': 'pool', 'events': len(s), 'positives': f'{int(L.loc[lab].pos.sum())}/{int(L.pos.sum())}',
                     'pos/labelled': round(L.loc[lab].pos.mean(), 2) if lab else None, 'unlabelled': len(s) - len(lab), 'manual useful': ''})
    R = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print(R.to_string(index=False))
    out = DATA / 'retrieval_v2' / 'eval_labels_v3.csv'
    R.to_csv(out, index=False)
    print(out)
