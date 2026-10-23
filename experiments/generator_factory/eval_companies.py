"""Factory v2-B on new companies, judged by judge v2 (gemma_summary input) on a stratified sample of three strata:
events of the selected generators (top-13 each), events only dropped probes found, random events outside both.
Per stratum: labelled, positives (potential ≥ 2 & significance ≥ 2 & mechanism ≥ 2), relevant (mechanism ≥ 2);
recall of the selected set over the whole corpus is estimated with the random stratum's positive rate.
python -X utf8 experiments/generator_factory/eval_companies.py U_150736 U_212593 U_9250"""
import json, sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
OUT, JV2 = HERE / 'out', HERE.parents[1] / 'experiments/judge_v2/out'
N_EVENTS = len(pd.read_parquet(HERE.parents[1] / 'data/news/events.parquet'))

for u in sys.argv[1:]:
    S = json.loads((OUT / f'judge_sets_{u}.json').read_text())
    f = JV2 / f'labels_v2_{u}.jsonl'
    L = pd.DataFrame(json.loads(l)['value'] for l in open(f, encoding='utf-8'))
    L = L[(L.input == 'gemma_summary') & (L.refusal == 'short')].drop_duplicates('event', keep='last').set_index('event')
    L['pos'] = (L.insight_potential >= 2) & (L.significance >= 2) & (L.mechanism_strength >= 2)
    L['rel'] = L.mechanism_strength >= 2
    print(f'\n===== {u}: {len(L)} labelled')
    est = {}
    for k in ('selected', 'dropped_only', 'random'):
        lab = [e for e in S[k] if e in L.index]
        p, r = int(L.loc[lab].pos.sum()), int(L.loc[lab].rel.sum())
        rate = p / max(1, len(lab))
        est[k] = rate * (len(S[k]) if k != 'random' else N_EVENTS - len(S['selected']) - len(S['dropped_only']))
        print(f'  {k:13} {len(lab):3}/{len(S[k]):3} labelled  pos {p:2} rel {r:2}  pos rate {rate:.3f}  -> est. positives in stratum {est[k]:.0f}')
    total = sum(est.values())
    print(f'  est. recall of selected generators over the corpus: {est["selected"] / max(1e-9, total):.0%} '
          f'(selected + dropped: {(est["selected"] + est["dropped_only"]) / max(1e-9, total):.0%}); '
          f'random stratum has {int(L.loc[[e for e in S["random"] if e in L.index]].pos.sum())} positives — wide interval')
    run = json.loads((OUT / f'factory_{u}_B.json').read_text(encoding='utf-8'))
    tops = json.loads((OUT / f'factory_{u}_B_tops.json').read_text())
    rows = []
    for p in run['selected']:
        t = tops.get(f"sel:{p['probe_id']}", [])
        lab = [e for e in t if e in L.index]
        rows.append({'channel': p['channel'], 'reason': p['reason'][:24], 'labelled': len(lab),
                     'pos': int(L.loc[lab].pos.sum()), 'rel': int(L.loc[lab].rel.sum())})
    print(pd.DataFrame(rows).sort_values(['pos', 'rel'], ascending=False).to_string(index=False))
    pos = L[L.pos]
    print('  positives:')
    for e, r in pos.iterrows():
        k = next(k for k in S if e in S[k]) if any(e in S[k] for k in S) else '?'
        print(f'    [{k[:4]}] s{r.significance} m{r.mechanism_strength} p{r.insight_potential} {r.channel[:22]:22} | {r.title[:80]}')
