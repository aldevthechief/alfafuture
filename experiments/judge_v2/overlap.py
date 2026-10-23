"""Who finds what: per-generator quotas of the current and the new (3 facets RRF) sets, attribution of positives,
overlap between generators (inside quotas and in their deeper top-50) and between the two sets. Judge-v2 labels:
gemma_summary verdict per event. Inputs from retrieval/new_generators.py (out/generator_tops.json).
python -X utf8 experiments/judge_v2/overlap.py"""
import itertools, json

import numpy as np
import pandas as pd

from judge import OUT, positive

T = json.loads((OUT / 'generator_tops.json').read_text(encoding='utf-8'))
L = pd.DataFrame(json.loads(l)['value'] for l in open(OUT / 'labels_v2_U_106783.jsonl', encoding='utf-8'))
L = L[(L.input == 'gemma_summary') & (L.refusal == 'short')].drop_duplicates('event', keep='last').set_index('event')
L['pos'] = L.apply(positive, axis=1)
L['rel'] = L.mechanism_strength >= 2


def quotas(gens, n):
    """Same rule as new_generators.candidate_set: smallest budget whose weighted quota union has >= n events."""
    w = np.array([T['weights'][g.split('|')[0]] if g.startswith('cur:') else 1.0 for g in gens])
    for b in range(n, 20 * n, 5):
        q = {g: T['tops'][g][:h] for g, h in zip(gens, np.maximum(1, np.round(b * w / w.sum())).astype(int))}
        if len(set().union(*q.values())) >= n:
            return q


def short(g):
    return g.split('|')[0].replace('cur:', 'C:').replace('new:', 'N:')


if __name__ == '__main__':
    cur = [g for g in T['tops'] if g.startswith('cur:')]
    new = [g for g in T['tops'] if g.startswith('new:') and g.endswith('|3')]
    for n in (100, 200):
        Q = {'current': quotas(cur, n), 'new': quotas(new, n)}
        S = {k: set().union(*q.values()) for k, q in Q.items()}
        print(f'\n===== budget {n}: current {len(S["current"])} events, new {len(S["new"])}, shared {len(S["current"] & S["new"])}')
        for k, q in Q.items():
            owners = pd.Series([g for g, es in q.items() for e in es]).value_counts()
            by = {}
            for g, es in q.items():
                for e in es:
                    by.setdefault(e, []).append(g)
            hist = pd.Series([len(v) for v in by.values()]).value_counts().sort_index().to_dict()
            print(f'  {k}: events found by 1/2/3+ generators: {hist}')
            rows = []
            for g, es in q.items():
                lab = [e for e in es if e in L.index]
                rows.append({'generator': short(g), 'quota': len(es), 'unique': sum(len(by[e]) == 1 for e in es),
                             'labelled': len(lab), 'positives': int(L.loc[lab].pos.sum()), 'relevant': int(L.loc[lab].rel.sum()),
                             'unique positives': sum(len(by[e]) == 1 and e in L.index and L.pos[e] for e in es)})
            print(pd.DataFrame(rows).sort_values('positives', ascending=False).to_string(index=False))
    # overlap of the retrieval itself: Jaccard of the generators' top-50 events
    print('\n===== top-50 overlap (Jaccard) between generators, pairs with > 0.1')
    gs = cur + new
    top = {g: set(T['tops'][g][:50]) for g in gs}
    pairs = sorted(((len(top[a] & top[b]) / len(top[a] | top[b]), a, b) for a, b in itertools.combinations(gs, 2)), reverse=True)
    for j, a, b in pairs:
        if j > 0.1:
            print(f'  {j:.2f}  {short(a)} ~ {short(b)}')
    print(f'  pairs with Jaccard 0: {sum(j == 0 for j, _, _ in pairs)} of {len(pairs)}')
    # new-only useful events at 200 with their generators and ranks
    q = quotas(new, 200); cs = set().union(*quotas(cur, 200).values())
    print('\n===== positives / relevant found only by the new set at budget 200 (generator: rank)')
    for e in sorted(set().union(*q.values()) - cs):
        if e in L.index and (L.pos[e] or L.rel[e]):
            who = ', '.join(f'{short(g)}:{T["tops"][g].index(e)}' for g, es in q.items() if e in es)
            r = L.loc[e]
            print(f"  {'POS' if r.pos else 'rel'} {e:5} s{r.significance} m{r.mechanism_strength} p{r.insight_potential} | {who:40} | {r.title[:80]}")
