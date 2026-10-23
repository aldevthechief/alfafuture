"""Generator sets compared on judge-v2 labels (gemma_summary input, one verdict per event for every event, old and new):
how many positives / relevant events each candidate set holds. Sets: out/generator_sets.json (retrieval/new_generators.py).
Pool = every labelled event; recall is relative to the positives found by any set or by the random sample.
python -X utf8 experiments/judge_v2/compare_sets.py"""
import json
from pathlib import Path

import pandas as pd

from judge import OUT, positive

if __name__ == '__main__':
    L = pd.DataFrame(json.loads(l)['value'] for l in open(OUT / 'labels_v2_U_106783.jsonl', encoding='utf-8'))
    L = L[(L.input == 'gemma_summary') & (L.refusal == 'short')].drop_duplicates('event', keep='last').set_index('event')
    L['pos'] = L.apply(positive, axis=1)
    L['rel'] = L.mechanism_strength >= 2  # relevant to this company, regardless of significance / insight
    sets = json.loads((OUT / 'generator_sets.json').read_text(encoding='utf-8'))
    useful = {u['event'] for u in json.loads((Path(__file__).parent / 'useful_negatives.json').read_text(encoding='utf-8'))['items']}
    P, R = set(L.index[L.pos]), set(L.index[L.rel])
    print(f'labelled events {len(L)}; positives {len(P)}; relevant (mechanism ≥ 2) {len(R)}; manual useful {len(useful)}')
    rows = []
    for k, ev in sets.items():
        name, n = k.rsplit('|', 1)
        ev = set(ev); lab = ev & set(L.index)
        rows.append({'set': name, 'budget': int(n), 'labelled': f'{len(lab)}/{len(ev)}',
                     'positives': f'{len(ev & P)}/{len(P)}', 'relevant': f'{len(ev & R)}/{len(R)}',
                     'useful (old manual)': f'{len(ev & useful)}/{len(useful)}',
                     'pos / labelled': round(len(ev & P) / max(1, len(lab)), 2), 'rel / labelled': round(len(ev & R) / max(1, len(lab)), 2)})
    print(pd.DataFrame(rows).to_string(index=False))
    cur, new = set(sets['current (orig. weights)|200']), set(sets['new, 1st facet only|200']) | set(sets['new, 3 facets RRF|200'])
    for name, s in (('only new sets (budget 200)', new - cur), ('only current (budget 200)', cur - new)):
        x = L.loc[sorted(s & set(L.index))]
        print(f'\n{name}: {len(s)} events, labelled {len(x)}, positives {int(x.pos.sum())}, relevant {int(x.rel.sum())}')
        for e, r in x[x.pos | x.rel].sort_values('insight_potential', ascending=False).iterrows():
            print(f"  {e:5} s{r.significance} m{r.mechanism_strength} p{r.insight_potential} {r.channel[:20]:20} | {r.title[:95]}")
