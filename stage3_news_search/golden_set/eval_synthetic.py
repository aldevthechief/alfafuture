"""Synthetic recall benchmark for candidate selection (no golden labels exist for real news).

Method ("planted needles"): for each golden-set company write news that SHOULD reach it — `direct` (its own industry) and
`chain` (only through a fact from the tables: supplier/customer industry, group, region, country, bank product) — plus
`hard negatives` that share surface words but carry no business consequence. Plant them into the real corpus and
measure the rank each gets for the target company. Stage A must keep needles inside the budget K that stage B/C will read.
Bias: needles below are hand-written by someone who saw the facets; v2 should generate them with an LLM from
stage2_company_facts/golden_set/company_facts_golden.json hypotheses only (blind to facets) — see stage3_news_search/README.md.
python -X utf8 stage3_news_search/golden_set/eval_synthetic.py [--k 50 100 200]
"""
import argparse, json, sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stage3_news_search.facet_retrieval.candidates import candidates

GOLD = json.loads(Path(__file__).with_name('synthetic_needles.json').read_text(encoding='utf-8'))
NEEDLES = {u: [(n['kind'], n['title'], n['text']) for n in v] for u, v in GOLD['needles'].items()}
HARD_NEG = [(n['title'], n['text']) for n in GOLD['hard_negatives']]


def corpus():
    rows = []
    for u, ns in NEEDLES.items():
        rows += [{'title': t, 'text': x, 'source': 'synthetic', 'tier': 1, 'date': date(2026, 10, 20), 'needle_for': u, 'kind': k}
                 for k, t, x in ns]
    rows += [{'title': t, 'text': x, 'source': 'synthetic', 'tier': 1, 'date': date(2026, 10, 20), 'needle_for': 'NEG', 'kind': 'neg'}
             for t, x in HARD_NEG]
    return pd.DataFrame(rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, nargs='+', default=[50, 100, 200])
    a = ap.parse_args()
    extra, rows = corpus(), []
    for u in NEEDLES:
        res, _ = candidates(u, top=max(a.k), extra=extra)
        res = res.reset_index(drop=True)
        for kind in ('direct', 'chain', 'neg'):
            mine = res[(res.get('needle_for') == u) & (res.get('kind') == kind)] if kind != 'neg' else res[res.get('kind') == 'neg']
            rank = int(mine.index.min()) + 1 if len(mine) else None
            rows.append({'company': u, 'kind': kind, 'rank': rank})
    r = pd.DataFrame(rows)
    print(r.pivot(index='company', columns='kind', values='rank').to_string())
    for kind in ('direct', 'chain'):
        x = r[r.kind == kind]['rank']
        print(kind, ' '.join(f'recall@{k}={(x.notna() & (x <= k)).mean():.0%}' for k in a.k))
    neg = r[r.kind == 'neg']['rank']
    print('hard negatives inside top-50 for', int((neg.notna() & (neg <= 50)).sum()), 'of', len(NEEDLES), 'companies')
