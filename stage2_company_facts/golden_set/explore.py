"""Exploratory dump for the golden-set companies (the analyst's raw view, not a product): python -X utf8 stage2_company_facts/golden_set/explore.py"""
import json, sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.data import load, money

pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60); pd.set_option('display.max_columns', 20)
U, P, S, F = load()
ids = json.loads(Path(__file__).with_name('companies.json').read_text(encoding='utf-8'))
cols = ['okved', 'MAINOKVED_NAME', 'REGION', 'REVENUE_COMPANY', 'CL_TYPE', 'FTS_COUNTRY_NAME', 'GROUP_ID', 'AMOUNT_WORKERS_FNS_GRADATION']


def cp_table(t, cp, gid):
    t = t.join(U[cols], on=cp).sort_values('PAY_AMT', ascending=False)
    t['avg_k'] = (t.PAY_AMT / t.PAY_CNT / 1e3).round(0)
    t['same_grp'] = t.GROUP_ID == gid
    t['REVENUE_COMPANY'] = (t.REVENUE_COMPANY / 1e6).round(0)
    t['PAY_AMT'] = (t.PAY_AMT / 1e6).round(1)
    return t[[cp, 'PAY_AMT', 'PAY_CNT', 'avg_k', 'okved', 'MAINOKVED_NAME', 'REGION', 'REVENUE_COMPANY', 'CL_TYPE', 'same_grp', 'FTS_COUNTRY_NAME']].head(12)


for u, meta in ids.items():
    r = U.loc[u]
    print(f'\n################ {u} {meta}\n', r[cols + ['COMPANY_AGE_YEARS_GRADATION', 'ECO_MARK_LABEL', 'OKFS_NAME', 'NEGATIVEMARK']].to_dict())
    out, inc = P[P.PAYER == u], P[P.BENEFICIARY == u]
    for name, t, cp in [('OUT', out, 'BENEFICIARY'), ('IN', inc, 'PAYER')]:
        if len(t):
            print(f'--- {name}: n={len(t)} sum={money(t.PAY_AMT.sum())} cnt={t.PAY_CNT.sum()} avg ticket={money(t.PAY_AMT.sum() / t.PAY_CNT.sum())}'
                  f' in-base={t[cp].isin(U.index).mean():.0%}')
            print(cp_table(t, cp, r.GROUP_ID).to_string(index=False))
    both = set(out.BENEFICIARY) & set(inc.PAYER)
    if both:
        print('--- two-way:', [(x, money(out[out.BENEFICIARY == x].PAY_AMT.sum()), money(inc[inc.PAYER == x].PAY_AMT.sum()),
                               U.at[x, 'MAINOKVED_NAME'] if x in U.index else '?') for x in list(both)[:8]])
    if u in S.index:
        print('--- self:', money(S.loc[u].PAY_AMT), 'ratio to revenue', round(S.loc[u].PAY_AMT / r.REVENUE_COMPANY, 2))
    g = U[U.GROUP_ID == r.GROUP_ID]
    if len(g) > 1:
        gm = set(g.index)
        flows = P[(P.PAYER == u) & P.BENEFICIARY.isin(gm) | (P.BENEFICIARY == u) & P.PAYER.isin(gm)]
        print(f'--- group n={len(g)} rev={money(g.REVENUE_COMPANY.sum())}; members (top by revenue):')
        print(g.sort_values('REVENUE_COMPANY', ascending=False)[['okved', 'MAINOKVED_NAME', 'REGION', 'REVENUE_COMPANY', 'CL_TYPE', 'OKFS_NAME']]
              .head(12).assign(REVENUE_COMPANY=lambda q: (q.REVENUE_COMPANY / 1e6).round(0)).to_string())
        print('--- flows with group members:', flows.assign(a=flows.PAYER.map(U.MAINOKVED_NAME), b=flows.BENEFICIARY.map(U.MAINOKVED_NAME))
              [['PAYER', 'BENEFICIARY', 'PAY_AMT', 'a', 'b']].to_string(index=False) if len(flows) else 'none')
        ge = P[P.PAYER.isin(gm) & ~P.BENEFICIARY.isin(gm)].assign(o=lambda q: q.BENEFICIARY.map(U.MAINOKVED_NAME)).groupby('o').PAY_AMT.sum().nlargest(6)
        gi = P[P.BENEFICIARY.isin(gm) & ~P.PAYER.isin(gm)].assign(o=lambda q: q.PAYER.map(U.MAINOKVED_NAME)).groupby('o').PAY_AMT.sum().nlargest(6)
        print('--- group external suppliers (known okved, mln):', (ge / 1e6).round(0).to_dict())
        print('--- group external customers (known okved, mln):', (gi / 1e6).round(0).to_dict())
    pf = F[F.SYNTHETIC_U == u]
    if len(pf):
        print('--- persons:', pf.to_dict('records'))
    peers = U[U.okved == r.okved]
    pr = peers[[c for c in U.columns if c.startswith('CL_PRODUCT')] + ['FTS_FLAG', 'NEGATIVEMARK']].mean().round(2).to_dict()
    print('--- peer product/flag rates (same okved):', pr)
    print('--- peer workers dist:', peers.AMOUNT_WORKERS_FNS_GRADATION.value_counts(normalize=True).round(2).to_dict())
