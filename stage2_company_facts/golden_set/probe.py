"""Targeted checks that try to refute golden-set hypotheses: python -X utf8 stage2_company_facts/golden_set/probe.py"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.data import load, money

U, P, S, F = load()
nm = U.MAINOKVED_NAME


def cps(x, side, top=8):
    t = P[P.PAYER == x] if side == 'out' else P[P.BENEFICIARY == x]
    c = t.BENEFICIARY if side == 'out' else t.PAYER
    k = t.assign(o=c.map(nm), r=c.map(U.REGION)).dropna(subset=['o'])
    return {'n': len(t), 'sum': money(t.PAY_AMT.sum()), 'known': f'{k.PAY_AMT.sum() / max(t.PAY_AMT.sum(), 1):.0%}',
            'by_okved': (k.groupby('o').PAY_AMT.sum().nlargest(top) / 1e6).round(1).to_dict(),
            'n_by_okved': k.o.value_counts().head(top).to_dict(), 'regions': (k.groupby('r').PAY_AMT.sum().nlargest(5) / 1e6).round(1).to_dict()}


print('U_55267 FTS flag/country:', U.loc['U_55267', ['FTS_FLAG', 'FTS_COUNTRY_NAME']].to_dict())
print('companies with countries but FTS_FLAG=0:', int((U.FTS_COUNTRY_NAME.notna() & (U.FTS_FLAG != 1)).sum()))
print('U_55267 payers:', cps('U_55267', 'in'))
print('U_55267 payees:', cps('U_55267', 'out'))

# SPV developer pattern: tiny revenue, large flows, pays a builder
rev = U.REVENUE_COMPANY
flow = P.groupby('BENEFICIARY').PAY_AMT.sum().add(P.groupby('PAYER').PAY_AMT.sum(), fill_value=0)
spv = U[U.okdiv.isin(['41']) & (rev < 1e6)].index
big = [x for x in spv if flow.get(x, 0) > 1e8]
print(f'41.x with revenue<1 mln and flows>100 mln: {len(big)} of {len(spv)}')
print('U_45078 (unknown payer of U_44767) degree in/out:', (P.BENEFICIARY == 'U_45078').sum(), (P.PAYER == 'U_45078').sum(),
      'its counterparties:', cps('U_45078', 'out', 5)['by_okved'], cps('U_45078', 'in', 5)['by_okved'])
print('U_110237 (builder paid by U_44767):', U.loc['U_110237', ['MAINOKVED_NAME', 'REVENUE_COMPANY', 'AMOUNT_WORKERS_FNS_GRADATION', 'GROUP_ID']].to_dict(),
      'payers:', cps('U_110237', 'in', 5)['by_okved'])

for x in ['U_98857', 'U_2602']:
    print(f'{x} (U_57162 big payee): in/out degree', (P.BENEFICIARY == x).sum(), (P.PAYER == x).sum(),
          '| payers:', cps(x, 'in', 6)['by_okved'], '| payees:', cps(x, 'out', 6)['by_okved'])
print('U_57162 payers:', cps('U_57162', 'in'))

print('U_68622 payers:', cps('U_68622', 'in', 10))
print('U_180417 (food wholesaler paying U_68622):', U.loc['U_180417', ['MAINOKVED_NAME', 'REVENUE_COMPANY', 'GROUP_ID']].to_dict(),
      'its payers:', cps('U_180417', 'in', 5)['by_okved'])
print('U_220924 -> U_68622:', P[(P.PAYER == 'U_220924') & (P.BENEFICIARY == 'U_68622')][['PAY_AMT', 'PAY_CNT']].to_dict('records'))

print('U_49717 payers:', cps('U_49717', 'in', 12))
print('U_174807 payers:', cps('U_174807', 'in', 10))
print('U_174807 group leader U_108274 payers:', cps('U_108274', 'in', 8))

print('U_76122 payer U_66319:', U.loc['U_66319', ['MAINOKVED_NAME', 'REVENUE_COMPANY', 'REGION', 'GROUP_ID']].to_dict(),
      '| U_66319 payees:', cps('U_66319', 'out', 8))
print('U_76122 Rostov payees:', P[(P.PAYER == 'U_76122') & P.BENEFICIARY.map(U.REGION).eq('Ростовская область')]
      .assign(o=lambda q: q.BENEFICIARY.map(nm)).groupby('o').PAY_AMT.sum().div(1e6).round(0).to_dict())
print('U_257637 payees:', cps('U_257637', 'out', 10))
for c in ['41.20', '62.01', '56.10']:
    g = U[U.okved == c]
    print(c, 'revenue per worker bucket median rev:', g.groupby('AMOUNT_WORKERS_FNS_GRADATION').REVENUE_COMPANY.median().div(1e6).round(0).to_dict())
