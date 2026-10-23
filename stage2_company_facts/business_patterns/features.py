"""Per-company fact features for pattern discovery: card, group, persons, payment graph. Only what the tables say.

Every feature is a number computed from the case tables, no interpretation. Payment features describe the *visible*
part of the flows (the bank sees only part of them), so they are compared with peers, never read as absolute truth.
python -X utf8 stage2_company_facts/business_patterns/features.py   -> data/companies/features.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.data import DATA, load

# OKVED divisions -> coarse sectors for counterparty mix
SECTORS = {'manuf': range(10, 34), 'energy': range(35, 40), 'build': range(41, 44), 'auto': [45], 'whole': [46], 'retail': [47],
           'transp': range(49, 54), 'horeca': [55, 56], 'it': [61, 62, 63], 'fin': range(64, 67), 'realty': [68],
           'prof': range(69, 76), 'admin': range(77, 83), 'agro_min': list(range(1, 4)) + list(range(5, 10))}
DIV2SECTOR = {f'{d:02d}': s for s, ds in SECTORS.items() for d in ds}
WORKERS = {'Менее 15': 0, '16-100': 1, '101-250': 2, 'Более 251': 3}
AGE = {'0-3': 0, '4-7': 1, '8-15': 2, '16-25': 3}
PRODUCTS = ['CL_PRODUCT_RKO_MARK', 'CL_PRODUCT_GAR_MARK', 'CL_PRODUCT_LOA_MARK', 'CL_PRODUCT_LEA_MARK', 'CL_PRODUCT_DEP_MARK',
            'CL_PRODUCT_ZPP_MARK', 'FACTORING_MARK']


def hhi(shares: pd.Series, key: pd.Series) -> pd.Series:
    return (shares ** 2).groupby(key).sum()


def side(P: pd.DataFrame, U: pd.DataFrame, me: str, cp: str, tag: str) -> pd.DataFrame:
    """Aggregates of one direction: me = PAYER (out) or BENEFICIARY (in)."""
    t = P[P[me].isin(U.index)].copy()
    tot = t.groupby(me).PAY_AMT.transform('sum')
    t['share'] = t.PAY_AMT / tot
    known = t[cp].isin(U.index)
    t['cp_div'] = t[cp].map(U.okdiv)
    t['cp_sector'] = t.cp_div.map(DIV2SECTOR).fillna('other')
    same_group = t[cp].map(U.GROUP_ID).eq(t[me].map(U.GROUP_ID)) & known
    same_cls = t[cp].map(U.cls).eq(t[me].map(U.cls)) & known
    other_region = known & t[cp].map(U.REGION).ne(t[me].map(U.REGION))
    large_cp = t[cp].map(U.REVENUE_COMPANY).ge(1e9)
    g = t.groupby(me)
    f = pd.DataFrame({
        f'{tag}_amt': g.PAY_AMT.sum(), f'{tag}_cnt': g.PAY_CNT.sum(), f'{tag}_n': g.size(),
        f'{tag}_top1': g.share.max(), f'{tag}_hhi': hhi(t.share, t[me]),
        f'{tag}_known': (t.share * known).groupby(t[me]).sum(),
        f'{tag}_group': (t.share * same_group).groupby(t[me]).sum(),
        f'{tag}_samecls': (t.share * same_cls).groupby(t[me]).sum(),
        f'{tag}_otherreg': (t.share * other_region).groupby(t[me]).sum(),
        f'{tag}_largecp': (t.share * large_cp).groupby(t[me]).sum(),
    })
    f[f'{tag}_ticket'] = f[f'{tag}_amt'] / f[f'{tag}_cnt']
    f[f'{tag}_cnt_per_cp'] = f[f'{tag}_cnt'] / f[f'{tag}_n']
    mix = (t[known].assign(s=t.share).pivot_table(index=me, columns='cp_sector', values='s', aggfunc='sum', fill_value=0))
    mix.columns = [f'{tag}_sec_{c}' for c in mix.columns]
    return f.join(mix)


def reciprocity(P: pd.DataFrame) -> pd.Series:
    """Share of a company's total flow (in+out) that goes to counterparties it both pays and is paid by."""
    pairs = set(zip(P.PAYER, P.BENEFICIARY))
    two_way = np.array([(b, a) in pairs for a, b in zip(P.PAYER, P.BENEFICIARY)])
    amt = pd.concat([P.assign(c=P.PAYER), P.assign(c=P.BENEFICIARY)])
    tw = np.concatenate([two_way, two_way])
    return (amt.PAY_AMT * tw).groupby(amt.c).sum() / amt.groupby('c').PAY_AMT.sum()


def pagerank(P: pd.DataFrame, d: float = 0.85, iters: int = 50) -> pd.Series:
    """Weighted PageRank on payer -> beneficiary (money flows to 'important receivers')."""
    nodes = pd.Index(pd.unique(pd.concat([P.PAYER, P.BENEFICIARY])))
    a, b = nodes.get_indexer(P.PAYER), nodes.get_indexer(P.BENEFICIARY)
    n = len(nodes)
    W = sp.csr_matrix((P.PAY_AMT.to_numpy(float), (a, b)), shape=(n, n))
    out = np.asarray(W.sum(1)).ravel()
    T = sp.diags(np.divide(1, out, out=np.zeros(n), where=out > 0)) @ W
    r = np.full(n, 1 / n)
    dangling = out == 0
    for _ in range(iters):
        r = d * (T.T @ r + r[dangling].sum() / n) + (1 - d) / n
    return pd.Series(r * n, index=nodes)


def build() -> pd.DataFrame:
    U, P, S, F = load()
    X = pd.DataFrame(index=U.index)
    X['okved'], X['cls'], X['okdiv'], X['region'] = U.okved, U.cls, U.okdiv, U.REGION
    X['revenue'] = U.REVENUE_COMPANY
    X['log_rev'] = np.log10(U.REVENUE_COMPANY)
    X['workers'] = U.AMOUNT_WORKERS_FNS_GRADATION.map(WORKERS)
    X['age'] = U.COMPANY_AGE_YEARS_GRADATION.map(AGE)
    X['eco_mark'] = U.ECO_MARK_LABEL
    X['client'] = U.CL_TYPE
    X['foreign'] = U.OKFS_NAME.str.contains('ИНОСТРАН').astype(int)
    X['state'] = U.OKFS_NAME.str.contains('ФЕДЕРАЛ|СУБЪЕКТ|МУНИЦИП').astype(int)
    X['negative'] = U.NEGATIVEMARK
    for c in PRODUCTS:
        X[c.replace('CL_PRODUCT_', 'prod_').replace('_MARK', '').lower()] = U[c].fillna(0)
    X['fts'] = U.FTS_FLAG.fillna(0)
    X['fts_countries'] = U.FTS_COUNTRY_NAME.fillna('').map(lambda s: len([c for c in s.split(',') if c.strip()]))
    # group
    grp = U.groupby('GROUP_ID')
    X['group_size'] = grp.SYNTHETIC_U.transform('size')
    X['group_rev_share'] = U.REVENUE_COMPANY / grp.REVENUE_COMPANY.transform('sum')
    X['group_divs'] = grp.okdiv.transform('nunique')
    X['group_regions'] = grp.REGION.transform('nunique')
    # persons
    per_company = F.groupby('SYNTHETIC_U').SYNTHETIC_F.nunique()
    links = F.merge(F, on='SYNTHETIC_F').query('SYNTHETIC_U_x != SYNTHETIC_U_y').groupby('SYNTHETIC_U_x').SYNTHETIC_U_y.nunique()
    X['persons'] = per_company.reindex(X.index).fillna(0)
    X['person_linked_cos'] = links.reindex(X.index).fillna(0)
    # payments
    X = X.join(side(P, U, 'PAYER', 'BENEFICIARY', 'out')).join(side(P, U, 'BENEFICIARY', 'PAYER', 'in'))
    for tag in ('out', 'in'):
        X[[f'{tag}_amt', f'{tag}_cnt', f'{tag}_n']] = X[[f'{tag}_amt', f'{tag}_cnt', f'{tag}_n']].fillna(0)
    X['self_amt'] = S.PAY_AMT.reindex(X.index).fillna(0)
    X['in_rev'] = X.in_amt / X.revenue
    X['out_rev'] = X.out_amt / X.revenue
    X['self_rev'] = X.self_amt / X.revenue
    flow = X.in_amt + X.out_amt
    X['balance'] = (X.in_amt - X.out_amt) / flow.where(flow > 0)
    X['passthrough'] = np.minimum(X.in_amt, X.out_amt) / np.maximum(X.in_amt, X.out_amt).where(flow > 0)
    X['reciprocity'] = reciprocity(P).reindex(X.index)
    X['pagerank'] = pagerank(P).reindex(X.index)
    sec = [c for c in X.columns if '_sec_' in c]
    X[sec] = X[sec].fillna(0)
    return X


if __name__ == '__main__':
    X = build()
    out = DATA / 'companies' / 'features.parquet'
    out.parent.mkdir(parents=True, exist_ok=True)
    X.to_parquet(out)
    print(X.shape, '->', out)
    assert X.index.is_unique and (X.in_top1.dropna() <= 1 + 1e-9).all() and (X.passthrough.dropna() <= 1 + 1e-9).all()
