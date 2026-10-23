"""Fact-patterns: combinations of fact flags that are rare in the market and shared by a cohesive set of companies.

1. `flags()` — every flag is a threshold on one feature from features.py (peer quantiles are taken within the OKVED
   division). A flag states a fact about the *visible* data, nothing more.
2. `discover()` — NMF on the company × flag matrix (idf-weighted, as in RolX role discovery) with bootstrap stability:
   which flags co-occur systematically. The catalogue PATTERNS below was read off the stable components (≥0.8)
   of runs on wholesale (46), construction (41-43) and all observable companies, then cross-checked by lift.
3. `catalogue()` — for each pattern: support, lift, cohesion beyond its own flags, OKVED enrichment, large companies,
   and a partial-visibility check (do members simply show less of their flows than peers?).

Only CLIENT companies with enough visible flows are used: for non-clients the bank sees almost no payments.
python -X utf8 stage2_company_facts/business_patterns/patterns.py [--discover 46 12]  -> data/companies/patterns*.parquet
"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.data import DATA

LARGE = 15e9


def observable(X: pd.DataFrame) -> pd.Series:
    flow_rev = (X.in_amt + X.out_amt) / X.revenue
    return (X.client == 'CLIENT') & (X.in_n + X.out_n >= 10) & (flow_rev >= 0.1) & (X.in_n >= 3) & (X.out_n >= 3)


def flags(O: pd.DataFrame) -> pd.DataFrame:
    """Fact flags of observable companies. Shares are of the visible flow; 'known' = counterparty is in the companies table."""
    q = lambda c, p: O.groupby('okdiv')[c].transform(lambda s: s.quantile(p))
    flow_rev = (O.in_amt + O.out_amt) / O.revenue
    f = {
        # flow shape
        'in_one_payer': O.in_top1 >= 0.8, 'out_one_payee': O.out_top1 >= 0.8,
        'pass_through': O.passthrough >= 0.85, 'net_receiver': O.balance >= 0.8, 'net_payer': O.balance <= -0.8,
        'flow_3x_rev': flow_rev >= 3, 'many_payers': O.in_n >= q('in_n', .9), 'many_payees': O.out_n >= q('out_n', .9),
        'small_in_tickets': O.in_ticket <= q('in_ticket', .1), 'big_in_tickets': O.in_ticket >= q('in_ticket', .9),
        'reciprocal': O.reciprocity >= 0.6, 'anon_in': O.in_known <= 0.05, 'anon_out': O.out_known <= 0.05,
        'interreg_sales': O.in_otherreg >= 0.6, 'interreg_buys': O.out_otherreg >= 0.6, 'large_customers': O.in_largecp >= 0.5,
        # group
        'in_from_group': O.in_group >= 0.5, 'out_to_group': O.out_group >= 0.5, 'big_group': O.group_size >= 20,
        'group_leader': (O.group_rev_share >= 0.5) & (O.group_size >= 5),
        'minor_member': (O.group_rev_share < 0.05) & (O.group_size >= 5), 'multi_ind_group': O.group_divs >= 5,
        # counterparty industries
        'buys_manuf': O.out_sec_manuf >= 0.3, 'sells_manuf': O.in_sec_manuf >= 0.3, 'buys_whole': O.out_sec_whole >= 0.5,
        'sells_whole': O.in_sec_whole >= 0.5, 'sells_retail': O.in_sec_retail >= 0.2, 'buys_build': O.out_sec_build >= 0.3,
        'sells_build': O.in_sec_build >= 0.3, 'buys_prof': O.out_sec_prof >= 0.2, 'pays_transport': O.out_sec_transp >= 0.2,
        'sells_transp': O.in_sec_transp >= 0.3, 'pays_realty': O.out_sec_realty >= 0.2, 'sells_realty': O.in_sec_realty >= 0.3,
        'same_cls_trade': (O.in_samecls >= 0.5) | (O.out_samecls >= 0.5),
        # card and bank products
        'tiny_staff_big_rev': (O.workers == 0) & (O.revenue >= q('revenue', .75)), 'big_staff': O.workers >= 2,
        'young': O.age == 0, 'fts_3plus': O.fts_countries >= 3, 'foreign_owned': O.foreign == 1,
        'guarantees': O.prod_gar == 1, 'deposits': O.prod_dep == 1, 'loans': O.prod_loa == 1, 'leasing': O.prod_lea == 1,
    }
    return pd.DataFrame(f).fillna(False).astype(bool)


# id, aspect, conjunction of flags, plain description of the facts (no interpretation)
PATTERNS = [
    ('FP01', 'поток', ['pass_through', 'flow_3x_rev'], 'видимые входящие ≈ исходящие, и оба потока ≥3× годовой выручки'),
    ('FP02', 'поток', ['in_one_payer', 'big_in_tickets'], '≥80% входящих от одного плательщика, средний платёж в верхних 10% отрасли'),
    ('FP03', 'поток', ['out_one_payee', 'anon_out'], '≥80% исходящих одному получателю, получатели почти все вне базы'),
    ('FP04', 'поток', ['many_payers', 'many_payees', 'big_staff'], 'плательщиков и получателей больше, чем у 90% отрасли; >100 сотрудников'),
    ('FP05', 'поток', ['many_payers', 'small_in_tickets'], 'много плательщиков при среднем входящем платеже в нижних 10% отрасли'),
    ('FP06', 'поток', ['young', 'flow_3x_rev'], 'компании 0–3 года, видимые обороты ≥3× выручки'),
    ('FP07', 'поток', ['big_in_tickets', 'tiny_staff_big_rev'], '<15 сотрудников, выручка в верхней четверти отрасли, крупные входящие платежи'),
    ('FP10', 'группа', ['in_from_group', 'out_to_group'], '≥50% и входящих, и исходящих — с компаниями своей группы'),
    ('FP11', 'группа', ['in_from_group', 'in_one_payer'], '≥80% входящих от одного плательщика, и это компания своей группы'),
    ('FP12', 'группа', ['big_group', 'minor_member', 'multi_ind_group'], 'группа ≥20 компаний из ≥5 разделов ОКВЭД, доля компании в выручке группы <5%'),
    ('FP20', 'контрагенты', ['buys_whole', 'interreg_buys'], '≥50% исходящих — оптовикам, ≥60% — в другие регионы'),
    ('FP21', 'контрагенты', ['sells_whole', 'large_customers'], '≥50% входящих от оптовиков, ≥50% — от компаний с выручкой ≥1 млрд'),
    ('FP22', 'контрагенты', ['buys_build', 'sells_build'], '≥30% входящих от строителей и ≥30% исходящих строителям'),
    ('FP23', 'контрагенты', ['out_to_group', 'buys_build', 'buys_prof'], 'исходящие в основном своей группе, строителям и проектным / инженерным компаниям'),
    ('FP24', 'контрагенты', ['pays_realty', 'sells_realty'], '≥20% исходящих и ≥30% входящих — компаниям операций с недвижимостью'),
    ('FP25', 'контрагенты', ['buys_manuf', 'sells_manuf'], '≥30% исходящих и ≥30% входящих — производственным компаниям'),
    ('FP26', 'контрагенты', ['pays_transport', 'sells_transp'], '≥20% исходящих и ≥30% входящих — транспортным компаниям'),
    ('FP27', 'контрагенты', ['sells_build', 'large_customers', 'interreg_sales'], 'входящие от крупных строителей из других регионов'),
    ('FP30', 'карточка', ['fts_3plus', 'pays_transport'], '≥3 страны в данных ФТС и ≥20% исходящих транспортным компаниям'),
    ('FP31', 'карточка', ['guarantees', 'deposits'], 'у компании есть банковские гарантии и депозиты в Альфе'),
]


def discover(Fl: pd.DataFrame, k: int, boots: int = 10) -> tuple[pd.DataFrame, np.ndarray]:
    """NMF components (flag weights, unit norm) and their bootstrap stability (mean matched cosine)."""
    idf = np.log(1 / Fl.mean().clip(1e-3))
    A = Fl.to_numpy(float) * idf.to_numpy()

    def fit(M, seed):
        H = NMF(n_components=k, init='nndsvda', random_state=seed, max_iter=800).fit(M).components_
        return H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-12)

    H, rng, sims = fit(A, 0), np.random.default_rng(0), []
    for b in range(boots):
        C = H @ fit(A[rng.choice(len(A), len(A))], b + 1).T
        r, c = linear_sum_assignment(-C)
        sims.append(C[r, c])
    return pd.DataFrame(H, columns=Fl.columns), np.mean(sims, 0)


def jaccard_mean(B: np.ndarray, rng, pairs: int = 3000) -> float:
    if len(B) < 2:
        return np.nan
    i, j = rng.integers(0, len(B), pairs), rng.integers(0, len(B), pairs)
    keep = i != j
    inter, union = (B[i[keep]] & B[j[keep]]).sum(1), (B[i[keep]] | B[j[keep]]).sum(1)
    return float(np.mean(np.divide(inter, union, out=np.zeros(len(inter)), where=union > 0)))


def catalogue(X: pd.DataFrame, O: pd.DataFrame, Fl: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    flow_rev = (O.in_amt + O.out_amt) / O.revenue
    rows, members = [], {}
    for pid, aspect, conj, text in PATTERNS:
        m = Fl[conj].all(axis=1)
        members[pid] = m
        rest = Fl.drop(columns=conj).to_numpy()
        # cohesion: are members similar on the flags that do NOT define the pattern? (1.0 = no more than random pairs)
        cohesion = jaccard_mean(rest[m.to_numpy()], rng) / jaccard_mean(rest, rng)
        div_share = O.okdiv[m].value_counts(normalize=True)
        enrich = (div_share / O.okdiv.value_counts(normalize=True)).loc[div_share.index]
        top = [f'{d} ×{enrich[d]:.1f} ({int(div_share[d] * m.sum())})' for d in div_share.index[:4]]
        peer_flow = flow_rev.groupby(O.okdiv).median()
        rows.append({'id': pid, 'aspect': aspect, 'flags': ' & '.join(conj), 'facts': text, 'n': int(m.sum()),
                     'share_%': round(100 * m.mean(), 2), 'lift': round(m.mean() / Fl[conj].mean().prod(), 1),
                     'cohesion': round(cohesion, 2), 'large': int((O.revenue[m] >= LARGE).sum()),
                     'flow_vs_peers': round(float((flow_rev[m] / O.okdiv[m].map(peer_flow)).median()), 2),
                     'top_divisions': ', '.join(top)})
    return pd.DataFrame(rows), pd.DataFrame(members)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--discover', nargs=2, metavar=('DIVS', 'K'), help='e.g. 46 12, or all 22')
    a = ap.parse_args()
    X = pd.read_parquet(DATA / 'companies' / 'features.parquet')
    O = X[observable(X)]
    Fl = flags(O)
    if a.discover:
        S = Fl if a.discover[0] == 'all' else Fl[O.okdiv.isin(a.discover[0].split(','))]
        H, stab = discover(S, int(a.discover[1]))
        for j in np.argsort(-stab):
            w = H.iloc[j].sort_values(ascending=False)
            print(f'stability {stab[j]:.2f}: ' + ', '.join(f'{f} {v:.2f}' for f, v in w.items() if v > 0.15))
    cat, members = catalogue(X, O, Fl)
    cat.to_parquet(DATA / 'companies' / 'patterns.parquet')
    members.to_parquet(DATA / 'companies' / 'pattern_members.parquet')
    pd.set_option('display.width', 250, 'display.max_colwidth', 60)
    print(f'observable companies: {len(O)} of {len(X)}; large observable: {int((O.revenue >= LARGE).sum())}')
    print(cat.drop(columns=['facts']).to_string(index=False))
    assert (cat.n > 0).all() and cat.id.is_unique
