"""Stage-2 output contract: one company profile per line -> data/companies/profiles.jsonl.

The profile is what stages 3-6 read instead of the case tables:
- `facts`: records of deterministic_facts.build() ({cat, text, depth, entities}, stage 3 reads them as is) plus
  `id` (fact:<cat>_<n>, the form a claim cites), `source` (table and columns behind the fact) and `kind` (fact | hypothesis);
- `card`: the fields of stage5 analyst_loop.profile.CompanyProfile, so `CompanyProfile(**profile['card'])` works without CSV;
- `scale`: size class by 209-FZ thresholds, the case criterion and revenue percentiles among peers;
- `search_terms`: deduplicated entities by category (activities, regions, countries, group activities) for retrieval.
Pure code, no LLM.
Usage: python -X utf8 stage2_company_facts/profile/profile.py --ids U_42222 U_31505   (no --ids = 115 companies >= 15 bn)
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.data import DATA, countries, load
from stage2_company_facts.deterministic_facts.facts import build, self_check
from stage2_company_facts.external_factors.factors import factors, search_terms
from stage2_company_facts.links.links import links
from stage2_company_facts.profile.scale import CLASS_RU, LARGE, ORDER, size_class
from stage2_company_facts.sensitivity.levers import sensitivity

PRODUCT_COLUMNS = {'CL_PRODUCT_LOA_MARK': 'loan', 'CL_PRODUCT_LEA_MARK': 'leasing', 'CL_PRODUCT_DEP_MARK': 'deposit',
                   'CL_PRODUCT_GAR_MARK': 'guarantee', 'FACTORING_MARK': 'factoring', 'CL_PRODUCT_RKO_MARK': 'rko',
                   'CL_PRODUCT_ZPP_MARK': 'salary'}
SOURCE = {'activity': 'df_Uid_data_hack.csv: MAINOKVED_CODE, MAINOKVED_NAME',
          'scale': 'df_Uid_data_hack.csv: REVENUE_COMPANY, AMOUNT_WORKERS_FNS_GRADATION',
          'geo': 'df_Uid_data_hack.csv: CITY, REGION',
          'profile': 'df_Uid_data_hack.csv: COMPANY_AGE_YEARS_GRADATION, ECO_MARK_LABEL, AMOUNT_WORKERS_*',
          'ownership': 'df_Uid_data_hack.csv: OKFS_NAME, NEGATIVEMARK',
          'bank': 'df_Uid_data_hack.csv: CL_TYPE, CL_PRODUCT_*_MARK, FACTORING_MARK',
          'trade': 'df_Uid_data_hack.csv: FTS_FLAG, FTS_COUNTRY_NAME',
          'data_quality': 'common.data.DQ (stage2_company_facts/deterministic_facts/data_quality.md)',
          'rank': 'df_Uid_data_hack.csv: REVENUE_COMPANY по всем компаниям',
          'persons': 'df_Fid_Uid_link_hack.csv'}
SOURCE_PREFIX = {'group': 'df_Uid_data_hack.csv: GROUP_ID (+ df_Uid_payment_hack.csv, df_Fid_Uid_link_hack.csv)',
                 'payments': 'df_Uid_payment_hack.csv: PAYER, BENEFICIARY, PAY_AMT, PAY_CNT (+ df_Uid_data_hack.csv контрагентов)'}
TERMS = {'activity': 'activities', 'geo': 'regions', 'trade': 'trade_countries', 'group_activity': 'group_activities'}


def source_of(cat: str) -> str:
    return SOURCE.get(cat) or next((v for k, v in SOURCE_PREFIX.items() if cat.startswith(k)), 'case tables')


def percentiles(U: pd.DataFrame, ids: list[str], min_n: int = 5) -> dict[str, dict]:
    """Revenue percentile (share of peers with revenue not above the company's, 0..100) per peer set."""
    rev = U.REVENUE_COMPANY.where(U.REVENUE_COMPANY > 1000)
    sets = {'all': pd.Series('all', index=U.index), 'okved': U.okved, 'okdiv': U.okdiv, 'region': U.REGION}
    out = {u: {} for u in ids}
    for name, key in sets.items():
        g = rev.groupby(key)
        p, n = g.rank(pct=True, method='max'), g.transform('count')
        for u in ids:
            if pd.notna(p.get(u)) and n[u] >= min_n:
                out[u][name] = {'percentile': round(float(p[u]) * 100, 1), 'peers': int(n[u])}
    return out


def payment_shares(P, U, company_id, me, other):
    """Known sector amounts divided by all payments, including unclassified counterparties."""
    df = P[P[me] == company_id]
    total = float(df.PAY_AMT.sum())
    df = df.assign(d=df[other].map(U.okdiv)).dropna(subset=['d'])
    return ({k: round(float(v / total), 4) for k, v in df.groupby('d').PAY_AMT.sum().items()} if total else {}), total


def profiles(ids: list[str] | None = None, news: bool = False) -> dict[str, dict]:
    """news: add `news_context` (recent news behind the strongest external factors, stage2_company_facts/news_context);
    off by default so the profile itself does not depend on the news corpus."""
    U, P, S, F = load()
    ids = [u for u in (ids or list(U.index[U.REVENUE_COMPANY >= LARGE])) if u in U.index]
    facts = build(ids)
    self_check(facts)
    pct = percentiles(U, ids)
    gsize, grev = U.groupby('GROUP_ID').size(), U.groupby('GROUP_ID').REVENUE_COMPANY.sum(min_count=1)
    out = {}
    for u in ids:
        r = U.loc[u]
        rev = float(r.REVENUE_COMPANY) if pd.notna(r.REVENUE_COMPANY) and r.REVENUE_COMPANY > 1000 else None
        n_group = int(gsize[r.GROUP_ID])
        group_rev = float(grev[r.GROUP_ID]) if n_group > 1 and pd.notna(grev[r.GROUP_ID]) else None
        cls, basis = size_class(r.REVENUE_COMPANY, r.AMOUNT_WORKERS_FNS_GRADATION)
        pay_out, out_total = payment_shares(P, U, u, 'PAYER', 'BENEFICIARY')
        pay_in, in_total = payment_shares(P, U, u, 'BENEFICIARY', 'PAYER')
        n_by_cat = {}
        typed = []
        for f in facts[u]:
            n_by_cat[f['cat']] = n_by_cat.get(f['cat'], 0) + 1
            typed.append({'id': f'fact:{f["cat"]}_{n_by_cat[f["cat"]]}', **f, 'source': source_of(f['cat']), 'kind': 'fact'})
        terms = {v: [] for v in TERMS.values()}
        for f in facts[u]:
            if f['cat'] in TERMS:  # 'ВЭД' is a tag of the FTS_FLAG fact, not a country
                terms[TERMS[f['cat']]] += [e for e in f['entities'] if e != 'ВЭД' and e not in terms[TERMS[f['cat']]]]
        card = {'id': u, 'okved': r.okved, 'okved_name': r.MAINOKVED_NAME, 'region': r.REGION, 'revenue': rev,
                'group_revenue': group_rev,
                'products': [v for k, v in PRODUCT_COLUMNS.items() if r[k] == 1] if r.CL_TYPE == 'CLIENT' else [],
                'fts_countries': countries(r.FTS_COUNTRY_NAME) if isinstance(r.FTS_COUNTRY_NAME, str) else [],
                'fts_flag': bool(r.FTS_FLAG == 1), 'pay_out': pay_out, 'pay_in': pay_in,
                'pay_out_total': out_total, 'pay_in_total': in_total}
        terms.update(search_terms(card))
        out[u] = {
            'id': u,
            'card': card,
            'scale': {'class': cls, 'class_ru': CLASS_RU.get(cls, 'н/д'), 'class_basis': basis,
                      'case_large': bool(rev and rev >= LARGE), 'group_size': n_group,
                      'group_case_large': bool(group_rev and group_rev >= LARGE),
                      'revenue_percentile': pct[u]},
            'data_quality': [x for x in r.DQ_FLAGS.split(',') if x],
            'search_terms': terms,
            'external_factors': factors(card, typed),
            'facts': typed,
        }
        out[u]['sensitivity'] = sensitivity(out[u], r.AMOUNT_WORKERS_FNS_GRADATION)
        out[u]['links'] = links(U, P, F, u)
    if news and out:
        from stage2_company_facts.news_context.context import NewsContext
        nc = NewsContext()
        for p in out.values():
            p['news_context'] = nc.for_profile(p)
    return out


def check(p: dict) -> None:
    ids = [f['id'] for f in p['facts']]
    assert len(ids) == len(set(ids)), f'duplicate fact id in {p["id"]}'
    assert p['scale']['class'] in ORDER + ['unknown']
    assert all(0 < v['percentile'] <= 100 for v in p['scale']['revenue_percentile'].values())
    for shares in (p['card']['pay_out'], p['card']['pay_in']):
        # The missing share belongs to counterparties whose sector is unknown; allow rounding to 4 decimals.
        tolerance = max(1e-3, len(shares) * 0.00005)
        assert all(0 <= v <= 1 for v in shares.values()), f'payment share out of range in {p["id"]}'
        assert sum(shares.values()) <= 1 + tolerance, f'payment shares exceed 1 in {p["id"]}'
    assert not (p['scale']['case_large'] and p['scale']['class'] != 'large')
    known = set(ids) | {f'card.{k}' for k in p['card']}
    assert all(b in known for x in p['external_factors'] for b in x['basis']), f'factor without a known basis in {p["id"]}'
    known |= {x['id'] for x in p['external_factors']} | {'base', 'card', 'scale.case_large'}
    assert all(0 <= s['weight'] <= 1 and set(s['reasons']) <= known for s in p['sensitivity'].values()), \
        f'sensitivity out of range or with an unknown reason in {p["id"]}'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ids', nargs='*')
    ap.add_argument('--out', default=str(DATA / 'companies' / 'profiles.jsonl'))
    ap.add_argument('--news', action='store_true', help='add news_context from the stage-1 news pool')
    a = ap.parse_args()
    res = profiles(a.ids, news=a.news)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        for p in res.values():
            check(p)
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    cls = pd.Series([p['scale']['class'] for p in res.values()]).value_counts().to_dict()
    n = [len(p['facts']) for p in res.values()]
    print(f'{len(res)} profiles -> {out}; facts per company: median {int(np.median(n)) if n else 0}; classes {cls}')
