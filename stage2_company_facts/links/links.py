"""Key links of a company: main suppliers and buyers by payments, its group and companies linked through shared persons.

Structured counterpart of the text facts of deterministic_facts (payments_*, group*, persons): stages 3-6 can follow a
link to the counterparty's sector (news about it reaches the company) without parsing strings. No LLM.
- counterparties: top-k by amount per side, share of the side's total, OKVED, region, scale class, same group,
  Alfa client; a counterparty outside the companies table keeps only id and amount (anonymised, no profile);
- concentration: top-1 / top-5 share and HHI per side (HHI >= 0.25 = dependence on few counterparties);
- group: size, total revenue, the company's share and rank, main activities, group regions, in-group payments;
- persons: companies linked through shared individuals outside the company's own group.
"""
import pandas as pd

from stage2_company_facts.profile.scale import size_class


def side(P, U, u, me, cp, k):
    g = P[P[me] == u].groupby(cp).agg(amount=('PAY_AMT', 'sum'), count=('PAY_CNT', 'sum')).sort_values('amount', ascending=False)
    total = float(g.amount.sum())
    if not total:
        return {'total': 0.0, 'n': 0, 'top1_share': None, 'top5_share': None, 'hhi': None, 'top': []}
    sh = g.amount / total
    top = []
    for x, row in g.head(k).iterrows():
        item = {'id': x, 'amount': float(row.amount), 'payments': int(row['count']), 'share': round(float(row.amount / total), 4)}
        if x in U.index:
            c = U.loc[x]
            item |= {'okved': c.okved, 'okved_name': c.MAINOKVED_NAME, 'region': c.REGION,
                     'scale_class': size_class(c.REVENUE_COMPANY, c.AMOUNT_WORKERS_FNS_GRADATION)[0],
                     'same_group': bool(c.GROUP_ID == U.at[u, 'GROUP_ID']), 'alfa_client': bool(c.CL_TYPE == 'CLIENT')}
        else:
            item['in_table'] = False
        top.append(item)
    return {'total': total, 'n': int(len(g)), 'top1_share': round(float(sh.iloc[0]), 4),
            'top5_share': round(float(sh.head(5).sum()), 4), 'hhi': round(float((sh ** 2).sum()), 4), 'top': top}


def group(U, P, u, top_k=5):
    gid = U.at[u, 'GROUP_ID']
    g = U[U.GROUP_ID == gid]
    if len(g) < 2:
        return None
    rev = g.REVENUE_COMPANY.where(g.REVENUE_COMPANY > 1000)
    total = float(rev.sum()) if rev.notna().any() else None
    acts = g.groupby(['okved', 'MAINOKVED_NAME']).agg(n=('SYNTHETIC_U', 'size'), revenue=('REVENUE_COMPANY', 'sum')) \
        .sort_values('revenue', ascending=False).head(top_k)
    members = set(g.index)
    inner = P[P.PAYER.isin(members) & P.BENEFICIARY.isin(members)]
    own = rev.get(u)
    return {'id': gid, 'size': int(len(g)), 'revenue_total': total,
            'company_share': round(float(own / total), 4) if total and pd.notna(own) else None,
            'company_rank': int(rev.rank(ascending=False, method='min')[u]) if pd.notna(own) else None,
            'activities': [{'okved': c, 'name': n, 'companies': int(r.n), 'revenue': float(r.revenue)} for (c, n), r in acts.iterrows()],
            'regions': g.REGION.value_counts().head(top_k).to_dict(),
            'alfa_clients': int((g.CL_TYPE == 'CLIENT').sum()),
            'inner_payments': {'amount': float(inner.PAY_AMT.sum()), 'edges': int(len(inner))}}


def persons(U, F, u, top_k=5):
    mine = F[F.SYNTHETIC_U == u].SYNTHETIC_F.unique()
    if not len(mine):
        return None
    other = F[F.SYNTHETIC_F.isin(mine) & (F.SYNTHETIC_U != u)].SYNTHETIC_U.unique()
    known = U.loc[[x for x in other if x in U.index]]
    outside = known[known.GROUP_ID != U.at[u, 'GROUP_ID']]
    return {'persons': int(len(mine)), 'linked_companies': int(len(other)), 'outside_group': int(len(outside)),
            'outside_activities': outside.MAINOKVED_NAME.value_counts().head(top_k).to_dict()}


def links(U, P, F, u, k=10) -> dict:
    return {'suppliers': side(P, U, u, 'PAYER', 'BENEFICIARY', k), 'buyers': side(P, U, u, 'BENEFICIARY', 'PAYER', k),
            'group': group(U, P, u), 'persons': persons(U, F, u)}
