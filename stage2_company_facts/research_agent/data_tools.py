"""Read-only tools over the case tables for the research agent.

No shell or eval: every tool is a fixed pandas query, `table_query` accepts a whitelisted filter spec only.
Each tool returns a compact text table; `as_tools()` exposes the methods as LangChain tools (docstring = description).
"""
from typing import Any, Literal

import pandas as pd
from langchain_core.tools import BaseTool, StructuredTool

from common.data import load, money

CARD = ['okved', 'MAINOKVED_NAME', 'REGION', 'CITY', 'REVENUE_COMPANY', 'AMOUNT_WORKERS_FNS_GRADATION', 'COMPANY_AGE_YEARS_GRADATION',
        'ECO_MARK_LABEL', 'OKFS_NAME', 'CL_TYPE', 'FTS_FLAG', 'FTS_COUNTRY_NAME', 'GROUP_ID', 'NEGATIVEMARK',
        'CL_PRODUCT_RKO_MARK', 'CL_PRODUCT_GAR_MARK', 'CL_PRODUCT_LOA_MARK', 'CL_PRODUCT_LEA_MARK', 'CL_PRODUCT_DEP_MARK',
        'CL_PRODUCT_ZPP_MARK', 'FACTORING_MARK']
OPS = {'==': lambda s, v: s == v, '!=': lambda s, v: s != v, '>': lambda s, v: s > v, '<': lambda s, v: s < v,
       '>=': lambda s, v: s >= v, '<=': lambda s, v: s <= v, 'in': lambda s, v: s.isin(v),
       'startswith': lambda s, v: s.astype(str).str.startswith(v),
       'contains': lambda s, v: s.astype(str).str.contains(v, case=False, regex=False)}
AGG = {'sum', 'mean', 'median', 'count', 'nunique', 'min', 'max'}
MAX_ROWS = 40


def as_text(df: pd.DataFrame, n: int = MAX_ROWS) -> str:
    if not len(df):
        return '(empty)'
    if len(df) > n:
        return df.head(n).to_string(index=False) + f'\n... ({len(df)} rows total, showing {n})'
    return df.to_string(index=False)


class CompanyDataTools:
    TOOL_NAMES = ['company_card', 'counterparties', 'payments_between', 'group_members', 'person_links', 'peer_stats', 'table_query']

    def __init__(self):
        self.companies, self.payments, self.self_payments, self.persons = load()

    def as_tools(self) -> list[BaseTool]:
        return [StructuredTool.from_function(getattr(self, name)) for name in self.TOOL_NAMES]

    def company_card(self, company_id: str) -> str:
        """Card of a company; for an id absent from the companies table, what its payment graph says about it."""
        U = self.companies
        if company_id not in U.index:
            return self._payment_profile(company_id)
        r = U.loc[company_id, CARD].to_dict()
        r['REVENUE_COMPANY'] = money(r['REVENUE_COMPANY'])
        return '\n'.join(f'{k}: {v}' for k, v in r.items() if pd.notna(v))

    def counterparties(self, company_id: str, direction: Literal['in', 'out'] = 'in',
                       group_by: Literal['none', 'okved', 'region', 'group'] = 'none', top: int = 15,
                       okved_prefix: str | None = None) -> str:
        """Payment counterparties of a company (in = who pays it, out = whom it pays): a list, or grouped by okved / region / group.
        okved_prefix keeps counterparties whose OKVED starts with it."""
        U, P = self.companies, self.payments
        me, cp = ('BENEFICIARY', 'PAYER') if direction == 'in' else ('PAYER', 'BENEFICIARY')
        t = P[P[me] == company_id]
        total = t.PAY_AMT.sum()
        t = t.assign(okved=t[cp].map(U.okved), name=t[cp].map(U.MAINOKVED_NAME), region=t[cp].map(U.REGION),
                     revenue=t[cp].map(U.REVENUE_COMPANY), group=t[cp].map(U.GROUP_ID), client=t[cp].map(U.CL_TYPE))
        if okved_prefix:
            t = t[t.okved.fillna('').str.startswith(okved_prefix)]
        head = (f'{direction}: {len(t)} counterparties, {money(t.PAY_AMT.sum())} of {money(total)} total, '
                f'{int(t.PAY_CNT.sum())} payments\n')
        if group_by != 'none':
            cols = {'okved': ['okved', 'name'], 'region': ['region'], 'group': ['group']}[group_by]
            g = t.groupby(cols, dropna=False).agg(amount=('PAY_AMT', 'sum'), n=(cp, 'nunique'), payments=('PAY_CNT', 'sum'))
            g = g.sort_values('amount', ascending=False).head(top).reset_index()
            g['share'] = (g.amount / total).round(3)
            g['amount'] = g.amount.map(money)
            return head + as_text(g, top)
        t = t.sort_values('PAY_AMT', ascending=False).head(top)
        t = t.assign(share=(t.PAY_AMT / total).round(3), avg=(t.PAY_AMT / t.PAY_CNT).map(money), PAY_AMT=t.PAY_AMT.map(money),
                     revenue=t.revenue.map(lambda v: money(v) if pd.notna(v) else 'not in base'))
        return head + as_text(t[[cp, 'PAY_AMT', 'share', 'PAY_CNT', 'avg', 'okved', 'name', 'region', 'revenue', 'group', 'client']], top)

    def payments_between(self, a: str, b: str) -> str:
        """Payments in both directions between two ids."""
        P = self.payments
        t = P[((P.PAYER == a) & (P.BENEFICIARY == b)) | ((P.PAYER == b) & (P.BENEFICIARY == a))]
        return as_text(t.assign(PAY_AMT=t.PAY_AMT.map(money)))

    def group_members(self, company_id: str, okved_prefix: str | None = None, top: int = 25) -> str:
        """Members of the company's group with OKVED, revenue and payment flows with the company."""
        U, P = self.companies, self.payments
        group = U.at[company_id, 'GROUP_ID']
        g = U[U.GROUP_ID == group]
        if okved_prefix:
            g = g[g.okved.str.startswith(okved_prefix)]
        members = set(g.index)
        flows = P[(P.PAYER == company_id) & P.BENEFICIARY.isin(members) | (P.BENEFICIARY == company_id) & P.PAYER.isin(members)]
        with_me = flows.groupby(flows.PAYER.where(flows.PAYER != company_id, flows.BENEFICIARY)).PAY_AMT.sum()
        g = g.assign(flow_with_company=g.index.map(with_me).map(lambda v: money(v) if pd.notna(v) else ''),
                     revenue=g.REVENUE_COMPANY.map(money)).sort_values('REVENUE_COMPANY', ascending=False)
        return (f'group {group}: {len(g)} members shown\n' +
                as_text(g[['SYNTHETIC_U', 'okved', 'MAINOKVED_NAME', 'REGION', 'revenue', 'CL_TYPE', 'OKFS_NAME', 'flow_with_company']], top))

    def person_links(self, company_id: str) -> str:
        """Individuals linked to the company and the other companies they link to."""
        U, F = self.companies, self.persons
        p = F[F.SYNTHETIC_U == company_id]
        if not len(p):
            return 'no linked persons'
        other = F[F.SYNTHETIC_F.isin(p.SYNTHETIC_F) & (F.SYNTHETIC_U != company_id)]
        other = other.assign(okved=other.SYNTHETIC_U.map(U.okved), name=other.SYNTHETIC_U.map(U.MAINOKVED_NAME),
                             revenue=other.SYNTHETIC_U.map(U.REVENUE_COMPANY), group=other.SYNTHETIC_U.map(U.GROUP_ID))
        return f'persons:\n{as_text(p)}\nother companies of these persons:\n{as_text(other)}'

    def peer_stats(self, okved: str, region: str | None = None, workers: str | None = None) -> str:
        """Peers by OKVED prefix (optionally region and headcount bucket): revenue quantiles, product and FTS rates,
        headcount mix, incoming payments / revenue (the payment-coverage baseline)."""
        U, P = self.companies, self.payments
        g = U[U.okved.str.startswith(okved)]
        if region:
            g = g[g.REGION == region]
        if workers:
            g = g[g.AMOUNT_WORKERS_FNS_GRADATION == workers]
        if not len(g):
            return 'no peers'
        q = g.REVENUE_COMPANY.quantile([.1, .25, .5, .75, .9, .99]).map(money).to_dict()
        coverage = P.groupby('BENEFICIARY').PAY_AMT.sum().reindex(g.index).fillna(0) / g.REVENUE_COMPANY
        rates = g[[c for c in CARD if c.endswith('_MARK') or c == 'FTS_FLAG']].mean().round(3).to_dict()
        return (f'{len(g)} peers; revenue quantiles {q}\nproduct/flag rates {rates}\n'
                f'workers {g.AMOUNT_WORKERS_FNS_GRADATION.value_counts(normalize=True).round(2).to_dict()}\n'
                f'incoming payments / revenue: median {coverage.median():.3f}, p90 {coverage.quantile(.9):.3f}')

    def table_query(self, table: Literal['companies', 'payments', 'self_payments', 'persons'],
                    filters: list[list[Any]] | None = None, group_by: list[str] | None = None,
                    agg: dict[str, str] | None = None, sort_by: str | None = None, top: int = 20) -> str:
        """Generic read-only query. filters: [[column, op, value], ...], op in == != > < >= <= in startswith contains.
        group_by: [columns]; agg: {column: sum|mean|median|count|nunique|min|max}; sort_by sorts descending.
        Example: table='payments', filters=[["PAYER", "==", "U_1"]], group_by=["BENEFICIARY"], agg={"PAY_AMT": "sum"},
        sort_by="PAY_AMT". Column names are upper case, as in the table headers."""
        tables = {'companies': self.companies.reset_index(drop=True), 'payments': self.payments,
                  'self_payments': self.self_payments.reset_index(), 'persons': self.persons}
        if table not in tables:
            return f'unknown table; use one of {list(tables)}'
        df = tables[table]
        for f in filters or []:
            if len(f) != 3 or f[0] not in df.columns or f[1] not in OPS:
                return f'bad filter {f}; columns: {list(df.columns)}'
            df = df[OPS[f[1]](df[f[0]], f[2])]
        if group_by:
            if any(c not in df.columns for c in group_by) or any(v not in AGG for v in (agg or {}).values()):
                return 'bad group_by/agg'
            df = df.groupby(group_by).agg(agg or {df.columns[0]: 'count'}).reset_index()
        if sort_by and sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=False)
        return f'{len(df)} rows\n' + as_text(df, min(int(top), MAX_ROWS))

    def _payment_profile(self, company_id: str) -> str:
        U, P = self.companies, self.payments
        out, inc = P[P.PAYER == company_id], P[P.BENEFICIARY == company_id]
        if not len(out) and not len(inc):
            return f'{company_id}: unknown id'
        lines = [f'{company_id}: NOT in companies table (no card). Payments: out {len(out)} edges {money(out.PAY_AMT.sum())}, '
                 f'in {len(inc)} edges {money(inc.PAY_AMT.sum())}']
        for label, t, cp in [('pays to', out, 'BENEFICIARY'), ('paid by', inc, 'PAYER')]:
            k = t.assign(o=t[cp].map(U.MAINOKVED_NAME)).groupby('o').PAY_AMT.sum().nlargest(6)
            if len(k):
                lines.append(f'{label} (known okved): ' + '; '.join(f'{o} {money(v)}' for o, v in k.items()))
        return '\n'.join(lines)
