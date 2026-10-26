"""Payment analytics over the same cleaned case tables as company facts. No agents or search."""
import math
import threading
from collections import OrderedDict

import pandas as pd

from backend.cabinet import company_name
from common.data import DQ, load


class PaymentAnalytics:
    def __init__(self, loader=load, capacity=128):
        self.loader, self.capacity = loader, capacity
        self.lock = threading.RLock()
        self.tables = None
        self.cache = OrderedDict()

    def _ensure(self):
        if self.tables is not None:
            return
        companies, payments, own, _ = self.loader()
        # Build adjacency once rather than filtering the entire payment CSV for every page request.
        outgoing = payments.groupby('PAYER', sort=False).indices
        incoming = payments.groupby('BENEFICIARY', sort=False).indices
        self.tables = companies, payments, own, outgoing, incoming

    def _company(self, company_id):
        with self.lock:
            self._ensure()
            if company_id in self.cache:
                self.cache.move_to_end(company_id)
                return self.cache[company_id]
            companies, payments, own, outgoing, incoming = self.tables
            if company_id not in companies.index:
                raise KeyError(company_id)
            company = companies.loc[company_id]
            result = dict(company_id=company_id, name=company_name(company_id, _text(company.get('MAINOKVED_NAME'))),
                          directions={})
            for direction, index, counterparty in [('out', outgoing, 'BENEFICIARY'), ('in', incoming, 'PAYER')]:
                edges = payments.iloc[index.get(company_id, [])]
                result['directions'][direction] = self._direction(edges, companies, counterparty, company.get('GROUP_ID'))
            self_edges = own.loc[[company_id]] if company_id in own.index else own.iloc[:0]
            result['self_transfers'] = _totals(self_edges)
            flags = _text(company.get('DQ_FLAGS')).split(',')
            result['data_notes'] = [DQ[flag][1] for flag in flags if flag in DQ]
            # One record per relationship, including companies present in both directions.
            relationships = {}
            for direction, flow in result['directions'].items():
                for row in flow['counterparties']:
                    record = relationships.setdefault(row['id'], {key:row[key] for key in
                        ('id','name','industry','industry_code','region','same_group','group_known')})
                    record[direction] = {key:row[key] for key in ('amount','count','average','share')}
            for record in relationships.values():
                record['relationship'] = 'both' if 'in' in record and 'out' in record else 'in' if 'in' in record else 'out'
                for direction in ('in','out'):
                    record.setdefault(direction, dict(amount=0.0,count=0,average=None,share=0.0))
                record['amount'] = record['in']['amount'] + record['out']['amount']
                record['count'] = record['in']['count'] + record['out']['count']
            result['relationships'] = sorted(relationships.values(),key=lambda row:(-row['amount'],row['id']))
            self.cache[company_id] = result
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
            return result

    @staticmethod
    def _direction(edges, companies, cp, own_group):
        edges = edges.assign(PAY_AMT=pd.to_numeric(edges.PAY_AMT,errors='coerce'),
                             PAY_CNT=pd.to_numeric(edges.PAY_CNT,errors='coerce'))
        totals = _totals(edges)
        if edges.empty:
            return dict(**totals, counterparties=[], industries=[], regions=[], top1_share=0.0, top5_share=0.0,
                        counterparties_count=0, within_group_share=0.0, known_industry_share=0.0, known_region_share=0.0)
        grouped = edges.groupby(cp, sort=False).agg(amount=('PAY_AMT','sum'), count=('PAY_CNT','sum'))
        metadata = companies.reindex(grouped.index)
        rows = []
        for identifier, numbers in grouped.iterrows():
            meta = metadata.loc[identifier]
            industry = _text(meta.get('MAINOKVED_NAME'))
            group = meta.get('GROUP_ID')
            rows.append(dict(id=str(identifier), name=company_name(str(identifier), industry),
                             industry=industry or None, industry_code=_text(meta.get('okved')) or None,
                             region=_text(meta.get('REGION')) or None,
                             amount=float(numbers.amount), count=int(numbers['count']),
                             average=float(numbers.amount / numbers['count']) if numbers['count'] else None,
                             share=float(numbers.amount / totals['amount']) if totals['amount'] else 0.0,
                             group_known=bool(pd.notna(own_group) and pd.notna(group)),
                             same_group=bool(pd.notna(own_group) and pd.notna(group) and group==own_group)))
        rows.sort(key=lambda row:(-row['amount'], row['id']))
        return dict(**totals, counterparties=rows, counterparties_count=len(rows),
                    top1_share=rows[0]['share'], top5_share=sum(row['share'] for row in rows[:5]),
                    within_group_share=sum(row['share'] for row in rows if row['same_group']),
                    known_industry_share=sum(row['share'] for row in rows if row['industry'] or row['industry_code']),
                    known_region_share=sum(row['share'] for row in rows if row['region']),
                    industries=_distribution(rows,'industry', 'Другие отрасли'),
                    regions=_distribution(rows,'region', 'Другие регионы'))

    def report(self, company_id, direction='out', query='', sort='amount', offset=0, limit=15):
        company = self._company(company_id)
        selected = company['directions'][direction]
        needle = query.strip().casefold()
        rows = [row for row in selected['counterparties'] if not needle or needle in ' '.join(
            str(row.get(key) or '') for key in ('id','name','industry','industry_code','region')).casefold()]
        if sort == 'count':
            rows.sort(key=lambda row:(-row['count'], -row['amount'], row['id']))
        return dict(company_id=company_id, name=company['name'], direction=direction,
                    summary={key:{field:values[field] for field in ('amount','count','average','counterparties_count')}
                             for key,values in company['directions'].items()},
                    flows={name:{key:value for key,value in flow.items() if key!='counterparties'}
                           for name,flow in company['directions'].items()},
                    counterparties=rows[offset:offset+limit], total=len(rows), offset=offset, limit=limit,
                    self_transfers=company['self_transfers'], data_notes=company['data_notes'],
                    period=None, currency='RUB', source='company_payment_csv')

    def counterparties(self, company_id, direction='all', query='', group='all', sort='amount', offset=0, limit=15):
        company = self._company(company_id)
        all_rows = company['relationships']
        needle = query.strip().casefold()
        rows = [row for row in all_rows
                if (direction=='all' or row['relationship']==direction or direction in ('in','out') and row['relationship']=='both')
                and (group=='all' or row['same_group'])
                and (not needle or needle in ' '.join(str(row.get(key) or '') for key in
                     ('id','name','industry','industry_code','region')).casefold())]
        if sort=='name':
            rows.sort(key=lambda row:(row['name'].casefold(),row['id']))
        elif sort in ('in','out'):
            rows.sort(key=lambda row:(-row[sort]['amount'],row['id']))
        elif sort=='count':
            rows.sort(key=lambda row:(-row['count'],-row['amount'],row['id']))
        return dict(company_id=company_id, name=company['name'], items=rows[offset:offset+limit],
                    total=len(rows), offset=offset, limit=limit,
                    summary=dict(total=len(all_rows), incoming=sum(row['relationship'] in ('in','both') for row in all_rows),
                                 outgoing=sum(row['relationship'] in ('out','both') for row in all_rows),
                                 both=sum(row['relationship']=='both' for row in all_rows),
                                 same_group=sum(row['same_group'] for row in all_rows)),
                    currency='RUB',period=None)

    def counterparty(self, company_id, counterparty_id):
        company = self._company(company_id)
        relationship = next((row for row in company['relationships'] if row['id']==counterparty_id),None)
        if relationship is None:
            raise KeyError(counterparty_id)
        companies = self.tables[0]
        available = counterparty_id in companies.index
        meta = companies.loc[counterparty_id] if available else {}
        profile = dict(available=available, city=_text(meta.get('CITY')) or None,
                       group_id=_text(meta.get('GROUP_ID')) or None,
                       revenue=_number(meta.get('REVENUE_COMPANY')),
                       age=_text(meta.get('COMPANY_AGE_YEARS_GRADATION')) or None,
                       staff=_text(meta.get('AMOUNT_WORKERS_FNS_GRADATION')) or None)
        flags = _text(meta.get('DQ_FLAGS')).split(',')
        return dict(company_id=company_id, counterparty=relationship, profile=profile,
                    data_notes=[DQ[flag][1] for flag in flags if flag in DQ],currency='RUB',period=None)


def _text(value):
    return '' if pd.isna(value) else str(value).strip()


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _totals(edges):
    if edges.empty:
        return dict(amount=0.0, count=0, average=None)
    amounts = pd.to_numeric(edges.PAY_AMT, errors='coerce')
    counts = pd.to_numeric(edges.PAY_CNT, errors='coerce')
    if (amounts.isna().any() or counts.isna().any() or (amounts < 0).any() or (counts < 0).any()
            or (counts % 1 != 0).any() or not all(math.isfinite(float(value)) for value in [*amounts, *counts])):
        raise ValueError('Invalid payment amounts or counts')
    amount, count = float(amounts.sum()), int(counts.sum())
    return dict(amount=amount, count=count, average=amount/count if count else None)


def _distribution(rows, field, other_label):
    grouped = {}
    for row in rows:
        if field == 'industry':
            key = (row['industry_code'] or '', row['industry'] or '')
            label = row['industry'] or (f'ОКВЭД {row["industry_code"]}' if row['industry_code'] else 'Отрасль неизвестна')
            unknown = not any(key)
        else:
            key, label = row[field] or '', row[field] or 'Регион неизвестен'
            unknown = not key
        item = grouped.setdefault(key, dict(label=label, amount=0.0, count=0, counterparties_count=0, share=0.0, unknown=unknown))
        for metric in ('amount','count','share'):
            item[metric] += row[metric]
        item['counterparties_count'] += 1
    ordered = sorted(grouped.values(),key=lambda item:-item['amount'])
    known = [item for item in ordered if not item['unknown']]
    result = known[:6] + [item for item in ordered if item['unknown']]
    if len(known)>6:
        result.append(dict(label=other_label, unknown=False,
                           **{metric:sum(item[metric] for item in known[6:]) for metric in ('amount','count','share','counterparties_count')}))
    return sorted(result,key=lambda item:-item['amount'])
