"""Shared data access and text helpers for all pipeline stages. Case tables live in data/ (gitignored)."""
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('ALFA_DATA', ROOT / 'data'))


# data-quality flags: flag -> (level, why). Values are kept; flagged companies should not be shown on the demo.
DQ = {'merged_id': ('hard', 'общий ID U_428697 у 12 разных компаний: платежи не атрибутируются — на демо не использовать'),
      'merged_id_partner': ('soft', 'не меньше 10% входящих или исходящих платежей компании — с общим ID U_428697; '
                                    'отнести их к одной из 12 компаний нельзя, рёбра удалены — платёжный профиль неполный'),
      'revenue_1000': ('soft', 'выручка ровно 1 000 ₽ — по словам организаторов, вероятно техническая заглушка'),
      'revenue_below_payroll': ('soft', 'выручка меньше минимального ФОТ для штата (МРОТ × 12 × нижняя граница градации)'),
      'products_incomplete': ('soft', 'действующий клиент без отметки РКО, но с платежами в графе — флаги продуктов неполные'),
      'no_payments_visible': ('soft', 'действующий клиент с РКО, но без единого платежа в выборке — платёжного профиля нет'),
      'status_vs_payments': ('soft', 'статус клиента противоречит платежам (переводы себе без счёта или платежи между '
                                     'не-клиентами) — статус и платежи сняты на разные даты'),
      'large_young': ('soft', 'выручка от 15 млрд ₽ при возрасте 0–3 года — вероятно новое или транзитное юрлицо'),
      'large_tiny_staff': ('soft', 'выручка от 15 млрд ₽ при штате до 15 чел. — вероятно транзитное юрлицо'),
      'staff_sources_disagree': ('soft', 'численность по СПАРК и по ФНС в разных градациях'),
      'flow_exceeds_revenue': ('soft', 'есть платёж с компанией другой группы, больший годовой выручки обеих сторон в 10+ раз'),
      'self_transfers_vs_revenue': ('soft', 'переводы между своими счетами больше годовой выручки в 1000+ раз'),
      'fts_inconsistent': ('soft', 'флаг ФТС и список стран противоречат друг другу'),
      'products_vs_status': ('soft', 'потенциальный клиент (без счёта), но с отметками о продуктах банка'),
      'geo_mismatch': ('soft', 'город регистрации не относится к указанному региону'),
      'stage_vs_age': ('soft', 'стадия «Рождение» у компании старше 7 лет')}
GEO_HOMONYMS = {('г. Благовещенск', 'Башкортостан (Республика)'), ('г. Киров', 'Калужская область'),
                ('г. Троицк', 'Челябинская область')}  # real towns, not errors
MIN_STAFF = {'16-100': 16, '101-250': 101, 'Более 251': 251}  # lower bound of the FNS headcount bucket
MIN_WAGE_YEAR = 12 * 22_440  # МРОТ 2025, ₽ per employee per year


def merged_partners(P, ids, share=0.1):
    """Counterparties with >= share of their in- or out-flow tied to the merged ids (edges cannot be split: see data_quality.md)."""
    P = P[P.PAYER != P.BENEFICIARY]
    out = []
    for me, cp in (('PAYER', 'BENEFICIARY'), ('BENEFICIARY', 'PAYER')):
        tot = P.groupby(me).PAY_AMT.sum()
        tied = P[P[cp].isin(ids)].groupby(me).PAY_AMT.sum()
        out += list(tied.index[tied / tot[tied.index] >= share])
    return set(out) - set(ids)


def dq_flags(U, merged, partners, P):
    """Comma-joined DQ keys per row ('' = clean). P = raw payments (self-loops included)."""
    rev, cl, ids = U.REVENUE_COMPANY, U.CL_TYPE, U.SYNTHETIC_U
    paying = ids.isin(set(P.PAYER) | set(P.BENEFICIARY))
    status = dict(zip(ids, cl))
    nc = P[(P.PAYER != P.BENEFICIARY) & P.PAYER.map(status).isin(['PROSPECT', 'CLOSE_CLIENT']) & P.BENEFICIARY.map(status).isin(['PROSPECT', 'CLOSE_CLIENT'])]
    selfp = set(P.PAYER[P.PAYER == P.BENEFICIARY])
    large = rev >= 15e9
    grp = dict(zip(ids, U.GROUP_ID))
    rv = dict(zip(ids, rev))
    big = P[(P.PAY_AMT > 10 * P.PAYER.map(rv)) & (P.PAY_AMT > 10 * P.BENEFICIARY.map(rv)) & (P.PAYER.map(grp) != P.BENEFICIARY.map(grp))]
    own = P[P.PAYER == P.BENEFICIARY].set_index('PAYER').PAY_AMT
    cr = U.groupby(['CITY', 'REGION']).SYNTHETIC_U.transform('size') / U.groupby('CITY').SYNTHETIC_U.transform('size')
    pairs = pd.Series(list(zip(U.CITY, U.REGION)), index=U.index)
    products = U[[c for c in U.columns if c.endswith('_MARK') and c != 'NEGATIVEMARK']].sum(axis=1) > 0
    hits = {'merged_id': merged, 'merged_id_partner': U.SYNTHETIC_U.isin(partners), 'revenue_1000': rev == 1000,
            'revenue_below_payroll': (rev != 1000) & (rev < U.AMOUNT_WORKERS_FNS_GRADATION.map(MIN_STAFF) * MIN_WAGE_YEAR),
            'products_incomplete': (cl == 'CLIENT') & (U.CL_PRODUCT_RKO_MARK == 0) & paying,
            'no_payments_visible': (cl == 'CLIENT') & (U.CL_PRODUCT_RKO_MARK == 1) & ~paying,
            'status_vs_payments': ((cl == 'PROSPECT') & ids.isin(selfp)) | ids.isin(set(nc.PAYER) | set(nc.BENEFICIARY)),
            'large_young': large & (U.COMPANY_AGE_YEARS_GRADATION == '0-3'),
            'large_tiny_staff': large & (U.AMOUNT_WORKERS_FNS_GRADATION == 'Менее 15'),
            'staff_sources_disagree': U.AMOUNT_WORKERS_SPARK_GRADATION.notna() &
                                      (U.AMOUNT_WORKERS_SPARK_GRADATION != U.AMOUNT_WORKERS_FNS_GRADATION),
            'flow_exceeds_revenue': ids.isin(set(big.PAYER) | set(big.BENEFICIARY)),
            'self_transfers_vs_revenue': ids.map(own).fillna(0) > 1000 * rev,
            'fts_inconsistent': (U.FTS_FLAG == 1) != U.FTS_COUNTRY_NAME.notna(),
            'products_vs_status': (cl == 'PROSPECT') & products,
            'geo_mismatch': (cr < 0.05) & (U.groupby('CITY').SYNTHETIC_U.transform('size') >= 50) & ~pairs.isin(GEO_HOMONYMS),
            'stage_vs_age': (U.ECO_MARK_LABEL == 'Рождение') & U.COMPANY_AGE_YEARS_GRADATION.isin(['8-15', '16-25'])}
    return pd.DataFrame(hits).apply(lambda r: ','.join(k for k, v in r.items() if v), axis=1)


def demo_ok(U):
    """Companies safe to show on the demo: no data-quality flag at all."""
    return U.DQ_FLAGS == ''


@lru_cache(None)
def load():
    """Tables shared by the fact builder and the agent tools: U (companies, indexed), P (edges w/o self-loops), S, F.
    Data-quality findings behind the cleaning: stage2_company_facts/deterministic_facts/data_quality.md."""
    U = pd.read_csv(DATA / 'df_Uid_data_hack.csv', sep=';', dtype={'MAINOKVED_CODE': str}, low_memory=False)
    dup = U.SYNTHETIC_U.duplicated(keep=False)
    merged = set(U.SYNTHETIC_U[dup])  # U_428697: 12 filled rows of different companies share one id (anonymisation)
    P = pd.read_csv(DATA / 'df_Uid_payment_hack.csv', sep=';')
    U['DQ_FLAGS'] = dq_flags(U, dup, merged_partners(P, merged), P)
    U.loc[dup, 'SYNTHETIC_U'] += '_' + (U[dup].groupby('SYNTHETIC_U').cumcount() + 1).astype(str)  # keep them in their groups
    okv = U.MAINOKVED_CODE.str.strip()
    U['okved'] = okv.where(~okv.str.match(r'^\d\d\.\d$'), okv + '0')  # '41.2' and '41.20' are the same class
    U['cls'], U['okdiv'] = U.okved.str[:5], U.okved.str[:2]
    U = U.set_index('SYNTHETIC_U', drop=False)
    P = P[~P.PAYER.isin(merged) & ~P.BENEFICIARY.isin(merged)]  # edges of 12 firms, not attributable to any (all dup pairs)
    P = P.drop_duplicates(['PAYER', 'BENEFICIARY'])  # no-op on the case data, guard for other inputs
    S = P[P.PAYER == P.BENEFICIARY].set_index('PAYER')  # same company on both sides
    P = P[P.PAYER != P.BENEFICIARY].reset_index(drop=True)
    F = pd.read_csv(DATA / 'df_Fid_Uid_link_hack.csv', sep=';').dropna(subset=['SYNTHETIC_U'])
    return U, P, S, F


def money(v):
    if pd.isna(v):
        return 'н/д'
    return f'{v / 1e9:.1f} млрд ₽' if v >= 1e9 else f'{v / 1e6:.0f} млн ₽' if v >= 1e6 else f'{v / 1e3:.0f} тыс. ₽'


def pct(v):
    return f'{v:.0%}' if v >= 0.01 else '<1%'


def plural(n, one, few, many):
    n = int(n)
    k = n % 100 if n % 100 in range(11, 15) else n % 10
    return one if k == 1 else few if 2 <= k <= 4 else many


def comp(n):
    return f'{int(n)} {plural(n, "компания", "компании", "компаний")}'


def share_of_revenue(x):
    return f'в {x:,.0f} раз больше годовой выручки'.replace(',', ' ') if x >= 10 else f'{pct(x)} годовой выручки'


def of_comp(n):  # genitive after "из"
    return f'{int(n)} {plural(n, "компании", "компаний", "компаний")}'


def countries(s):
    # FTS stores upper case; keep short acronyms (ОАЭ, США, КНДР) upper
    return [c.strip() if len(c.strip()) <= 4 else c.strip().title() for c in s.split(',') if c.strip()]
