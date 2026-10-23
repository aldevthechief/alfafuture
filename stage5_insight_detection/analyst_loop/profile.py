"""Company profile for the analyst loop: numbers from the case tables and channel exposures computed by fixed rules.

`exposure[topic]` in 0..1 says how strongly a macro or sector channel reaches this company; `reasons[topic]` lists the
facts behind it, so the impact estimate and the UI can show why. Rules work for any OKVED (a new code falls back to its
division and then to the base weights), nothing is tuned to particular companies. Pure code, no LLM.
"""
from dataclasses import dataclass, field

from stage5_insight_detection.analyst_loop.facts import Fact, money_value

BASE = {'инфляция': 0.3, 'бюджет_и_налоги': 0.4, 'ставка': 0.3, 'кредит': 0.2, 'рынок_труда': 0.3, 'курс': 0.2}
# OKVED division (2 digits) -> channel weights; a 5-char class may refine it (precious metals)
SECTOR = {
    **{d: {'строительство': 1.0, 'ставка': 0.9, 'кредит': 0.9, 'металлургия': 0.5} for d in ('41', '42', '43')},
    '68': {'строительство': 0.8, 'ставка': 0.9, 'кредит': 0.9},
    '45': {'потребление': 0.8, 'курс': 0.6, 'кредит': 0.6, 'инфляция': 0.6},
    '46': {'потребление': 0.5, 'курс': 0.6, 'инфляция': 0.6, 'внешняя_торговля': 0.5},
    '47': {'потребление': 1.0, 'инфляция': 0.8, 'курс': 0.5, 'рынок_труда': 0.5},
    **{d: {'потребление': 0.8, 'инфляция': 0.8, 'энергия_и_сырьё': 0.4} for d in ('10', '11')},
    '23': {'строительство': 0.9, 'энергия_и_сырьё': 0.6},
    '24': {'металлургия': 1.0, 'курс': 0.7, 'энергия_и_сырьё': 0.5, 'внешняя_торговля': 0.6},
    **{d: {'металлургия': 0.6, 'промышленность': 0.8, 'ставка': 0.5, 'инвестиции': 0.6} for d in ('25', '27', '28', '29', '30')},
    **{d: {'энергия_и_сырьё': 1.0, 'курс': 0.8, 'бюджет_и_налоги': 0.6, 'внешняя_торговля': 0.7} for d in ('05', '06', '08', '09', '19')},
    '07': {'металлургия': 0.7, 'курс': 0.7, 'внешняя_торговля': 0.6},
    '35': {'энергия_и_сырьё': 0.8, 'бюджет_и_налоги': 0.5},
    **{d: {'энергия_и_сырьё': 0.7, 'внешняя_торговля': 0.5, 'потребление': 0.3} for d in ('49', '50', '51', '52', '53')},
    **{d: {'потребление': 0.9, 'инфляция': 0.6, 'рынок_труда': 0.6} for d in ('55', '56')},
    **{d: {'рынок_труда': 0.7, 'инвестиции': 0.5} for d in ('62', '63')},
    **{d: {'ставка': 1.0, 'облигации': 0.9, 'кредит': 0.9} for d in ('64', '65', '66')},
    **{d: {'курс': 0.5, 'внешняя_торговля': 0.6, 'бюджет_и_налоги': 0.4, 'потребление': 0.4} for d in ('01', '02', '03')},
    **{d: {'рынок_труда': 0.6, 'инвестиции': 0.6, 'бюджет_и_налоги': 0.4} for d in ('69', '70', '71', '72', '73', '74')},
    **{d: {'потребление': 0.6, 'рынок_труда': 0.6, 'инфляция': 0.5, 'бюджет_и_налоги': 0.5} for d in ('85', '86', '87', '88')},
    **{d: {'потребление': 0.5, 'инвестиции': 0.5, 'курс': 0.4} for d in ('58', '59', '60', '61')},
}
# 5-char classes whose business differs from their division: fuel wholesale and retail sell oil products, not goods at large
CLASS = {'24.41': {'золото': 1.0}, '07.29': {'золото': 0.9}, '32.12': {'золото': 0.8},
         '46.71': {'энергия_и_сырьё': 1.0, 'курс': 0.7}, '47.30': {'энергия_и_сырьё': 0.9, 'потребление': 0.8},
         '35.11': {'энергия_и_сырьё': 0.9}, '35.21': {'энергия_и_сырьё': 0.9}, '68.10': {'строительство': 0.9, 'кредит': 1.0},
         '64.91': {'кредит': 1.0, 'ставка': 1.0}, '41.20': {'строительство': 1.0, 'кредит': 1.0},
         '42.11': {'строительство': 1.0, 'бюджет_и_налоги': 0.7}, '24.10': {'металлургия': 1.0, 'внешняя_торговля': 0.8},
         '21.20': {'курс': 0.7, 'внешняя_торговля': 0.6}}
NAME_WEIGHT = 0.8  # a topic named by the OKVED name itself («топливом», «строительство», «металлов»): works for any code
PRODUCT_LABELS = {'loan': 'кредитные продукты', 'leasing': 'лизинг', 'deposit': 'депозиты', 'guarantee': 'банковские гарантии',
                  'factoring': 'факторинг', 'rko': 'расчётно-кассовое обслуживание', 'salary': 'зарплатный проект'}
PRODUCT_COLUMNS = {'CL_PRODUCT_LOA_MARK': 'loan', 'CL_PRODUCT_LEA_MARK': 'leasing', 'CL_PRODUCT_DEP_MARK': 'deposit',
                   'CL_PRODUCT_GAR_MARK': 'guarantee', 'FACTORING_MARK': 'factoring', 'CL_PRODUCT_RKO_MARK': 'rko',
                   'CL_PRODUCT_ZPP_MARK': 'salary'}
LARGE = 15e9


@dataclass
class CompanyProfile:
    id: str
    okved: str
    okved_name: str
    region: str
    revenue: float | None  # rub, 2025; None when the table has the 1000 placeholder or nothing
    group_revenue: float | None = None
    products: list[str] = field(default_factory=list)  # keys of PRODUCT_LABELS
    fts_countries: list[str] = field(default_factory=list)
    fts_flag: bool = False
    pay_out: dict[str, float] = field(default_factory=dict)  # supplier OKVED division -> share of outgoing payments
    pay_in: dict[str, float] = field(default_factory=dict)  # customer OKVED division -> share of incoming payments
    pay_out_total: float = 0.0
    pay_in_total: float = 0.0
    exposure: dict[str, float] = field(default_factory=dict)
    reasons: dict[str, list[str]] = field(default_factory=dict)
    facts: dict[str, Fact] = field(default_factory=dict)

    def __post_init__(self):
        if not self.facts:
            self.facts = self._facts()
        if not self.exposure:
            self.exposure, self.reasons = self._exposure()

    @property
    def okdiv(self) -> str:
        return self.okved[:2]

    def card(self) -> str:
        """Short card for prompts (classifier, assessment)."""
        top = ', '.join(f'ОКВЭД {k} {v:.0%}' for k, v in sorted(self.pay_out.items(), key=lambda kv: -kv[1])[:3])
        return '\n'.join(filter(None, [
            f'ОКВЭД {self.okved}: {self.okved_name}; регион: {self.region}',
            f'Выручка 2025: {self.facts["fact:revenue"].text()}' if 'fact:revenue' in self.facts else '',
            'Продукты Альфа-Банка: ' + ', '.join(PRODUCT_LABELS[p] for p in self.products) if self.products else '',
            'Внешнеторговые страны (ФТС, без направления): ' + ', '.join(self.fts_countries) if self.fts_countries else '',
            f'Крупнейшие отрасли поставщиков по платежам: {top}' if top else '']))

    def _facts(self) -> dict[str, Fact]:
        out = [Fact('fact:okved', 'Основной ОКВЭД', f'{self.okved} {self.okved_name}', source='df_Uid_data_hack.csv'),
               Fact('fact:region', 'Регион регистрации', self.region, source='df_Uid_data_hack.csv')]
        if self.revenue:
            v, u = money_value(self.revenue)
            out.append(Fact('fact:revenue', 'Выручка за 2025 год', v, u, 'REVENUE_COMPANY', 'df_Uid_data_hack.csv'))
        if self.group_revenue:
            v, u = money_value(self.group_revenue)
            out.append(Fact('fact:group_revenue', 'Суммарная выручка группы', v, u, 'sum(REVENUE_COMPANY) по GROUP_ID',
                            'df_Uid_data_hack.csv'))
        for p in self.products:
            out.append(Fact(f'fact:product_{p}', 'Продукт Альфа-Банка', PRODUCT_LABELS[p], source='df_Uid_data_hack.csv'))
        if self.fts_countries or self.fts_flag:
            out.append(Fact('fact:fts', 'Внешнеторговые операции (ФТС, направление не указано)',
                            ', '.join(self.fts_countries) or 'есть', source='df_Uid_data_hack.csv'))
        for side, shares, total in (('out', self.pay_out, self.pay_out_total), ('in', self.pay_in, self.pay_in_total)):
            who = 'поставщикам' if side == 'out' else 'от покупателей'
            for div, share in sorted(shares.items(), key=lambda kv: -kv[1]):
                out.append(Fact(f'fact:pay_{side}_{div}', f'Доля платежей {who} с ОКВЭД {div}.*', round(share * 100, 1), '%',
                                f'sum(PAY_AMT {who} ОКВЭД {div}) / sum(PAY_AMT {who}); всего {money_value(total)[0]} '
                                f'{money_value(total)[1]}', 'df_Uid_payment_hack.csv'))
        return {f.id: f for f in out}

    def _exposure(self) -> tuple[dict[str, float], dict[str, list[str]]]:
        exp, why = dict(BASE), {k: ['fact:okved'] for k in BASE}

        def bump(topic, w, fact):
            if w > exp.get(topic, 0):
                exp[topic] = w
            why.setdefault(topic, []).append(fact)

        for topic, w in {**SECTOR.get(self.okdiv, {}), **CLASS.get(self.okved[:5], {})}.items():
            bump(topic, w, 'fact:okved')
        from stage5_insight_detection.bank_analytics.library import topics_of
        for topic in topics_of(self.okved_name or ''):
            bump(topic, NAME_WEIGHT, 'fact:okved')
        if 'loan' in self.products:
            bump('ставка', 1.0, 'fact:product_loan'), bump('кредит', 1.0, 'fact:product_loan')
        if 'leasing' in self.products:
            bump('ставка', 0.8, 'fact:product_leasing')
        if 'deposit' in self.products:
            bump('ставка', 0.6, 'fact:product_deposit'), bump('облигации', 0.5, 'fact:product_deposit')
        if self.fts_countries or self.fts_flag:
            bump('курс', 0.9, 'fact:fts'), bump('внешняя_торговля', 0.9, 'fact:fts')
        if (self.group_revenue or 0) >= LARGE or (self.revenue or 0) >= LARGE:
            bump('облигации', 0.6, 'fact:group_revenue' if self.group_revenue else 'fact:revenue')
        for side, shares in (('out', self.pay_out), ('in', self.pay_in)):  # chain: counterparties' sectors reach us at 0.6
            for div, share in shares.items():
                if share < 0.1:
                    continue
                for topic, w in SECTOR.get(div, {}).items():
                    bump(topic, round(min(1.0, w * 0.6 * min(1.0, share * 2)), 2), f'fact:pay_{side}_{div}')
        return exp, {k: list(dict.fromkeys(v)) for k, v in why.items()}

    @classmethod
    def from_tables(cls, company_id: str) -> 'CompanyProfile':
        from common.data import load
        from stage2_company_facts.deterministic_facts.facts import countries
        U, P, S, F = load()
        r = U.loc[company_id]
        revenue = float(r.REVENUE_COMPANY) if r.REVENUE_COMPANY and r.REVENUE_COMPANY > 1000 else None  # 1000 = placeholder
        group = U[U.GROUP_ID == r.GROUP_ID]
        div = U.okdiv.to_dict()

        def shares(df, other):
            total = float(df.PAY_AMT.sum())
            df = df.assign(d=df[other].map(div)).dropna(subset=['d'])
            return ({k: float(v / total) for k, v in df.groupby('d').PAY_AMT.sum().items()} if total else {}), total

        pay_out, out_total = shares(P[P.PAYER == company_id], 'BENEFICIARY')
        pay_in, in_total = shares(P[P.BENEFICIARY == company_id], 'PAYER')
        return cls(id=company_id, okved=r.okved, okved_name=r.MAINOKVED_NAME, region=r.REGION, revenue=revenue,
                   group_revenue=float(group.REVENUE_COMPANY.sum()) if len(group) > 1 else None,
                   products=[v for k, v in PRODUCT_COLUMNS.items() if r.get(k) == 1],
                   fts_countries=countries(r.FTS_COUNTRY_NAME) if isinstance(r.FTS_COUNTRY_NAME, str) else [],
                   fts_flag=r.FTS_FLAG == 1, pay_out=pay_out, pay_in=pay_in, pay_out_total=out_total, pay_in_total=in_total)
