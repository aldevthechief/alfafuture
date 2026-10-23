"""Deterministic company facts from the case tables -> data/companies/facts.jsonl.

One line per company: {"id", "facts": [{"cat", "text", "depth", "entities"}]}.
- only verifiable facts; missing data produces no line (no "не состоит в группе" etc.);
- FTS has a country but no direction -> never phrased as export/import;
- data-quality flags (common.data.DQ, data_quality.md) -> a 'data_quality' caveat; values are kept as given;
- depth 1 = own card, 2 = rank / group / persons / direct counterparties, 3 = second-hop counterparties.
Usage: python -X utf8 stage2_company_facts/deterministic_facts/facts.py [--ids U_1 U_2 ...]   (no --ids = all companies, ~15 min)
"""
import argparse, json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.data import DATA, DQ, comp, countries, load, money, of_comp, pct, plural, share_of_revenue

OKFS_FOREIGN = {'СОБСТВЕННОСТЬ ИНОСТРАННЫХ ЮРИДИЧЕСКИХ ЛИЦ': 'иностранный собственник — юридическое лицо',
                'СОБСТВЕННОСТЬ ИНОСТРАННЫХ ГРАЖДАН И ЛИЦ БЕЗ ГРАЖДАНСТВА': 'иностранный собственник — физическое лицо',
                'СОВМЕСТНАЯ ЧАСТНАЯ И ИНОСТРАННАЯ СОБСТВЕННОСТЬ': 'совместная российская частная и иностранная собственность',
                'СМЕШАННАЯ ИНОСТРАННАЯ СОБСТВЕННОСТЬ': 'смешанная иностранная собственность'}
OKFS_STATE = {'ФЕДЕРАЛЬНАЯ СОБСТВЕННОСТЬ', 'СОБСТВЕННОСТЬ СУБЪЕКТОВ РОССИЙСКОЙ ФЕДЕРАЦИИ', 'МУНИЦИПАЛЬНАЯ СОБСТВЕННОСТЬ',
              'СМЕШАННАЯ РОССИЙСКАЯ СОБСТВЕННОСТЬ С ДОЛЕЙ ФЕДЕРАЛЬНОЙ СОБСТВЕННОСТИ',
              'СМЕШАННАЯ РОССИЙСКАЯ СОБСТВЕННОСТЬ С ДОЛЕЙ СОБСТВЕННОСТИ СУБЪЕКТОВ РОССИЙСКОЙ ФЕДЕРАЦИИ'}
PRODUCTS = {'CL_PRODUCT_RKO_MARK': 'расчётно-кассовое обслуживание', 'CL_PRODUCT_GAR_MARK': 'банковские гарантии',
            'CL_PRODUCT_LOA_MARK': 'кредитные продукты', 'CL_PRODUCT_LEA_MARK': 'лизинг',
            'CL_PRODUCT_DEP_MARK': 'депозиты', 'CL_PRODUCT_ZPP_MARK': 'зарплатный проект', 'FACTORING_MARK': 'факторинг'}
CLTYPE = {'CLIENT': 'Действующий клиент Альфа-Банка', 'CLOSE_CLIENT': 'Бывший клиент Альфа-Банка',
          'PROSPECT': 'Потенциальный клиент Альфа-Банка'}
FTS_MAX_COUNTRIES = 3  # field is truncated: 3 countries as frequent as 1, never more


def build(ids=None, top_k=8, *, use_profiles=False):
    """Raw table facts by default; the pipeline can opt into stored profiles and their explanatory context."""
    if use_profiles:
        if ids is None:
            raise ValueError('Profile-backed facts require explicit company IDs')
        from stage2_company_facts.profile.integration import facts_from_profiles
        return facts_from_profiles(ids)
    U, P, S, F = load()
    ids = list(U.index if ids is None else ids)
    target = set(ids)
    facts = {u: [] for u in ids}

    def add(u, cat, text, depth=1, entities=()):
        facts[u].append({'cat': cat, 'text': text, 'depth': depth, 'entities': [e for e in entities if e]})

    name_of = dict(zip(U.okved, U.MAINOKVED_NAME))
    T = U.loc[ids]

    # --- own card ---
    div_fts = U.groupby('okdiv').FTS_FLAG.mean()
    med_rev = U.groupby(['okved', 'AMOUNT_WORKERS_FNS_GRADATION']).REVENUE_COMPANY.median()
    for r in T.itertuples():
        u = r.SYNTHETIC_U
        add(u, 'activity', f'Основной вид деятельности (ОКВЭД {r.okved}): {r.MAINOKVED_NAME}', 1, [r.okved, r.MAINOKVED_NAME])
        if r.cls != r.okved and r.cls in name_of:
            add(u, 'activity', f'Группировка ОКВЭД {r.cls}: {name_of[r.cls]}', 1, [r.cls, name_of[r.cls]])
        rev = r.REVENUE_COMPANY
        if pd.notna(rev):
            add(u, 'scale', f'Годовая выручка {money(rev)}', 1)
        if rev >= 15e9:
            add(u, 'scale', 'Крупный бизнес по критерию кейса (выручка свыше 15 млрд ₽)', 1)
        for flag in filter(None, r.DQ_FLAGS.split(',')):
            add(u, 'data_quality', f'Качество данных ({DQ[flag][0]}): {DQ[flag][1]}', 1)
        add(u, 'geo', f'Зарегистрирована: {r.CITY}, {r.REGION}', 1, [r.CITY, r.REGION])
        add(u, 'profile', f'Возраст компании {r.COMPANY_AGE_YEARS_GRADATION} лет', 1)
        if isinstance(r.ECO_MARK_LABEL, str):
            add(u, 'profile', f'Стадия развития бизнеса: «{r.ECO_MARK_LABEL}»', 1)
        add(u, 'profile', f'Численность по ФНС: {r.AMOUNT_WORKERS_FNS_GRADATION.lower()} чел.', 1)
        if isinstance(r.AMOUNT_WORKERS_SPARK_GRADATION, str):
            add(u, 'profile', f'Численность по СПАРК: {r.AMOUNT_WORKERS_SPARK_GRADATION.lower()} чел.', 1)
        hi = {'Менее 15': 15, '16-100': 100, '101-250': 250}.get(r.AMOUNT_WORKERS_FNS_GRADATION)
        if hi and pd.notna(rev):  # headcount bucket is capped -> this is a lower bound
            add(u, 'scale', f'Выручка на сотрудника не менее {money(rev / hi)} (штат не больше {hi} чел.)', 1)
        m = med_rev.get((r.okved, r.AMOUNT_WORKERS_FNS_GRADATION))
        if m and m > 0 and pd.notna(rev):
            add(u, 'scale', f'Выручка в {rev / m:.2g} раза от медианы компаний того же ОКВЭД с таким же штатом '
                            f'({money(m)})', 2)
        own = OKFS_FOREIGN.get(r.OKFS_NAME) or r.OKFS_NAME.capitalize()
        add(u, 'ownership', f'Форма собственности: {own}', 1)
        if r.OKFS_NAME in OKFS_FOREIGN:
            add(u, 'ownership', 'Иностранное участие в собственности', 1, ['иностранный собственник'])
        if r.OKFS_NAME in OKFS_STATE:
            add(u, 'ownership', 'Государственное или муниципальное участие в собственности', 1, ['госсобственность'])
        if r.NEGATIVEMARK == 1:
            add(u, 'ownership', 'Есть признак ключевого собственника (поле NEGATIVEMARK)', 1)
        add(u, 'bank', CLTYPE[r.CL_TYPE], 1)
        used = [v for k, v in PRODUCTS.items() if getattr(r, k) == 1]
        if used:  # flags are set for 37% of former and 3% of prospective clients -> not necessarily current products
            add(u, 'bank', ('Продукты Альфа-Банка: ' if r.CL_TYPE == 'CLIENT' else
                            'Отметки о продуктах Альфа-Банка (клиент не действующий, продукты могут быть прошлыми): ')
                + ', '.join(used), 1, used)
        if r.FTS_FLAG == 1:
            add(u, 'trade', f'Участник внешнеэкономической деятельности (есть в базе ФТС); '
                            f'в разделе ОКВЭД {r.okdiv} таких {pct(div_fts[r.okdiv])} компаний', 1, ['ВЭД'])
        if isinstance(r.FTS_COUNTRY_NAME, str):  # 206 companies have countries with FTS_FLAG=0: keep the country, it is data
            cs = countries(r.FTS_COUNTRY_NAME)
            for c in cs:  # one line per country: each is a separate retrieval key
                add(u, 'trade', f'Внешнеторговые операции с контрагентами из страны {c} (данные ФТС, направление не указано)', 1, [c])
            if len(cs) >= FTS_MAX_COUNTRIES:
                add(u, 'data_quality', f'Список стран ФТС в данных обрезан до {FTS_MAX_COUNTRIES} — у компании могут быть и другие '
                                       'страны-контрагенты', 1)

    # --- ranks (computed on all companies, emitted for targets) ---
    def ranks(keys, label, min_n=5):
        g = U.groupby(keys).REVENUE_COMPANY
        df = pd.DataFrame({'k': g.rank(ascending=False, method='min'), 'n': g.transform('count'), 's': g.transform('sum'),
                           'med': g.transform('median')}).loc[ids]
        for u, row in df.iterrows():
            if row.n < min_n or pd.isna(row.k):  # n and k count known revenues only
                continue
            key = ' / '.join(str(U.at[u, c]) for c in keys)
            v, top = U.at[u, 'REVENUE_COMPANY'], row.k / row.n
            txt = f'{int(row.k)}-е место по выручке среди {int(row.n)} компаний {label} ({key})'
            if top <= 0.25:
                txt += f', топ-{max(1, int(np.ceil(top * 100)))}%'
            elif top >= 0.5:
                txt += f', ниже медианы (выручка {v / row.med:.2g}× медианы)' if row.med else ''
            if v / row.s >= 0.01:
                txt += f'; {pct(v / row.s)} суммарной выручки этих компаний в данных'
            add(u, 'rank', txt, 2)
    ranks(['okved'], 'с тем же ОКВЭД')
    if any(len(U.at[u, 'okved']) > 5 for u in ids):
        ranks(['cls'], 'в группировке ОКВЭД')
    ranks(['okdiv'], 'в разделе ОКВЭД')
    ranks(['REGION'], 'в регионе')
    ranks(['REGION', 'okved'], 'с тем же ОКВЭД в регионе')
    pr = U.REVENUE_COMPANY.rank(pct=True, ascending=False)
    for u in ids:
        if pr[u] <= 0.10:
            add(u, 'rank', f'Входит в топ-{max(1, int(np.ceil(pr[u] * 100)))}% всех компаний датасета по выручке', 2)

    # --- group ---
    gsize = U.groupby('GROUP_ID').size()
    grev = U.groupby('GROUP_ID').REVENUE_COMPANY.sum()
    real = grev[gsize > 1]
    grank = real.rank(ascending=False, method='min')
    links = F.merge(F, on='SYNTHETIC_F').query('SYNTHETIC_U_x != SYNTHETIC_U_y')
    for gid in T.GROUP_ID.unique():
        g = U[U.GROUP_ID == gid]
        if len(g) < 2:
            continue  # one-company "group" -> no facts
        t = g.REVENUE_COMPANY.sum(min_count=1)  # all unknown -> NaN, not 0
        acts = g.groupby(['okved', 'MAINOKVED_NAME']).REVENUE_COMPANY.agg(['size', 'sum', 'count'])
        acts = acts.assign(sum=acts['sum'].where(acts['count'] > 0)).sort_values('sum', ascending=False)
        divs = g.okdiv.nunique()
        regs = g.REGION.value_counts()
        foreign = g[g.OKFS_NAME.isin(OKFS_FOREIGN)]
        cs = sorted({c for s in g.FTS_COUNTRY_NAME.dropna() for c in countries(s)})
        prod = {v: int(g[k].sum()) for k, v in PRODUCTS.items() if g[k].sum()}
        gm = set(g.index)
        ge = P[P.PAYER.isin(gm) | P.BENEFICIARY.isin(gm)]
        inner = ge[ge.PAYER.isin(gm) & ge.BENEFICIARY.isin(gm)]
        gl = links[links.SYNTHETIC_U_x.isin(gm)]
        gr = g.REVENUE_COMPANY.rank(ascending=False, method='min')
        unknown = g.REVENUE_COMPANY.isna().sum()
        for u in (x for x in g.index if x in target):
            r = U.loc[u]
            add(u, 'group', f'Входит в группу из {of_comp(len(g))} с суммарной выручкой {money(t)}' +
                (f' (выручка неизвестна у {unknown} из них)' if unknown else ''), 2)
            k = gr[u]
            if pd.notna(k):
                add(u, 'group', f'Доля в выручке группы {pct(r.REVENUE_COMPANY / t)}; '
                                f'{"крупнейшая компания группы" if k == 1 else f"{int(k)}-я по выручке в группе"}', 2)
            if t >= 15e9 and not r.REVENUE_COMPANY >= 15e9:
                add(u, 'group', 'Группа — крупный бизнес по критерию кейса (свыше 15 млрд ₽), сама компания нет', 2)
            add(u, 'group', f'Группа — {int(grank[gid])}-я по суммарной выручке среди {len(real)} групп в данных', 2)
            add(u, 'group', f'Компании группы работают в {divs} {plural(divs, "разделе", "разделах", "разделах")} ОКВЭД', 2)
            same = g[(g.okved == r.okved) & (g.index != u)]
            if len(same):
                add(u, 'group', f'В группе есть ещё {comp(len(same))} с тем же ОКВЭД на {money(same.REVENUE_COMPANY.sum(min_count=1))}', 2)
            for (code, nm), row in acts.head(top_k).iterrows():
                if code == r.okved:
                    continue
                add(u, 'group_activity', f'В группе {comp(row["size"])} с ОКВЭД {code} «{nm}», выручка {money(row["sum"])}',
                    2, [code, nm])
            if len(regs) > 1:
                add(u, 'group_geo', f'Компании группы зарегистрированы в {len(regs)} регионах: ' +
                    ', '.join(f'{k} ({v})' for k, v in regs.head(top_k).items()), 2, list(regs.index[:top_k]))
            add(u, 'group_bank', f'Действующих клиентов Альфа-Банка в группе: {(g.CL_TYPE == "CLIENT").sum()} из {len(g)}', 2)
            if prod:
                add(u, 'group_bank', 'Продукты Альфа-Банка в группе: ' + ', '.join(f'{k} — {comp(v)}' for k, v in prod.items()), 2)
            if len(foreign):
                add(u, 'group_ownership', f'В группе {comp(len(foreign))} с иностранным участием в собственности', 2,
                    ['иностранный собственник'])
            if cs:
                add(u, 'group_trade', 'Внешнеторговые связи компаний группы (ФТС): ' + ', '.join(cs), 2, cs)
            if len(inner):
                add(u, 'group_payments', f'Платежи между компаниями группы: {money(inner.PAY_AMT.sum())}, '
                                         f'{len(inner)} {plural(len(inner), "связь", "связи", "связей")}', 2)
            if len(gl):
                add(u, 'group_persons', f'Компании группы связаны общими физлицами с {gl.SYNTHETIC_U_y.nunique()} компаниями', 2)

    # --- persons ---
    for u, g in F[F.SYNTHETIC_U.isin(target)].groupby('SYNTHETIC_U'):  # CITY_NAME / RC_NAME = bank office serving the person, not home
        regs = [x for x in g.RC_NAME.dropna().unique() if x not in ('Не определено', 'ИНН не определён')]
        add(u, 'persons', f'Связанных физических лиц: {g.SYNTHETIC_F.nunique()}' +
            (f'; обслуживаются в региональных центрах Альфа-Банка: {", ".join(regs)}' if regs else ''), 2)
        cities = [c for c in g.CITY_NAME.dropna().unique() if c.lower() not in str(U.at[u, 'CITY']).lower()]
        if cities:
            add(u, 'persons', 'Связанные физлица обслуживаются в Альфа-Банке в других городах, не в городе регистрации компании: ' + ', '.join(cities[:top_k]), 2)
    for u, g in links[links.SYNTHETIC_U_x.isin(target)].groupby('SYNTHETIC_U_x'):
        others = g.SYNTHETIC_U_y.unique()
        known = U.loc[[x for x in others if x in U.index]]
        add(u, 'persons', f'Через общих физлиц связана с {comp(len(others))}', 2)
        outside = known[known.GROUP_ID != U.at[u, 'GROUP_ID']]
        if len(outside):
            acts = outside.MAINOKVED_NAME.value_counts()
            add(u, 'persons', f'Из них вне своей группы — {comp(len(outside))} с выручкой {money(outside.REVENUE_COMPANY.sum(min_count=1))}; '
                              'виды деятельности: ' + ', '.join(f'{k} ({v})' for k, v in acts.head(top_k).items()),
                2, list(acts.index[:top_k]))

    # --- payments ---
    for u in ids:
        if u in S.index:
            s = S.loc[u]
            add(u, 'payments', f'Платежи, где плательщик и получатель — сама компания: {money(s.PAY_AMT)} ({int(s.PAY_CNT)} шт.)', 2)
    E = P[P.PAYER.isin(target) | P.BENEFICIARY.isin(target)]
    tw = E.merge(P, left_on=['PAYER', 'BENEFICIARY'], right_on=['BENEFICIARY', 'PAYER'])
    tw['vol'] = tw.PAY_AMT_x + tw.PAY_AMT_y
    for u, g in tw[tw.PAYER_x.isin(target)].groupby('PAYER_x'):
        add(u, 'payments', f'Двусторонние платёжные связи (платят друг другу): {g.BENEFICIARY_x.nunique()} '
                           f'{plural(g.BENEFICIARY_x.nunique(), "контрагент", "контрагента", "контрагентов")}', 2)
        for x in g.sort_values('vol', ascending=False).head(3).itertuples():
            who = (f'ОКВЭД {U.at[x.BENEFICIARY_x, "okved"]} «{U.at[x.BENEFICIARY_x, "MAINOKVED_NAME"]}»'
                   if x.BENEFICIARY_x in U.index else 'контрагент вне базы')
            add(u, 'payments', f'Двусторонний поток: компания заплатила {money(x.PAY_AMT_x)}, получила {money(x.PAY_AMT_y)} ({who})', 2)
    inc_tot, out_tot = P.groupby('BENEFICIARY').PAY_AMT.sum(), P.groupby('PAYER').PAY_AMT.sum()
    inc_ratio = (inc_tot / U.REVENUE_COMPANY).dropna()
    out_ratio = (out_tot / U.REVENUE_COMPANY).dropna()
    med = {'in': inc_ratio.groupby(U.okdiv).median(), 'out': out_ratio.groupby(U.okdiv).median()}

    def describe(x, gid):
        if x not in U.index:
            return None
        c = U.loc[x]
        return (f'ОКВЭД {c.okved} «{c.MAINOKVED_NAME}», {c.REGION}, выручка {money(c.REVENUE_COMPANY)}' +
                (', та же группа компаний' if c.GROUP_ID == gid else ''))

    for me, cp, cat, verb, key in [('PAYER', 'BENEFICIARY', 'payments_out', 'Исходящие платежи', 'out'),
                                   ('BENEFICIARY', 'PAYER', 'payments_in', 'Входящие платежи', 'in')]:
        sub = E[E[me].isin(target)]
        sub = sub.assign(okv=sub[cp].map(U.okved), oname=sub[cp].map(U.MAINOKVED_NAME), reg=sub[cp].map(U.REGION),
                         fts=sub[cp].map(U.FTS_COUNTRY_NAME), grp=sub[cp].map(U.GROUP_ID), cl=sub[cp].map(U.CL_TYPE),
                         rev=sub[cp].map(U.REVENUE_COMPANY))
        for u, g in sub.groupby(me):
            r = U.loc[u]
            tot = g.PAY_AMT.sum()
            g = g.sort_values('PAY_AMT', ascending=False)
            m = med[key].get(r.okdiv, np.nan)
            add(u, cat, f'{verb}: {len(g)} {plural(len(g), "контрагент", "контрагента", "контрагентов")}, {money(tot)}, '
                        f'{int(g.PAY_CNT.sum())} платежей' +
                        (f'; это {share_of_revenue(tot / r.REVENUE_COMPANY)} (медиана по разделу ОКВЭД {r.okdiv}: {pct(m)})'
                         if pd.notna(r.REVENUE_COMPANY) else ''), 2)
            n_cnt = int(g.PAY_CNT.sum())
            add(u, cat, f'{verb}: средний платёж {money(tot / n_cnt)}, в среднем {n_cnt / len(g):.1f} платежа на контрагента', 2)
            x0, a0 = g[cp].iloc[0], g.PAY_AMT.iloc[0]
            their = (inc_tot if key == 'out' else out_tot).get(x0, 0)  # counterparty's total on the opposite side
            if their:
                add(u, cat, f'Доля компании {"во входящих" if key == "out" else "в исходящих"} платежах крупнейшего контрагента: '
                            f'{pct(a0 / their)} (контрагент {"получил" if key == "out" else "заплатил"} всего {money(their)})', 2)
            if len(g) >= 2:
                add(u, cat, f'Концентрация ({verb.lower()}): крупнейший контрагент {pct(g.PAY_AMT.iloc[0] / tot)}, '
                            f'топ-5 {pct(g.PAY_AMT.head(5).sum() / tot)}, топ-10 {pct(g.PAY_AMT.head(10).sum() / tot)}', 2)
            top = describe(g[cp].iloc[0], r.GROUP_ID)
            if top:
                add(u, cat, f'Крупнейший контрагент ({verb.lower()}, {pct(g.PAY_AMT.iloc[0] / tot)}): {top}', 2,
                    [g.okv.iloc[0], g.oname.iloc[0], g.reg.iloc[0]])
            k = g.dropna(subset=['okv'])
            if not len(k):
                continue
            add(u, cat, f'{verb}: вид деятельности контрагента известен для {pct(k.PAY_AMT.sum() / tot)} суммы', 2)
            for (code, nm), v in k.groupby(['okv', 'oname']).PAY_AMT.sum().sort_values(ascending=False).head(top_k).items():
                n = (k.okv == code).sum()
                add(u, cat, f'{verb} — {comp(n)} с ОКВЭД {code} «{nm}»: {money(v)} ({pct(v / tot)} суммы)', 2, [code, nm])
            for reg, v in k.groupby('reg').PAY_AMT.sum().sort_values(ascending=False).head(top_k).items():
                if v >= 1e6:
                    add(u, cat, f'{verb} — контрагенты из региона {reg}: {money(v)} ({pct(v / tot)} суммы)', 2, [reg])
            other_reg = k[k.reg != r.REGION].PAY_AMT.sum()
            add(u, cat, f'{verb}: доля контрагентов из других регионов {pct(other_reg / k.PAY_AMT.sum())} известной суммы', 2)
            big = k[k.rev >= 15e9]
            if len(big):
                add(u, cat, f'{verb} — крупный бизнес (выручка свыше 15 млрд ₽): {comp(big[cp].nunique())}, '
                            f'{money(big.PAY_AMT.sum())}', 2)
            cc = pd.Series([c for s in k.fts.dropna() for c in countries(s)]).value_counts()
            if len(cc):
                add(u, cat, f'{verb} — среди контрагентов участники ВЭД со связями со странами: ' +
                    ', '.join(f'{c} ({n})' for c, n in cc.head(top_k).items()), 2, list(cc.index[:top_k]))
            ing = k[k.grp == r.GROUP_ID]
            if len(ing):
                add(u, cat, f'{verb} внутри своей группы: {money(ing.PAY_AMT.sum())} ({pct(ing.PAY_AMT.sum() / tot)} суммы); ' +
                    '; '.join(f'{o} ({rg}) {money(v)}' for (o, rg), v in ing.groupby(['oname', 'reg']).PAY_AMT.sum()
                              .sort_values(ascending=False).head(4).items()), 2)
            nc = k[k.cl != 'CLIENT']
            if len(nc):
                add(u, cat, f'{verb} — контрагенты, не являющиеся действующими клиентами Альфа-Банка: '
                            f'{comp(len(nc))} на {money(nc.PAY_AMT.sum())}', 2)

    # --- second hop: who the counterparties pay / are paid by (known OKVED only) ---
    for me, cp, cat, verb in [('PAYER', 'BENEFICIARY', 'payments_hop2', 'Поставщики получателей платежей компании'),
                              ('BENEFICIARY', 'PAYER', 'payments_hop2', 'Плательщики плательщиков компании')]:
        first = E[E[me].isin(target)][[me, cp, 'PAY_AMT']].rename(columns={me: 'u', cp: 'mid', 'PAY_AMT': 'w'})
        second = P[[me, cp, 'PAY_AMT']].rename(columns={me: 'mid', cp: 'far'})
        h = first.merge(second, on='mid')
        h = h[h.far != h.u]
        h['okv'] = h.far.map(U.okved)
        h['oname'] = h.far.map(U.MAINOKVED_NAME)
        for u, g in h.dropna(subset=['okv']).groupby('u'):
            top = g.groupby(['okv', 'oname']).PAY_AMT.sum().sort_values(ascending=False).head(5)
            add(u, cat, f'{verb} (второй уровень цепочки): ' + '; '.join(f'ОКВЭД {c} «{n}» {money(v)}' for (c, n), v in top.items()),
                3, [n for (_, n) in top.index])

    return facts


def self_check(facts):
    allf = [x['text'] for fs in facts.values() for x in fs]
    assert not any(t.startswith(('Не ', 'Нет ')) for t in allf), 'absence phrased as a fact'
    assert not any('экспорт' in t.lower() or 'импорт' in t.lower() for t in allf), 'FTS direction is unknown'
    assert not any(re.search(r'\bnan\b|\binf\b', t) for t in allf), 'missing value leaked into a fact'
    assert all(any(x['cat'] == 'activity' for x in fs) for fs in facts.values())
    for u, fs in facts.items():
        texts = [x['text'] for x in fs]
        assert len(texts) == len(set(texts)), f'duplicate fact for {u}'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ids', nargs='*')
    ap.add_argument('--out', default=str(DATA / 'companies' / 'facts.jsonl'))
    ap.add_argument('--top-k', type=int, default=8)
    a = ap.parse_args()
    facts = build(a.ids, a.top_k)
    self_check(facts)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        for u, fs in facts.items():
            f.write(json.dumps({'id': u, 'facts': fs}, ensure_ascii=False) + '\n')
    n = [len(fs) for fs in facts.values()]
    print(f'{len(facts)} companies -> {out}; facts per company: median {int(np.median(n))}, max {max(n)}')
