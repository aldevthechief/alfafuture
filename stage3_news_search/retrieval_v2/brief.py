"""Deterministic company brief for the generator factory: stage-2 fact texts regrouped by impact channel (variant (d) of
experiments/briefs, the factory's input in every evaluated run). Copied, not imported: experiments/ stay frozen records."""
import re


def strength(p: float) -> str:
    return 'сильная' if p >= 30 else 'средняя' if p >= 5 else 'слабая'


def pct(s: str) -> float:
    m = re.search(r'(<?)(\d+)% суммы', s)
    return 0.5 if m and m[1] else float(m[2]) if m else 0.0


def deterministic(facts: list[dict]) -> str:
    """Code-built brief: fact texts regrouped by impact channel, each counterparty link with its share of the flow and a
    strength label; FTS countries always «не измерена». shortcut: parses stage-2 fact texts, breaks if facts.py wording
    changes; build from features.parquet if this format wins."""
    by = {}
    for f in facts:
        by.setdefault(f['cat'], []).append(f['text'])
    first = lambda cat, pre='': next((t for t in by.get(cat, []) if t.startswith(pre)), None)
    act = first('activity', 'Основной')
    lines = [f"Отрасль: {act.split(': ', 1)[1]} ({re.search(r'ОКВЭД [\d.]+', act)[0]}). "
             f"{first('scale', 'Годовая')}; {first('profile', 'Численность').replace('Численность по ФНС: ', 'штат ')}"]
    if r := first('rank'):
        lines.append(f'Место: {r.split(";")[0]}.')
    if any('иностран' in t for t in by.get('ownership', [])):
        lines.append('Собственник: иностранное участие.')

    for cat, title, who in (('payments_in', 'Покупатели (входящие платежи)', 'плательщиков'),
                            ('payments_out', 'Поставщики (исходящие платежи)', 'получателей')):
        ts = by.get(cat, [])
        if not ts:
            lines.append(f'{title}: нет данных (платежи видны в основном у клиентов Альфа-Банка).')
            continue
        tot = next(t for t in ts if re.match(r'(Входящие|Исходящие) платежи: \d+ контраг', t))
        share = tot.split('; это ')[1] if '; это ' in tot else ''
        conc = next((t.split(': ', 1)[1] for t in ts if t.startswith('Концентрация')), '')
        known = next((re.search(r'известен для (.+? суммы)', t)[1] for t in ts if 'известен для' in t), '?')
        lines.append(f'{title}: {share}. Концентрация: {conc}. Отрасль {who} известна для {known}.')
        m = re.match(r'(<?)(\d+)% годовой', share)
        if m and (m[1] or int(m[2]) < 30):  # a small visible flow: its shares say little about the business
            lines.append(f'  - виден только поток в {m[1]}{m[2]}% выручки: доли ниже описывают его, а не весь бизнес')
        elif 'раз больше' in share:
            lines.append('  - поток во много раз больше выручки: аномалия: обороты не отражены в выручке (проектная компания, транзит или неполная выручка в данных)')
        if big := next((t for t in ts if t.startswith('Крупнейший контрагент')), None):
            lines.append(f'  - крупнейший: {big.split(": ", 1)[1]}')
        elif conc:  # facts.py names the largest counterparty only when it is in the company table
            lines.append('  - крупнейший: контрагент вне базы, отрасль неизвестна')
        if dep := next((t for t in ts if t.startswith('Доля компании')), None):
            p = int(re.search(r': <?(\d+)%', dep)[1])  # '<1%' too
            if p >= 50:
                lines.append(f'  - крупнейший плательщик отправляет этой компании {p}% всех своих платежей' if cat == 'payments_in' else
                             f'  - крупнейший получатель получает от этой компании {p}% всех своих поступлений')
        for t in [t for t in ts if ' с ОКВЭД ' in t and pct(t) >= 1][:4]:
            m = re.search(r'— (.+?) с ОКВЭД ([\d.]+) «(.+?)»: (.+)$', t)
            lines.append(f'  - {m[3]} ({m[2]}, {m[1]}): {m[4]} — {strength(pct(t))}')
        if grp := next((t for t in ts if 'внутри своей группы' in t), None):
            lines.append(f'  - внутри группы: {re.search(r": (.+? суммы\))", grp)[1]}')

    if prods := first('bank', 'Продукты'):
        credit = any(w in prods for w in ('кредит', 'лизинг'))
        lines.append(f"Альфа-Банк: {prods.split(': ', 1)[1]}{' — чувствительность к ставке' if credit else ''}.")
    countries = [re.search(r'из страны (.+?) \(', t)[1] for t in by.get('trade', []) if 'из страны' in t]
    if countries:
        lines.append(f"ВЭД (ФТС): {', '.join(countries)} — сила не измерена: направление (ввоз или вывоз) и объём неизвестны, "
                     'это не доля бизнеса.')
    else:
        lines.append('ВЭД: в базе ФТС нет.')
    geo = first('geo').split(': ', 1)[1]
    other = [re.search(r'(\d+%|<1%) известной', t)[1] for t in by.get('payments_in', []) + by.get('payments_out', [])
             if 'других регионов' in t]
    lines.append(f"Регион: {geo}" + (f"; доля контрагентов из других регионов: входящие {other[0]}, исходящие {other[1]}"
                                     if len(other) == 2 else '') + '.')
    if g := by.get('group'):
        acts = [re.sub(r'^В группе ', '', t) for t in by.get('group_activity', [])[:3]]
        gp = first('group_payments')
        lines.append(f"Группа: {g[0].replace('Входит в группу', 'группа')}; {g[1][0].lower() + g[1][1:]}."
                     + (f" Другие компании: {'; '.join(acts)}." if acts else '') + (f' {gp}.' if gp else ''))
    lines.append('Неизвестно: направление и объём ВЭД; отрасль контрагентов вне названной доли суммы; период платежей; '
                 'структура выручки по продуктам.')
    return '\n'.join(lines)
