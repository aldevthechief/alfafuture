"""Task 4: the computation tool behind DataRequest(kind='computation'). Pure code, no LLM imports, only checked numbers:
the company's facts from the case tables, the event's figures and the bank's figures that are printed in its text or
confirmed on a chart (a value measured off a chart, «≈», or read by the model alone is never used).

Four calculations, chosen by the words of the request:
- change: the event's figure against the bank's for the same indicator and unit, in п.п. (or %) and relative %;
- share: the company's revenue in its group's revenue;
- forecast on revenue: the bank's forecast (%) applied to the company's revenue, a hypothesis with its assumption;
- units: a number of the request converted between трлн / млрд / млн / тыс. ₽ or between п.п. and б.п.
Anything else, or no checked numbers for it, is answered with the reason, so no request stays open without one.
"""
import re
from math import isfinite

from stage5_insight_detection.analyst_loop.facts import Fact, money_value
from stage5_insight_detection.bank_analytics.library import numeric, topics_of

MONEY = {'трлн': 1e12, 'млрд': 1e9, 'млн': 1e6, 'тыс': 1e3}
NUMBER = re.compile(r'([-+−]?(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)(?:[.,]\d+)?)\s*'
                    r'(трлн|млрд|млн|тыс|п\.\s?п\.|пп|б\.\s?п\.|бп|%)?', re.I)
SUPPORTED = 'изменение показателя к позиции Банка, доля выручки в группе, прогноз Банка на выручку, перевод единиц'


def checked_bank(d) -> list[tuple[str, dict]]:
    """(fragment id, figure) of the bank fragments in the dossier that are checked: printed in text or confirmed."""
    return [(m.fragment.id, x) for m in d.bank for x in m.fragment.figures
            if not m.fragment.stale and (not d.event.date or not m.fragment.date or m.fragment.date <= d.event.date.isoformat())
            and x.get('verified', True) and x.get('status', '') not in ('approx', 'unverified')
            and isinstance(x.get('value'), (int, float)) and isfinite(x['value'])]


def revenue(d) -> tuple[float, str] | None:
    """The company's revenue in ₽ and the fact id it comes from."""
    f = d.profile.facts.get('fact:revenue')
    if f is None or not isinstance(f.value, (int, float)) or not isfinite(f.value) or f.value <= 0:
        return None
    scale = next((v for k, v in MONEY.items() if f.unit.startswith(k)), 1.0)
    return float(f.value) * scale, f.id


def currencies(text: str) -> set[str]:
    return {code for code, pattern in [('USD', r'доллар|\busd\b|\$'), ('CNY', r'юан|\bcny\b|\bcnh\b'),
                                      ('EUR', r'евро|\beur\b|€')]
            if re.search(pattern, text, re.I)}


def change(d, topics: list[str], n: int, query: str = '') -> tuple[list[Fact], str]:
    years = lambda text: set(re.findall(r'\b(?:19|20)\d{2}\b', text))
    requested_years, requested_currency = years(query), currencies(query)
    forecast = bool(re.search(r'прогноз|ожида', query, re.I))
    if len(requested_currency) > 1:
        return [], 'укажите одну валюту и одну пару показателей за вызов'
    pairs = {}
    for t in topics:
        news = [x for x in d.event.figures if x.indicator == t and x.kind == 'actual'
                and x.measure != 'change' and isfinite(x.value)]
        bank = [(fid, x) for fid, x in checked_bank(d) if x['indicator'] == t
                and x['kind'] == ('forecast' if forecast else 'actual')]
        for nf in news:
            for fid, bf in bank:
                if bf['unit'] != nf.unit or bf.get('measure') == 'change':
                    continue
                ny, by = years(nf.period), years(bf.get('period', ''))
                if requested_years:
                    # One year selects the bank target; two years may explicitly request a cross-year comparison.
                    if not by or not by <= requested_years:
                        continue
                    if len(requested_years) > 1 and (not ny or not ny <= requested_years or ny == by):
                        continue
                elif nf.period or bf.get('period'):
                    if nf.period and bf.get('period'):
                        if nf.period != bf['period']:
                            continue
                    elif nf.period or forecast or not d.event.date or by != {str(d.event.date.year)}:
                        continue
                if forecast and not bf.get('period'):
                    continue
                nc, bc = currencies(nf.sentence + ' ' + nf.unit), currencies(bf.get('sentence', '') + ' ' + bf['unit'])
                if t == 'курс' and (len(nc) != 1 or nc != bc or requested_currency and nc != requested_currency):
                    continue
                key = (t, nf.value, nf.unit, nf.period, bf['value'], bf['kind'], bf.get('period'), tuple(sorted(nc)))
                pairs.setdefault(key, (t, nf, fid, bf))
    if len(pairs) > 1:
        return [], 'несколько подходящих пар: уточните показатель, валюту и период; случайная пара не выбирается'
    if pairs:
        t, nf, fid, bf = next(iter(pairs.values()))
        diff = round(nf.value - bf['value'], 2)
        unit = 'п.п.' if nf.unit == '%' else nf.unit
        rel = round(100 * diff / bf['value'], 1) if bf['value'] else None
        label = f'Изменение показателя «{t}» к цифре Банка ({bf["kind"]}, {bf["period"] or "период не указан"})'
        facts = [Fact(f'fact:calc_{n}_change', label, diff, unit, f'{nf.value} ({d.event.id}) − {bf["value"]} ({fid})',
                      f'{fid}; {d.event.id}')]
        if rel is not None:
            facts.append(Fact(f'fact:calc_{n}_change_rel', label + ', относительное', rel, '%',
                              f'({nf.value} − {bf["value"]}) / {bf["value"]} × 100', f'{fid}; {d.event.id}'))
        return facts, ''
    return [], f'нет однозначной пары проверенных чисел по темам {", ".join(topics) or "запроса"}: проверьте период, тип факт/прогноз, валюту и единицы'


def share(d, n: int) -> tuple[list[Fact], str]:
    rev, group = d.profile.facts.get('fact:revenue'), d.profile.facts.get('fact:group_revenue')
    r, g = revenue(d), None
    if group is not None and isinstance(group.value, (int, float)) and isfinite(group.value):
        g = float(group.value) * next((v for k, v in MONEY.items() if group.unit.startswith(k)), 1.0)
    if not r or not g or g <= 0:
        return [], 'нет выручки компании или выручки группы в таблицах кейса'
    return [Fact(f'fact:calc_{n}_share', 'Доля выручки компании в выручке группы', round(100 * r[0] / g, 1), '%',
                 'fact:revenue / fact:group_revenue × 100', 'df_Uid_data_hack.csv')], ''


def forecast_on_revenue(d, topics: list[str], n: int) -> tuple[list[Fact], str]:
    r = revenue(d)
    if not r:
        return [], 'нет выручки компании в таблицах кейса (или стоит заглушка)'
    for t in topics or ['инфляция']:
        fc = [(fid, x) for fid, x in checked_bank(d) if x['indicator'] == t and x['kind'] == 'forecast' and x['unit'] == '%']
        if fc:
            fid, x = fc[0]
            v, u = money_value(r[0] * x['value'] / 100)
            return [Fact(f'fact:calc_{n}_forecast', f'Выручка × прогноз Банка «{t}» {x["value"]}% ({x["period"] or "срок не указан"})',
                         v, u, f'{r[1]} × {x["value"]}% ({fid}); допущение: показатель переносится на выручку один к одному',
                         f'{r[1]}; {fid}', kind='hypothesis')], ''
    return [], f'нет проверенного прогноза Банка в % по темам {", ".join(topics) or "инфляция"} среди подобранной аналитики'


def units(query: str, n: int) -> tuple[list[Fact], str]:
    q = query.lower()
    m = NUMBER.search(q)
    target = re.search(r'\bв\s+(трлн|млрд|млн|тыс|б\.\s?п\.|бп|п\.\s?п\.|пп)', q)
    if not m or not m.group(2) or not target:
        return [], 'в запросе нет числа с единицей и единицы, в которую перевести'
    value, src, dst = numeric(m.group(1)), m.group(2).replace(' ', ''), target.group(1).replace(' ', '')
    bp = {'б.п.': 1, 'бп': 1, 'п.п.': 100, 'пп': 100}
    if src in MONEY and dst in MONEY:
        out = value * MONEY[src] / MONEY[dst]
    elif src in bp and dst in bp:
        out = value * bp[src] / bp[dst]
    else:
        return [], f'единицы {src} и {dst} не переводятся друг в друга'
    return [Fact(f'fact:calc_{n}_units', f'{value:g} {src} в {dst}', round(out, 4), dst,
                 f'{value:g} {src} → {dst}', 'число из запроса, не подтверждено таблицами', kind='hypothesis')], ''


def compute(query: str, d, n: int = 1) -> tuple[list[Fact], str]:
    """Facts for the request, or none and the reason."""
    q = query.lower()
    topics = topics_of(query) or list(d.event.topics)
    if re.search(r'перев\w*|\bв\s+(трлн|млрд|млн|тыс|б\.\s?п|бп|п\.\s?п|пп)\b', q):
        return units(query, n)
    if 'доля' in q or 'долю' in q:
        return share(d, n)
    if 'выручк' in q and ('прогноз' in q or topics):
        return forecast_on_revenue(d, [t for t in topics_of(query)], n)
    if re.search(r'изменени|разниц|на сколько|отклонени|в %', q):
        return change(d, topics, n, query)
    return [], f'расчёт не из поддерживаемых ({SUPPORTED}) или для него нет проверенных чисел'
