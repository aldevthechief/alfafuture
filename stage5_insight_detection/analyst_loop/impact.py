"""Task 5: consequences of the event for the client's business, computed by code from the profile, the event and the
bank's figures. The model only explains them later (loop.py), it never computes.

For every event topic the company is exposed to (exposure >= 0.3) a consequence gets:
- sign and mechanism from a fixed table (topic, direction of the indicator, profile) — e.g. the key rate goes up and the
  company has bank loans -> debt service gets more expensive (-); the company has deposits -> higher yield (+);
  an unknown FTS direction gives sign 0 and a two-sided hypothesis, never "exporter";
- a quantitative estimate where the data allow it: the indicator delta from the news or the bank's figures times a base
  from the profile (revenue, payments to the affected sector). Assumptions that the data cannot confirm (share of debt in
  revenue, pass-through) are `Fact(kind='hypothesis')`, shown as hypotheses;
- an impact level by fixed criteria (IMPACT_CRITERIA): high needs a number.
Pure code, no LLM.
"""
import re
from dataclasses import asdict, dataclass, field

from stage5_insight_detection.analyst_loop.event import EventCard
from stage5_insight_detection.analyst_loop.facts import Fact, money_value
from stage5_insight_detection.analyst_loop.profile import CompanyProfile
from stage5_insight_detection.bank_analytics.match import BankMatch

DEBT_TO_REVENUE = 0.2  # assumption for borrowers: floating-rate debt = 20% of revenue (no debt data in the case tables)
LIQUIDITY_TO_REVENUE = 0.05  # assumption for deposit holders: free cash = 5% of revenue
COST_SHARE = 0.3  # assumption: the affected input is 30% of costs
# assumption: payroll as a share of revenue by OKVED division (order of magnitude, not case data); the rest PAYROLL_DEFAULT
PAYROLL_SHARE = {'47': 0.08, '46': 0.04, '45': 0.06, '41': 0.15, '42': 0.15, '43': 0.18, '49': 0.15, '52': 0.15, '55': 0.25,
                 '56': 0.25, '62': 0.35, '63': 0.3, '69': 0.35, '70': 0.3, '71': 0.35, '85': 0.45, '86': 0.4}
PAYROLL_DEFAULT = 0.12
HIGH_SCORE, MID_SCORE, HIGH_SHARE = 0.55, 0.3, 0.01
IMPACT_CRITERIA = ('score = чувствительность компании к каналу (0–1) × важность события (1–5)/5 × подтверждение '
                   '(1.0 — есть свежая аналитика Банка или ≥2 источника, иначе 0.7). high: score ≥ 0.55 и есть количественная '
                   'оценка ≥ 1% выручки; mid: score ≥ 0.3; иначе low. Без числа уровень не выше mid.')
BORROWER_DIVS = {'41', '42', '43', '68', '64', '65', '66'}
STEEL_USERS = {'25', '27', '28', '29', '30', '41', '42', '43'}
COMMODITY_PRODUCERS = {'05', '06', '07', '08', '09', '19'}
FUEL_USERS = {'49', '50', '51', '52', '53', '23', '24', '01'}
DIESEL_USERS = {'49', '50', '51', '52', '53', '01', '41', '42', '43'}  # trucks, ships, tractors, building machines: diesel
FUEL_GRADES = {'дизель': re.compile(r'дизел|дт\b|солярк', re.I), 'бензин': re.compile(r'бензин|аи-?9\d', re.I)}
# a change over a week or a day is not a yearly one: the code never carries it over to a yearly budget
SHORT_HORIZON = re.compile(r'за\s+(?:неделю|недели|сутки|день|торговую сессию)|недельн|за\s+\d+\s+(?:дн|недел)|'
                           r'в период с \d{1,2}(?:\s+\w+)? по \d{1,2}\s+\w+', re.I)


@dataclass
class Consequence:
    id: str
    topic: str
    move: str  # up / down / flat / ?
    sign: int  # +1 opportunity, -1 risk, 0 depends / watch
    mechanism: str
    exposure: float
    facts: list[str]
    bank_refs: list[str]
    news_refs: list[str]
    estimate: Fact | None = None
    assumptions: list[Fact] = field(default_factory=list)
    horizon: str = ''
    score: float = 0.0
    level: str = 'low'
    kind: str = 'ai'  # ai (derived from facts) | hypothesis (sign or base is an assumption)
    level_why: str = ''  # the threshold and evidence behind the impact level

    def dict(self) -> dict:
        d = asdict(self)
        d['estimate'] = self.estimate.dict() if self.estimate else None
        return d


def mechanisms(topic: str, move: str, p: CompanyProfile) -> list[tuple[int, str, str]]:
    """(sign, mechanism, kind) for one topic; several lines when the company sits on both sides (loans and deposits)."""
    s = {'up': 1, 'down': -1}.get(move, 0)
    d = p.okdiv
    out = []
    if topic in ('ставка', 'облигации'):
        if 'loan' in p.products or 'leasing' in p.products or d in BORROWER_DIVS:
            what = 'облигационного долга' if topic == 'облигации' else 'кредитов и лизинга'
            out.append((-s, f'стоимость обслуживания и рефинансирования {what}', 'ai' if p.products else 'hypothesis'))
        if 'deposit' in p.products:
            out.append((s, 'доходность размещения свободной ликвидности', 'ai'))
        if not out:
            out.append((-s, 'стоимость заёмного финансирования и инвестиционный спрос', 'hypothesis'))
    elif topic == 'инфляция':
        out.append((-s, 'рост издержек и давление на маржу' if d != '47' else 'номинальная выручка растёт, маржа под давлением', 'ai'))
    elif topic == 'потребление':
        out.append((s, 'спрос со стороны населения', 'ai' if p.exposure.get('потребление', 0) >= 0.5 else 'hypothesis'))
    elif topic in ('строительство', 'инвестиции', 'промышленность'):
        out.append((s, f'спрос со стороны {"строительства" if topic == "строительство" else "инвестиционных и промышленных заказчиков"}',
                    'ai' if p.exposure.get(topic, 0) >= 0.6 else 'hypothesis'))
    elif topic == 'металлургия':
        if d in ('24', '07'):
            out.append((s, 'цены реализации металлопродукции', 'ai'))
        elif d in STEEL_USERS:
            out.append((-s, 'стоимость металла в себестоимости', 'ai'))
    elif topic == 'энергия_и_сырьё':
        if d in COMMODITY_PRODUCERS:
            out.append((s, 'цены реализации сырья', 'ai'))
        elif d in FUEL_USERS:
            out.append((-s, 'затраты на топливо и энергию', 'ai'))
    elif topic == 'золото' and p.exposure.get('золото', 0) >= 0.8:
        out.append((s, 'цены реализации драгметаллов', 'ai'))
    elif topic == 'курс':
        out.append((0, 'валютная выручка или валютные закупки: знак зависит от направления ВЭД, в данных ФТС его нет', 'hypothesis'))
    elif topic in ('бюджет_и_налоги', 'рынок_труда'):
        out.append((-s, 'налоговая нагрузка' if topic == 'бюджет_и_налоги' else 'фонд оплаты труда и доступность кадров', 'ai'))
    elif topic in ('внешняя_торговля', 'кредит'):
        out.append((0, 'логистика и расчёты по ВЭД' if topic == 'внешняя_торговля' else 'доступность кредита', 'hypothesis'))
    return out


def indicator_delta(topic: str, event: EventCard, matches: list[BankMatch]) -> tuple[float, str, str, list[str]] | None:
    """Only explicit changes or an explicit pair of levels are deltas. A lone «до 8%» is a level, never −8%.
    For the key rate a bank actual can supply the baseline; otherwise a 1 pp sensitivity stays a hypothesis.
    A change in a percentage-valued level is measured in pp, not percent of revenue/costs.
    """
    move = event.moves.get(topic)
    fresh = sorted([m.fragment for m in matches if not m.fragment.stale and (
        not event.date or not m.fragment.date or m.fragment.date <= event.date.isoformat()[:10])],
        key=lambda f: f.date or '', reverse=True)
    sure = [(f, x) for f in fresh for x in f.figures if x.get('verified', True)]  # a chart value read off the scale is no base
    bank_actual = [(f, x) for f, x in sure if x['indicator'] == topic and x['kind'] == 'actual' and x['unit'] == '%'
                   and x.get('measure') != 'change']
    horizon = next((x['period'] for f, x in sure if x['indicator'] == topic and x['kind'] == 'forecast'), '')
    news = [x for x in event.figures if x.indicator == topic and x.kind == 'actual'
            and x.unit in ('%', 'п.п.', 'пп', 'б.п.', 'бп')]
    changes = [x for x in news if x.measure == 'change']
    def signed(x):
        return -abs(x.value) if move == 'down' else x.value
    pp = [x for x in changes if x.unit in ('п.п.', 'пп', 'б.п.', 'бп')]
    if pp:
        return signed(pp[0]) * (0.01 if pp[0].unit in ('б.п.', 'бп') else 1), 'п.п.', horizon, [event.id]
    relative = [x for x in changes if x.unit == '%']
    if relative:
        return signed(relative[0]), '%', relative[0].period or horizon, [event.id]
    levels = [x for x in news if x.measure == 'level' and x.unit == '%']
    # Compare only an explicit «с X% до Y%» pair in the same sentence, never unrelated historic figures.
    for old, new in zip(levels, levels[1:]):
        if old.sentence == new.sentence and re.search(r'\bс\s+[-+−]?\d+(?:[.,]\d+)?\s*%\s+до\s+', old.sentence, re.I):
            return round(new.value - old.value, 2), 'п.п.', horizon, [event.id]
    if topic == 'ставка' and levels and bank_actual:
        f, x = bank_actual[0]
        return round(levels[0].value - x['value'], 2), 'п.п.', horizon, [event.id, f.id]
    if topic == 'ставка' and move in ('up', 'down'):
        return (1.0 if move == 'up' else -1.0), 'п.п.', horizon, ['чувствительность на 1 п.п.']
    return None


def change_sentence(topic: str, event: EventCard) -> str:
    """The sentence of the news change indicator_delta takes (an explicit change first), for the checks below."""
    news = [x for x in event.figures if x.indicator == topic and x.kind == 'actual' and x.measure == 'change'
            and x.unit in ('%', 'п.п.', 'пп', 'б.п.', 'бп')]
    news.sort(key=lambda x: x.unit == '%')  # pp first, as indicator_delta
    return news[0].sentence if news else ''


def no_estimate_why(topic: str, mechanism: str, sentence: str, p: CompanyProfile) -> str:
    """Why the news change is not the company's own cost line or year: then no estimate is made (empty: estimate).
    1) fuel: the company runs on diesel and the news gives a petrol grade (Аи-92), not the same indicator;
    2) horizon: a change over a week or a day is not carried over to a year."""
    if topic == 'энергия_и_сырьё' and 'топлив' in mechanism and p.okdiv in DIESEL_USERS:
        said = {k for k, r in FUEL_GRADES.items() if r.search(sentence)}
        if said == {'бензин'}:
            return 'в новости изменение цены бензина, у компании затраты на дизель: показатель не тот, оценки нет'
    if SHORT_HORIZON.search(sentence):
        return 'изменение за неделю/день: на годовой бюджет не переносится, оценки нет'
    return ''


def estimate(topic: str, mechanism: str, delta, p: CompanyProfile, n: int) -> tuple[Fact | None, list[Fact]]:
    if delta is None or not p.revenue:
        return None, []
    value, unit, _, _ = delta
    rev = p.revenue
    if topic in ('ставка', 'облигации') and unit != 'п.п.':
        return None, []  # a relative change in a rate needs its baseline, not a share of revenue
    if topic in ('ставка', 'облигации') and unit == 'п.п.':
        if 'ликвидност' in mechanism:
            a = Fact(f'fact:assume_liquidity_{n}', 'Допущение: свободная ликвидность', *money_value(rev * LIQUIDITY_TO_REVENUE),
                     f'{LIQUIDITY_TO_REVENUE:.0%} × выручка', 'допущение модели расчёта', 'hypothesis')
            v, u = money_value(rev * LIQUIDITY_TO_REVENUE * value / 100)
        else:
            a = Fact(f'fact:assume_debt_{n}', 'Допущение: плавающий долг', *money_value(rev * DEBT_TO_REVENUE),
                     f'{DEBT_TO_REVENUE:.0%} × выручка (данных о долге в кейсе нет)', 'допущение модели расчёта', 'hypothesis')
            v, u = money_value(rev * DEBT_TO_REVENUE * value / 100)
        est = Fact(f'fact:estimate_{n}', 'Изменение процентных потоков в год', v, u,
                   f'{a.value} {a.unit} × {value:+} п.п.', 'расчёт кода', 'hypothesis')
        return est, [a]
    if unit == '%' and topic == 'рынок_труда':
        share = PAYROLL_SHARE.get(p.okdiv, PAYROLL_DEFAULT)
        a = Fact(f'fact:assume_payroll_{n}', 'Допущение: доля ФОТ в выручке по отрасли', round(share * 100), '%',
                 f'ОКВЭД {p.okdiv}: оценка порядка для отрасли, в данных кейса ФОТ нет', 'допущение модели расчёта', 'hypothesis')
        v, u = money_value(rev * share * value / 100)
        return Fact(f'fact:estimate_{n}', 'Изменение фонда оплаты труда в год', v, u,
                    f'выручка × {share:.0%} (доля ФОТ) × {value:+}%', 'расчёт кода', 'hypothesis'), [a]
    if unit == '%':
        share = 1.0 if mechanism.startswith('спрос') or 'цены реализации' in mechanism or 'выручка' in mechanism else COST_SHARE
        a = Fact(f'fact:assume_share_{n}', 'Допущение: доля затронутой статьи', round(share * 100), '%',
                 'спрос и цены реализации — вся выручка; издержки — 30% выручки', 'допущение модели расчёта', 'hypothesis')
        v, u = money_value(rev * share * value / 100)
        return Fact(f'fact:estimate_{n}', 'Изменение выручки или затрат в год', v, u,
                    f'выручка × {share:.0%} × {value:+}%', 'расчёт кода', 'hypothesis'), [a]
    return None, []


def level_why(score: float, est: Fact | None, revenue: float | None, assumptions: list[Fact] | None = None) -> tuple[str, str]:
    """The level by IMPACT_CRITERIA and the reason in words."""
    big = False
    if est is not None and revenue:
        mult = {'трлн ₽': 1e12, 'млрд ₽': 1e9, 'млн ₽': 1e6, 'тыс. ₽': 1e3, '₽': 1}.get(est.unit, 1)
        big = abs(float(est.value) * mult) >= HIGH_SHARE * revenue
    numbered = est is not None and (assumptions is None or bool(assumptions))
    if score >= HIGH_SCORE and big and numbered:
        return 'high', f'score {score:.2f} ≥ {HIGH_SCORE} и оценка ≥ {HIGH_SHARE:.0%} выручки с допущениями'
    if score >= HIGH_SCORE:
        why = 'нет количественной оценки' if est is None else 'оценка меньше 1% выручки' if not big else 'у оценки нет допущений'
        return 'mid', f'score {score:.2f} ≥ {HIGH_SCORE}, но {why}: не выше mid'
    if score >= MID_SCORE:
        return 'mid', f'score {score:.2f} ≥ {MID_SCORE}'
    return 'low', f'score {score:.2f} < {MID_SCORE}'


def level_of(score: float, est: Fact | None, revenue: float | None) -> str:
    return level_why(score, est, revenue)[0]


def assess(event: EventCard, profile: CompanyProfile, matches: list[BankMatch], confirmations: int = 0,
           min_exposure: float = 0.3, deltas: dict | None = None) -> tuple[list[Consequence], list[Fact]]:
    """Returns the consequences (strongest first) and every computed fact they cite. deltas: topic -> the delta of a
    bank thesis (theses.delta), used instead of the news figures."""
    out, facts, n = [], [], 0
    confirmed = 1.0 if any(not m.fragment.stale for m in matches) or confirmations >= 1 or len(event.sources) >= 2 else 0.7
    for topic in event.topics:
        exp = profile.exposure.get(topic, 0.0)
        if exp < min_exposure:
            continue
        bank = [m.fragment.id for m in matches if topic in m.fragment.topics]
        delta = (deltas or {}).get(topic) if getattr(event, 'origin', 'news') == 'bank' else indicator_delta(topic, event, matches)
        from_news = delta is not None and getattr(event, 'origin', 'news') == 'news'
        for sign, mech, kind in mechanisms(topic, event.moves.get(topic, '?'), profile):
            n += 1
            skip = no_estimate_why(topic, mech, change_sentence(topic, event), profile) if from_news else ''
            est, assumptions = estimate(topic, mech, delta, profile, n) if sign and not skip else (None, [])
            if skip:
                mech, kind = f'{mech} ({skip})', 'hypothesis'
            fact_ids = list(dict.fromkeys(profile.reasons.get(topic, ['fact:okved'])))
            if delta:
                rough = any(str(x).startswith('≈') for x in delta[3])
                thesis = getattr(event, 'origin', 'news') == 'bank' and str(delta[3][0]).startswith('bank:')
                how = (('прогноз Банка минус текущее значение, напечатанное Банком' if len(delta[3]) > 1 else
                        'изменение показателя из аналитики Банка') if thesis else
                       'новость минус значение, снятое кодом с графика Банка (≈, по пикселям)' if rough else
                       'новость минус последнее фактическое значение в аналитике Банка' if len(delta[3]) > 1 else
                       'расчёт на 1 п.п. в направлении события (в новости нет числа)' if delta[3][0].startswith('чувств') else
                       'из новости')
                d = Fact(f'fact:delta_{n}', f'Изменение показателя «{topic}»' + (' (≈)' if rough else '') +
                         (' по аналитике Банка' if thesis else ''), delta[0], delta[1], how,
                         ', '.join(x for x in delta[3] if not str(x).startswith('≈')),
                         'hypothesis' if delta[3][0].startswith('чувств') or rough else 'fact')
                facts.append(d)
                fact_ids.append(d.id)
            score = round(exp * event.importance / 5 * confirmed, 3)
            if not sign:
                score = round(score * 0.6, 3)  # unknown direction is worth watching, not acting on
            level, why = level_why(score, est, profile.revenue, assumptions)
            if skip:
                why += f'; {skip}'
            c = Consequence(f'C{n}', topic, event.moves.get(topic, '?'), sign, mech, exp, fact_ids, bank, [event.id],
                            est, assumptions, delta[2] if delta else '', score, level, kind, why)
            if est:
                facts += [est, *assumptions]
                c.facts += [est.id, *[a.id for a in assumptions]]
            out.append(c)
    out.sort(key=lambda c: -c.score)
    return out, facts
