"""Theses of the Bank: its own positions as the event of an insight of origin «bank» (no news needed).

1. collect(): theses from the report digests (digest.py: bank_views, checked against their pages) and from the own-view
   sentences of the library fragments («мы ожидаем…»). A thesis: indicator, topic, direction, period, number, report,
   page, report date and the fragment that prints it (bank:<file>:p<page>.<n>, the only ref a claim may cite). Only
   checked numbers: the number must be a figure of a text or table fragment of that page (a chart: a label the pixels
   confirmed); a value with «≈», read off a chart scale or marked unverified makes no thesis. Investment advice of the
   reports (portfolios, target prices), views of the securities market (NOT_BUSINESS) and weekly or monthly prints
   (GRANULAR) make no thesis either: the card gives business recommendations only. Own-view sentences of the fragments
   are the fallback for a report without a digest: a thesis only with one topic and a checked forecast in %, never a
   previous forecast («ранее»).
   A thesis lives SHELF_LIFE days by the report kind (a rate or inflation view as long as a rate report, wherever it
   is); an expired thesis gives no insight.
2. select(): the reverse match for a company. The thesis becomes an EventCard (event.from_thesis) and goes through the
   same impact.assess as a news event: channel exposure of the profile (OKVED topics, counterparties' sectors,
   products), mechanism and level by IMPACT_CRITERIA; a thesis passes at level mid or high. One thesis per topic, the
   freshest report first. A number reaches the estimate only as a delta the code derives from the bank's data
   (delta()): a growth rate as is, a rate forecast minus the current rate printed by the bank.
3. ShownLog: antispam. A thesis is shown to a company once until a new report or a change of the position (the key is
   report, topic, direction and number), at most MAX_PER_WINDOW bank insights per company in WINDOW_DAYS days; the log
   lives in data/insights/ (gitignored: it names case companies).
Pure code, no LLM.
"""
import hashlib, json, re, sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.text import stems
from common.news_corpus import write_json
from stage5_insight_detection.analyst_loop import impact
from stage5_insight_detection.analyst_loop.event import EventCard
from stage5_insight_detection.analyst_loop.profile import CompanyProfile
from stage5_insight_detection.bank_analytics.library import (FORECAST, NUMBER, OWN_VIEW, SCALE_MARK, BankLibrary, Fragment,
                                                             direction, figures_of, is_own_view, numeric, sentences, topics_of)
from stage5_insight_detection.bank_analytics.match import BankMatch

ROOT = Path(__file__).resolve().parents[2]
DIGEST_DIR = ROOT / 'data' / 'analytics' / 'text'
SHOWN = ROOT / 'data' / 'insights' / 'bank_theses_shown.json'

# days a thesis lives after its report, by the report kind
SHELF_LIFE = {'ставка': 60, 'инфляция': 60, 'макро': 92, 'стратегия': 92, 'отрасль': 92}
SHORT_TOPICS = {'ставка', 'инфляция'}  # a rate or inflation view ages as fast as a rate report, in any report
REPORT_KIND = [('ставка', re.compile(r'policy_rate|ключев\w* ставк', re.I)), ('инфляция', re.compile(r'inflation|инфляц', re.I)),
               ('стратегия', re.compile(r'strategy|стратеги|тактическ|бонд|bond', re.I)),
               ('отрасль', re.compile(r'steel|стал[ьи]|gold|золот|отрасл|сектор', re.I))]
IMPORTANCE = {True: 4, False: 3}  # a thesis with a checked number / a qualitative one, as the event importance (1–5)
LEVELS = ('mid', 'high')  # a thesis reaches a company at these impact levels only
MAX_PER_WINDOW, WINDOW_DAYS = 2, 7
SHAKY = re.compile(r'≈|~|по шкале|не провер|искаж|примерно|около', re.I)
ADVICE = re.compile(r'рекоменд|портфел|целев\w* цен|дивиденд|котиров|флоутер|офз|покупат|продават|спекулятив|эмитент|'
                    r'кредитн\w* качеств|спред', re.I)
GRANULAR = re.compile(r'недельн|н/н|м/м|месячн', re.I)  # a weekly or monthly print is a data point, not a position
EARLIER = re.compile(r'ранее|прежн|предыдущ|был[аио]? (на уровне|\d)', re.I)
COST_TOPICS = {'рынок_труда', 'инфляция'}  # a growth rate here moves a cost level (payroll, prices) by its sign
NOT_BUSINESS = {'облигации', 'ставка_мир'}  # views of the securities market and foreign central banks
FALL = re.compile(r'(спад|паден\w*|снижени\w*|сокращени\w*)\s+(на\s+)?(?=\d)', re.I)  # «спад на 4%» is −4%
CHANGE = re.compile(r'г/г|м/м|рост|прирост|темп|динамик|изменени', re.I)  # the value is a change, not a level
TOPIC_LABEL = {'ставка': 'Ключевая ставка', 'инфляция': 'Инфляция', 'курс': 'Курс рубля', 'потребление': 'Потребительский спрос',
               'кредит': 'Кредитование', 'бюджет_и_налоги': 'Бюджет и налоги', 'облигации': 'Рынок облигаций',
               'золото': 'Золото', 'металлургия': 'Металлургия', 'строительство': 'Строительство',
               'промышленность': 'Промышленность', 'энергия_и_сырьё': 'Энергия и сырьё', 'рынок_труда': 'Зарплаты и рынок труда',
               'инвестиции': 'Инвестиции', 'внешняя_торговля': 'Внешняя торговля', 'ставка_мир': 'Ставки мировых ЦБ'}


@dataclass
class Thesis:
    id: str  # thesis:<file stem>:p<page>:<topic>:<n>
    topic: str
    indicator: str
    direction: str  # up | down | flat, as the bank says it
    move: str  # how the level moves: a growth rate of +10% is «up» even when the bank calls it a slowdown
    value: float | None
    unit: str
    value_text: str
    period: str
    kind: str  # forecast | actual
    report: str
    file: str
    page: int
    report_date: str
    fragment: str  # bank:<file stem>:p<page>.<n> that prints the thesis: the ref of every claim about it
    quote: str
    source: str  # digest | fragment
    report_kind: str
    expires: str

    def fresh(self, as_of: date) -> bool:
        return date.fromisoformat(self.report_date) <= as_of <= date.fromisoformat(self.expires)

    def key(self) -> str:
        """A new report or another direction or number is a new thesis for the antispam log."""
        return f'{self.file}|{self.report_date}|{self.topic}|{self.direction}|{self.value}|{self.unit}|{self.period}'

    def text(self) -> str:
        return f'{self.indicator}: {self.value_text}' + (f', {self.period}' if self.period else '')

    def dict(self) -> dict:
        return asdict(self)


@dataclass
class Pick:
    """A thesis that reaches the company: its level, channel and the delta the estimate may use."""
    thesis: Thesis
    score: float
    level: str
    sign: int
    channel: str
    reasons: list[str]
    delta: tuple | None = None

    def dict(self) -> dict:
        return {'thesis': self.thesis.id, 'score': self.score, 'level': self.level, 'sign': self.sign, 'channel': self.channel,
                'reasons': self.reasons, 'delta': list(self.delta) if self.delta else None}


def report_kind(file: str, title: str) -> str:
    return next((k for k, rx in REPORT_KIND if rx.search(f'{file} {title}')), 'макро')


def shelf_life(kind: str, topic: str) -> int:
    return min(SHELF_LIFE[kind], SHELF_LIFE[topic]) if topic in SHORT_TOPICS else SHELF_LIFE[kind]


def is_change(topic: str, label: str) -> bool:
    """The value is a change (inflation, «рост … г/г»), not a level (the key rate, unemployment)."""
    return topic == 'инфляция' or bool(CHANGE.search(label))


def kind_of(text: str, period: str, report_date: str) -> str:
    """forecast: the words say so, or the horizon is the end of a year or a later year than the report's."""
    later = any(int(y) > int(report_date[:4]) for y in re.findall(r'20\d{2}', period))
    return 'forecast' if FORECAST.search(text) or later or re.search(r'конец|к концу|ближайш', period, re.I) else 'actual'


def checked(x: dict) -> bool:
    """A figure the code may rely on: printed in a text or table, or a chart label confirmed on the pixels."""
    return x.get('verified', True) and x.get('status', '') not in ('approx', 'unverified') and \
        not SHAKY.search(x.get('sentence', '')) and SCALE_MARK not in x.get('sentence', '')


def printed(value: float, fragments: list[Fragment], unit: str = '', topic: str = '') -> Fragment | None:
    """The fragment of the page that prints this number as a checked figure (text and tables first)."""
    for f in sorted(fragments, key=lambda f: (f.kind == 'chart', not f.own_view)):
        if any(abs(x['value'] - value) < 1e-6 and checked(x)
               and (not unit or x['unit'] == unit) and (not topic or x['indicator'] == topic)
               for x in f.figures) and (f.kind != 'chart' or not f.unverified):
            return f
    return None


def quoted(quote: str, fragments: list[Fragment]) -> Fragment | None:
    """The fragment of the page that holds most stems of the quote (a qualitative thesis has no number to check)."""
    q = set(stems(quote))
    if len(q) < 3:
        return None
    best = max(fragments, key=lambda f: len(q & set(stems(f.text))), default=None)
    return best if best is not None and len(q & set(stems(best.text))) >= 0.7 * len(q) else None


def level_move(topic: str, label: str, value: float | None, unit: str, said: str) -> str:
    """A growth rate of a cost in % moves its level by the sign (wages +10% г/г: the payroll grows, though the bank may
    call it a slowdown); demand, investment and levels keep the bank's word (a slowdown of retail is a risk for a shop)."""
    if value is not None and unit == '%' and topic in COST_TOPICS and is_change(topic, label):
        return 'up' if value > 0 else 'down' if value < 0 else 'flat'
    return said


def first_number(text: str, unit_hint: str = '') -> tuple[float, str] | None:
    """The first number of the text with its unit (the hint's unit when the text has none: «%» of «%н/н»); a fall named
    before it («спад на 4%») makes it negative."""
    for m in NUMBER.finditer(text):
        value = numeric(m.group(1))
        hint = NUMBER.match(f'0 {unit_hint.strip()}')
        unit = (m.group(2) or (hint.group(2) if hint else '') or '').lower().replace(' ', '')
        if not unit and 1990 <= value <= 2100:
            continue  # a year is a period
        if value > 0 and FALL.search(text[:m.start()][-25:] + text[m.start():m.end()]):
            value = -value
        return value, unit
    return None


def make(i: int, topic: str, label: str, said: str, number, period: str, kind: str, quote: str, f: Fragment, source: str,
         value_text: str) -> Thesis:
    rk = report_kind(f.file, f.report)
    when = date.fromisoformat(f.date)
    value, unit = number if number else (None, '')
    stem = Path(f.file).stem
    return Thesis(f'thesis:{stem}:p{f.page}:{topic}:{i}', topic, label, said,
                  level_move(topic, f'{label} {value_text}', value, unit, said), value, unit, value_text, period, kind,
                  f.report, f.file, f.page, f.date, f.id, quote[:300], source, rk,
                  (when + timedelta(days=shelf_life(rk, topic))).isoformat())


def from_digest(view: dict, i: int, page_fragments: list[Fragment]) -> Thesis | None:
    """A bank view of a digest -> a thesis, if the digest checked it and the page prints its number (or its quote)."""
    label, value_text, quote = view.get('indicator', ''), str(view.get('value', '')), view.get('quote', '')
    topics = [t for t in topics_of(label) or topics_of(quote) if t not in NOT_BUSINESS]
    if view.get('unverified') or not topics or not page_fragments or SHAKY.search(f'{value_text} {view.get("unit", "")}'):
        return None
    if ADVICE.search(f'{label} {value_text} {quote}') or GRANULAR.search(f'{label} {value_text}'):
        return None
    number = first_number(value_text, (view.get('unit') or '').strip())
    code, model = direction(value_text) or direction(quote), view.get('direction') or ''
    said = code or model  # the code's reading of the words first
    if said not in ('up', 'down', 'flat') or not number and (model not in ('up', 'down', 'flat') or code and code != model):
        return None  # a thesis without a number needs the digest's direction and the words not saying otherwise
    page_fragments = [f for f in page_fragments if f.own_view and topics[0] in f.topics]
    f = (printed(abs(number[0]), page_fragments, number[1], topics[0]) or
         printed(number[0], page_fragments, number[1], topics[0])) if number else quoted(quote, page_fragments)
    if f is None or not f.date:
        return None
    period = view.get('horizon') or ''
    return make(i, topics[0], label, said, number, period, kind_of(f'{label} {quote}', period, f.date), quote or value_text, f,
                'digest', value_text)


def from_fragment(f: Fragment) -> list[Thesis]:
    """Own-view sentences of a text fragment («мы ожидаем…») with a topic and a direction, without «≈»."""
    out = []
    for s in sentences(f.text):
        if not is_own_view(s) or SHAKY.search(s) or SCALE_MARK in s or ADVICE.search(s) or EARLIER.search(s) or GRANULAR.search(s):
            continue
        said, topics = direction(s), topics_of(s)
        if not said or len(topics) != 1 or topics[0] in NOT_BUSINESS:
            continue
        fig = next((x for x in figures_of(s) if x.indicator == topics[0] and x.unit == '%' and x.kind == 'forecast'), None)
        if fig is None:
            continue
        value_text = f'{fig.value:g}'.replace('.', ',') + '%'
        out.append(make(len(out) + 1, topics[0], TOPIC_LABEL.get(topics[0], topics[0]), said, (fig.value, '%'), fig.period,
                        'forecast', s, f, 'fragment', value_text))
    return out


def candidates(library: BankLibrary, folder: Path = DIGEST_DIR) -> list[Thesis]:
    """Every thesis of the library, expired ones included (the digest's views first, one per report, topic and number)."""
    by_page: dict[tuple[str, int], list[Fragment]] = {}
    for f in library.fragments:
        by_page.setdefault((Path(f.file).stem, f.page), []).append(f)
    out, digested = [], set()
    for stem in sorted({Path(f.file).stem for f in library.fragments if f.date}):
        path = folder / f'{stem}.digest.json'
        if not path.exists():
            continue
        from stage5_insight_detection.bank_analytics.digest import ReportDigest, verify
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
            md = folder / f'{stem}.md'
            if payload.get('source_sha256') and (not md.exists() or
                    hashlib.sha256(md.read_bytes()).hexdigest() != payload['source_sha256']):
                continue  # do not promote a stale digest to a current bank position
            pages = {}
            for f in library.fragments:
                if Path(f.file).stem == stem:
                    pages[f.page] = pages.get(f.page, '') + '\n' + f.brief_text()
            digest = verify(ReportDigest.model_validate(payload), pages)
        except (ValueError, TypeError, OSError):
            continue
        for i, view in enumerate(digest.bank_views, 1):
            v = view.model_dump()
            t = from_digest(v, i, by_page.get((stem, v.get('page') or 0), []))
            if t:
                out.append(t)
                digested.add(stem)
    for f in library.fragments:
        if f.own_view and f.kind == 'text' and f.date and Path(f.file).stem not in digested:
            out += from_fragment(f)
    seen, unique = set(), []
    for t in out:
        k = (t.file, t.topic, t.value if t.value is not None else t.fragment)
        if k not in seen:
            seen.add(k)
            unique.append(t)
    return unique


def collect(library: BankLibrary, as_of: date | None = None, folder: Path = DIGEST_DIR) -> list[Thesis]:
    """The theses alive on `as_of`."""
    as_of = as_of or date.today()
    return [t for t in candidates(library, folder) if t.fresh(as_of)]


def delta(t: Thesis, library: BankLibrary) -> tuple | None:
    """The change the estimate may use, derived by code from the bank's data: a growth rate in % as is; a key rate
    forecast minus the current key rate the same report prints as a checked actual; anything else none."""
    if t.value is None or t.unit != '%':
        return None
    if t.topic == 'ставка' and t.kind == 'forecast':
        now = [(f, x) for f in library.fragments if f.file == t.file for x in f.figures
               if x['indicator'] == 'ставка' and x['kind'] == 'actual' and x['unit'] == '%'
               and x.get('measure') != 'change' and checked(x)]
        if not now:
            return None
        f, x = now[0]
        return round(t.value - x['value'], 2), 'п.п.', t.period, [t.fragment, f.id]
    if is_change(t.topic, f'{t.indicator} {t.value_text}') and t.move == ('up' if t.value > 0 else 'down'):
        return t.value, '%', t.period, [t.fragment]  # a slowdown to +1.5% is no «+1.5% of revenue» for a risk
    return None


def select(theses: list[Thesis], profile: CompanyProfile, library: BankLibrary) -> list[Pick]:
    """Theses that reach the company at level mid or higher, one per topic (the freshest report, then the strongest),
    strongest first."""
    picks = []
    for t in theses:
        f = library.by_id.get(t.fragment)
        if f is None:
            continue
        d = delta(t, library)
        event = EventCard.from_thesis(t, IMPORTANCE[t.value is not None])
        cons, _ = impact.assess(event, profile, [BankMatch(f, 1.0, {}, '')], deltas={t.topic: d})
        c = next((c for c in cons if c.topic == t.topic and c.level in LEVELS), None)
        if c is None:
            continue
        why = ', '.join(profile.reasons.get(t.topic, ['fact:okved']))
        picks.append(Pick(t, c.score, c.level, c.sign, f'{c.mechanism} (чувствительность {c.exposure:.1f}: {why})',
                          profile.reasons.get(t.topic, ['fact:okved']), d))
    best = {}
    for p in sorted(picks, key=lambda p: (p.thesis.report_date, p.score, p.delta is not None), reverse=True):
        best.setdefault(p.thesis.topic, p)
    return sorted(best.values(), key=lambda p: -p.score)


class ShownLog:
    """Which theses each company has seen: {company: [{'key', 'thesis', 'how', 'at'}]} in a gitignored JSON."""

    def __init__(self, path: Path = SHOWN):
        self.path = path
        self.rows: dict[str, list[dict]] = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}

    def blocked(self, company: str, t: Thesis, now: datetime) -> str:
        """Why the thesis is not shown now, or ''."""
        rows = self.rows.get(company, [])
        if any(r['key'] == t.key() for r in rows):
            return 'тезис уже показан компании: снова — только с новым отчётом или сменой позиции'
        recent = [r for r in rows if r['how'] == 'insight' and now - datetime.fromisoformat(r['at']) < timedelta(days=WINDOW_DAYS)]
        if len(recent) >= MAX_PER_WINDOW:
            return f'лимит: не более {MAX_PER_WINDOW} инсайтов по аналитике Банка за {WINDOW_DAYS} дней'
        return ''

    def record(self, company: str, t: Thesis, now: datetime, how: str = 'insight') -> None:
        """how: insight (its own card, counts to the limit) | news (in the bank view of a news insight)."""
        self.rows.setdefault(company, []).append({'key': t.key(), 'thesis': t.id, 'how': how, 'at': now.isoformat(timespec='seconds')})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_json(self.path, self.rows)


def attach(result: dict, t: Thesis, fragment: Fragment) -> None:
    """The thesis into the bank view of a news insight on the same topic (instead of its own card): a code-made claim
    citing the fragment that prints it."""
    ins = result['insight']
    if not fragment.own_view or fragment.stale:
        return
    if not any(t.fragment in c.get('sources', []) for c in ins.get('bank_view', [])):
        ins.setdefault('bank_view', []).append({'text': f'{t.text()} ({t.report}, {t.report_date}, стр. {t.page})',
                                                'sources': [t.fragment], 'kind': 'bank', 'unconfirmed': False})
    if t.fragment not in {b['id'] for b in result.get('bank', [])}:
        result.setdefault('bank', []).append(BankMatch(fragment, 1.0, {'thesis': t.id}, 'тезис Банка').dict())
    brief = (result.get('verdict') or {}).get('brief')
    if brief is not None:
        if t.fragment not in brief['refs']:
            brief['refs'].append(t.fragment)
        if t.fragment not in {b['id'] for b in brief['bank']}:
            brief['bank'].append(BankMatch(fragment, 1.0, {'thesis': t.id}, 'тезис Банка').dict())
        brief['not_bank_position'] = False
    ins['not_bank_position'] = False
    result.setdefault('bank_theses', []).append(t.dict())


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='theses of the bank for companies (code only, no LLM)')
    ap.add_argument('company_id', nargs='*')
    ap.add_argument('--as-of', type=date.fromisoformat, default=None)
    a = ap.parse_args()
    lib = BankLibrary.load(as_of=a.as_of)
    alive = collect(lib, a.as_of)
    print(f'{len(candidates(lib))} theses, {len(alive)} alive')
    for cid in a.company_id:
        from stage2_company_facts.profile.integration import load_company
        for p in select(alive, CompanyProfile(**load_company(cid).card), lib):
            print(f'{cid} [{p.level} {p.score:.2f}] {p.thesis.topic}: {p.thesis.report} ({p.thesis.report_date}); {p.channel}')
