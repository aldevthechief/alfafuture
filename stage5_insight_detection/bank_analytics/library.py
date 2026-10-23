"""Task 1: parse the bank's analytics into citable fragments -> data/analytics/fragments.jsonl.

Folder loader: drop a new PDF into data/analytics/ and run this script; only new or changed files are parsed (registry
by hash of the file and its OCR). Per page the text layer is used if the PDF has one, otherwise the OCR output of ocr.py:
the full run (data/analytics/ocr_full) cleaned by pages.clean_page with the old run as the fallback for broken pages, or
the old run alone (`--ocr` runs the vision model for pages that have none yet).
A page is cut by its structure: every [График] block is one `chart` fragment, every Markdown table one `table` fragment
(a long one cut by rows, the header repeated), the rest `text` paragraphs. Chart values the model read off the scale
("≈") or of a block flagged ⚠ stay in the text marked "по шкале, не проверено" and in figures with verified=False: kept,
not dropped, so the agent and the card can say what the chart shows while every computation (impact, enrich, audit)
skips them; such a fragment is `unverified`. Table cells become figures with the column header as the period
("2026П" is a forecast). Every fragment carries what the later steps need, computed by code:
- id `bank:<file stem>:p<page>.<n>`, kind, report title, date, page, heading (bold lead "...:" or a short first line;
  "график Банка, рис. N, стр. M: <title>" for a chart, the table title for a table);
- topics from a keyword map (ставка, инфляция, курс, потребление ...), the same map tags news, so matching is symmetric;
- figures: numbers with unit, indicator, period and kind (forecast / actual) taken from the sentence around them;
- own_view: the sentence is the bank's own position ("мы ожидаем", "наш прогноз"), not a quote of the CB or Minfin;
- stance: direction of the bank's view (up / down / flat) for consistency checks;
- freshness against `as_of`: fresh <= 60 days, aging <= 365, stale > 365, unknown when no date could be found.
Report dates come from reports.json; a new PDF gets its date from the file name (22072026, 11Sep, August26, 15Apr2021)
or from the first page ("11 сентября 2026 г."). Deterministic, no LLM except the optional OCR.
python -X utf8 stage5_insight_detection/bank_analytics/library.py [--ocr] [--rebuild] [--as-of 2026-10-20]
python -X utf8 stage5_insight_detection/bank_analytics/library.py --search "ключевая ставка прогноз"
"""
import argparse, calendar, hashlib, json, re, sys
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.search import StemIndex
from stage5_insight_detection.bank_analytics.ocr import OCR_DIR, OCR_FULL_DIR, PDF_DIR
from stage5_insight_detection.bank_analytics.digitize import PHANTOM, annotate, page_grounded, page_records
from stage5_insight_detection.bank_analytics import tables as table_reading
from stage5_insight_detection.bank_analytics import chart_novelty
from stage5_insight_detection.bank_analytics.pages import CHART_BLOCK, clean_page
from stage5_insight_detection.bank_analytics.search import REPORTS, paragraphs

FRAGMENTS = PDF_DIR / 'fragments.jsonl'
REGISTRY = PDF_DIR / 'registry.json'
PARSER_VERSION = 'combined-pdf-v4-monthly-forecasts'

# topic -> word beginnings; shared with news tagging (analyst_loop) so a news item and a fragment speak one vocabulary
TOPICS = {
    'ставка': ['ключев', 'ставк', 'дкп', 'денежно-кредит', 'смягчен', 'ужесточен'],
    'инфляция': ['инфляц', 'ипц', 'рост цен', 'дефлятор', 'инфляционн'],
    'курс': ['курс', 'рубл', 'доллар', 'юан', 'валют', 'евро'],
    'потребление': ['потреб', 'розничн', 'спрос населен', 'потребительск', 'оборот торговли', 'расходы населен'],
    'кредит': ['кредит', 'займ', 'заем', 'заёмщик', 'ипотек', 'фондирован'],
    'бюджет_и_налоги': ['бюджет', 'минфин', 'дефицит', 'профицит', 'налог', 'ндс', 'ндфл', 'пошлин'],
    'облигации': ['облигац', 'офз', 'доходност', 'купон', 'размещен', 'эмитент', 'дюрац', 'спред'],
    'золото': ['золот', 'драгмет', 'драгоценн'],
    # no «стали»: it is also a verb («россияне стали экономить»)
    'металлургия': ['сталь', 'стальн', 'сталелит', 'цен на стал', 'металлург', 'прокат', 'чугун', 'железн руд', 'северстал', 'нлмк', 'ммк'],
    'строительство': ['строител', 'стройк', 'жиль', 'девелоп', 'ввод жиль', 'цемент'],
    'промышленность': ['промышлен', 'обрабатывающ', 'производств', 'загрузк мощност', 'pmi', 'индекс деловой'],
    'энергия_и_сырьё': ['нефт', 'газ', 'топлив', 'бензин', 'дизел', 'уголь', 'urals', 'brent', 'экспорт сырь'],
    'рынок_труда': ['безработ', 'занятост', 'зарплат', 'заработн', 'дефицит кадр', 'рынок труда'],
    'инвестиции': ['инвестиц', 'капвложен', 'капитальн вложен', 'capex'],
    'внешняя_торговля': ['импорт', 'экспорт', 'внешнеторгов', 'торгов баланс', 'санкц', 'логистик'],
    'ставка_мир': ['фрс', 'fomc', 'ецб', 'федеральной резервн', 'федеральная резервн'],
}
# a rate of a foreign central bank is not the key rate of the Bank of Russia: «ставка ФРС», «ЦБ Индии повысил ставку»
FOREIGN_CB = re.compile(r'фрс|fomc|ецб|федеральн\w* резервн|(цб|центробанк\w*|нацбанк\w*|народн\w* банк\w*) (сша|индии|китая|турции|'
                        r'бразилии|японии|англии|казахстана|евро\w*)', re.I)
RUSSIAN_CB = re.compile(r'банк\w* росси|цб рф|центробанк\w* рф|ключев\w* ставк\w* (в россии|цб рф)|\bрф\b|росси', re.I)
DISCLAIMER = re.compile(r'генеральная лицензия цб рф|несанкционированное копирование|не делают? заявлений', re.I)
TOPIC_KEYS = {name: re.compile('|'.join(r'(?<![а-яa-z])' + re.escape(k) for k in keys)) for name, keys in TOPICS.items()}
FORECAST = re.compile(r'прогноз|ожида(ем|ет|ют|ется|лось|ли|л|ла)\b|к концу|оценива|предполага|сценари|допуска|вероятн|до конца года', re.I)
OWN_VIEW = re.compile(r'\b(мы|нам|наш\w*)\b[^.]{0,80}?(ожида|жд[её]м|прогноз|считаем|полагаем|рекоменд|взгляд|мнени|оценк|оценива|видим|сохраня)'
                      r'|(ожида|прогноз|рекоменд)\w*[^.]{0,40}\b(мы|наш\w*)\b', re.I)
QUOTED_VIEW = re.compile(r'(?:по словам|как (?:заяв|сообщ|отмеч))|'
                         r'(?:\bцб\b|банк\w* росси\w*|минфин\w*|росстат\w*|\bфрс\b)[^.?!]{0,100}[:«]', re.I)
# «рос\w*» would take «Россия»: growth words are spelled out; «сократилось», «упало» are falls too
UP = re.compile(r'(?<![а-яё])(не\s+)?(рост\w*|росл\w*|вырос\w*|выраст\w*|ускор\w*|повыш\w*|повыс\w*|подня\w*|подним\w*|увелич\w*|ужесточ\w*|'
                r'ослаблен\w* рубл\w*|подорожа\w*|подъем\w*|подъём\w*|дорожа\w*)', re.I)
DOWN = re.compile(r'(?<![а-яё])(не\s+)?(сниж\w*|сниз\w*|замедл\w*|паден\w*|упал\w*|упад\w*|сокращ\w*|сократ\w*|смягч\w*|'
                  r'укреплен\w* рубл\w*|подешев\w*|дешеве\w*|уменьш\w*|спад\w*)', re.I)
FLAT = re.compile(r'(сохран\w*|без изменени\w*|пауз\w*|неизменн\w*|стабилиз\w*)', re.I)
# the rouble is the subject: its rise is a fall of the exchange rate (руб./$), its fall a rise («ослабление рубля» = up)
RUB_STRONGER = re.compile(r'рубл\w* (на подъем|на подъём|укреп\w*|вырос\w*|раст\w*|подорожа\w*)|укреплен\w* (курса )?рубл', re.I)
RUB_WEAKER = re.compile(r'рубл\w* (упал\w*|подешев\w*|ослаб\w*|сниз\w*|теря\w*)|ослаблен\w* (курса )?рубл', re.I)
NUMBER = re.compile(r'(?<![\w.,])([-+−]?(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)(?:[.,]\d+)?)\s*'
                    r'(%|п\.\s?п\.|пп\b|б\.\s?п\.|бп\b|(?:трлн|млрд|млн|тыс\.?)(?:\s*(?:руб\.?|₽|\$|долл\.?))?|руб\.?|₽|\$|долл\.?)?',
                    re.I)
PERIOD = re.compile(r'(\d[КQ]\d{2}|[1-4]\s?кв\w*\.?\s?20\d{2}|[1-2]П\d{2}|\d{1,2}М\d{2}|20\d{2}\s?г?)', re.I)
MONTHS_RU = {m: i for i, m in enumerate(['январ', 'феврал', 'март', 'апрел', 'ма', 'июн', 'июл', 'август', 'сентябр', 'октябр',
                                         'ноябр', 'декабр'], 1)}
MONTHS_EN = {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}
FULL_EN = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}


def topics_of(text: str) -> list[str]:
    """Topics whose word beginnings start a word of the text (not «санкц» inside «несанкционированное»); a disclaimer has
    none; a rate of a foreign central bank without a word on Russia is «ставка_мир», not the key rate."""
    t = f' {text.lower().replace("ё", "е")} '
    if DISCLAIMER.search(t):
        return []
    out = [name for name, rx in TOPIC_KEYS.items() if rx.search(t)]
    if 'ставка' in out and FOREIGN_CB.search(t) and not RUSSIAN_CB.search(t):
        out = [x for x in out if x != 'ставка'] + ([] if 'ставка_мир' in out else ['ставка_мир'])
    return out


@dataclass
class Figure:
    value: float
    unit: str
    indicator: str  # topic of the sentence; '' if none
    kind: str  # forecast | actual
    period: str
    sentence: str
    verified: bool = True  # False: a chart value not printed or not confirmed (status says which)
    status: str = ''  # charts: verified (printed, confirmed on the pixels) | approx (measured by code) | unverified (model)
    measure: str = 'unknown'  # level | change | unknown; independent of chart verification status


@dataclass
class Fragment:
    id: str
    file: str
    report: str
    date: str | None
    page: int
    n: int
    heading: str
    text: str
    topics: list[str]
    figures: list[dict]
    own_view: bool
    stance: dict = field(default_factory=dict)  # topic -> up / down / flat, from own-view sentences first
    freshness: str = 'unknown'
    age_days: int | None = None
    kind: str = 'text'  # text | chart | table
    unverified: bool = False  # holds chart values read off the scale

    def caption(self) -> str:
        """Title of the fragment in prompts and on the card."""
        return f'{self.report} — {self.heading}' if self.kind in ('chart', 'table') else self.report

    def brief_text(self) -> str:
        """Text for a model prompt. A chart loses what is not confirmed: the page OCR's "Значения", the ⚠ line, labels
        the pixels did not confirm (one note instead); the confirmed labels stay, the code's values go as «≈ N»;
        "Что видно" is marked as the model's description. text/ and the card keep it all."""
        if self.kind != 'chart':
            return self.text
        out, note_at, in_values, hidden, measured = [], None, False, False, False
        for line in self.text.split('\n'):
            s = line.strip()
            if s.startswith('⚠') or s.startswith('Подписи на графике (прочитаны моделью'):
                note_at = len(out) if note_at is None else note_at
                continue
            if s.startswith('Оцифровано кодом'):
                measured = True
                out.append(line.replace('Оцифровано кодом по графику', 'По графику, оцифровано кодом (≈)'))
                continue
            if measured and s.startswith('|') and not s.startswith('| дата'):
                out.append(re.sub(r'(?<=\| )([-−]?\d)', r'≈\1', line))  # every measured value is shown as «≈ N»
                continue
            if s.startswith('Значения') and not s.startswith('Значения не определены'):
                in_values = True
                head, _, rest = s.partition(':')
                hidden = True  # the page OCR's values: a whole-page reading made most of them up
                items = [i.strip() for i in rest.split(';') if i.strip()]
                keep = []
                if hidden or len(keep) < len(items):
                    note_at = len(out) if note_at is None else note_at
                if keep:
                    out.append('Значения: ' + '; '.join(keep))
                continue
            if in_values and not re.match(r'(Что видно|Источник|Подписи на графике|Оцифровано кодом|Значения не определены)', s):
                note_at = len(out) if note_at is None else note_at
                continue  # every continuation line of the page OCR's values
            in_values = False
            if s.startswith('Что видно'):
                line = 'Описание графика (распознано моделью):' + s.partition(':')[2]
            out.append(line)
        if note_at is not None:
            out.insert(note_at, f'Значения на графике распознаны ненадёжно и не передаются; см. PDF, стр. {self.page}.')
        return '\n'.join(out)

    def approx(self) -> set[float]:
        """Values no printed number supports (the model's reading, the code's measurement): never a confirmation as such."""
        sure = {abs(x['value']) for x in self.figures if x.get('verified', True)}
        return {abs(x['value']) for x in self.figures if not x.get('verified', True)} - sure

    def measured(self) -> set[float]:
        """Values the code measured on the chart: a claim may use them only as «≈ N» or «около N»."""
        return {abs(x['value']) for x in self.figures if x.get('status') == 'approx'}

    @property
    def stale(self) -> bool:
        return self.freshness in ('stale', 'unknown')


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?;])\s+(?=[А-ЯЁA-Z«"(])|\n', text) if s.strip()]


def direction(sentence: str) -> str | None:
    """up / down / flat of a sentence; a negated move («не снизилась», «не вырос») is no move; for the rouble as the subject
    the direction is that of the exchange rate."""
    if RUB_STRONGER.search(sentence) or RUB_WEAKER.search(sentence):
        return 'down' if RUB_STRONGER.search(sentence) else 'up'
    up = any(not m.group(1) for m in UP.finditer(sentence))
    down = any(not m.group(1) for m in DOWN.finditer(sentence))
    negated = any(m.group(1) for rx in (UP, DOWN) for m in rx.finditer(sentence))
    if FLAT.search(sentence) or (negated and not up and not down):
        return 'flat'  # "оставить неизменной, снижение не рассматривалось" is a pause, not a cut
    if up != down:
        return 'up' if up else 'down'
    return None


def measurement(sentence: str, match: re.Match) -> str:
    """Classify the numeric mention, not the whole sentence (it may contain both «на 2%» and «до 8%»)."""
    before = sentence[:match.start()].lower()
    unit = (match.group(2) or '').lower().replace(' ', '')
    if re.search(r'(?:\bдо|\bс|\bот|уровн[ея]|состав\w*|достиг\w*)\s*$', before):
        return 'level'
    if re.search(r'\bна\s*$', before) or match.group(1).startswith(('+', '-', '−')):
        return 'change'
    if unit in ('п.п.', 'пп', 'б.п.', 'бп'):
        return 'change'
    if re.search(r'\b(?:рост|прирост|снижение|сокращение|падение)\s*(?:на\s*)?$', before):
        return 'change'
    return 'unknown'


def figures_of(text: str) -> list[Figure]:
    out = []
    for s in sentences(text):
        topic = (topics_of(s) or [''])[0]
        kind = 'forecast' if FORECAST.search(s) else 'actual'
        periods = [(m.start(), m.group(1).strip()) for m in PERIOD.finditer(s)]
        for m in NUMBER.finditer(s):
            raw, unit = m.group(1), (m.group(2) or '').lower().replace(' ', '')
            if not unit and not topic:
                continue  # a bare number outside any topic is a page number, a count or a year
            value = numeric(raw)
            if not unit and 1990 <= value <= 2100:
                continue  # years are periods, not figures
            after = [p for pos, p in periods if m.end() <= pos <= m.end() + 40]
            before = [p for pos, p in periods if pos < m.start()]
            period = after[0] if after else before[-1] if before else ''
            out.append(Figure(value, unit, topic, kind, period, s[:300], measure=measurement(s, m)))
    return out


def heading_and_body(paragraph: str) -> tuple[str, str]:
    first, _, rest = paragraph.partition('\n')
    if len(first) < 120 and not first.rstrip().endswith('.') and rest:
        return first.strip(' #*'), paragraph
    lead = re.match(r'^(.{15,200}?):\s', paragraph)
    return (lead.group(1).strip(' *') if lead else ''), paragraph


def stance_of(text: str) -> tuple[bool, dict]:
    own = [s for s in sentences(text) if is_own_view(s)]
    stance = {}
    for s in own:
        d = direction(s)
        for t in topics_of(s):
            if d and t not in stance:
                stance[t] = d
    return bool(own), stance


def is_own_view(sentence: str) -> bool:
    match = OWN_VIEW.search(sentence)
    return bool(match and not QUOTED_VIEW.search(sentence[:match.start()]))


def date_from_name(name: str, default_year: int) -> date | None:
    stem = Path(name).stem
    m = re.search(r'(?<!\d)(\d{2})(\d{2})(20\d{2})(?!\d)', stem)
    if m:
        try:
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            pass
    m = re.search(r'(?<![A-Za-z])(\d{1,2})(' + '|'.join(MONTHS_EN) + r')[a-z]*(20\d{2})?', stem, re.I)
    if m:
        return date(int(m.group(3) or default_year), MONTHS_EN[m.group(2).lower()[:3]], int(m.group(1)))
    m = re.search(r'(' + '|'.join(FULL_EN) + r')(\d{2})(?!\d)', stem, re.I)
    if m:
        y, mo = 2000 + int(m.group(2)), FULL_EN[m.group(1).lower()]
        return date(y, mo, calendar.monthrange(y, mo)[1])
    return None


def date_from_text(text: str) -> date | None:
    m = re.search(r'(\d{1,2})\s+([а-яё]+)\s+(20\d{2})\s*г', text.lower())
    if m:
        month = next((i for k, i in sorted(MONTHS_RU.items(), key=lambda kv: -len(kv[0])) if m.group(2).startswith(k)), None)
        if month:
            return date(int(m.group(3)), month, int(m.group(1)))
    m = re.search(r'(\d{2})\.(\d{2})\.(20\d{2})', text)
    return date(int(m.group(3)), int(m.group(2)), int(m.group(1))) if m else None


def freshness(report_date: date | None, as_of: date) -> tuple[str, int | None]:
    if report_date is None:
        return 'unknown', None
    age = (as_of - report_date).days
    if age < 0:
        return 'unknown', age
    return ('fresh' if age <= 60 else 'aging' if age <= 365 else 'stale'), age


def text_layer(pdf: Path) -> tuple[list[str], bool]:
    """Page texts of the PDF's own text layer and whether it is usable (on average 50+ chars a page; scans have none)."""
    import pymupdf
    with pymupdf.open(pdf) as doc:
        layer = [page.get_text() for page in doc]
    return layer, sum(len(t.strip()) for t in layer) > 100 * len(layer) * 0.5


def page_texts(pdf: Path, ocr_dir: Path = OCR_DIR, ocr=None) -> list[str]:
    """Choose text per page: mixed PDFs may contain both text pages and scans."""
    layer, _ = text_layer(pdf)
    cache = ocr_dir / f'{pdf.stem}.jsonl'
    if ocr is not None:
        missing = [n for n, text in enumerate(layer, 1) if len(text.strip()) < 50]
        read = ocr.transcribe(pdf, only_pages=missing) if missing else layer
        return [text if len(text.strip()) >= 50 else read[n] or text for n, text in enumerate(layer)]
    if cache.exists():
        manifest = ocr_dir / f'{pdf.stem}.source.json'
        if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')).get('pdf_sha256') != hashlib.sha256(pdf.read_bytes()).hexdigest():
            return layer  # the source PDF was replaced; do not cite its previous OCR
        pages = {int(d['key']): d['value'] for d in map(json.loads, filter(None, cache.read_text(encoding='utf-8').splitlines()))}
        return [text if len(text.strip()) >= 50 else pages.get(n, '') or text for n, text in enumerate(layer, 1)]
    return layer


def best_pages(pdf: Path, full_dir: Path = OCR_FULL_DIR, ocr_dir: Path = OCR_DIR, ocr=None, report_year: int | None = None,
               charts: str | None = None) -> list[str]:
    """Text layer if present; else the full OCR run cleaned, with the old run under pages where the full one broke
    (pages.clean_page, the same as export_text.py), the charts read crop by crop where charts.py ran
    (digitize.annotate) and the tables read three times and reconciled where tables.py ran; else the old run.
    `charts` (chart_novelty.MODES, default env BANK_CHARTS or all): selective keeps only the charts that may say what
    the report's text does not, off keeps none; a dropped chart leaves no line for the agent."""
    old = page_texts(pdf, ocr_dir, ocr)
    main = page_texts(pdf, full_dir) if (full_dir / f'{pdf.stem}.jsonl').exists() else old
    report = chart_report(pdf, main, old, report_year, charts)
    pages = [prepare_page(pdf, n, raw, fallback, report_year, charts, report)[0]
             for n, (raw, fallback) in enumerate(zip(main, old), 1)]
    return ['\n'.join(ln for ln in p.split('\n') if not ln.startswith(chart_novelty.SKIPPED)) for p in pages]


def chart_report(pdf: Path, main: list[str], old: list[str], report_year: int | None, charts: str | None = None):
    """The report's cleaned text for chart_novelty under `selective`; None for the other modes."""
    if chart_novelty.mode_of(charts) != 'selective':
        return None
    return chart_novelty.Report(pdf.stem, [clean_page(raw, fallback, report_year)[0] for raw, fallback in zip(main, old)])


def prepare_page(pdf: Path, n: int, raw: str, fallback: str = '', report_year: int | None = None, mode: str | None = None,
                 report: 'chart_novelty.Report | None' = None):
    """One preparation path for fragments and the Markdown used by report digests. `mode` and `report`: which charts
    are kept (chart_novelty.select; selective needs the report from chart_report)."""
    charts = pdf.parent / 'charts'
    from stage5_insight_detection.bank_analytics.assets import current
    text, stats = clean_page(raw, fallback, report_year)
    fresh = current(pdf, charts)
    text, records = chart_novelty.select(text, page_records(pdf.stem, n, charts) if fresh else [], report, n,
                                         chart_novelty.mode_of(mode))
    if fresh:
        text = annotate(text, records, page_grounded(pdf.stem, n, charts))
    tables = pdf.parent / 'tables'
    return table_reading.replace_tables(text, table_reading.page_records(pdf.stem, n, tables) if current(pdf, tables) else None), stats


UNIT_HINT = re.compile(r'(%|п\.\s?п\.|б\.\s?п\.|трлн|млрд|млн|руб|₽|\$|долл)', re.I)
FORECAST_COLUMN = re.compile(r'\d\s*[ПпEeFf]\b|прогноз|\bАБ\b', re.I)
SEPARATOR_ROW = re.compile(r'^:?-{2,}:?$')
SCALE_MARK = 'по шкале, не проверено'
CHECKED_TABLE = '[таблица сверена'
TABLE_CHARS = 1200


def numeric(raw: str) -> float:
    return float(re.sub(r'\s', '', raw).replace(',', '.').replace('−', '-'))


def hinted_unit(text: str) -> str:
    hints = UNIT_HINT.findall(text)
    if not hints:
        return ''
    scales = [h for h in hints if h.lower() in ('трлн', 'млрд', 'млн')]
    currencies = [h for h in hints if h.lower() in ('руб', '₽', '$', 'долл')]
    return ' '.join((scales[-1], currencies[-1])) if scales and currencies else hints[-1].lower().replace(' ', '')


def cells(line: str) -> list[str]:
    return [c.strip().strip('*').strip() for c in line.strip().strip('|').split('|')]


def structure(text: str) -> list[tuple[str, str, str]]:
    """Page -> (kind, title, body) segments in reading order: [График] blocks, Markdown tables (with the line above as
    the title when it is short), and the text between them."""
    out = []
    for part in re.split(r'(\[График\].*?\[/График\])', text, flags=re.S):
        if CHART_BLOCK.fullmatch(part.strip()):
            title = part.strip().split('\n', 1)[0].replace('[График]', '').strip()
            if out and out[-1][0] == 'text' and title:  # the chart title printed above the block is part of the chart
                kept = [ln for ln in out[-1][2].rstrip().split('\n') if ln.strip().rstrip('*') != title.rstrip('*')]
                out[-1] = ('text', '', '\n'.join(kept))
                if not out[-1][2].strip():
                    out.pop()
            out.append(('chart', '', part.strip()))
            continue
        buf, table = [], []
        for line in part.split('\n') + ['']:
            if line.strip().startswith('|'):
                table.append(line)
                continue
            if table:
                title = ''
                while buf and not buf[-1].strip():
                    buf.pop()
                if buf and len(buf[-1].strip()) < 200 and not buf[-1].strip().endswith('.'):
                    title = buf.pop().strip()
                if '\n'.join(buf).strip():
                    out.append(('text', '', '\n'.join(buf)))
                out.append(('table', title, '\n'.join(table)))
                buf, table = [], []
            buf.append(line)
        if '\n'.join(buf).strip():
            out.append(('text', '', '\n'.join(buf)))
    return out


def chart_values(block: str) -> str:
    m = re.search(r'Значения[^:]*:(.*?)(?=\n(?:Что видно|Источник|⚠)|\[/График\]|$)', block, re.S)
    return m.group(1) if m else ''


def checked_figures(block: str, title: str, unit: str, indicator: str) -> list[Figure]:
    """Figures of the lines digitize.annotate added: printed labels confirmed on the pixels (verified), labels it could
    not confirm (unverified), the code's table «дата | ряд…» and last/min/max lines (approx)."""
    out, series = [], []
    approx = False
    for line in block.split('\n'):
        s = line.strip()
        if s.startswith('Подписи на графике'):
            ok = 'сверены' in s.split(':', 1)[0]
            for item in s.split(':', 1)[1].split(';'):
                label, _, value = item.rpartition(':')
                m = NUMBER.search(value)
                if m:
                    out.append(Figure(numeric(m.group(1)), (m.group(2) or unit).lower(), indicator,
                                      'actual', label.split('—', 1)[-1].strip()[:40], f'{title}: {item.strip()}'[:300], ok,
                                      'verified' if ok else 'unverified'))
        elif s.startswith('Оцифровано кодом'):
            approx = True
        elif approx and s.startswith('|') and not series:
            series = cells(s)[1:]
        elif approx and s.startswith('|') and not SEPARATOR_ROW.match(cells(s)[0] or '--'):
            row = cells(s)
            for name, cell in zip(series, row[1:]):
                m = NUMBER.search(cell)
                if m:
                    out.append(Figure(numeric(m.group(1)), unit, indicator, 'actual', row[0][:40],
                                      f'{title} | {name} | {row[0]}: ≈{cell} (оцифровано кодом)'[:300], False, 'approx'))
        elif approx and 'последняя точка ≈' in s:
            name, _, rest = s.partition(':')
            for what, m in re.findall(r'(последняя точка|минимум|максимум) ≈ ([-−]?\d+(?:,\d+)?)', rest):
                out.append(Figure(float(m.replace(',', '.').replace('−', '-')), unit, indicator, 'actual', what,
                                  f'{title} | {name}: {what} ≈{m} (оцифровано кодом)'[:300], False, 'approx'))
    return out


def chart_figures(block: str, title: str, suspect: bool) -> list[Figure]:
    """Numbers of the "Значения" field (the page OCR: unverified, a whole-page reading made most of them up) and of the
    lines digitize.annotate added (checked_figures)."""
    unit_hint = hinted_unit(title.rsplit(',', 1)[-1])
    indicator = (topics_of(title) or topics_of(block) or [''])[0]
    out = []
    for item in re.split(r'[;\n]', chart_values(block)):
        label, _, text = item.rpartition(':')
        if '—' in text:  # "ряд — максимум ≈16,8 (июль '26)": numbers in the series name are not values
            label, _, text = text.partition('—')
        base = label.split('—', 1)[1].strip() if '—' in label else label.strip(' -')
        prev = 0
        for m in NUMBER.finditer(text):
            value = numeric(m.group(1))
            if not m.group(2) and 1990 <= value <= 2100 and value.is_integer() or text[max(0, m.start() - 1):m.start()] in "'’":
                continue  # a year, or the year of a month label (мае '24)
            ctx = re.sub(r'^[\s,;)(]+|[\s(≈:=]+$', '', text[prev:m.start()])
            after = re.match(r'\s*\(([^)]{1,30})\)', text[m.end():])
            prev = m.end() + (after.end() if after else 0)
            period = ' '.join(p for p in (ctx or base, after.group(1) if after else '') if p).strip()
            kind = 'forecast' if FORECAST.search(item) or FORECAST_COLUMN.search(period) else 'actual'
            out.append(Figure(value, (m.group(2) or unit_hint).lower().replace(' ', ''), indicator, kind, period[:40],
                              f'{title}: {item.strip()}'[:300], verified=False, status='unverified'))
    return out + checked_figures(block, title, unit_hint, indicator)


def chart_text(block: str, label: str, suspect: bool) -> str:
    """The block for the agent: labelled, values read off the scale marked; the [График] tags dropped."""
    lines = [label]
    for line in block.replace('[График]', '').replace('[/График]', '').strip().split('\n')[1:]:
        if line.startswith('Значения') and suspect:
            line = f'Значения ({SCALE_MARK}): ' + line.split(':', 1)[-1].strip()
        elif ('≈' in line or 'по шкале' in line) and not line.startswith(('Оцифровано', '|')) and 'последняя точка ≈' not in line:
            line = re.sub(r'≈\s*по шкале', '≈', line)
            line = re.sub(r'(≈[^;]*)', lambda m: f'{m.group(1).rstrip()} [{SCALE_MARK}]', line)
        lines.append(line)
    return '\n'.join(ln for ln in lines if ln.strip())


def table_parts(title: str, body: str) -> list[tuple[str, list[Figure]]]:
    """Table -> chunks of up to TABLE_CHARS with the header repeated, and figures from the cells."""
    rows = [cells(ln) for ln in body.split('\n') if ln.strip()]
    rows = [r for r in rows if not all(SEPARATOR_ROW.match(c) or not c for c in r)]
    if not rows:
        return []
    header, data = rows[0], rows[1:] or rows[:1]
    period_header = any(re.fullmatch(r'(?:19|20)\d{2}(?:[ПпEeFf]|\s*г\.?)?', c) or PERIOD.fullmatch(c) for c in header[1:])
    if not period_header and sum(bool(NUMBER.fullmatch(c.rstrip('.,'))) for c in header[1:]) > len(header[1:]) / 2:
        header, data = [''] * len(header), rows  # no header row: the first row is data, the periods are unknown
    head = [title] if title else []
    head += ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    chunks, cur = [], []
    for r in data:
        line = '| ' + ' | '.join(r) + ' |'
        if cur and len('\n'.join(head + cur + [line])) > TABLE_CHARS:
            chunks.append(cur)
            cur = []
        cur.append(line)
    chunks.append(cur)
    out = []
    for chunk in chunks:
        figs = []
        for r in (cells(ln) for ln in chunk):
            label = r[0]
            indicator = (topics_of(label) or topics_of(title) or [''])[0]
            for j, cell in enumerate(r[1:], 1):
                m = NUMBER.search(cell)
                if not m:
                    continue
                value = numeric(m.group(1))
                period = header[j] if j < len(header) else ''
                if not m.group(2) and 1990 <= value <= 2100 and value.is_integer() and not label:
                    continue
                unit = m.group(2) or hinted_unit(label) or hinted_unit(title)
                kind = 'forecast' if FORECAST_COLUMN.search(period) or FORECAST_COLUMN.search(label) else 'actual'
                checked = CHECKED_TABLE in title  # tables.py: three readings reconciled cell by cell
                shaky = checked and cell.rstrip().endswith('(?)')
                figs.append(Figure(value, unit, indicator, kind, period[:40], f'{title} | {label} | {period}: {cell}'[:300],
                                   not shaky, ('unverified' if shaky else 'verified') if checked else ''))
        out.append(('\n'.join(head + chunk), figs))
    return out


def page_fragments(text: str, page_no: int) -> list[dict]:
    """Fragment fields (kind, heading, text, figures, own_view, stance, unverified) of one page in reading order."""
    out, charts = [], 0
    for kind, title, body in structure(text):
        if kind == 'chart' and PHANTOM in body:
            continue  # a chart block the page OCR invented: grounding found no chart for it, the agent does not get it
        if kind == 'chart':
            charts += 1
            title = body.split('\n', 1)[0].replace('[График]', '').strip() or 'без названия'
            n = re.search(r'Рис\.?\s*(\d+)', title)
            label = f'график Банка, рис. {n.group(1) if n else charts}, стр. {page_no}: {title}'
            suspect = '⚠' in body
            figs = chart_figures(body, title, suspect)
            out.append({'kind': 'chart', 'heading': label, 'text': chart_text(body, label, suspect), 'figures': figs,
                        'own_view': False, 'stance': {},  # unverified: only the model's reading, nothing measured or confirmed
                        'unverified': (suspect or any(f.status == 'unverified' for f in figs)) and not any(f.status in ('verified', 'approx') for f in figs)})
        elif kind == 'table':
            for i, (chunk, figs) in enumerate(table_parts(title, body)):
                heading = f'таблица Банка, стр. {page_no}: {title or "без названия"}' + (f' (часть {i + 1})' if i else '')
                out.append({'kind': 'table', 'heading': heading, 'text': chunk, 'figures': figs, 'own_view': False,
                            'stance': {}, 'unverified': False})
        else:
            for para in paragraphs(body):
                heading, para = heading_and_body(para)
                own, stance = stance_of(para)
                out.append({'kind': 'text', 'heading': heading, 'text': para, 'figures': figures_of(para), 'own_view': own,
                            'stance': stance, 'unverified': False})
    return out


def parse_report(file: str, pages: list[str], meta: dict | None, as_of: date) -> list[Fragment]:
    meta = meta or {}
    when = date.fromisoformat(meta['date']) if meta.get('date') else \
        date_from_name(file, as_of.year) or date_from_text('\n'.join(pages[:1]))
    title = meta.get('title') or next((ln.strip() for ln in (pages[0] if pages else '').splitlines() if len(ln.strip()) > 10),
                                      Path(file).stem)
    fresh, age = freshness(when, as_of)
    out = []
    for page_no, text in enumerate(pages, 1):
        for n, p in enumerate(page_fragments(text, page_no), 1):
            if p['kind'] == 'chart':
                from stage5_insight_detection.bank_analytics.digitize import PERIOD_SUFFIX, period_end
                for figure in p['figures']:
                    end = period_end(figure.period)
                    if PERIOD_SUFFIX.search(figure.period) or (when and end is not None and end > when.year * 12 + when.month - 1):
                        figure.kind = 'forecast'  # keep verified/approx status; a future date does not verify a value
            out.append(Fragment(id=f'bank:{Path(file).stem}:p{page_no}.{n}', file=file, report=title,
                                date=when.isoformat() if when else None, page=page_no, n=n, heading=p['heading'],
                                text=p['text'], topics=topics_of(p['text']), figures=[asdict(f) for f in p['figures']],
                                own_view=p['own_view'], stance=p['stance'], freshness=fresh, age_days=age, kind=p['kind'],
                                unverified=p['unverified']))
    return out


def file_hash(path: Path, *extra: Path) -> str:
    """Hash of the file and of any existing extra files (its OCR caches): new OCR output re-parses the report."""
    h = hashlib.sha1(path.read_bytes())
    for e in extra:
        if e.exists():
            h.update(e.read_bytes())
    return h.hexdigest()[:16]


class BankLibrary:
    """All fragments + a stem TF-IDF index. `refresh()` parses only new or changed PDFs of the folder."""

    def __init__(self, fragments: list[Fragment]):
        self.fragments = fragments
        self.by_id = {f.id: f for f in fragments}
        self.index = StemIndex([f'{f.heading}. {f.text}' for f in fragments], min_df=1, max_df=1.0) if fragments else None

    @classmethod
    def from_pages(cls, reports: dict[str, list[str]], as_of: date, meta: list[dict] = REPORTS) -> 'BankLibrary':
        """reports: file name -> page texts (tests, or OCR output already in memory)."""
        by_file = {r['file']: r for r in meta}
        return cls([f for file, pages in reports.items() for f in parse_report(file, pages, by_file.get(file), as_of)])

    @classmethod
    def refresh(cls, pdf_dir: Path = PDF_DIR, ocr_dir: Path = OCR_DIR, as_of: date | None = None, ocr=None,
                out: Path = FRAGMENTS, registry: Path = REGISTRY, full_dir: Path = OCR_FULL_DIR, rebuild: bool = False,
                charts: str | None = None) -> 'BankLibrary':
        as_of = as_of or date.today()
        mode = chart_novelty.mode_of(charts)
        known = json.loads(registry.read_text(encoding='utf-8')) if registry.exists() and not rebuild else {}  # rebuild: parser changed
        old = cls.load(out, as_of).fragments if out.exists() and not rebuild else []
        by_file = {r['file']: r for r in REPORTS}
        pdfs = sorted(pdf_dir.glob('*.pdf'))
        present = {p.name for p in pdfs}
        known = {name: h for name, h in known.items() if name in present}
        keep, fresh = [f for f in old if f.file in known], []
        def signature(pdf):
            assets = [full_dir / f'{pdf.stem}.jsonl', ocr_dir / f'{pdf.stem}.jsonl']
            assets += sorted((pdf_dir / 'charts').glob(f'{pdf.stem}_p*.json'))
            assets += sorted((pdf_dir / 'tables').glob(f'{pdf.stem}_p*.json'))
            assets += [pdf_dir / folder / f'{pdf.stem}.source.json' for folder in ('charts', 'tables')]
            version = json.dumps([PARSER_VERSION, by_file.get(pdf.name), as_of.year] + ([] if mode == 'all' else [mode]),
                                 sort_keys=True)
            return hashlib.sha256((file_hash(pdf, *assets) + version).encode()).hexdigest()
        for pdf in pdfs:
            h = signature(pdf)
            if known.get(pdf.name) == h and any(f.file == pdf.name for f in keep):
                continue
            keep = [f for f in keep if f.file != pdf.name]
            meta = by_file.get(pdf.name) or {}
            when = date.fromisoformat(meta['date']) if meta.get('date') else date_from_name(pdf.name, as_of.year)
            fresh += parse_report(pdf.name, best_pages(pdf, full_dir, ocr_dir, ocr, when.year if when else None, mode), meta, as_of)
            known[pdf.name] = signature(pdf)  # optional OCR may have written new pages
        lib = cls(keep + fresh)
        lib.save(out)
        from common.news_corpus import write_json
        write_json(registry, known)
        return lib

    @classmethod
    def load(cls, path: Path = FRAGMENTS, as_of: date | None = None) -> 'BankLibrary':
        """Freshness is recomputed for `as_of`: materials age between the parse and the check."""
        as_of = as_of or date.today()
        frags = []
        for line in filter(None, path.read_text(encoding='utf-8').splitlines()):
            f = Fragment(**json.loads(line))
            if f.kind == 'text':
                f.own_view, f.stance = stance_of(f.text)  # migrate old cached attribution
            f.freshness, f.age_days = freshness(date.fromisoformat(f.date) if f.date else None, as_of)
            frags.append(f)
        return cls(frags)

    def save(self, path: Path = FRAGMENTS) -> None:
        from common.news_corpus import _atomic
        _atomic(path, lambda p: p.write_text(''.join(json.dumps(asdict(f), ensure_ascii=False) + '\n'
                                                    for f in self.fragments), encoding='utf-8'))

    def search(self, query: str, k: int = 8) -> list[tuple[Fragment, float]]:
        if self.index is None:
            return []
        return [(self.fragments[i], s) for i, s in self.index.search(query, k)]

    def scores(self, query: str):
        return self.index.scores(query) if self.index is not None else []


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ocr', action='store_true', help='run the vision model for pages without text')
    ap.add_argument('--as-of', type=date.fromisoformat, default=None)
    ap.add_argument('--rebuild', action='store_true', help='parse every PDF again (after a change of the parser)')
    ap.add_argument('--search', default='')
    ap.add_argument('--charts', choices=chart_novelty.MODES, default=None,
                    help='all: every chart; selective: only charts with what the text lacks; off: none (default env BANK_CHARTS or all)')
    a = ap.parse_args()
    if a.search:
        for f, s in BankLibrary.load(as_of=a.as_of).search(a.search):
            print(f'{s:.3f} {f.id} {f.date} {f.freshness}{" OWN" if f.own_view else ""} {f.topics}\n  {f.text[:250]}')
    else:
        from stage5_insight_detection.bank_analytics.ocr import AnalyticsOCR
        lib = BankLibrary.refresh(as_of=a.as_of, ocr=AnalyticsOCR() if a.ocr else None, rebuild=a.rebuild, charts=a.charts)
        print(f'{FRAGMENTS}: {len(lib.fragments)} fragments ({sum(f.kind == "chart" for f in lib.fragments)} charts, mode '
              f'{chart_novelty.mode_of(a.charts)}), {sum(f.own_view for f in lib.fragments)} with the bank\'s own view, '
              f'{sum(not f.text.strip() for f in lib.fragments)} empty')
