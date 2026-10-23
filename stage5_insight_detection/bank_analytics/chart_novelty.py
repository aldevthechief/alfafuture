"""Selective chart processing: a chart is read and shown only when it may say what the report's text does not.

The bank's text usually retells a chart's conclusion («инфляция замедлилась до 6,3% (см Рис.1)»), the chart adds the
history behind it. So each chart is first looked up in the recognized text (Report.related):
- its title printed as a caption, the references to it («см Рис.4», «рисунок 4»),
- the paragraphs of its page and the page before that talk about the same indicator (shared stems of the title).
That text is then checked for what the chart holds (coverage): the indicator, the period (the chart's last date), the
forecast (points after the report date or marked «2026П», «2027F», «АБ»), the values (printed labels confirmed on the
pixels or by Tesseract; before reading, the numbers of the page OCR's block) and the caveats (a «*» footnote, «оценка»,
«предварительно», «без учёта»). A number matches at the chart's precision: 6,3 on the chart is 6,32 in the text; a value
measured by code («≈») within 3%; a year is never a value. Verdicts, in order of priority:
- `forecast`    — the chart has forecast points the related text does not give (priority 0);
- `new_values`  — confirmed values the related text does not give (1);
- `only_chart`  — no related paragraph names the indicator: the chart is the only place it appears (2);
- `duplicate`   — the related text gives every confirmed and forecast number;
- `history`     — the text names the indicator, the chart adds its shape and «≈» values only;
- `empty`       — the reading failed.
The first three are new information: under `selective` they go to the separate chart analysis, forecasts first.

Mode (`--charts` of charts.py, export_text.py, library.py; env BANK_CHARTS; default `all`):
- `all`       — every chart is read (charts.py) and shown with the code's note (export_text, library), as before;
- `selective` — charts.py reads only the charts whose page OCR block may hold something new (block_plan; a crop that
                matches no block is read: unknown is not a duplicate), pages with a forecast first, so a budget stop
                drops the least important; export_text and library keep only the new charts, a dropped one leaves the
                line SKIPPED with its title and the reason (the library drops the line too);
- `off`       — no chart is read, every chart block becomes the SKIPPED line, the library has no chart fragments.
No model is called here.
python -X utf8 stage5_insight_detection/bank_analytics/chart_novelty.py [stem ...] [-v]   (verdict per chart, free)
"""
import argparse, json, os, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stage5_insight_detection.bank_analytics.digitize import (PHANTOM, chart_points, checked_labels, forecast_points,
                                                              month_index, page_records, report_date, similar)

MODES = ('all', 'selective', 'off')
PRIORITY = {'forecast': 0, 'new_values': 1, 'only_chart': 2}
NEW = tuple(PRIORITY)
SKIPPED = '[График не добавлен]'
BLOCK = re.compile(r'\[График\].*?\[/График\]', re.S)
VALUE = re.compile(r'(?<![\w.,])[-+−–]?\d{1,3}(?:[  ]\d{3})+(?:[.,]\d+)?(?![\w])|(?<![\w.,])[-+−–]?\d+(?:[.,]\d+)?')
FORECAST_MARK = re.compile(r'\d\s*[ПпFfEe]\b|прогноз|\bАБ\b', re.I)
FORECAST_WORDS = re.compile(r'прогноз|ожида|ждем|ждём|предполага|оценива|сценари|к концу', re.I)
CAVEAT = re.compile(r'\*|оценк|предварит|без учет|без учёт|с учетом|с учётом|сезонн|скоррект|пересмотр|по данным на', re.I)
FIGURE = re.compile(r'Рис\.?\s*(\d+)', re.I)
CAPTION = re.compile(r'^\s*Рис\.?\s*\d+\s*[.:]?[^\n]{3,}$', re.I | re.M)
WORD = re.compile(r'[а-яёa-z]{4,}', re.I)
GENERIC = {'динам', 'измен', 'показ', 'струк', 'разре', 'перио', 'значе', 'табли', 'графи', 'итого', 'накоп', 'начал',
           'конец', 'альфа', 'банка', 'банк', 'трлн', 'млрд', 'данны', 'основ', 'ключе', 'урове', 'сравн', 'годов',
           'месяч', 'недел', 'рынка', 'рынок', 'числа', 'объем', 'объём', 'доля', 'долл'}
REASONS = {'forecast': 'прогнозные точки, которых нет в связанном тексте',
           'new_values': 'подписанные значения, которых нет в связанном тексте',
           'only_chart': 'показатель графика в тексте не упоминается',
           'duplicate': 'значения графика уже есть в связанном тексте',
           'history': 'текст называет показатель, график добавляет только форму ряда и значения со шкалы (≈)',
           'empty': 'график не прочитан', 'off': 'обработка графиков выключена'}


def mode_of(value: str | None = None) -> str:
    mode = (value or os.environ.get('BANK_CHARTS') or 'all').strip().lower()
    if mode not in MODES:
        raise ValueError(f'chart mode {mode!r}: one of {", ".join(MODES)}')
    return mode


def number(text: str) -> tuple[float, int] | None:
    """'6,32' -> (6.32, 2 decimals); '69 572' -> (69572.0, 0); None for anything that is not a number."""
    m = VALUE.search(str(text))
    if not m:
        return None
    raw = re.sub(r'[  ]', '', m.group()).replace('−', '-').replace('–', '-').replace(',', '.')
    try:
        return float(raw), len(raw.split('.', 1)[1]) if '.' in raw else 0
    except ValueError:
        return None


def is_year(value: float, decimals: int) -> bool:
    return decimals == 0 and 1990 <= value <= 2100


def text_numbers(text: str) -> list[float]:
    """Numbers of a text, years left out, signs dropped: «снижение на 0,2%» and a chart's −0,2 are the same fact."""
    out = []
    for m in VALUE.finditer(text):
        n = number(m.group())
        if n and not is_year(*n):
            out.append(abs(n[0]))
    return out


def in_text(value: str, numbers: list[float]) -> bool:
    """The text has the chart's number at the chart's precision (6,3 on the chart is 6,32 in the text); a value the code
    measured («≈ 13,47») within 3%."""
    n = number(value)
    if n is None:
        return True  # not a number: nothing new to say
    v, d = abs(n[0]), n[1]
    tol = max(0.5 * 10 ** -d, 0.03 * v if '≈' in str(value) else 0) + 1e-9
    return any(abs(t - v) <= tol for t in numbers)


LATIN = str.maketrans('ABCEHKMOPTXYaceopxy', 'АВСЕНКМОРТХУасеорху')  # «MMK» in a title is «ММК» in the text
ABBR = re.compile(r'\b[А-ЯЁ]{3}\b')


def stems(text: str) -> set[str]:
    """First five letters of the title's words (four letters and more) and its three-letter abbreviations (ММК, ВВП)."""
    text = text.translate(LATIN)
    return ({w.lower()[:5] for w in WORD.findall(text)} | {w.lower() for w in ABBR.findall(text)}) - GENERIC


def period_tokens(label: str) -> set[str]:
    """What a text would say about a chart's last date: «авг.26» -> {'август', '2026'}, «2К26» -> {'2026', '2к26', 'квартал'}."""
    t, out = str(label).lower(), set()
    out |= set(re.findall(r'20\d{2}', t))
    if m := re.search(r'[а-яёa-z]+\.?\s*[\'’]?(\d{2})\b', t):
        out.add('20' + m.group(1))
    i = month_index(t)
    if i is not None:
        out.add(('январ', 'феврал', 'март', 'апрел', 'ма', 'июн', 'июл', 'август', 'сентябр', 'октябр', 'ноябр', 'декабр')[i % 12])
        out.add(str(i // 12))
    if m := re.search(r'([1-4])\s*[кq]\s*(\d{2})', t):
        out |= {f'{m.group(1)}к{m.group(2)}', '20' + m.group(2)}
    return out


class Report:
    """A report's text as a reader gets it, for finding what it says about a chart: paragraphs per page, the chart
    blocks and SKIPPED lines left out, the tables replaced by those tables.py read three times and reconciled."""

    def __init__(self, stem: str, pages: list[str], tables: bool = True):
        from stage5_insight_detection.bank_analytics import tables as table_reading
        self.stem, self.paras = stem, []
        when = report_date(stem)
        self.year = when.year if when else None
        for n, text in enumerate(pages, 1):
            if tables:
                text = table_reading.replace_tables(text, table_reading.page_records(stem, n))
            for p in re.split(r'\n\s*\n', BLOCK.sub('\n\n', text)):
                p = '\n'.join(ln for ln in p.strip().split('\n') if not ln.startswith(SKIPPED))
                if p.strip():
                    self.paras.append((n, p.strip()))

    def related(self, title: str, page: int) -> dict:
        """The text about a chart: its caption (the title printed above it), the paragraphs that refer to it («см Рис.4»)
        anywhere in the report, the paragraphs of its page and the page before that share a word of its title."""
        fig = FIGURE.search(title or '')
        ref = re.compile(rf'рис(?:\.|унк\w*)?\s*{fig.group(1)}(?!\d)', re.I) if fig else None
        keys = stems(FIGURE.sub('', title or ''))
        need = 1 if len(keys) <= 2 else 2
        caption, refs, near = False, [], []
        for n, p in self.paras:
            first = p.split('\n', 1)[0]
            if n in (page, page - 1) and similar(first, title) >= 0.8:
                caption = True  # the title printed as the caption, not a paragraph about the chart
                continue
            if ref and ref.search(p):
                refs.append(p)
            elif n in (page, page - 1) and keys and len(keys & stems(p)) >= need:
                near.append(p)
        text = '\n\n'.join(refs + near)
        return {'caption': caption, 'refs': len(refs), 'paragraphs': len(refs) + len(near), 'text': text,
                'numbers': text_numbers(text)}


def record_facts(record: dict) -> dict:
    """What a read chart holds (charts.py record): title, confirmed values, forecast values, its dates, a caveat."""
    read = record.get('read') or {}
    sure, _ = checked_labels(record)
    points = chart_points(record, sure)
    _, ahead = forecast_points(record, points)
    title = (read.get('title') or record.get('title') or '').strip()
    has_series = bool(points or read.get('series') or any(s.get('last') is not None for s in (record.get('digitized') or {}).get('series') or []))
    return {'title': title, 'values': [str(v) for _, _, v, _ in sure if not is_year(*(number(v) or (0, 1)))],
            'forecast': [v for _, _, v in ahead], 'forecast_mark': bool(ahead), 'periods': [x for _, x, _ in points],
            'caveat': bool(CAVEAT.search(f'{title} {read.get("source") or ""}')), 'content': has_series}


def block_facts(block: str, year: int | None = None) -> dict:
    """What the page OCR's block says a chart holds, before any paid reading. The OCR makes values up: good enough to
    decide what is worth reading, never a fact."""
    head, _, body = block.partition('\n')
    title = head.replace('[График]', '').strip()
    values, forecast = [], []
    for line in body.split('\n'):
        if not line.lower().lstrip().startswith(('значения', '-')):
            continue
        for item in line.split(':', 1)[-1].split(';'):  # «16; 14 (2026П)»: an item marked as a forecast is a forecast
            for m in VALUE.finditer(item):
                if not is_year(*(number(m.group()) or (0, 1))) and not re.match(r'\d*\s*[ПпFfEe]\b', item[m.end():]):
                    (forecast if FORECAST_MARK.search(item) else values).append(m.group())
    later = year and any(int(y) > year for y in re.findall(r'\b(20\d{2})\b', block))  # «ставка в 2026-2027 гг.»
    axis = re.search(r'Ось X:[^\n;]*?[–-]\s*([^\n;]+)', body)
    return {'title': title, 'values': values, 'forecast': forecast, 'forecast_mark': bool(FORECAST_MARK.search(block) or later),
            'periods': [axis.group(1).strip()] if axis else [], 'caveat': bool(CAVEAT.search(title)), 'content': True}


def coverage(facts: dict, rel: dict) -> dict:
    """Which of the chart's indicator, period, forecast, values and caveats the related text gives (None: the chart has
    none of it), and the numbers it lacks."""
    text = rel['text'].lower()
    nums = rel['numbers']
    missing_values = [v for v in facts['values'] if not in_text(v, nums)]
    missing_forecast = [v for v in facts['forecast'] if not in_text(v, nums)]
    last = period_tokens(facts['periods'][-1]) if facts['periods'] else set()
    return {
        'indicator': bool(rel['paragraphs']),
        'period': (any(t in text for t in last) if last else None),
        # a forecast the block marks (a later year, «П») without values to check is never taken as told: priority
        'forecast': (bool(FORECAST_WORDS.search(text)) and not missing_forecast and bool(facts['forecast'])
                     if facts['forecast_mark'] else None),
        'values': (not missing_values if facts['values'] else None),
        'caveats': (bool(CAVEAT.search(text)) if facts['caveat'] else None),
        'missing_values': missing_values, 'missing_forecast': missing_forecast}


def verdict(facts: dict, rel: dict) -> dict:
    cov = coverage(facts, rel)
    if cov['forecast'] is False:
        kind = 'forecast'
    elif cov['values'] is False:
        kind = 'new_values'
    elif not facts['content'] and not facts['values']:
        kind = 'empty'
    elif not cov['indicator']:
        kind = 'only_chart'
    elif facts['values'] or facts['forecast']:
        kind = 'duplicate'
    else:
        kind = 'history'
    gaps = [name for name, label in (('period', 'период'), ('caveats', 'оговорки')) if cov[name] is False]
    reason = REASONS[kind] + (f'; в тексте нет: {", ".join(dict(period="период", caveats="оговорки")[g] for g in gaps)}' if gaps else '')
    return {'title': facts['title'], 'kind': kind, 'new': kind in NEW, 'priority': PRIORITY.get(kind, 9), 'reason': reason,
            'coverage': {k: cov[k] for k in ('indicator', 'period', 'forecast', 'values', 'caveats')},
            'refs': rel['refs'], 'paragraphs': rel['paragraphs'], 'caption': rel['caption'],
            'missing': cov['missing_values'][:12], 'missing_forecast': cov['missing_forecast'][:12]}


def novelty(record: dict, report: Report, page: int) -> dict:
    """Verdict for one record of charts.py against the text about it in its report."""
    facts = record_facts(record)
    return dict(verdict(facts, report.related(facts['title'], page)), crop=record.get('crop'), page=page)


def skipped_line(title: str, reason: str) -> str:
    return f'{SKIPPED} {title.strip() or "без названия"}: {reason}.'


def select(text: str, records: list[dict], report: 'Report | None', page: int, mode: str) -> tuple[str, list[dict]]:
    """A page for digitize.annotate under `mode`: the records to annotate with, and the page text where the blocks of
    dropped charts (selective: not new; off: every one) are the SKIPPED line. A block is a record's by its title, as in
    annotate; a block no record matches stays for annotate (it marks phantoms) under selective, goes under off."""
    if mode == 'all':
        return text, records
    if mode == 'off':
        return BLOCK.sub(lambda m: skipped_line(m.group().split('\n', 1)[0].replace('[График]', ''), REASONS['off']), text), []
    verdicts = [novelty(r, report, page) for r in records]
    keep = [r for r, v in zip(records, verdicts) if v['new']]
    dropped = [(r, v) for r, v in zip(records, verdicts) if not v['new']]
    name = lambda r: r.get('title') or (r.get('read') or {}).get('title', '')

    def drop(m):
        title = m.group().split('\n', 1)[0].replace('[График]', '').strip()
        if PHANTOM in m.group() or not dropped:
            return m.group()
        best = max(dropped, key=lambda rv: similar(name(rv[0]), title))
        if similar(name(best[0]), title) < 0.6:
            return m.group()
        dropped.remove(best)
        return skipped_line(title, best[1]['reason'])
    return BLOCK.sub(drop, text), keep


def clean_pages(pdf: Path) -> list[str]:
    """The page texts export_text and library start from (pages.clean_page of the full OCR run, the old run as backup)."""
    from stage5_insight_detection.bank_analytics.library import page_texts
    from stage5_insight_detection.bank_analytics.ocr import OCR_DIR, OCR_FULL_DIR
    from stage5_insight_detection.bank_analytics.pages import clean_page
    when = report_date(pdf.stem)
    return [clean_page(main, old, when.year if when else None)[0]
            for main, old in zip(page_texts(pdf, OCR_FULL_DIR), page_texts(pdf, OCR_DIR))]


def block_plan(pdf: Path) -> dict[int, list[dict]]:
    """For charts.py under selective, before any paid call: page -> verdicts of the page OCR's chart blocks."""
    pages = clean_pages(pdf)
    report = Report(pdf.stem, pages)
    plan = {}
    for n, text in enumerate(pages, 1):
        facts = [block_facts(b, report.year) for b in BLOCK.findall(text) if PHANTOM not in b]
        # the captions printed outside the blocks: the OCR cuts a block's title («Рис.8») or writes no block at all,
        # and the caption may be the only place saying «Прогноз Альфа-Банка ...»
        captions = {FIGURE.match(c.strip()).group(1): c.strip() for c in CAPTION.findall(BLOCK.sub('', text))}
        for f in facts:
            fig = FIGURE.match(f['title'])
            caption = captions.pop(fig.group(1), '') if fig else ''
            if len(caption) > len(f['title']):
                f['title'] = caption
                f['forecast_mark'] = f['forecast_mark'] or bool(FORECAST_MARK.search(caption) or report.year and any(
                    int(y) > report.year for y in re.findall(r'\b(20\d{2})\b', caption)))
        facts += [block_facts(f'[График] {c}\n[/График]', report.year) for c in captions.values()
                  if not any(similar(c, f['title']) >= 0.6 for f in facts)]
        if facts:
            plan[n] = [verdict(f, report.related(f['title'], n)) for f in facts]
    return plan


def page_priority(blocks: list[dict] | None) -> int:
    """The most important new chart of a page (0 forecast, 1 values, 2 only in the chart); a page without blocks is read
    after them (1, it may hold anything); 9: nothing new."""
    if not blocks:
        return 1
    return min(b['priority'] for b in blocks)


def worth_reading(title: str, blocks: list[dict] | None) -> bool:
    """A crop is read unless its title matches a block of the page judged not new; no block like it: read (unknown)."""
    if not blocks:
        return True
    best = max(blocks, key=lambda b: similar(b['title'], title or ''))
    return best['new'] or similar(best['title'], title or '') < 0.6


def report_verdicts(stem: str, pages: list[str]) -> list[dict]:
    """Verdict of every chart charts.py read in a report (pages: its cleaned page texts, pages.clean_page)."""
    report = Report(stem, pages)
    return [novelty(r, report, n) for n, _ in enumerate(pages, 1) for r in page_records(stem, n)]


if __name__ == '__main__':
    from stage5_insight_detection.bank_analytics.ocr import PDF_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument('stems', nargs='*')
    ap.add_argument('-v', action='store_true', help='coverage and the numbers that make a chart new')
    a = ap.parse_args()
    total: dict[str, int] = {}
    mark = {True: '+', False: '-', None: '.'}
    for pdf in sorted(PDF_DIR.glob('*.pdf')):
        if a.stems and pdf.stem not in a.stems:
            continue
        for v in report_verdicts(pdf.stem, clean_pages(pdf)):
            total[v['kind']] = total.get(v['kind'], 0) + 1
            cov = ''.join(mark[v['coverage'][k]] for k in ('indicator', 'period', 'forecast', 'values', 'caveats'))
            print(f'{v["kind"]:10} {cov} refs {v["refs"]} paras {v["paragraphs"]:<2} p{v["page"]:<3} {v["crop"]:44} {v["title"][:50]}')
            if a.v and v['new']:
                print('           ', v['reason'], '|', '; '.join((v['missing_forecast'] or v['missing'])[:8]))
    print('coverage: indicator period forecast values caveats (+ given, - missing, . not on the chart)')
    print(json.dumps(total, ensure_ascii=False), f'new: {sum(total.get(k, 0) for k in NEW)} of {sum(total.values())}')
