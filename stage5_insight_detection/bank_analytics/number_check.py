"""Number check for insights built from the bank's reports (no LLM): every number the agent quotes as the bank's view must stand
on the report page it cites, and a table value must not be an old one when a newer report has another value in the same row.

- `numbers(text)`: numeric tokens in a canonical form: «14,0%» = «14.0» = «14» (comma or dot, trailing zeros dropped), the sign is
  ignored («−4,0%» = «4»), dates, years, and tags like 4К26 / 1П26 / 2027П are not numbers, «1 200» is 1200.
- `check_numbers(rows, view, sources)`: warnings in the words the agent should act on; nothing is dropped silently.
- The same macro tables repeat in several reports with different values («Курс, конец года» for 2027: 95,0 in the July and September
  macro statistics, 93,0 in the 11.09 rate report), so a row is compared by its label and column header across reports.
"""
import re
from decimal import Decimal

DASH = str.maketrans('–—−‑', '----')
# 31.07.2026 and 31.07 (a decimal is written with a comma in the reports, a dotted dd.mm is a date unless a % follows)
MONTH = r'(?:январ|феврал|март|апрел|ма[яй]|июн|июл|август|сентябр|октябр|ноябр|декабр)[а-я]*'
DATE = re.compile(r'(?<!\d)(?:19|20)\d\d-\d\d-\d\d(?!\d)|\b\d{1,2}\s+' + MONTH + r'\b|\b\d{1,2}\s?(?:-|–|до)\s?\d{1,2}\s+' + MONTH + r'\b|\b\d{1,2}\.\d{1,2}\.\d{2,4}\b|\b(?:0[1-9]|[12]\d|3[01])\.(?:0[1-9]|1[0-2])\b(?!\s*%|\d)')
TAG = re.compile(r'\b\d+[A-Za-zА-Яа-яЁё]+\d*\b|\b[A-Za-zА-Яа-яЁё]+\d+\b')
THOUSANDS = re.compile(r'(?<![\d.,])\d{1,3}(?:[   ]\d{3})+(?![\d]|[.,]\d)')
NUMBER = re.compile(r'(?<![\d.,])\d+(?:[.,]\d+)?')
UNIT_AFTER = re.compile(r'\s*(%|п\.\s?п)')


def canon(token: str) -> str:
    return format(Decimal(token.replace(',', '.')).normalize(), 'f')


def numbers(text: str, keep_small: bool = False) -> dict[str, str]:
    """{canonical number: first raw spelling}. Bare integers below 10 count only with a % or «п.п.» after them (counts, months), unless keep_small."""
    text = DATE.sub(' ', text.translate(DASH))
    text = TAG.sub(' ', text)
    out = {}
    for m in THOUSANDS.finditer(text):
        raw = m.group(0)
        out.setdefault(str(int(re.sub(r'\D', '', raw))), raw)
        text = text.replace(raw, ' ' * len(raw), 1)
    for m in NUMBER.finditer(text):
        raw = m.group(0)
        c = canon(raw)
        if re.fullmatch(r'\d+', raw) and 1990 <= int(raw) <= 2040:
            continue  # a year
        if not keep_small and re.fullmatch(r'\d', raw) and not UNIT_AFTER.match(text[m.end():m.end() + 6]):
            continue
        out.setdefault(c, raw)
    return out


def norm_label(label: str) -> str:
    return re.sub(r'[^\wа-яё%$/]+', ' ', label.lower().replace('(?)', '')).strip()


def norm_period(cell: str) -> str:
    return re.sub(r'\s+', '', re.sub(r'\(\?\)', '', cell.lower())).rstrip('пp')


def table_cells(rows: list[dict]):
    """(report, date, page, label, period, canonical value, raw cell) for every numeric cell of the tables."""
    for f in rows:
        if f['kind'] not in ('table', 'chart'):
            continue
        header = None
        for line in f['text'].split('\n'):
            if not line.strip().startswith('|') or re.match(r'^\|[-:| ]+$', line.strip()):
                continue
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            if not any(cells):
                continue
            if header is None:
                header = cells
                continue
            if len(cells) != len(header) or not cells[0]:
                continue
            for h, c in zip(header[1:], cells[1:]):
                vals = numbers(c)
                if len(vals) == 1 and h:
                    yield f['report'], f['date'], f['page'], norm_label(cells[0]), norm_period(h), next(iter(vals)), c


def cited(sources: list[str], reports: list[dict]) -> tuple[list[tuple[dict, list[int]]], list[str]]:
    """Report and pages of every source «title, date, стр. N–M»; the sources that could not be read."""
    found, bad = [], []
    for s in sources:
        flat = s.translate(DASH).lower()
        iso = re.findall(r'\d{4}-\d{2}-\d{2}', flat) + [f'{y}-{m}-{d}' for d, m, y in re.findall(r'\b(\d{2})\.(\d{2})\.(\d{4})\b', flat)]
        report = next((r for r in reports if r['date'] in iso), None) or next(
            (r for r in reports if r['title'].translate(DASH).lower() in flat), None)
        pages = []
        for chunk in re.findall(r'стр\.?\s*([\d\s,\-]+)', flat):
            for part in re.split(r'\s*,\s*', chunk.strip(' ,')):
                m = re.fullmatch(r'(\d+)(?:\s*-\s*(\d+))?', part.strip())
                if m:
                    pages += range(int(m.group(1)), int(m.group(2) or m.group(1)) + 1)
        if report and pages:
            found.append((report, sorted(set(pages))))
        else:
            bad.append(s)
    return found, bad


def check_numbers(rows: list[dict], reports: list[dict], view: str, sources: list[str], read_from_picture: str = '') -> list[str]:
    """Warnings for the agent about the numbers in `view` (the bank's position as quoted) against the pages in `sources`."""
    want = numbers(view)
    for c in numbers(read_from_picture):  # numbers the agent marked as read from the picture (evidence kind=picture) are approximate by its own word
        want.pop(c, None)
    found, bad = cited(sources, reports)
    out = [f'источник «{s[:80]}» не удалось разобрать: пиши «название отчёта, дата, стр. N»' for s in bad]
    text_nums, chart_nums = set(), set()
    where, where_keys = [], []
    for r, pages in found:
        if (r['title'], tuple(pages)) not in [(w[0], w[1]) for w in where_keys]:
            where_keys.append((r['title'], tuple(pages)))
            where.append(f"«{r['title']}» стр. {', '.join(map(str, pages))}")
        for f in rows:
            if f['report'] == r['title'] and f['page'] in pages:
                (chart_nums if f['kind'] == 'chart' and not f['text'].lstrip().startswith(('Табл', 'стр')) else text_nums).update(numbers(f['text']))
    if found:
        for c, raw in want.items():
            if c in text_nums:
                continue
            if c in chart_nums:
                out.append(f'число {raw} найдено только в тексте графика (чтение картинки, приблизительно) на {"; ".join(where)}: если так и есть, его можно оформить доказательством kind=picture')
            else:
                out.append(f'число {raw} есть в твоём описании позиции Банка, но его нет в тексте {"; ".join(where)} — возможно, оно '
                           f'из другого отчёта или строки таблицы, или это ошибка распознавания (можно сверить через read_page или look_at_page)')
    # a value of a cited table row that a newer report states differently
    cells = list(table_cells(rows))
    seen = set()
    for rep, date, page, label, period, c, cell in cells:
        if c not in want or not any(r['title'] == rep and page in pages for r, pages in found):
            continue
        for rep2, date2, _, label2, period2, c2, cell2 in cells:
            if label2 == label and period2 == period and date2 > date and c2 != c and c2 not in want and (label, period, date2) not in seen:
                seen.add((label, period, date2))
                out.append(f'число {want[c]} — значение строки «{label}» за {period} из отчёта «{rep}» от {date}, но в более свежем отчёте '
                           f'«{rep2}» от {date2} в этой строке {cell2}: возможно, стоит взять новое значение и показать, что взгляд изменился')
    return out
