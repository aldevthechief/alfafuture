"""Which parsed fragments of the bank's analytics are retrievable (analytic-retrieve v1, approved 22.10): prose without legal
text / parser notes / captions / chart text / label lists, tables with a header row (headers of split tables repaired),
no charts. Deterministic, no LLM. See metadata_audit.md for the audit behind every rule."""
import re


LEGAL = re.compile(r'Все права защищены|Генеральная лицензия|исключительной собственностью|несанкционированное|не делают? заявлений|предназначен', re.I)
CHART_META = re.compile(r'Ось [XY]|Тип: (линия|столбцы|круговая|комбинир)|Ряды \(легенда\)|Легенда:|Значения по оси|\[по шкале')  # a chart description that leaked into text
CAPTION = re.compile(r'^\s*[#*]*\s*(Рис\.?|Табл\.?|Таблица|График|Источник|Примечани|©|Диаграмма)', re.I)


def prose_reason(f: dict) -> str:
    """'' if the fragment is a prose paragraph of the report; else why not. Deterministic split of the OCR text from what
    the vision model made of charts: `kind` already separates [График] blocks and tables, this drops what still leaks into
    kind == 'text': legal disclaimers (a quarter of the text characters), parser notes, captions and sources, axis / legend
    lists and flattened table columns (symbols, many short lines), anything too short to be a statement."""
    t = f['text'].strip()
    if f['kind'] != 'text': return f['kind']
    if 'Резервное распознавание' in t: return 'parser note'
    if LEGAL.search(t): return 'legal'
    if CAPTION.match(t) and len(t) < 300: return 'caption / source'
    if len(t) < 120: return 'short'
    if CHART_META.search(t): return 'chart text'
    if sum(c.isalpha() for c in t) / len(t) < 0.6: return 'digits / symbols'
    lines = [x for x in t.split('\n') if x.strip()]
    if len(lines) >= 5 and sum(map(len, lines)) / len(lines) < 40 and not re.search(r'[.!?]\s', t): return 'label list'  # slides have short lines with sentences
    if len(re.findall(r'[а-яёa-z]{3,}', t.lower())) < 15: return 'few words'
    if not re.search(r'[.!?;:»)]\s|[.!?»)]\s*$', t): return 'no sentence end'
    return ''


def table_has_header(f: dict) -> bool:
    """A table fragment whose first non-empty row names the columns (years, words); a piece of a split table whose value
    columns are plain numbers has no header and cannot be read (1H26 appendix: the 2026П column is unlabelled)."""
    rows = [[x.strip() for x in r.strip().strip('|').split('|')] for r in f['text'].split('\n')[1:]
            if r.strip().startswith('|') and not re.match(r'^\|[-:| ]+$', r.strip())]
    vals = [x for x in next((r for r in rows if any(r)), [''])[1:] if x]
    return not vals or any(re.search('[A-Za-zА-Яа-я]', x) or re.fullmatch(r'(?:19|20)\d\d', x) for x in vals)


def repair_headers(rows: list[dict]) -> list[dict]:
    """The OCR splits a long table into pieces and only the first has the header row: a headless piece takes the header of
    the previous piece on the same page with the same number of columns (1H26 appendix: the forecast columns 2026П / 2027П)."""
    cells = lambda r: [x.strip() for x in r.strip().strip('|').split('|')]
    body = lambda f: [r.strip() for r in f['text'].split('\n')[1:] if r.strip().startswith('|') and not re.match(r'^\|[-:| ]+$', r.strip())]
    last = {}
    for f in rows:
        if f['kind'] != 'table':
            continue
        data = [r for r in body(f) if any(cells(r))]
        if not data:
            continue
        key, n = (f['file'], f['page']), len(cells(data[0]))
        if table_has_header(f):
            last[key] = (data[0], n)
        elif key in last and last[key][1] == n:
            f['text'] = '\n'.join([f['text'].split('\n')[0], last[key][0], '|' + '---|' * n, *data]); f['repaired'] = True
    return rows


TABLE_CHART = re.compile(r'стр\. \d+: (Таблица|Табл\.)')


def use_fragment(f: dict) -> bool:
    """Sources of analytic-retrieve v1 (approved 22.10): prose of the reports without legal text, parser notes, captions,
    chart text and label lists; tables that have a header row (the macro forecast table lives only there), also a table the
    parser took for a chart (title «Таблица…»: 1H26 July indicators and forecasts); no charts."""
    if f['kind'] == 'table':
        return table_has_header(f)
    if f['kind'] == 'chart':
        return bool(TABLE_CHART.search(f['text'].split('\n')[0])) and not f.get('unverified')
    return prose_reason(f) in ('', 'few words', 'no sentence end')
