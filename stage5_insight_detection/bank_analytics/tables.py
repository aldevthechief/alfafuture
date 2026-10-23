"""Tables of the bank's PDFs read crop by crop: three independent readings reconciled cell by cell.

A whole-page OCR breaks dense tables (digits read as I/O/Z, rows lost, a table spilled into one cell per line). Here:
1. the vision model finds the tables of a page (prompts/table_boxes.md, kind table/chart, 0-1000 boxes);
2. each table is rendered from the PDF itself (PyMuPDF clip) at 300 dpi; a long one is cut into bands of up to
   BAND_ROWS text rows, the header strip repeated on top of every band;
3. three readings of every band: Gemma and Qwen3-VL with prompts/ocr_table.md (printed cells only, «[неразборчиво]»
   instead of a guess) and Tesseract (rus+eng, word boxes; the image scaled to ~30 px digits, Otsu, contrast up;
   cells assembled by code: rows by y, columns by the x gaps shared by the rows);
4. reconcile: cells are normalised (thousand separators, comma/point, minus/dash, %, I/O/Z/S -> 1/0/2/5 in numbers,
   Latin/Cyrillic look-alikes) and compared; all three equal -> «проверено», otherwise «не проверено» with the
   majority (or the vision reading) kept and marked. Automatic checks: totals of rows and columns, shares summing to
   100 % (±0.5), a year or period in the header.
Scoring against a hand-made reference (data/analytics/eval/tables/<id>.csv): the share of reference cells found in the
right order in the best-matching row of a candidate.
python -X utf8 stage5_insight_detection/bank_analytics/tables.py --eval-before
python -X utf8 stage5_insight_detection/bank_analytics/tables.py --pilot --budget 5
python -X utf8 stage5_insight_detection/bank_analytics/tables.py --all --budget 7   (pages done before are skipped)
"""
import argparse, csv, json, re, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.data import DATA

EVAL_DIR = DATA / 'analytics' / 'eval' / 'tables'
TABLE_DIR = DATA / 'analytics' / 'tables'
LOOKALIKE = str.maketrans('АВЕКМНОРСТУХаеорсухіІ', 'ABEKMHOPCTYXaeopcyxiI')
DIGIT_FIX = str.maketrans('IOZSlоОЗ', '10251002')
NUMERIC = re.compile(r'^[-+]?[\dIOZSlоОЗ]+([.,][\dIOZSlоОЗ]+)?[%x]?\*?$', re.I)


def norm(cell: str) -> str:
    """One spelling for comparison: no spaces, look-alikes and dashes unified, comma for decimals, digits fixed in numbers."""
    s = re.sub(r'\s+', '', str(cell or '')).replace(' ', '').translate(LOOKALIKE)
    s = re.sub(r'^[—–−]$', '-', s)
    s = re.sub(r'(?<=\d)[−–—]|^[−–—](?=\d)', '-', s).replace('−', '-')
    if NUMERIC.match(s):
        s = s.translate(DIGIT_FIX).replace('.', ',')
        s = re.sub(r'(?<=\d)x$', 'x', s, flags=re.I)
    return s.lower()


def lcs(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence: cells right and in the right order."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def read_reference(path: Path) -> list[list[str]]:
    with open(path, encoding='utf-8') as f:
        return [row for row in csv.reader(f, delimiter=';')]


def candidate_rows(text: str) -> list[list[str]]:
    """Every line of a page split into cells: Markdown tables by «|», tab-separated lines, else runs of 2+ spaces."""
    rows = []
    for line in text.split('\n'):
        s = line.strip()
        if not s or re.fullmatch(r'\|?[\s:|-]+\|?', s):
            continue
        parts = s.strip('|').split('|') if '|' in s else s.split('\t') if '\t' in s else re.split(r'\s{2,}', s)
        rows.append([p.strip() for p in parts])
    return rows


def score(reference: list[list[str]], rows: list[list[str]]) -> dict:
    """Share of reference cells (header row included, label column too) found in order in the best candidate row."""
    cands = [[norm(c.replace(MARK, '')) for c in r if norm(c.replace(MARK, ''))] for r in rows]  # «(?)» is a status, not text
    total = right = 0
    misses = []
    for ref in reference:
        want = [norm(c) for c in ref if norm(c)]
        if len(want) <= 1:
            continue  # a section row («Russia», «До 1 года»): a label only
        best = max(cands, key=lambda c: lcs(want, c), default=[])
        got = lcs(want, best)
        total += len(want)
        right += got
        if got < len(want):
            misses.append({'ref': ref[:6], 'got': best[:8], 'right': got, 'of': len(want)})
    return {'cells': total, 'right': right, 'share': round(right / total, 4) if total else 0.0, 'misses': misses}


PILOT = {  # reference id -> (pdf stem, page)
    'fi_p26_t1': ('FI_Strategy_3Q26_22072026', 26), 'fi_p24_t1': ('FI_Strategy_3Q26_22072026', 24),
    'steels_p3_t1': ('ALFA_Russian_Steels_1К21PreviewUpdate_15Apr2021', 3),
    'steels_p2_t1': ('ALFA_Russian_Steels_1К21PreviewUpdate_15Apr2021', 2),
    'steels_p2_t2': ('ALFA_Russian_Steels_1К21PreviewUpdate_15Apr2021', 2),
    'steels_p2_t3': ('ALFA_Russian_Steels_1К21PreviewUpdate_15Apr2021', 2),
    'macro_p4_t1': ('ALFA-macro_stat1H26', 4), '7m26_p3_t1': ('ALFA-7M26', 3), '7m26_p3_t2': ('ALFA-7M26', 3),
}


def page_of_text(md: str, page: int) -> str:
    m = re.search(rf'^## Страница {page}\s*$(.*?)(?=^## Страница \d+\s*$|\Z)', md, re.S | re.M)
    return m.group(1) if m else ''


def eval_before() -> dict:
    """The metric «до»: what text/ and fragments.jsonl have now, against the reference."""
    frags = [json.loads(line) for line in (DATA / 'analytics' / 'fragments.jsonl').read_text(encoding='utf-8').splitlines() if line]
    out = {}
    for rid, (stem, page) in PILOT.items():
        ref = read_reference(EVAL_DIR / f'{rid}.csv')
        md = (DATA / 'analytics' / 'text' / f'{stem}.md').read_text(encoding='utf-8')
        page_frags = '\n'.join(f['text'] for f in frags if Path(f['file']).stem == stem and f['page'] == page)
        out[rid] = {'text': score(ref, candidate_rows(page_of_text(md, page))), 'fragments': score(ref, candidate_rows(page_frags))}
    return out


def summary(results: dict, key: str) -> str:
    cells = sum(r[key]['cells'] for r in results.values())
    right = sum(r[key]['right'] for r in results.values())
    return f'{right}/{cells} = {right / cells:.1%}' if cells else '—'


BAND_ROWS, DPI = 12, 300


def text_ink(gray: np.ndarray, threshold: int = 170) -> np.ndarray:
    """Ink of the text only: the ruling lines of the table (a row or a column of ink across half the image) removed."""
    ink = gray < threshold
    ink[:, ink.mean(0) > 0.5] = False
    ink[ink.mean(1) > 0.5] = False
    return ink


def line_height(gray: np.ndarray, threshold: int = 170) -> float:
    """Median height of one text line (before the lines of a cell are merged)."""
    from stage5_insight_detection.bank_analytics.digitize import groups
    lines = groups(text_ink(gray, threshold).sum(1) > 0, gap=2)
    return float(np.median([b - a + 1 for a, b in lines])) if lines else 0.0


def ink_rows(gray: np.ndarray, threshold: int = 170) -> list[tuple[int, int]]:
    """Table rows of an image. With horizontal rules: the text between two rules is one row (a cell may hold two lines).
    Without: text lines, merged only when they nearly touch (under 0.12 of a line)."""
    from stage5_insight_detection.bank_analytics.digitize import groups
    ink = text_ink(gray, threshold)
    lines = groups(ink.sum(1) > 0, gap=2)
    if not lines:
        return []
    rules = groups((gray < threshold).mean(1) > 0.5, gap=2)
    if len(rules) >= 3 and len(rules) >= 0.4 * len(lines):  # a rule under every row (not just between sections)
        edges = [0] + [(a + b) // 2 for a, b in rules] + [gray.shape[0] - 1]
        rows = []
        for top, bottom in zip(edges, edges[1:]):
            inside = [(a, b) for a, b in lines if top <= (a + b) / 2 <= bottom]
            if inside:
                rows.append((inside[0][0], inside[-1][1]))
        return rows
    height = np.median([b - a + 1 for a, b in lines])
    merged = [list(lines[0])]
    for a, b in lines[1:]:
        if a - merged[-1][1] < 0.12 * height:  # lines of one cell nearly touch; rows of a shaded table do not
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def render_table(pdf: Path, page: int, bbox: list[float], dpi: int = DPI) -> np.ndarray:
    """The table from the PDF at `dpi`: the model's box grown to the nearest white band (a box cut at the labels or the
    last column is common), at most 6 % of the page wide and 2 % high."""
    import pymupdf
    from stage5_insight_detection.bank_analytics.digitize import snap_box, to_rgb
    with pymupdf.open(pdf) as doc:
        p = doc[page - 1]
        w, h = p.rect.width, p.rect.height
        x1, y1, x2, y2 = [v / 1000 for v in bbox]
        clip = pymupdf.Rect(max(0, (x1 - 0.06) * w), max(0, (y1 - 0.02) * h), min(w, (x2 + 0.06) * w), min(h, (y2 + 0.02) * h))
        rgb = to_rgb(p.get_pixmap(dpi=dpi, clip=clip).tobytes('png'))
    k = dpi / 72
    inner = (int((x1 * w - clip.x0) * k), int((y1 * h - clip.y0) * k), int((x2 * w - clip.x0) * k), int((y2 * h - clip.y0) * k))
    X1, Y1, X2, Y2 = snap_box(rgb, inner, gap=int(0.08 * k * 72 / 2.54 * 2))  # a white band of ~1.6 mm ends the table
    return rgb[Y1:Y2 + 1, X1:X2 + 1]


def title_above(pdf: Path, page: int, bbox: list[float], known: list[str]) -> str:
    """The table's title: the line closest to the table, in the strip from 3 % above its box to 1.5 % inside it (read by
    Tesseract), that matches one of the titles grounding gave for this page; grounding's own title for the box
    otherwise. Grounding alone mixed up the titles of three similar tables (Severstal / MMK / NLMK)."""
    import pymupdf
    from stage5_insight_detection.bank_analytics.digitize import similar
    with pymupdf.open(pdf) as doc:
        p = doc[page - 1]
        w, h = p.rect.width, p.rect.height
        x1, y1, x2, _ = [v / 1000 for v in bbox]
        clip = pymupdf.Rect(max(0, (x1 - 0.06) * w), max(0, (y1 - 0.03) * h), min(w, (x2 + 0.02) * w), min(h, (y1 + 0.015) * h))
        pix = p.get_pixmap(dpi=300, clip=clip)
    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    lines = [' '.join(row).strip() for row in tesseract_cells(rgb) if len(' '.join(row).strip()) >= 3]
    hits = [(ln, max(known, key=lambda k: similar(k, ln))) for ln in lines if known]
    hits = [k for ln, k in hits if similar(k, ln) >= 0.6]
    return hits[-1] if hits else ''


OVERLAP = 2


def bands(rgb: np.ndarray, max_rows: int = BAND_ROWS, overlap: int = OVERLAP) -> list[np.ndarray]:
    """A long table cut into bands of up to max_rows text rows (cut in the white gap between rows), each band after the
    first starting `overlap` rows early (glued back by row labels: merge_bands) and topped by the header strip."""
    gray = rgb.mean(axis=2)
    rows = ink_rows(gray)
    if len(rows) <= max_rows + 2:  # rows merged across a cell's lines: count the text lines too
        from stage5_insight_detection.bank_analytics.digitize import groups
        lines = groups(text_ink(gray).sum(1) > 0, gap=2)
        if len(lines) <= 1.5 * (max_rows + 2):
            return [rgb]
        rows = [(a, b) for a, b in lines]
    head = rgb[:max(0, rows[1][0] - 2)]
    cut = lambda i: (rows[i - 1][1] + rows[i][0]) // 2
    out, start = [], 0
    for i in range(max_rows, len(rows), max_rows):
        out.append(rgb[start:cut(i)] if not out else np.vstack([head, rgb[start:cut(i)]]))
        start = cut(max(1, i - overlap))
    out.append(np.vstack([head, rgb[start:]]))
    return out


def label_of(row: list[str]) -> str:
    return norm(row[0]) if row else ''


def merge_bands(parts: list[list[list[str]]], overlap: int = OVERLAP) -> tuple[list[list[str]], set[int]]:
    """Rows of overlapping bands glued by row labels: a leading row of a band whose label is among the last rows already
    taken is a duplicate and dropped. If no leading row matched, the junction is unsure: nothing is dropped and those
    rows are returned as doubtful (their cells become «не проверено»). Each part has its header row first."""
    rows, doubtful = [], set()
    for i, part in enumerate(parts):
        body = part if i == 0 else part[1:]  # the repeated header once
        if i == 0 or not overlap:
            rows += body
            continue
        tail = {label_of(r) for r in rows[-(overlap + 2):] if label_of(r)}
        lead = body[:overlap + 1]
        dup = [label_of(r) in tail for r in lead]
        if any(dup):
            last = max(k for k, d in enumerate(dup) if d)
            body = body[last + 1:]
        else:
            doubtful |= set(range(len(rows), len(rows) + len(lead)))
        rows += body
    return rows, doubtful


def tesseract_cells(rgb: np.ndarray) -> list[list[str]]:
    """Tesseract (rus+eng) on a table image: scaled to ~30 px digits, grey, contrast up, Otsu; words to rows by the ink
    rows of the image and to columns by the x gaps no row has a word in."""
    from common.ocr_local import word_data
    from PIL import Image, ImageOps
    from stage5_insight_detection.bank_analytics.digitize import groups
    gray = rgb.mean(axis=2)
    rows = ink_rows(gray)
    if not rows:
        return []
    gray = gray.copy()  # ruling lines out with their anti-aliased edges: Tesseract reads them as «|» or «о» and glues the cells
    for axis in (0, 1):
        lines = np.nonzero((gray < 200).mean(axis) > 0.5)[0]
        for i in lines:
            if axis == 0:
                gray[:, max(0, i - 3):i + 4] = 255
            else:
                gray[max(0, i - 3):i + 4] = 255
    line_h = line_height(gray) or 20.0
    k = max(0.5, min(4.0, 30 / max(1.0, 0.75 * line_h)))  # a digit is about 3/4 of a line
    img = ImageOps.autocontrast(Image.fromarray(gray.astype(np.uint8)).resize((int(rgb.shape[1] * k), int(rgb.shape[0] * k)),
                                                                                 Image.LANCZOS))
    a = np.asarray(img).astype(float)
    hist = np.bincount(a.astype(int).ravel(), minlength=256)  # Otsu threshold
    total, sum_all, w_b, sum_b, best, thr = a.size, (np.arange(256) * hist).sum(), 0, 0.0, -1, 128
    for t in range(256):
        w_b += hist[t]
        if not w_b or w_b == total:
            continue
        sum_b += t * hist[t]
        m_b, m_f = sum_b / w_b, (sum_all - sum_b) / (total - w_b)
        between = w_b * (total - w_b) * (m_b - m_f) ** 2
        if between > best:
            best, thr = between, t
    binary = (a > thr).astype(np.uint8) * 255
    if (binary == 0).mean() > 0.5:  # light text on a dark band (a coloured header): invert
        binary = 255 - binary
    data = word_data(Image.fromarray(binary), psm=6)
    words = [(data['left'][i] / k, data['top'][i] / k, data['width'][i] / k, data['height'][i] / k, data['text'][i].strip())
             for i in range(len(data['text'])) if data['text'][i].strip() and float(data['conf'][i]) > 0]
    if not words:
        return []
    width = rgb.shape[1]
    cover = np.zeros(width, int)
    for x, y, w, h, t in words:
        cover[int(x):int(x + w) + 1] += 1
    gap = max(8, int(0.9 * line_h))  # a column gap is wider than the space between the words of a label
    cols = groups(cover > 0, gap=gap)
    out = []
    for r0, r1 in rows:
        in_row = [wd for wd in words if r0 - 3 <= wd[1] + wd[3] / 2 <= r1 + 3]
        if not in_row:
            continue
        cells = []
        for c0, c1 in cols:
            cells.append(' '.join(t for x, y, w, h, t in sorted(in_row) if c0 <= x + w / 2 <= c1))
        out.append(cells)
    return out


def align(base: list[str], other: list[str]) -> list[str | None]:
    """Cells of `other` placed under the cells of `base` (difflib on the normalised cells; equal lengths: by index)."""
    if len(base) == len(other):
        return list(other)
    from difflib import SequenceMatcher
    out: list[str | None] = [None] * len(base)
    sm = SequenceMatcher(None, [norm(c) for c in base], [norm(c) for c in other], autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ('equal', 'replace') and i2 - i1 == j2 - j1:
            for k in range(i2 - i1):
                out[i1 + k] = other[j1 + k]
    return out


def match_row(row: list[str], rows: list[list[str]], used: set) -> int | None:
    """Index of the row of another reading that matches: by its label first (a value that sits under another label is not
    confirmed by it), by the overlap of cells only when no label matches; each row used once."""
    from difflib import SequenceMatcher
    want = [norm(c) for c in row]
    label = want[0] if want else ''
    if len(label) >= 3:
        same = [i for i, r in enumerate(rows) if i not in used and r and
                SequenceMatcher(None, label, norm(r[0])).ratio() >= 0.8]
        if same:
            return max(same, key=lambda i: lcs(want, [norm(c) for c in rows[i]]))
    best, best_i = 0, None
    for i, r in enumerate(rows):
        if i in used:
            continue
        got = lcs(want, [norm(c) for c in r])
        if got > best:
            best, best_i = got, i
    return best_i if best >= max(2, len(want) // 3) else None


def widen(row: list[str], width: int, guide: list[str] | None) -> list[str]:
    """A row one or more cells short of the header placed by a full row of another reading (Tesseract keeps its cells at
    their x): its cells go under the guide cells they equal, the rest in order into the free places between them."""
    if len(row) >= width or not guide or len(guide) != width:
        return row + [''] * (width - len(row)) if len(row) < width else row
    from difflib import SequenceMatcher
    out: list[str | None] = [None] * width
    taken = set()
    sm = SequenceMatcher(None, [norm(c) for c in row], [norm(c) for c in guide], autojunk=False)
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            out[blk.b + k] = row[blk.a + k]
            taken.add(blk.a + k)
    rest = [c for i, c in enumerate(row) if i not in taken]
    for j in range(width):  # unmatched cells keep their order, before the next anchored cell
        if out[j] is None and rest:
            nxt = next((k for k in range(j + 1, width) if out[k] is not None), width)
            free = sum(1 for k in range(j, nxt) if out[k] is None)
            if len(rest) >= free or j > 0:
                out[j] = rest.pop(0)
    return [c if c is not None else '' for c in out]


THOUSANDS = re.compile(r'^-?\d{1,3}( \d{3})+$|^-?\d{4,}$')
DECIMAL = re.compile(r'^-?\d+,\d{1,2}%?x?$')


def column_style(cells: list[str]) -> str:
    """thousands («17 760», «992») or decimal («21,2», «0,8x») by most of the column's numbers; '' if unclear."""
    th = sum(bool(THOUSANDS.match(c.strip())) for c in cells)
    de = sum(bool(DECIMAL.match(norm(c))) for c in cells)
    return 'thousands' if th > 2 * de and th >= 2 else 'decimal' if de > 2 * th and de >= 2 else ''


def separator_key(value: str, style: str) -> str:
    """Same digits, another separator («9 215» / «9,215»): one key when the column says which it is."""
    n = norm(value)
    if style == 'thousands' and re.fullmatch(r'-?\d{1,3}([,.]\d{3})+', n):
        return n.replace(',', '').replace('.', '')
    if style == 'decimal' and re.fullmatch(r'-?\d+ \d{1,2}', value.strip()):
        return n  # «0 8» read for «0,8»: no key, it stays a disagreement
    return n


UNREADABLE = ('', '[неразборчиво]', None)


def reconcile(readings: dict[str, list[list[str]]], base: str = 'gemma', doubtful: dict[str, set] | None = None) -> list[list[dict]]:
    """Cell by cell: {'value', 'status'} with status verified (all three equal), two (two equal, the third empty or
    unreadable) or unverified (they differ: the majority or the base reading's value is kept). Rows are matched by
    label; a row short of the header is placed by a full row of another reading; «9 215» and «9,215» agree when the
    column shows its numbers are thousands; a row from an unsure band junction (doubtful) is unverified throughout."""
    doubtful = doubtful or {}

    def agreement(name):  # how many cells of this reading another reading has in its best row
        rows = readings.get(name) or []
        return sum(lcs([norm(c) for c in r], max(([norm(c) for c in o] for k, rs in readings.items() if k != name for o in rs),
                                                  key=lambda o: lcs([norm(c) for c in r], o), default=[])) for r in rows)
    base = max([k for k in readings if readings[k]], key=agreement, default=base)  # the reading the others confirm most
    order = [base] + [k for k in readings if k != base]
    rows_base = readings.get(base) or max(readings.values(), key=len)
    width = max((len(r) for r in rows_base[:1]), default=0)
    others = {k: readings[k] for k in order[1:]}
    used = {k: set() for k in others}
    matched = []
    for bi, row in enumerate(rows_base):
        found = {}
        for k, rows in others.items():
            i = match_row(row, rows, used[k])
            if i is not None:
                used[k].add(i)
            found[k] = rows[i] if i is not None else None
        if len(row) < width:  # a lost cell: place the row by a full row of another reading, Tesseract first
            guide = next((found[k] for k in ('tesseract', *others) if found.get(k) and len(found[k]) == width), None)
            row = widen(list(row), width, guide)
        matched.append((bi, row, found))
    styles = [column_style([r[j] for _, r, _ in matched[1:] if j < len(r)]) for j in range(width)]
    out = []
    for bi, row, found in matched:
        cols = {base: list(row)}
        for k, r in found.items():
            cols[k] = align(row, r) if r is not None else [None] * len(row)
        unsure = bi in doubtful.get(base, set())
        cells = []
        for j, v in enumerate(row):
            style = styles[j] if j < len(styles) else ''
            vals = {k: cols[k][j] for k in cols}
            normed = {k: separator_key(x, style) for k, x in vals.items() if x not in UNREADABLE and norm(x) not in ('[неразборчиво]',)}
            counts: dict[str, int] = {}
            for x in normed.values():
                counts[x] = counts.get(x, 0) + 1
            top = max(counts, key=counts.get) if counts else ''
            n_top, n_read = counts.get(top, 0), len(normed)
            status = 'verified' if n_top == 3 else 'two' if n_top == 2 and n_read == 2 else 'unverified'
            first = next((vals[k] for k in order if vals.get(k) not in UNREADABLE), v)  # the base lost it: another reading's
            value = next((vals[k] for k in order if vals.get(k) not in UNREADABLE and separator_key(vals[k], style) == top), v) \
                if n_top >= 2 else first
            if style == 'thousands' and n_top >= 2 and re.fullmatch(r'-?\d{1,3}([,.]\d{3})+', norm(value or '')):
                value = re.sub(r'[,.]', ' ', value)  # written as the column writes its thousands
            cells.append({'value': value or '', 'status': ('unverified' if unsure else status) if top else 'empty'})
        out.append(cells)
    return out


def auto_checks(header: list[str], rows: list[list[dict]]) -> list[str]:
    """Totals of rows, shares summing to 100 % (±0.5), a year or a period in the header. Returns the findings."""
    found = []
    if header and any(re.search(r'(19|20)\d{2}|\d[КQ]\d{2}|FY\d{2}|\d{2}\.\d{2}\.\d{4}|[Пп]\b', h) for h in header):
        found.append('шапка: год или период есть')
    num = lambda c: float(norm(c['value']).rstrip('%x*').replace(',', '.')) if re.fullmatch(r'-?\d+(,\d+)?%?', norm(c['value'])) else None
    for r in rows:
        label = norm(r[0]['value']) if r else ''
        vals = [num(c) for c in r[1:]]
        if re.search(r'итого|всего|total', label) and any(v is not None for v in vals):
            found.append(f'итог «{r[0]["value"]}» — нужна сверка с суммой строк (по столбцам)')
    for j in range(1, max((len(r) for r in rows), default=0)):
        col = [num(r[j]) for r in rows if j < len(r) and r[j]['value'].strip().endswith('%')]
        if len(col) >= 3 and all(v is not None for v in col) and 99.5 <= sum(col) <= 100.5:
            found.append(f'столбец {j}: доли дают 100% ({sum(col):.1f})')
    return found


def reading_rows(answer) -> list[list[str]]:
    """The model's JSON {header, rows} as rows (the header first); anything else is no reading."""
    if not isinstance(answer, dict):
        return []
    rows = [[str(c) for c in answer['header']]] if isinstance(answer.get('header'), list) and answer['header'] else []
    rows += [[str(c) for c in r] for r in answer.get('rows') or [] if isinstance(r, list)]
    return rows


def run_pages(pages: list[tuple[str, int]], budget: float, likely: set | None = None, *,
              pdf_dir: Path | None = None, out_dir: Path = TABLE_DIR, callbacks: list | None = None) -> list[dict]:
    """Find the tables of each page (Qwen grounding), cut them into bands, read every band three times; one JSON per page
    in data/analytics/tables/. A page done before is loaded, not paid again. When Qwen sees no table, Gemma looks again
    only on `likely` pages (a table fragment there already): elsewhere «no table» is the usual answer."""
    import time
    from stage5_insight_detection.bank_analytics.charts import ChartReader, ChartReadError, GROUND_DPI, render
    from stage5_insight_detection.bank_analytics.digitize import to_png
    from stage5_insight_detection.bank_analytics.ocr import PDF_DIR
    from stage5_insight_detection.bank_analytics.digest import BudgetExceeded
    if not pages:
        return []
    pdf_dir = pdf_dir or PDF_DIR
    qwen, gemma = ChartReader('qwen', callbacks=callbacks), ChartReader('gemma', callbacks=callbacks)
    for r in (qwen, gemma):  # a dense table is a long answer
        r.model = r.model.model_copy(update={'max_tokens': 4000})
    spent = lambda: qwen.meter.cost + gemma.meter.cost

    def ask(reader, prompt, png, what):
        if spent() + max(qwen.max_call, gemma.max_call, 0.05) > budget:
            raise BudgetExceeded(f'budget: spent {spent():.2f} of {budget} ₽ before {what}')
        reader.budget = reader.meter.cost + max(0, budget - spent())  # retries share the page run's remaining budget
        return reader.ask(prompt, png, what)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = []
    def read_page(stem, page):
        from stage5_insight_detection.bank_analytics.assets import prepare
        from stage5_insight_detection.bank_analytics.ocr import PROMPTS
        pdf = pdf_dir / f'{stem}.pdf'
        prepare(pdf, out_dir, PROMPTS.render('table_boxes') + PROMPTS.render('ocr_table'))
        path = out_dir / f'{stem}_p{page}.json'
        if path.exists():
            return json.loads(path.read_text(encoding='utf-8'))
        t0, c0 = time.time(), spent()
        page_png = render(pdf, page, GROUND_DPI)
        boxes, _ = ask(qwen, 'table_boxes', page_png, f'{stem} p{page} tables')
        found = [b for b in boxes or [] if isinstance(b, dict) and b.get('kind') == 'table' and len(b.get('bbox') or []) == 4]
        if not found and (likely is None or (stem, page) in likely):  # Qwen saw no table: ask the other model once
            boxes, _ = ask(gemma, 'table_boxes', page_png, f'{stem} p{page} tables (gemma)')
            found = [b for b in boxes or [] if isinstance(b, dict) and b.get('kind') == 'table' and len(b.get('bbox') or []) == 4]
        tables = []
        found = distinct_tables([{'bbox': b['bbox'], **b} for b in found])
        for n, b in enumerate(found, 1):
            rgb = render_table(pdf, page, b['bbox'])
            parts = {'gemma': [], 'qwen': [], 'tesseract': []}
            for i, band in enumerate(bands(rgb)):
                png = to_png(band)
                for name, reader in (('gemma', gemma), ('qwen', qwen)):
                    parts[name].append(reading_rows(ask(reader, 'ocr_table', png, f'{stem} p{page} t{n} band{i} {name}')[0]))
                parts['tesseract'].append(tesseract_cells(band))
            readings = {k: merge_bands(v)[0] for k, v in parts.items()}
            known = [str(x.get('title') or '') for x in boxes or [] if isinstance(x, dict) and x.get('title')]
            tables.append({'n': n, 'title': b.get('title', ''), 'title_above': title_above(pdf, page, b['bbox'], known), 'bbox': b['bbox'],
                           'bands': len(bands(rgb)), 'overlap': OVERLAP, 'band_readings': parts, 'readings': readings})
        rec = {'stem': stem, 'page': page, 'boxes': boxes, 'tables': tables, 'cost': round(spent() - c0, 4), 'sec': round(time.time() - t0, 1)}
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding='utf-8')
        print(stem, page, len(tables), 'tables', rec['cost'], '₽', rec['sec'], 's', flush=True)
        return rec

    for stem, page in pages:
        try:
            out.append(read_page(stem, page))
        except ChartReadError as error:
            print(error, flush=True)
    print(f'spent {spent():.2f} ₽')
    return out


PILOT_PAGES = sorted(set(PILOT.values()))


def page_records(stem: str, page: int, folder: Path | None = None) -> dict | None:
    path = (folder or TABLE_DIR) / f'{stem}_p{page}.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def eval_after() -> dict:
    """The metric «после» for every reading alone and for the reconciliation, plus how many reconciled cells got
    «проверено» and how many of those are right (the rule «2 of 3» counted apart)."""
    out = {}
    for rid, (stem, page) in PILOT.items():
        rec = page_records(stem, page)
        if not rec:
            continue
        ref = read_reference(EVAL_DIR / f'{rid}.csv')
        res = {}
        for name in ('gemma', 'qwen', 'tesseract'):
            res[name] = score(ref, [r for t in distinct_tables(rec['tables']) for r in table_readings(t)[0][name]])
        merged = [reconcile(*table_readings(t)) for t in distinct_tables(rec['tables'])]
        res['reconciled'] = score(ref, [[c['value'] for c in r] for t in merged for r in t])
        refs = [[norm(c) for c in r if norm(c)] for r in ref]
        st = {'verified': [0, 0], 'two': [0, 0], 'unverified': [0, 0]}
        for t in merged:
            for r in t:
                vals = [norm(c['value']) for c in r]
                best = max(refs, key=lambda x: lcs(x, [v for v in vals if v]), default=[])
                if lcs(best, [v for v in vals if v]) < max(2, len(best) // 2):
                    continue  # a row of another table on the same page
                for c, v in zip(r, vals):
                    if c['status'] in st and v:
                        st[c['status']][0] += 1
                        st[c['status']][1] += v in best
        res['status'] = st
        out[rid] = res
    return out


def distinct_tables(tables: list[dict]) -> list[dict]:
    """Tables of a page without the duplicates grounding sometimes returns (the same box twice, a box of no height)."""
    def iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        area = lambda r: max(0, r[2] - r[0]) * max(0, r[3] - r[1])
        return inter / (area(a) + area(b) - inter) if inter else 0.0
    out = []
    for t in tables:
        if t['bbox'][3] - t['bbox'][1] < 10:
            continue
        if not any(iou(t['bbox'], o['bbox']) > 0.7 for o in out):
            out.append(t)
    return out


def looks_like_table(rows: list[list[str]]) -> bool:
    """A reading that is a table, not a paragraph or a chart read as one: 3+ rows, 3+ cells in a typical row, few long
    cells (a paragraph puts whole sentences in a cell), and numbers in at least 40 % of the cells after the label."""
    if len(rows) < 3:
        return False
    widths = sorted(len(r) for r in rows)
    cells = [c for r in rows for c in r[1:] if str(c).strip()]
    if widths[len(widths) // 2] < 3 or not cells:
        return False
    long = sum(len(str(c)) > 60 for r in rows for c in r) / max(1, sum(len(r) for r in rows))
    numeric = sum(bool(re.search(r'\d', str(c))) for c in cells) / len(cells)
    return long < 0.1 and numeric >= 0.4


def real_tables(tables: list[dict]) -> list[dict]:
    """Distinct tables that at least two of the three readings show as a table (grounding also boxes paragraphs and
    charts as tables)."""
    return [t for t in distinct_tables(tables) if sum(looks_like_table(table_readings(t)[0].get(k) or []) for k in
                                                      ('gemma', 'qwen', 'tesseract')) >= 2]


MARK = ' (?)'
PROMPT_SECTIONS = {norm(x) for x in re.findall(r'«([^»]+)»', next(
    (ln for ln in (Path(__file__).with_name('prompts') / 'ocr_table.md').read_text(encoding='utf-8').splitlines()
     if 'Строка-раздел' in ln), ''))}
CAPTION = '[таблица сверена: три независимых чтения по кропу; «(?)» — чтения разошлись]'


NUMBER_TOKEN = re.compile(r'^[-−–]?\d[\d,.]*%?\*?$')


def split_label(row: list[str], width: int) -> list[str]:
    """Tesseract glues the first columns to the label when they stand close («Ставка, % 6,3% 4,3% 8,5%» in one cell):
    a row short of `width` gets the numbers at the end of its first cell back as cells, as many as it lacks."""
    if not row or len(row) >= width:
        return row
    tokens = str(row[0]).split()
    k = 0
    while k < min(width - len(row), len(tokens)) and NUMBER_TOKEN.match(tokens[len(tokens) - 1 - k]):
        k += 1
    return [' '.join(tokens[:len(tokens) - k])] + tokens[len(tokens) - k:] + list(row[1:]) if k else row


def table_readings(t: dict) -> tuple[dict, str, dict]:
    """The readings of a table (bands glued by labels when the bands were kept) and its doubtful junction rows; Tesseract's
    rows with columns glued to the label split back to the models' width."""
    if t.get('band_readings'):
        glued = {k: merge_bands(v, t.get('overlap', 0)) for k, v in t['band_readings'].items()}
        readings, doubtful = {k: v[0] for k, v in glued.items()}, {k: v[1] for k, v in glued.items()}
    else:
        readings, doubtful = dict(t['readings']), {}
    width = max((len(r) for k, rows in readings.items() if k != 'tesseract' for r in rows or []), default=0)
    if readings.get('tesseract'):
        readings['tesseract'] = [split_label(r, width) for r in readings['tesseract']]
    return readings, 'gemma', doubtful


def reconciled_markdown(rec: dict) -> list[str]:
    """Markdown of the reconciled tables of a page: a cell where the readings disagree gets «(?)». Two readings that agree
    while the third could not read the cell count as agreement (the pilot: 389 of 389 right)."""
    out = []
    for t in real_tables(rec['tables']):
        readings = table_readings(t)
        rows = reconcile(*readings)
        if not rows:
            continue
        # a row of nothing (a ruling line read as a row), or a bare section label copied from the prompt's examples: Qwen
        # writes «До 1 года», «Russia» into tables that have no such row; kept when Tesseract sees it on the crop
        seen = norm(' '.join(c for r in readings[0].get('tesseract') or [] for c in r))
        leaked = lambda c: norm(c['value']) in PROMPT_SECTIONS and c['status'] != 'verified' and norm(c['value']) not in seen
        rows = [r for r in rows if any(c['value'].strip() for c in r[1:]) or (norm(r[0]['value']) and not leaked(r[0]))]
        cell = lambda c: c['value'].replace('|', '/') + (MARK if c['status'] == 'unverified' else '')
        width = max(len(r) for r in rows)
        if len(rows[0]) == width - 1 and all(len(r) == width for r in rows[1:]):  # the header without its empty corner
            rows[0] = [{'value': '', 'status': rows[0][0]['status']}] + rows[0]
        lines = [f'{t.get("title_above") or t.get("title") or "Таблица"} {CAPTION}',
                 '| ' + ' | '.join(cell(c) for c in rows[0]) + ' |', '|' + '---|' * width]
        lines += ['| ' + ' | '.join(cell(c) for c in r) + ' |' for r in rows[1:]]
        out.append('\n'.join(lines))
    return out


def replace_tables(text: str, rec: dict | None) -> str:
    """Replace only tables matched by row labels; keep charts and unrelated page tables."""
    if not rec or not real_tables(rec.get('tables') or []):
        return text
    rendered = reconciled_markdown(rec)
    if not rendered:
        return text
    labels = {norm(row.split('|')[1]) for md in rendered for row in md.splitlines()[3:]
              if row.strip().startswith('|') and len(row.split('|')) > 2
              and re.search(r'[а-яёa-z]', row.split('|')[1], re.I)}
    def replace_part(part):
        if part.lstrip().startswith('[График]'):
            return part
        lines, kept, block = part.split('\n'), [], []
        def flush():
            if block:
                row_labels = {norm(re.split(r'\t|\|', ln.strip().strip('|'))[0]) for ln in block}
                if not row_labels & labels:
                    kept.extend(block)
                block.clear()
        for ln in lines:
            if ln.strip().startswith('|') or ln.count('\t') >= 2:
                block.append(ln)
            else:
                flush()
                kept.append(ln)
        flush()
        return '\n'.join(kept)
    parts = re.split(r'(\[График\].*?\[/График\])', text, flags=re.S)
    return ''.join(replace_part(p) for p in parts).rstrip() + '\n\n' + '\n\n'.join(rendered)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-before', action='store_true')
    ap.add_argument('--pilot', action='store_true', help='find, cut and read the tables of the pilot pages')
    ap.add_argument('--eval-after', action='store_true')
    ap.add_argument('--all', action='store_true', help='every page of the 9 reports except the pilot ones')
    ap.add_argument('--budget', type=float, default=5.0)
    a = ap.parse_args()
    if a.pilot:
        run_pages(PILOT_PAGES, a.budget)
    if a.all:  # every page of every report: pages with a table fragment first, then the rest (charts, text)
        import pymupdf
        from stage5_insight_detection.bank_analytics.digest import BudgetExceeded
        from stage5_insight_detection.bank_analytics.ocr import PDF_DIR
        from stage5_insight_detection.bank_analytics.library import BankLibrary
        lib = BankLibrary.refresh()
        likely = {(Path(f.file).stem, f.page) for f in lib.fragments if f.kind == 'table'}
        every = []
        for pdf in sorted(PDF_DIR.glob('*.pdf')):
            with pymupdf.open(pdf) as doc:
                every += [(pdf.stem, n) for n in range(1, doc.page_count + 1)]
        todo = [pg for pg in every if pg not in PILOT_PAGES]
        todo.sort(key=lambda pg: pg not in likely)
        print(len(todo), 'pages,', sum(pg in likely for pg in todo), 'with a table fragment', flush=True)
        try:
            run_pages(todo, a.budget, likely)
        except BudgetExceeded as e:
            done = {(r['stem'], r['page']) for r in (json.loads(f.read_text(encoding='utf-8')) for f in TABLE_DIR.glob('*.json'))
                    if 'stem' in r and 'page' in r}
            print(e, '| pages left:', [pg for pg in todo if pg not in done])
    if a.eval_after:
        res = eval_after()
        for rid, r in res.items():
            print(rid, ' '.join(f'{k} {r[k]["right"]}/{r[k]["cells"]}' for k in ('gemma', 'qwen', 'tesseract', 'reconciled')), r['status'])
        for k in ('gemma', 'qwen', 'tesseract', 'reconciled'):
            print('ИТОГО', k, summary(res, k))
        tot = {s: [sum(r['status'][s][i] for r in res.values()) for i in (0, 1)] for s in ('verified', 'two', 'unverified')}
        print('статусы (ячеек, из них верных):', tot)
    if a.eval_before:
        res = eval_before()
        for rid, r in res.items():
            print(f'{rid:14} text {r["text"]["right"]:4}/{r["text"]["cells"]:<4} {r["text"]["share"]:.1%}   '
                  f'fragments {r["fragments"]["right"]:4}/{r["fragments"]["cells"]:<4} {r["fragments"]["share"]:.1%}')
        print('ИТОГО text', summary(res, 'text'), '| fragments', summary(res, 'fragments'))
