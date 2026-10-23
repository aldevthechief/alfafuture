"""Read chart values from the pixels of a crop: code, no model. A vision model reads the printed text of a chart well
(ticks, dates, legend, data labels) but makes up values that are not printed; their position on the image is exact.

Input: an RGB crop (charts.py) and what the model read from it (prompts/ocr_chart.md): y ticks, x labels, series.
1. Y scale: dark text rows in the tick label band (left or right of the plot) are grouped into blocks; their number must
   equal the number of ticks; a least-squares line maps pixel -> value. A tick more than 0.1 of a step off the line
   rejects the chart.
2. X: dark pixels in the label band under the plot (labels are rotated) are grouped into columns with gaps > 3 px; their
   number must equal the number of labels; a group's centre is the x of that point.
3. Series by colour (red, black, grey, green, yellow, blue) in the plot without the legend; the legend gives the order:
   a colour's sample is the first of its pixels with legend text to the right.
   - line with markers: in columns x±6 the top and bottom of the marker; a peak is top + half a marker, a trough is
     bottom - half, else the middle; half a marker = 10th percentile of the half-ranges;
   - dotted or plain line: median of the colour's rows in x±3, widened to x±9 when empty;
   - bars: the far edge of the bar from the zero line.
4. Checks: values inside the scale, one value per label, series masks distinct. Anything off -> "значения не определены"
   with the reason, never a guess.
Accuracy on Alfa charts: ±0.2 of a scale step (the card says so). Values are `approx`: direction and size, not facts.
python -X utf8 stage5_insight_detection/bank_analytics/digitize.py <crop.png> <gemma.json>
"""
import json, re, sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pymupdf

COLORS = {  # name -> mask over int RGB arrays
    'red': lambda R, G, B: (R > 150) & (G < 130) & (B < 130) & (R - G > 60),
    'black': lambda R, G, B: (R < 90) & (G < 90) & (B < 90),
    'grey': lambda R, G, B: (R >= 95) & (R < 185) & (abs(R - G) < 18) & (abs(G - B) < 18),
    'green': lambda R, G, B: (G > 110) & (G - R > 40) & (G - B > 20),
    'yellow': lambda R, G, B: (R > 190) & (G > 130) & (B < 100) & (R - B > 110),
    'blue': lambda R, G, B: (B > 140) & (B - R > 50) & (B - G > 20),
}
RU_COLORS = {'красн': 'red', 'чёрн': 'black', 'черн': 'black', 'сер': 'grey', 'зелён': 'green', 'зелен': 'green',
             'жёлт': 'yellow', 'желт': 'yellow', 'оранж': 'yellow', 'син': 'blue', 'голуб': 'blue'}
# what the pilot allowed (Рис. 2 within 0.3 step on average; no value more than a step off on the pilot charts): lines with
# markers or dotted get values at every date; a plain (daily) line only its last point, its min and max were 1.6 steps
# off; bars, stacked bars, areas and pies had no pilot -> «значения не определены» (bars are measured, but not enabled)
LINES = ('линия', 'пунктир', 'линия с маркерами')
LAST_ONLY = ('линия',)
NUM = re.compile(r'[-−–]?\d+(?:[.,]\d+)?')


def to_rgb(png: bytes) -> np.ndarray:
    pix = pymupdf.Pixmap(png)
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    return a[:, :, :3].astype(int)


def to_png(rgb: np.ndarray) -> bytes:
    h, w = rgb.shape[:2]
    return pymupdf.Pixmap(pymupdf.csRGB, w, h, np.ascontiguousarray(rgb.astype(np.uint8)).tobytes(), 0).tobytes('png')


def tick_value(s: str) -> float | None:
    m = NUM.search(str(s).replace(' ', ''))
    return float(m.group().replace(',', '.').replace('−', '-').replace('–', '-')) if m else None


def groups(profile: np.ndarray, gap: int, min_size: int = 1) -> list[tuple[int, int]]:
    """Runs of non-zero entries of a profile joined across gaps of up to `gap` zeros: [(start, end)]."""
    idx = np.nonzero(profile)[0]
    if not len(idx):
        return []
    out, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > gap + 1:
            out.append((start, prev))
            start = i
        prev = i
    out.append((start, prev))
    return [g for g in out if g[1] - g[0] + 1 >= min_size]


def snap_box(rgb: np.ndarray, inner: tuple[int, int, int, int], white: int = 235, gap: int = 6) -> tuple[int, int, int, int]:
    """Grow the box (x1, y1, x2, y2) from the model's box `inner` outwards until a white band of `gap` px: what touches
    the chart (labels, legend) comes in, a neighbour beyond a white band stays out. A neighbour inside the model's box
    stays in: the box is not shrunk (tried: cutting by blocks loses titles more often than it removes neighbours)."""
    h, w = rgb.shape[:2]
    ink = rgb.min(axis=2) < white
    x1, y1, x2, y2 = inner

    def grow(pos, step, limit, line):
        blank = 0
        while pos != limit:
            nxt = pos + step
            blank = blank + 1 if not line(nxt).any() else 0
            if blank >= gap:
                return nxt - step * (gap - 1)
            pos = nxt
        return pos
    y1 = grow(y1, -1, 0, lambda y: ink[y, x1:x2])
    y2 = grow(y2, 1, h - 1, lambda y: ink[y, x1:x2])
    x1 = grow(x1, -1, 0, lambda x: ink[y1:y2, x])
    x2 = grow(x2, 1, w - 1, lambda x: ink[y1:y2, x])
    return x1, y1, x2, y2


@dataclass
class Digitized:
    ok: bool
    reason: str = ''
    x_labels: list[str] = field(default_factory=list)
    series: list[dict] = field(default_factory=list)  # {'name', 'color', 'values': [float | None] per label, 'last', 'min', 'max'}
    step: float = 0.0  # one scale step in value units: accuracy is ±0.2 of it
    px: dict = field(default_factory=dict)  # pixel calibration on the crop: {'a', 'b'} (y = a*value + b), 'xs' of the dates

    def table(self) -> str:
        """Markdown for text/: «дата | ряд 1 | ряд 2» when the dates matched, the last point, min and max of each series."""
        if not self.ok:
            return f'Значения не определены кодом ({self.reason}).'
        fmt = lambda v: '—' if v is None else f'{v:.2f}'.rstrip('0').rstrip('.').replace('.', ',')
        out = [f'Оцифровано кодом по графику, точность ≈ ±0,2 деления шкалы (±{fmt(0.2 * self.step)}).']
        if self.x_labels:
            out += ['| дата | ' + ' | '.join(s['name'] for s in self.series) + ' |', '|' + '---|' * (len(self.series) + 1)]
            out += [f'| {x} | ' + ' | '.join(fmt(s['values'][i] if i < len(s['values']) else None) for s in self.series) + ' |' for i, x in enumerate(self.x_labels)]
        out += [f'{s["name"]}: значения не определены ({s["note"]})' if s.get('note') else
                f'{s["name"]}: последняя точка ≈ {fmt(s["last"])}' +
                (f', минимум ≈ {fmt(s["min"])}, максимум ≈ {fmt(s["max"])}' if s.get('min') is not None else '')
                for s in self.series]
        return '\n'.join(out)


def even_ticks(raw: list) -> list[float] | None:
    """Tick values as an even progression from the largest: a misread or doubled tick is corrected, nonsense is None."""
    vals = sorted({v for v in (tick_value(t) for t in raw) if v is not None}, reverse=True)
    if len(vals) < 3:
        return None
    diffs = [round(a - b, 6) for a, b in zip(vals, vals[1:])]
    step = max(set(diffs), key=diffs.count)
    best, run = [], [vals[0]]  # the longest chain with the common step: a stray tick (12,0 among 16,25…13,50) drops out
    for a, b in zip(vals, vals[1:]):
        run = run + [b] if round(a - b, 6) == step else [b]
        best = run if len(run) > len(best) else best
    return best if len(best) >= 3 and len(best) >= len(vals) / 2 else None


def grid_fit(centres: list[float], ticks: list[float], heights: list[float] | None = None, edges: list[float] | None = None):
    """The ink blocks that sit on an even grid of len(ticks) rows: try every spacing between two blocks and every tick
    for the first one, count blocks within 0.12 spacing of the grid. Needs 60% of the ticks and both the top and bottom
    one (a split or a merged label is a miss, not a wrong scale). Returns (fit, error in steps) or None."""
    n, best = len(ticks), None
    for i, ci in enumerate(centres):
        for cj in centres[i + 1:]:
            for k in range(1, n):
                d = (cj - ci) / k
                if d < 8:
                    continue
                for t in range(0, n - k):
                    grid = [ci + (q - t) * d for q in range(n)]
                    hits = {}
                    for q, g in enumerate(grid):
                        c = min(centres, key=lambda c: abs(c - g))
                        if abs(c - g) <= 0.12 * d:
                            hits[q] = c
                    if len(hits) < max(3, 0.6 * n) or 0 not in hits or n - 1 not in hits:
                        continue
                    if heights:  # tick labels are one text size: a title or a legend line on the grid is not a tick
                        hs = [heights[centres.index(c)] for c in hits.values()]
                        if max(hs[0], hs[-1]) > 1.8 * np.median(hs) or min(hs[0], hs[-1]) < 0.45 * np.median(hs):
                            continue  # the outer ticks decide the scale: no title, no speck of a date label there
                    if edges:  # tick labels are aligned to the axis: an outer «tick» off that edge is a date label under
                        # the plot or a neighbour's label above it (Бонд Бриф p4_5: the grid slid one step down onto the dates)
                        es = [edges[centres.index(c)] for c in hits.values()]
                        if max(abs(es[0] - np.median(es)), abs(es[-1] - np.median(es))) > 8:
                            continue
                    fit = np.poly1d(np.polyfit([ticks[q] for q in hits], list(hits.values()), 1))
                    err = max(abs(fit(ticks[q]) - c) for q, c in hits.items()) / d
                    key = (-len(hits), err)
                    if err <= 0.1 and (best is None or key < best[0]):
                        best = (key, fit, err)
    return (best[1], best[2]) if best else None


def y_scale(dark: np.ndarray, ticks: list[float], rows: tuple[int, int]):
    """Tick labels: a column window at the left or right edge whose ink blocks sit on an even grid (grid_fit).
    Returns (fit, (x_from, x_to)) or a reason."""
    h, w = dark.shape
    r1, r2 = rows
    best = None
    for x0 in list(range(0, w // 4, 8)) + list(range(w - 30, 3 * w // 4, -8)):
        for width in (40, 70, 110):
            a, b = (x0, x0 + width) if x0 < w / 2 else (x0 + 30 - width, x0 + 30)
            if a < 0 or b > w:
                continue
            blocks = groups(dark[r1:r2, a:b].sum(1), gap=4, min_size=4)
            centres = [r1 + (p + q) / 2 for p, q in blocks]
            cols = [np.nonzero(dark[r1 + p:r1 + q + 1, a:b].any(0))[0] for p, q in blocks]
            edges = [float(c.max() if x0 < w / 2 else c.min()) if len(c) else 0.0 for c in cols]  # the side facing the plot
            found = grid_fit(centres, ticks, [q - p + 1 for p, q in blocks], edges) if len(centres) <= 40 else None
            if found and abs(found[0](ticks[0]) - found[0](ticks[-1])) < 0.3 * h:
                found = None  # a scale shorter than 30% of the crop is text rows that happen to be even, not the axis
            if found and (best is None or found[1] < best[0]):
                best = (found[1], found[0], (a, b))
        if best and x0 > w / 2:
            break
    return (best[1], best[2]) if best else 'не найдена шкала: подписи делений не ложатся на ровную сетку'


def x_positions(dark: np.ndarray, n: int, below: int, cols: tuple[int, int]) -> list[float] | str:
    """Centres of the label columns in the first ink block under the plot (rotated labels)."""
    rows = [r for r in groups(dark[below:, cols[0]:cols[1]].sum(1), gap=4) if r[1] - r[0] >= 5]  # a speck is not text
    if not rows:
        return 'нет подписей под графиком'
    a, b = rows[0][0] + below, rows[0][1] + below
    found = groups(dark[a:b + 1, cols[0]:cols[1]].sum(0), gap=3, min_size=2)
    if len(found) != n:
        return f'подписей оси X {len(found)}, в ответе модели {n}'
    diagonal = np.median([q - p for p, q in found]) > 0.6 * (b - a)  # a label at 45° ends at its tick, a vertical one is on it
    return [cols[0] + (q if diagonal else (p + q) / 2) for p, q in found]


def legend_order(masks: dict[str, np.ndarray], dark: np.ndarray) -> list[tuple[str, int, int]]:
    """(colour, y, x) of each colour's legend sample in reading order. A sample is a thin horizontal stroke (12+ pixels
    of the colour in 30 px, a dotted line too, little of it 6 px above and below) with legend text 20-90 px to its
    right; a colour without such a sample (anti-aliased text, a band, gridlines) is not a series."""
    h, w = dark.shape
    out = []
    for name, m in masks.items():
        ys, xs = np.nonzero(m)
        for y, x in sorted(zip(ys, xs)):
            if x + 90 >= w or y < 6 or y + 7 >= h:
                continue
            seg = m[y, x:x + 30].sum()
            thin = m[y - 6, x:x + 30].sum() + m[y + 6, x:x + 30].sum()
            text = dark[y - 6:y + 7, x + 32:x + 90]
            blank_left = x >= 8 and not dark[y - 4:y + 5, x - 8:x - 1].any() and not m[y - 4:y + 5, x - 8:x - 1].any()
            clean = name == 'black' or dark[y, x:x + 30].sum() < 4  # grey next to black is the edge of a letter
            if seg >= 12 and thin < 6 and text.sum() > 15 and blank_left and clean and \
                    (name != 'black' or not dark[y - 6:y + 7, x + 28:x + 33].any()):
                out.append((name, int(y), int(x)))
                break
    return sorted(out, key=lambda t: (t[1] // 15, t[2]))


def digitize(rgb: np.ndarray, read: dict, scale: dict | None = None, words: list | None = None) -> Digitized:
    """read: the model's JSON for this crop (y_axis.ticks, x_axis.ticks, series[].name); scale: the one code found from
    Tesseract's tick labels ({'a', 'b', 'ticks'}, chart_reading.scale_from_words), used when the model gave no even
    ticks. A failure of the code is a «значения не определены» with the reason, never an exception mid-run."""
    try:
        return _digitize(rgb, read, scale, words)
    except Exception as e:  # noqa: BLE001
        return Digitized(False, f'ошибка оцифровки: {type(e).__name__}')


def near(mask: np.ndarray, r: int) -> np.ndarray:
    """Pixels within r px of the mask (a square dilation by shifts)."""
    out = mask.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= np.roll(np.roll(mask, dy, 0), dx, 1)
    return out


def grey_unreliable(m: np.ndarray, rgb: np.ndarray, plot: tuple[int, int, int, int], tick_rows: list[float]) -> str:
    """Clean a grey series mask in place and say why it cannot be trusted ('' if it can). Grey is also the colour of
    gridlines (rows at the ticks) and of the anti-aliased edges of black text (date labels inside the plot): both are
    removed; if they were a large part of the mask, or the rest is not one line across the plot, the series is skipped."""
    total = m.sum()
    if not total:
        return 'нет пикселей'
    black = (rgb.max(axis=2) < 75) & ~m
    m &= ~near(black, 2)  # grey next to black text: the edge of a letter
    width = plot[2] - plot[0]
    for t in tick_rows:  # a dashed gridline: a grey row at a tick
        for y in range(max(0, int(t) - 2), min(m.shape[0], int(t) + 3)):
            if m[y, plot[0]:plot[2]].mean() > 0.25:
                m[y] = False
    removed = 1 - m.sum() / total
    if removed > 0.35:
        return f'{removed:.0%} серых пикселей — сетка и края подписей'
    cols = m[:, plot[0]:plot[2]].any(0)
    if cols.mean() < 0.5:
        return f'серая линия покрывает {cols.mean():.0%} ширины'
    if np.median(m[:, plot[0]:plot[2]].sum(0)[cols]) > 0.06 * (plot[3] - plot[1]):
        return 'в колонках больше одной серой линии'
    return ''


MONTHS = ('янв', 'фев', 'мар', 'апр', 'ма', 'июн', 'июл', 'авг', 'сен', 'окт', 'ноя', 'дек')
MONTHS_EN = ('jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec')
MONTH_LABEL = re.compile(r'^\s*([a-zа-яё]+)\.?\s*[\'’]?(\d{2}|\d{4})\s*$', re.I)


def month_index(label: str) -> int | None:
    """«сен.23», «Сен 2023», «sep-23» -> year * 12 + month (0-based); anything else None."""
    m = MONTH_LABEL.match(str(label).replace('-', ' ').replace('_', ' '))
    if not m:
        return None
    word, year = m.group(1).lower(), int(m.group(2))
    mon = next((i for i, p in enumerate(MONTHS) if word.startswith(p)), None)
    mon = mon if mon is not None and not (mon == 4 and word.startswith('мар')) else \
        next((i for i, p in enumerate(MONTHS_EN) if word.startswith(p)), mon)
    if mon is None:
        return None
    return (year + 2000 if year < 100 else year) * 12 + mon


def month_label(index: int, like: str) -> str:
    """A month in the style of the chart's own labels («сен.23» -> «авг.26»)."""
    m = MONTH_LABEL.match(str(like).replace('-', ' ').replace('_', ' '))
    year, mon = divmod(index, 12)
    word = ('янв', 'фев', 'мар', 'апр', 'май', 'июн', 'июл', 'авг', 'сен', 'окт', 'ноя', 'дек')[mon]
    if m and m.group(1)[:1].isupper():
        word = word.capitalize()
    if m and len(m.group(2)) == 4:
        return f'{word} {year}'
    return f'{word}.{year % 100:02d}'


def axis_ticks(rgb: np.ndarray, plot: tuple[int, int, int, int], y_bottom: float) -> list[float]:
    """x of the tick marks under the x axis: the longest light row near the bottom of the scale is the axis, short
    vertical marks under it at the same x in 3+ rows are its ticks, kept only when evenly spaced."""
    ink = rgb.min(axis=2) < 235
    left, right = plot[0], plot[2]
    rows = range(max(0, int(y_bottom) - 8), min(ink.shape[0] - 8, int(y_bottom) + 9))
    if not len(rows):
        return []
    axis = max(rows, key=lambda y: ink[y, left:right].sum())
    if ink[axis, left:right].mean() < 0.6:
        return []
    marks = [{(a + b) / 2 for a, b in groups(ink[y, left:right].astype(int), gap=1) if b - a <= 4} for y in range(axis + 1, axis + 7)]
    found = sorted(marks[1]) if len(marks) > 1 else []
    found = [x for x in found if sum(any(abs(x - o) <= 2 for o in row) for row in marks) >= 3]
    while len(found) >= 3:  # a stray mark (the corner of the axes, a label stroke) sits closer than a tick spacing
        diffs = np.diff(found)
        i = int(np.argmin(diffs))
        if diffs[i] >= 0.5 * np.median(diffs):
            break
        other = lambda j: abs(found[j + 1] - found[j - 1]) if 0 < j < len(found) - 1 else np.median(diffs)
        drop = min((i, i + 1), key=lambda j: abs(other(j) - np.median(diffs)))
        found.pop(drop)
    if len(found) < 3:
        return []
    diffs = np.diff(found)
    if diffs.min() < 0.9 * np.median(diffs) or diffs.max() > 1.1 * np.median(diffs):
        return []
    return [left + x for x in found]


def monthly_points(trace_x: list, trace: list, labels: list[str], ticks: list[float], xs, pitch_min: float = 8):
    """A plain monthly line whose labels skip months (сен.23, дек.23, …): the line bends at every month, so its value at
    each month is the knot of a piecewise-linear least-squares fit to the traced line with knots one month apart. The
    month grid comes from the axis tick marks (between categories: len(labels) + 1 marks, on them: len(labels)) or from
    the label columns, the month step from the labels. Returns ([month index], [y px], rms px, month at x = c0, c0, px
    per month) or a reason."""
    months = [month_index(lb) for lb in labels]
    if len(months) < 3 or None in months:
        return 'подписи оси X не месяцы'
    k = {b - a for a, b in zip(months, months[1:])}
    if len(k) != 1 or min(k) < 1:
        return 'подписи оси X идут не через равное число месяцев'
    k = k.pop()
    if len(ticks) == len(labels) + 1:  # marks between categories: a label's first month is half a month after its mark
        pitch = float(np.median(np.diff(ticks))) / k
        c0 = ticks[0] + pitch / 2
    elif len(ticks) == len(labels):
        pitch, c0 = float(np.median(np.diff(ticks))) / k, ticks[0]
    elif not isinstance(xs, str) and len(xs) == len(labels):
        pitch, c0 = float(np.median(np.diff(xs))) / k, xs[0]
    else:
        return f'не найдена сетка месяцев: меток оси {len(ticks)}, подписей {len(labels)}'
    if pitch < pitch_min:
        return f'месяц уже {pitch_min:g} px: точки не различить'
    x0, x1 = trace_x[0], trace_x[-1]
    i0, i1 = round((x0 - c0) / pitch), round((x1 - c0) / pitch)
    if abs(x0 - (c0 + i0 * pitch)) > 0.3 * pitch or abs(x1 - (c0 + i1 * pitch)) > 0.3 * pitch or i1 - i0 < 2:
        return 'концы линии не на сетке месяцев'
    tx, ty = np.asarray(trace_x, float), np.asarray(trace, float)

    def fit(c, p):
        knots = c + p * np.arange(i0, i1 + 1)
        basis = np.clip(1 - np.abs(tx[:, None] - knots[None, :]) / p, 0, None)  # hat functions: linear between knots
        y, *_ = np.linalg.lstsq(basis, ty, rcond=None)
        return float(np.sqrt(np.mean((basis @ y - ty) ** 2))), y, (basis > 0.5).sum(0)
    # the marks give the grid to a pixel or two; the bends of the line itself place it exactly (Бонд Бриф p4: 1.7 -> 0.7 px)
    rms, c0, pitch = min((fit(c, p)[0], c, p) for c in c0 + pitch * np.linspace(-0.25, 0.25, 21)
                         for p in pitch * np.linspace(0.985, 1.015, 13))
    rms, y, seen = fit(c0, pitch)
    return [months[0] + i for i in range(i0, i1 + 1)], [float(v) if n else None for v, n in zip(y, seen)], rms, \
        months[0] - i0, c0, pitch


def _digitize(rgb: np.ndarray, read: dict, scale_words: dict | None = None, words: list | None = None) -> Digitized:
    ticks = even_ticks((read.get('y_axis') or {}).get('ticks') or [])
    if not ticks and scale_words:  # the model gave no even ticks: the ones code found by Tesseract
        ticks = list(scale_words['ticks'])
    labels = [str(t) for t in (read.get('x_axis') or {}).get('ticks') or []]
    styles_all = [str(s.get('style') or '').lower() for s in read.get('series') or []]
    if any(st and st not in LINES for st in styles_all):  # bars and areas are chart_reading's: the lines only here
        lines_only = [s for s, st in zip(read.get('series') or [], styles_all) if st in LINES]
        if not lines_only:
            return Digitized(False, f'тип ряда «{next(st for st in styles_all if st and st not in LINES)}»: линий нет')
        read = dict(read, series=lines_only)
    read_series = read.get('series') or []
    names = [s.get('name', '') for s in read_series]
    if not ticks or not names:
        return Digitized(False, 'нет ровной шкалы или рядов в ответе модели')
    styles = [str(s.get('style') or '').lower() for s in read.get('series') or []]
    said = [next((v for k, v in RU_COLORS.items() if k in str(s.get('color') or '').lower()), None) for s in read.get('series') or []]
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    dark = (R < 120) & (G < 120) & (B < 120)
    h, w = dark.shape
    scale = y_scale(dark, ticks, (0, h))
    if not isinstance(scale, str) and scale_words and len(scale_words['ticks']) >= 4 and \
            abs(scale[0][1] - scale_words['a']) > 0.2 * abs(scale_words['a']):
        # Tesseract read 4+ tick numbers in one even column and they disagree on the pixels per unit: the dark blocks
        # fitted another grid (Gold p5: grey tick labels, a window through «4%» split it and fitted half the spacing)
        scale = 'шкала по блокам не сошлась с подписями Tesseract'
    if isinstance(scale, str) and scale_words and len(scale_words['ticks']) >= 4:  # Tesseract's tick boxes instead
        ticks = list(scale_words['ticks'])
        scale = (np.poly1d([scale_words['a'], scale_words['b']]), tuple(int(round(v)) for v in scale_words.get('band') or (0, 0.12 * w)))
    if isinstance(scale, str):
        return Digitized(False, scale)
    fit, band = scale
    step = abs(ticks[0] - ticks[1])
    y_top, y_bottom = fit(ticks[0]), fit(ticks[-1])
    rows = np.zeros(h, bool)  # the band is a search window, wider than the labels: the plot starts at their ink
    for t in ticks:
        rows[max(0, int(fit(t)) - 6):int(fit(t)) + 7] = True
    ink = dark[rows, int(band[0]):int(band[1])].sum(0)
    if ink.any():
        g = max(groups(ink, gap=8), key=lambda q: ink[q[0]:q[1] + 1].sum())
        band = (band[0] + g[0], band[0] + g[1])
    left, right = (band[1] + 8, w) if band[0] < w / 2 else (0, band[0] - 8)  # tick label edges are not data
    plot = (left, max(0, int(y_top) - 3), right, min(h, int(y_bottom) + 4))  # a series never leaves the scale
    masks = {name: f(R, G, B) for name, f in COLORS.items()}
    present = {k: v for k, v in masks.items() if v[plot[1]:plot[3], plot[0]:plot[2]].sum() > 80}
    found = legend_order(present, dark)
    samples = {}
    palette_ok = (all(said) and len(set(said)) == len(said)) or len(found) == len(names)
    if words and not palette_ok:  # the palette cannot tell the series apart (shades of one colour): each series' own
        # colour from its legend stroke, light and dark grey apart
        from stage5_insight_detection.bank_analytics.chart_reading import legend_samples
        samples = legend_samples(rgb, words, names)
    if len(samples) == len(names) and len(names) > 1:
        cols = [np.array(samples[i][0], float) for i in range(len(names))]
        gaps = [min(float(np.abs(c - o).max()) for j, o in enumerate(cols) if j != i) for i, c in enumerate(cols)]
        if min(gaps) >= 20:  # the strokes differ enough to tell the series apart
            for i, c in enumerate(cols):
                tol = min(40.0, gaps[i] / 2)
                masks[f'sample{i}'] = (np.abs(rgb - c).max(axis=2) <= tol) & (rgb.min(axis=2) < 235)
            for i, c in enumerate(cols):  # a light series loses the halo of darker lines (anti-aliased edges are light)
                darker = [masks[f'sample{j}'] for j, o in enumerate(cols) if o.mean() < c.mean() - 30]
                if darker:
                    masks[f'sample{i}'] &= ~near(np.logical_or.reduce(darker), 2)
            found = [(f'sample{i}', samples[i][1], samples[i][2]) for i in range(len(names))]
            said = [None] * len(names)
    if all(said) and len(set(said)) == len(said):  # the model named each series' colour: no guessing of the order
        where = {c: (y, x) for c, y, x in found}
        found = [(c, *where.get(c, (-100, -100))) for c in said]
    if len(found) != len(names):
        return Digitized(False, f'образцов в легенде {len(found)} ({", ".join(c for c, _, _ in found)}), рядов {len(names)}')
    xs = x_positions(dark, len(labels), plot[3] + 2, (left, right)) if labels else 'нет подписей оси X'
    to_value = lambda y: (y - fit[0]) / fit[1]
    lo, hi = ticks[-1] - step / 2, ticks[0] + step / 2
    x_ticks = axis_ticks(rgb, plot, y_bottom)
    monthly, grid = {}, None  # series -> [(month index, value)] for plain monthly lines; grid: month at c0, c0, pitch
    series = []
    for (color, ly, lx), name in zip(found, names):
        if re.search(r'прав\w* (шкал|ос)', name.lower()):  # on the second axis: the left-axis calibration does not apply
            series.append({'name': name, 'color': color, 'values': [], 'last': None, 'min': None, 'max': None,
                           'note': 'вторая ось: значения не определены'})
            continue
        m = masks[color].copy()
        if ly >= 0:
            m[max(0, ly - 14):ly + 15, max(0, lx - 4):min(w, lx + 60)] = False  # the legend sample is not data
        m[:plot[1]] = m[plot[3]:] = False
        m[:, :plot[0]] = m[:, plot[2]:] = False
        m[m.sum(1) > 0.6 * (plot[2] - plot[0])] = False  # an axis or gridline in the series colour is not the series
        grey_why = grey_unreliable(m, rgb, plot, [fit(t) for t in ticks]) if color == 'grey' else ''
        if grey_why:  # a grey series is measured only when its mask separates from gridlines and text edges
            series.append({'name': name, 'color': color, 'values': [], 'last': None, 'min': None, 'max': None,
                           'note': f'серый ряд не отличить от сетки и подписей: {grey_why}'})
            continue
        cols = np.nonzero(m.any(0))[0]
        if len(cols) < 0.3 * (plot[2] - plot[0]):
            return Digitized(False, f'ряд «{name}» ({color}) не найден на графике')
        ys_of = lambda x0, x1: np.nonzero(m[:, max(0, x0):x1 + 1].any(1))[0]
        line = max(groups(m.any(0).astype(int), gap=6), key=lambda g: g[1] - g[0])
        long_runs = [g[1] for g in groups(m.any(0).astype(int), gap=2) if g[1] - g[0] >= min(0.05 * w, line[1] - line[0])]
        if not long_runs:
            series.append({'name': name, 'color': color, 'values': [], 'last': None, 'min': None, 'max': None,
                           'note': 'линия ряда не прослеживается'})
            continue
        end = max(long_runs)
        if right == w and end >= w - 3:  # the line runs into the crop's edge: the chart is cut, its end is not on it
            series.append({'name': name, 'color': color, 'values': [], 'last': None, 'min': None, 'max': None,
                           'note': 'линия упирается в край вырезки: график обрезан, конец ряда не виден'})
            continue
        line = (line[0], end)  # where the line ends: a data label printed beside it is not the line
        step_px = abs(fit[1] * step)
        start = [np.median(np.nonzero(m[:, x])[0]) for x in range(line[0], min(line[1], line[0] + 15)) if m[:, x].any()]
        trace, trace_x, prev, prev_x = [], [], float(np.median(start)) if start else None, line[0]  # one y per column
        for x in range(line[0], line[1] + 1):
            rows = np.nonzero(m[:, x])[0]
            if not len(rows):
                continue
            flag = np.zeros(h, int)
            flag[rows] = 1
            a, b = min(groups(flag, gap=8), key=lambda bl: abs((bl[0] + bl[1]) / 2 - prev)) if prev is not None else (rows.min(), rows.max())
            y = (a + b) / 2
            if prev is None or abs(y - prev) <= step_px * min(3, 0.5 + 0.1 * (x - prev_x)):  # the legend is a jump, the line is not
                trace.append(y)
                trace_x.append(x)
                prev, prev_x = y, x
        is_bar = (styles[names.index(name)] if names.index(name) < len(styles) else '') == 'столбцы'
        if not is_bar and len(trace) < 0.3 * (line[1] - line[0]) and isinstance(xs, str):
            return Digitized(False, f'ряд «{name}» ({color}) прослеживается меньше чем на 30% ширины')
        values = []
        if not isinstance(xs, str):
            spans, prev, jump = [], None, 3 * abs(fit[1] * step)  # a jump of 3+ scale steps is checked wider
            tick_rows = [fit(t) for t in ticks]

            def nearest(x, width, ref):
                rows = ys_of(int(x) - width, int(x) + width)
                if not len(rows):
                    return None
                flag = np.zeros(h, int)
                flag[rows] = 1
                blocks = groups(flag, gap=8)  # several blocks in a column: the line and e.g. the legend text
                # a speck of the zero line or a gridline between bars is not the line (Inflation_tracker p2: the weekly
                # line read 0,00 from 25.05 on, stuck to the axis left between the bars)
                blocks = [bl for bl in blocks if bl[1] - bl[0] > 3 or min(abs((bl[0] + bl[1]) / 2 - r) for r in tick_rows) > 2] or blocks
                a, b = min(blocks, key=lambda bl: abs((bl[0] + bl[1]) / 2 - ref)) if ref is not None else \
                    (min(bl[0] for bl in blocks), max(bl[1] for bl in blocks))
                rows = rows[(rows >= a) & (rows <= b)]
                return rows.min(), rows.max(), float(np.median(rows))
            for x in xs:
                sp = nearest(x, 3, prev) or nearest(x, 9, prev)
                if sp and prev is not None and abs(sp[2] - prev) > jump:
                    wide = nearest(x, 9, prev)
                    sp = wide if wide and abs(wide[2] - prev) <= jump else None  # not the line: undetermined
                spans.append(sp)
                prev = sp[2] if sp else prev
            half = np.percentile([(s[1] - s[0]) / 2 for s in spans if s], 10) if any(spans) else 0
            markers = half > 4
            mids = [None if s is None else (s[0] + s[1]) / 2 for s in spans]
            for i, sp in enumerate(spans):
                if sp is None:
                    values.append(None)
                    continue
                y = sp[2]
                if markers:  # line with markers: a peak is the marker's top + half, a trough its bottom - half
                    around = [mids[j] for j in (i - 1, i + 1) if 0 <= j < len(mids) and mids[j] is not None]
                    y = sp[0] + half if around and mids[i] < min(around) else sp[1] - half if around and mids[i] > max(around) else mids[i]
                values.append(round(float(to_value(y)), 2))
        if (styles[names.index(name)] if names.index(name) < len(styles) else '') == 'столбцы':  # a bar: its far edge from the zero line
            if isinstance(xs, str):
                return Digitized(False, f'столбцы без сопоставленных дат: {xs}')
            zero = fit(0) if ticks[-1] <= 0 <= ticks[0] else fit(ticks[-1])
            values = []
            for x in xs:
                rows = ys_of(int(x) - 2, int(x) + 2)
                values.append(None if not len(rows) else
                              round(float(to_value(rows.min() if rows.min() < zero - 1 else rows.max())), 2))
            trace = []
        known = [v for v in values if v is not None]
        traced = bool(trace) and len(trace) >= 0.3 * (line[1] - line[0])  # a dotted line breaks the trace, the dates do not
        if not known and not traced:
            return Digitized(False, f'ряд «{name}» не прослеживается')
        from_trace = traced and trace_x[-1] > (xs[-1] if known else 0)
        last = round(float(to_value(np.median(trace[-3:]))), 2) if from_trace else known[-1]
        last_x = float(trace_x[-1]) if from_trace else float(xs[max(i for i, v in enumerate(values) if v is not None)])
        pool = known + ([round(float(to_value(y)), 2) for y in (max(trace), min(trace))] if traced else [])
        vmin, vmax = min(pool), max(pool)
        if not all(lo <= v <= hi for v in [last, vmin, vmax] + [v for v in values if v is not None] if v is not None):
            return Digitized(False, f'значение ряда «{name}» вне шкалы')
        alt = None
        if (styles[names.index(name)] if names.index(name) < len(styles) else '') in LAST_ONLY:
            rows = np.nonzero(m[:, line[0]:line[1] + 1].any(1))[0]  # candidates kept for the pilot of min / max
            thick = np.median([np.ptp(np.nonzero(m[:, x])[0]) + 1 for x in range(line[0], line[1] + 1) if m[:, x].any()] or [1])
            span = max(1, line[1] - line[0])
            alt = {'trace': [vmin, vmax], 'coverage': round(len(trace_x) / span, 3) if trace_x else 0,
                   'start': round((trace_x[0] - line[0]) / span, 3) if trace_x else 1,
                   'mask': [round(float(to_value(rows.max() - thick / 2)), 2), round(float(to_value(rows.min() + thick / 2)), 2)] if len(rows) else None}
            values, vmin, vmax = [], None, None  # a plain daily line: only its last point passed the pilot
            got = monthly_points(trace_x, trace, labels, x_ticks, xs) if traced and len(trace) >= 0.8 * (line[1] - line[0]) \
                else 'линия прослеживается не на всей ширине'
            if isinstance(got, str):
                alt['monthly'] = got
            elif got[2] > 0.15 * step_px:
                alt['monthly'] = f'ломаная по месяцам не ложится на линию: {got[2]:.1f} px'
            else:
                pts = [(mo, round(float(to_value(yv)), 2)) for mo, yv in zip(got[0], got[1]) if yv is not None]
                if all(lo <= v <= hi for _, v in pts):
                    monthly[name] = pts
                    grid = got[3:]
                    alt['monthly'] = f'{len(pts)} точек, отклонение {got[2]:.1f} px'
        said_last = [tick_value(v) for v in re.findall(r'[-−]?\d[\d ]*(?:[.,]\d+)?', str(read_series[names.index(name)].get('last_point') or ''))]
        said_last = [v for v in said_last if v is not None]
        greyish = color == 'grey' or (color.startswith('sample') and float(np.ptp(samples[int(color[6:])][0])) < 30)
        if greyish and len(said_last) >= 2 and not min(said_last) - step <= last <= max(said_last) + step:
            # grey is also the colour of gridlines and date labels: the model's «между A и B» decides when they disagree
            series.append({'name': name, 'color': color, 'values': [], 'last': None, 'min': None, 'max': None,
                           'note': f'серый ряд не отличить от сетки и подписей: код {last:g}, модель {min(said_last):g}–{max(said_last):g}'})
            continue
        series.append({'name': name, 'color': color, 'values': values, 'last': last, 'min': vmin, 'max': vmax, 'last_x': last_x,
                       **({'minmax_candidates': alt} if alt else {})})
    if len({s['color'] for s in series}) < len(series):
        return Digitized(False, 'два ряда одного цвета')
    if monthly and all(s['name'] in monthly or not s['values'] for s in series):
        # plain monthly lines: a value at every month, the dates are the months themselves
        first, last = min(mo for pts in monthly.values() for mo, _ in pts), max(mo for pts in monthly.values() for mo, _ in pts)
        dates = [month_label(mo, labels[0]) for mo in range(first, last + 1)]
        for s in series:
            if s['name'] in monthly:
                at = dict(monthly[s['name']])
                s['values'] = [at.get(mo) for mo in range(first, last + 1)]
                known = [v for v in s['values'] if v is not None]
                s['min'], s['max'] = min(known), max(known)
                s['last'] = next(v for v in reversed(s['values']) if v is not None)
        px = {'a': float(fit[1]), 'b': float(fit[0]), 'xs': [float(grid[1] + (mo - grid[0]) * grid[2]) for mo in range(first, last + 1)]}
        return Digitized(True, '', dates, series, step, px)
    dated = not isinstance(xs, str) and any(s['values'] for s in series)
    px = {'a': float(fit[1]), 'b': float(fit[0]), 'xs': [float(x) for x in xs] if dated else []}
    return Digitized(True, '' if not isinstance(xs, str) else f'даты не сопоставлены: {xs}', labels if dated else [], series, step, px)


def similar(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    norm = lambda t: re.sub(r'[^а-яёa-z0-9]', '', str(t).lower())
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def fmt_num(v: float) -> str:
    return f'{v:,.2f}'.rstrip('0').rstrip('.').replace(',', ' ').replace('.', ',') if abs(v) >= 1000 else \
        f'{v:.2f}'.rstrip('0').rstrip('.').replace('.', ',')


def checked_labels(record: dict) -> tuple[list[tuple], list[tuple]]:
    """Printed labels the model read, split into confirmed and not: confirmed by the pixels (a measured line passes
    within half a scale step with the same sign, or a bar's length fits one k across the chart) or by Tesseract (it read
    the same number on the crop). Each item: (series, x, value as printed, how)."""
    from stage5_insight_detection.bank_analytics.chart_reading import number
    read, dg, code = record.get('read') or {}, record.get('digitized') or {}, record.get('code') or {}
    by_name = {s['name']: s for s in dg.get('series') or []} if dg.get('ok') else {}
    labels, step = dg.get('x_labels') or [], dg.get('step') or 0
    tess = [number(t) for t in code.get('numbers') or []]
    bar_ok = {(s['name'], str(lb['x']), lb['value']): lb['pixels'] for s in ((code.get('bars') or {}).get('series') or [])
              for lb in s['labels']}
    pie_ok = {(sl['name'], sl['label']) for sl in (code.get('pie') or {}).get('slices') or [] if sl['pixels']}
    sure, unsure, placed = [], [], set()
    if (code.get('bars') or {}).get('stacked'):  # stacked bars: the segment a label fits names its series, not the model
        for bs in code['bars']['series']:
            for lb in bs['labels']:
                if lb['pixels']:
                    sure.append((bs['name'], lb['x'], fmt_num(lb['value']), 'пиксели+Tesseract' if lb['tesseract'] else 'пиксели'))
                    placed.add((str(lb['x']), lb['value']))
    for s in read.get('series') or []:
        for pv in s.get('printed_values') or []:
            v = number(str(pv.get('value')))
            if v is None or (str(pv.get('x')), v) in placed:
                continue
            line = by_name.get(s.get('name'), {})
            at = labels.index(pv.get('x')) if pv.get('x') in labels else None
            near = line.get('values', [])[at] if at is not None and at < len(line.get('values', [])) else \
                line.get('last') if pv.get('x') and pv.get('x') in (labels[-1:] + [(read.get('x_axis') or {}).get('last')]) else None
            how = []
            if near is not None and abs(v - near) <= 0.5 * step and (v >= 0) == (near >= -0.25 * step):
                how.append('пиксели')
            if bar_ok.get((s.get('name', ''), str(pv.get('x')), v)):
                how.append('пиксели')
            if (s.get('name', ''), v) in pie_ok:
                how.append('пиксели')
            if any(t is not None and abs(t - v) < 1e-6 for t in tess):
                how.append('Tesseract')
            item = (s.get('name', ''), pv.get('x'), pv.get('value'), '+'.join(dict.fromkeys(how)))
            (sure if how else unsure).append(item)
    return sure, unsure


PERIOD_SUFFIX = re.compile(r'\d\s*[ПпFfEe]\)?\s*$|прогноз|\bАБ\b', re.I)  # «2026П», «Июл.26П», «2027F», «1К21 АБ»


def period_end(label: str) -> int | None:
    """The last month of a period on the x axis as year * 12 + month (0-based): «дек.27», «2027П», «3К26», «1П26», «11.07.2026»;
    None for anything else (a category, a sector)."""
    t = PERIOD_SUFFIX.sub(lambda m: m.group()[0] if m.group()[0].isdigit() else '', str(label).strip()).strip()
    if month_index(t) is not None:
        return month_index(t)
    if m := re.fullmatch(r'(20\d{2})\s*(?:г\.?)?', t):
        return int(m.group(1)) * 12 + 11
    if m := re.fullmatch(r'([1-4])\s*[КкQq]\s*(\d{2}|20\d{2})', t):
        return (int(m.group(2)) % 100 + 2000) * 12 + 3 * int(m.group(1)) - 1
    if m := re.fullmatch(r'([12])\s*[ПпHh]\s*(\d{2}|20\d{2})', t):
        return (int(m.group(2)) % 100 + 2000) * 12 + 6 * int(m.group(1)) - 1
    if m := re.fullmatch(r'\d{1,2}\.(\d{1,2})\.(20\d{2})', t):
        return int(m.group(2)) * 12 + int(m.group(1)) - 1
    return None


def report_date(stem: str):
    """The report's date from reports.json (search.REPORTS), None for a report not listed there."""
    from datetime import date
    path = Path(__file__).with_name('reports.json')
    try:
        reports = json.loads(path.read_text(encoding='utf-8'))['reports']
    except (OSError, ValueError, KeyError):
        return None
    row = next((r for r in reports if Path(r['file']).stem == stem and r.get('date')), None)
    return date.fromisoformat(row['date']) if row else None


def forecast_points(record: dict, points: list[tuple[str, str, str]]) -> tuple[str, list[tuple[str, str, str]]]:
    """The points of a chart that lie in the future for its report: a period ending after the report's month, or a label
    marked as a forecast («2026П», «2027F»). These are what the bank (or the market) expects, the part of a chart the text
    rarely repeats. Returns (the report date as text or '', [(series, x, value)])."""
    stem = re.sub(r'_p\d+_\d+\.png$', '', record.get('crop', ''))
    when = report_date(stem)
    now = when.year * 12 + when.month - 1 if when else None
    out = []
    for name, x, v in points:
        end = period_end(x)
        if PERIOD_SUFFIX.search(str(x)) or (now is not None and end is not None and end > now):
            out.append((name, x, v))
    return (when.strftime('%d.%m.%Y') if when else ''), out


def page_of(record: dict) -> str:
    m = re.search(r'_p(\d+)_\d+\.png$', record.get('crop', ''))
    return m.group(1) if m else '?'


def chart_points(record: dict, sure: list[tuple] | None = None) -> list[tuple[str, str, str]]:
    """Every point of a chart as (series, x, value as text): the printed labels confirmed first, then what the code
    measured on the lines and bars («≈ N»)."""
    dg, bars = record.get('digitized') or {}, (record.get('code') or {}).get('bars') or {}
    sure = checked_labels(record)[0] if sure is None else sure
    points = [(n, str(xv), str(v)) for n, xv, v, _ in sure]
    seen = {(n, x) for n, x, _ in points}
    for s in (dg.get('series') or []) if dg.get('ok') else []:
        points += [(s['name'], x, '≈ ' + fmt_num(v)) for x, v in zip(dg.get('x_labels') or [], s.get('values') or [])
                   if v is not None and (s['name'], x) not in seen]
    for s in bars.get('series') or []:
        points += [(s['name'], str(c), '≈ ' + fmt_num(v)) for c, v in zip(bars.get('categories') or [], s.get('values') or [])
                   if v is not None and (s['name'], str(c)) not in seen and not any(str(c) == x for _, x, _ in points)]
    return points


def describe(record: dict) -> str:
    """The chart in words, written by code from what was confirmed or measured: what it shows (title, units, period,
    source), each series' last point, direction, minimum and maximum with dates, bars by size. A number the code only
    measured keeps its «≈»; a number nothing confirmed is not used («см. PDF, стр. N»)."""
    read, dg, code = record.get('read') or {}, record.get('digitized') or {}, record.get('code') or {}
    title = (read.get('title') or record.get('title') or 'график').strip()
    unit = (read.get('y_axis') or {}).get('unit') or (title.rsplit(',', 1)[1].strip() if ',' in title else '')
    x = read.get('x_axis') or {}
    first, last = (dg['x_labels'][0], dg['x_labels'][-1]) if dg.get('ok') and dg.get('x_labels') else (x.get('first'), x.get('last'))
    period = f', период {first} – {last}' if first and last else ''  # the measured dates: a line may go past the last label
    source = f'; источник: {read["source"].strip()}' if str(read.get('source') or '').strip() else ''
    parts = [f'График «{title}»{f" ({unit})" if unit and unit not in title else ""}{period}{source}.']
    sure, _ = checked_labels(record)
    by_series: dict[str, list[tuple]] = {}
    for name, xv, val, how in sure:
        by_series.setdefault(name, []).append((xv, val))
    said = set()
    for s in (dg.get('series') or []) if dg.get('ok') else []:
        if s.get('note') or s.get('last') is None:
            continue
        said.add(s['name'])
        dated = [(dg['x_labels'][i], v) for i, v in enumerate(s.get('values') or []) if v is not None and i < len(dg.get('x_labels') or [])]
        text = f'{s["name"] or "ряд"}: последняя точка ≈ {fmt_num(s["last"])}'
        if dated:
            first, last = dated[0], dated[-1]
            lo, hi = min(dated, key=lambda d: d[1]), max(dated, key=lambda d: d[1])
            trend = 'рост' if last[1] > first[1] + 0.5 * (dg.get('step') or 0) else 'снижение' if last[1] < first[1] - 0.5 * (dg.get('step') or 0) else 'без большого изменения'
            text += (f' ({last[0]}); за период {trend}: с ≈ {fmt_num(first[1])} ({first[0]}); минимум ≈ {fmt_num(lo[1])} ({lo[0]}), '
                     f'максимум ≈ {fmt_num(hi[1])} ({hi[0]})')
        elif s.get('min') is not None:
            text += f'; минимум ≈ {fmt_num(s["min"])}, максимум ≈ {fmt_num(s["max"])}'
        parts.append(text + '.')
    pie = code.get('pie') or {}
    if pie.get('slices'):
        shares = [f'{sl["name"]} {fmt_num(sl["label"]) if sl["pixels"] else "≈ " + fmt_num(sl["share"])}%' for sl in pie['slices']]
        parts.append('Доли по убыванию (круговая диаграмма): ' + ', '.join(shares) + '.')
        said |= {sl['name'] for sl in pie['slices']}
    bars = code.get('bars') or {}
    for s in bars.get('series') or []:
        cats = bars.get('categories') or []
        printed = [(lb['x'], lb['value']) for lb in s['labels'] if lb['pixels'] or lb['tesseract']]
        measured = [(cats[i], v) for i, v in enumerate(s.get('values') or []) if v is not None and i < len(cats)]
        values = printed or [(c, v) for c, v in measured]
        if not values or s['name'] in said:
            continue
        said.add(s['name'])
        mark = '' if printed else '≈ '
        ranked = sorted(values, key=lambda cv: -cv[1])
        text = f'{s["name"] or "столбцы"}: последний столбец ({values[-1][0]}) {mark}{fmt_num(values[-1][1])}'
        if len(values) > 2:
            text += (f'; больше всего — {ranked[0][0]} ({mark}{fmt_num(ranked[0][1])}), меньше всего — {ranked[-1][0]} '
                     f'({mark}{fmt_num(ranked[-1][1])})')
        if 3 <= len(values) <= 12:
            text += '; по убыванию: ' + ', '.join(f'{c} {mark}{fmt_num(v)}' for c, v in ranked)
        parts.append(text + '.')
    for name, vals in by_series.items():  # printed labels confirmed for a series nothing else described
        if name not in said:
            parts.append(f'{name or "подписи"}: ' + '; '.join(f'{xv}: {v}' for xv, v in vals) + '.')
            said.add(name)
    if len(parts) == 1:
        parts.append(f'Значения на графике не подтверждены: см. PDF, стр. {page_of(record)}.')
    when, ahead = forecast_points(record, chart_points(record, sure))
    if ahead:
        parts.append(f'ПРОГНОЗ (периоды позже даты отчёта{" " + when if when else ""} или помеченные как прогноз): ' +
                     '; '.join(f'{n + " — " if n else ""}{x}: {v}' for n, x, v in ahead[:24]) + '.')
    return ' '.join(parts)


def chart_note(record: dict) -> list[str]:
    """Lines for a chart block from one record of charts.py: the code's description of the chart, the printed labels
    confirmed (pixels, Tesseract) and not, the bars and lines the code measured (approx)."""
    dg, code = record.get('digitized') or {}, record.get('code') or {}
    sure, unsure = checked_labels(record)
    out = ['Описание графика (составлено кодом): ' + describe(record)]
    if sure:
        out.append('Подписи на графике (сверены: ' + ', '.join(sorted({h for *_, h in sure})) + '): ' +
                   '; '.join(f'{n} — {x}: {v}' for n, x, v, _ in sure))
    if unsure:
        out.append('Подписи на графике (прочитаны моделью, не подтверждены пикселями): ' + '; '.join(f'{n} — {x}: {v}' for n, x, v, _ in unsure))
    bars = code.get('bars') or {}
    if bars.get('scale') and any(any(v is not None for v in s['values']) for s in bars.get('series') or []):
        cats = bars.get('categories') or []
        out.append('Оцифровано кодом по графику (столбцы), точность ≈ ±0,2 деления шкалы.')
        out.append('| категория | ' + ' | '.join(s['name'] or 'столбцы' for s in bars['series']) + ' |')
        out.append('|' + '---|' * (len(bars['series']) + 1))
        for i, c in enumerate(cats):
            out.append(f'| {c} | ' + ' | '.join('—' if i >= len(s['values']) or s['values'][i] is None else fmt_num(s['values'][i])
                                               for s in bars['series']) + ' |')
    if dg and not (code.get('pie') or {}).get('slices'):  # a pie measured by angle: no «values not determined» line
        out += Digitized(**dg).table().split('\n')
    return out


PHANTOM = '⚠ график не найден на странице, вероятно ошибка распознавания'


def annotate(text: str, records: list[dict], grounded: bool = False) -> str:
    """Add chart_note lines to the [График] blocks of a page, a record to the block with the most similar title. On a page
    where grounding looked for charts (`grounded`), a block of the page OCR that matches none of the charts it found is
    marked PHANTOM: the whole-page OCR invents chart blocks on pages of text (22 of 23 such blocks in the nine reports),
    so the library leaves it out and text/ keeps it marked."""
    free = list(records)

    def add(m):
        block = m.group()
        title = block.split('\n', 1)[0].replace('[График]', '').strip()
        best = max(free, key=lambda r: similar(r.get('title') or (r.get('read') or {}).get('title', ''), title), default=None)
        if best is None or similar(best.get('title') or '', title) < 0.6:
            if grounded:
                head, _, rest = block.partition('\n')
                return f'{head}\n{PHANTOM}\n{rest}'
            return block
        free.remove(best)
        return block.replace('[/График]', '\n'.join(chart_note(best)) + '\n[/График]')
    text = re.sub(r'\[График\].*?\[/График\]', add, text, flags=re.S)
    for r in free:  # the page OCR wrote no block for it (a table of dashes, a loop): the crop reading is the block
        note = chart_note(r)
        if note:
            text += f'\n\n[График] {r.get("title") or (r.get("read") or {}).get("title", "")}\n' + '\n'.join(note) + '\n[/График]'
    return text


def page_grounded(stem: str, page: int, folder: Path | None = None) -> bool:
    """charts.py looked for the charts of this page (its boxes file exists, even with no chart in it)."""
    from common.data import DATA
    return ((folder or DATA / 'analytics' / 'charts') / f'{stem}_p{page}.boxes.json').exists()


def page_records(stem: str, page: int, folder: Path | None = None) -> list[dict]:
    """What charts.py stored for a page (the cheap model's reading of the crops plus the code's values), [] if nothing."""
    from common.data import DATA
    folder = folder or DATA / 'analytics' / 'charts'
    for model in ('gemma', 'qwen'):
        path = folder / f'{stem}_p{page}.{model}.json'
        if path.exists():
            return json.loads(path.read_text(encoding='utf-8'))
    return []


if __name__ == '__main__':
    crop, answer = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
    read = next((r['read'] for r in answer if r['crop'] == crop.name), answer) if isinstance(answer, list) else answer
    print(digitize(to_rgb(crop.read_bytes()), read).table())
