"""Charts read by code, no model: Tesseract words, a scale from the tick labels, bars by their shape, labels checked.

digitize.py measures lines; this module adds what the pilot left out, with nothing tuned to the nine reports:
- chart_words: Tesseract (rus+eng) on the crop scaled to ~30 px digits, every word with its box; the numbers are
  normalised as in tables.py. A second reading of what is printed: a label the model read and Tesseract also read
  is confirmed (`verified`).
- scale_from_words: tick labels found by code (numbers stacked in one column at the left or right edge, evenly
  spaced in value and in pixels) and a pixel -> value line fitted to them: no ticks from the model needed.
- find_bars: bars are solid rectangles (they survive an erosion that removes text strokes) standing on a common
  baseline, stacked segments on the segment below; legend swatches stand on nothing. Categories are the groups of
  bars split by the wider gaps, series the order of bars in a group.
- bar_values: with a scale, the far edge of each bar (a stacked segment: its own height); without one, only the
  lengths, which are enough to check printed labels: value = k * length for one k across the chart (±3 %).
python -X utf8 stage5_insight_detection/bank_analytics/chart_reading.py <crop.png>
"""
import re, sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stage5_insight_detection.bank_analytics.digitize import groups, to_rgb
from stage5_insight_detection.bank_analytics.tables import norm

NUMBER = re.compile(r'^[-+]?\d+(?:[ ,.]\d{3})*(?:,\d+)?%?x?$')


def number(text: str) -> float | None:
    """A printed number as a float («1 528», «-0,8%», «12,9%», «1,5x»); None for anything else."""
    s = norm(text).rstrip('%x*').replace('−', '-')
    if not re.fullmatch(r'[-+]?\d+(,\d+)?', s):
        return None
    return float(s.replace(',', '.'))


@dataclass
class Word:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def value(self) -> float | None:
        return number(self.text)


def chart_words(rgb: np.ndarray) -> list[Word]:
    """Tesseract words of a chart crop with boxes in crop pixels. The crop is scaled so that a line of text is ~40 px
    (digits ~30 px), turned grey and contrast-stretched; sparse text mode, as labels are scattered over the plot."""
    from common.ocr_local import word_data
    from PIL import Image, ImageOps
    gray = rgb.mean(axis=2)
    ink = gray < 160
    lines = groups(ink.sum(1) > 0, gap=1)
    heights = [b - a + 1 for a, b in lines if 6 <= b - a + 1 <= 80]
    k = max(1.0, min(4.0, 40 / np.median(heights))) if heights else 2.0
    img = ImageOps.autocontrast(Image.fromarray(gray.astype(np.uint8)).resize((int(rgb.shape[1] * k), int(rgb.shape[0] * k)),
                                                                                 Image.LANCZOS))
    data = word_data(img, psm=11)
    return [Word(data['text'][i].strip(), data['left'][i] / k, data['top'][i] / k, (data['left'][i] + data['width'][i]) / k,
                 (data['top'][i] + data['height'][i]) / k)
            for i in range(len(data['text'])) if data['text'][i].strip() and float(data['conf'][i]) > 0]


def scale_from_words(words: list[Word], shape: tuple[int, int]):
    """Tick labels by code: at least 4 numbers whose boxes line up in one column (right edges within 2 % of the width),
    evenly spaced in value (one step) and lying on a line in pixels (every tick within 0.1 step). Returns
    (fit y = a*value + b as np.poly1d, ticks from top to bottom, (x0, x1) of the column) or None."""
    h, w = shape
    nums = [wd for wd in words if wd.value is not None and abs(wd.y1 - wd.y0) < 0.08 * h]
    best = None
    for anchor in nums:
        col = [wd for wd in nums if abs(wd.x1 - anchor.x1) <= 0.02 * w]
        col = sorted({round(wd.cy): wd for wd in col}.values(), key=lambda wd: wd.cy)
        if len(col) < 4:
            continue
        vals = [wd.value for wd in col]
        steps = [round(a - b, 6) for a, b in zip(vals, vals[1:])]
        step = max(set(steps), key=steps.count)
        if step <= 0 or steps.count(step) < len(steps) - 1:
            continue  # values must fall evenly from the top: one misread tick at most
        run = [col[0]]
        for prev, cur in zip(col, col[1:]):
            if round(prev.value - cur.value, 6) == step:
                run.append(cur)
        if len(run) < 4:
            continue
        fit = np.poly1d(np.polyfit([wd.value for wd in run], [wd.cy for wd in run], 1))
        step_px = abs(fit[1] * step)
        err = max(abs(fit(wd.value) - wd.cy) for wd in run) / step_px if step_px else 1
        if err <= 0.1 and step_px >= 8 and (best is None or len(run) > len(best[1])):
            best = (fit, [wd.value for wd in run], (min(wd.x0 for wd in run), max(wd.x1 for wd in run)))
    return best


@dataclass
class Bar:
    x0: int
    x1: int
    y0: int  # top
    y1: int  # bottom
    rgb: tuple[int, int, int]
    stacked_on: int | None = None  # index of the segment below in a stack

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def width(self) -> int:
        return self.x1 - self.x0 + 1

    @property
    def height(self) -> int:
        return self.y1 - self.y0 + 1


def find_bars(rgb: np.ndarray, words: list[Word] | None = None) -> tuple[list[Bar], int | None]:
    """Bars of a chart crop and the baseline row. Solid rectangles of one colour (colours quantised, so two bars that
    touch are two bars) survive an opening by a 5x5 square, text strokes do not. The baseline is the edge the most bar
    area stands on (a count would let small blocks of bold text win); a bar stands on it, a stacked segment on the
    segment below; a legend swatch stands on nothing."""
    from scipy import ndimage
    smooth = np.stack([ndimage.median_filter(rgb[..., c], size=3) for c in range(3)], axis=-1)  # JPEG noise off the fills
    gray = smooth.mean(axis=2)
    paint = (smooth.min(axis=2) < 232) & (gray > 25)  # coloured or grey fill, not white, not black text
    q = (smooth // 32).astype(np.int32)
    key = q[..., 0] * 10000 + q[..., 1] * 100 + q[..., 2]
    keys, counts = np.unique(key[paint], return_counts=True)
    boxes = []
    for k in keys[counts >= 200]:  # a palette colour: every pixel within 36 of it (one bar never splits at a bucket edge)
        centre = smooth[paint & (key == k)].mean(axis=0)
        same = paint & (np.abs(smooth - centre).max(axis=2) <= 36)
        solid = ndimage.binary_opening(same, structure=np.ones((5, 5)))
        labels, _ = ndimage.label(solid)
        for i, sl in enumerate(ndimage.find_objects(labels), 1):
            if sl is None:
                continue
            ys, xs = sl
            part = labels[sl] == i
            h, w = part.shape
            if w < 6 or h < 2 or part.mean() < 0.8 or w > 0.4 * rgb.shape[1]:
                continue
            colour = tuple(int(c) for c in np.median(rgb[sl][part], axis=0))
            boxes.append(Bar(xs.start, xs.stop - 1, ys.start, ys.stop - 1, colour))
    unique: list[Bar] = []
    for b in sorted(boxes, key=lambda b: -b.width * b.height):  # neighbouring palette colours find the same bar twice
        if not any(min(b.x1, u.x1) - max(b.x0, u.x0) > 0.8 * b.width and min(b.y1, u.y1) - max(b.y0, u.y0) > 0.8 * b.height
                   for u in unique):
            unique.append(b)
    boxes = unique
    if not boxes:
        return [], None
    area: dict[int, float] = {}
    for b in boxes:  # a bar's bottom (positive) or top (negative) edge is a baseline candidate, weighted by its area
        for e in (b.y1, b.y0):
            area[e // 3] = area.get(e // 3, 0) + b.width * b.height
    base = max(area, key=area.get) * 3 + 1
    on_base = [b for b in boxes if abs(b.y1 - base) <= 4 or abs(b.y0 - base) <= 4]
    if len(on_base) < 2:
        return [], None
    out = list(on_base)
    added = True
    while added:  # stacked segments: right above (or below, for negatives) a bar already taken, same x span
        added = False
        for b in boxes:
            if b in out:
                continue
            for j, o in enumerate(out):
                if abs(b.x0 - o.x0) <= 3 and abs(b.x1 - o.x1) <= 3 and (0 <= o.y0 - b.y1 <= 4 or 0 <= b.y0 - o.y1 <= 4):
                    b.stacked_on = j
                    out.append(b)
                    added = True
                    break
    return sorted(out, key=lambda b: (b.cx, -b.y1)), base


def categories(bars: list[Bar], words: list[Word] | None = None, base: int | None = None) -> list[list[Bar]]:
    """Bars on the baseline grouped into categories: by the category labels under the baseline when there are any (a bar
    goes to the nearest label), else by the wider gaps between neighbours (grouped bars stand close)."""
    ground = sorted([b for b in bars if b.stacked_on is None], key=lambda b: b.cx)
    if not ground:
        return []
    if words and base is not None:
        bottom = max(b.y1 for b in ground)
        below = [w for w in words if w.y0 >= bottom - 2 and w.y0 <= bottom + 0.25 * (bottom - min(b.y0 for b in ground) + 1)]
        row = sorted(below, key=lambda w: w.cx)
        if 2 <= len(row) <= len(ground) and all(min(abs(b.cx - w.cx) for w in row) < 1.5 * max(b.width for b in ground) for b in ground):
            out: dict[int, list[Bar]] = {}
            for b in ground:
                i = min(range(len(row)), key=lambda i: abs(row[i].cx - b.cx))
                out.setdefault(i, []).append(b)
            return [out[i] for i in sorted(out)]
    gaps = [n.x0 - p.x1 for p, n in zip(ground, ground[1:])]
    if not gaps:
        return [ground]
    small, large = min(gaps), max(gaps)
    cut = (small + large) / 2 if large > 2 * max(small, 1) else -1  # gaps alike: every bar is a category of its own
    out_list, cur = [], [ground[0]]
    for g, b in zip(gaps, ground[1:]):
        if g > cut:
            out_list.append(cur)
            cur = []
        cur.append(b)
    return out_list + [cur]


def labels_fit(pairs: list[tuple[float, float]], tolerance: float = 0.03) -> list[bool]:
    """Printed values against bar lengths in pixels (signed): one k for the whole chart, value ≈ k * length within
    `tolerance` of the largest value. A label that fits is confirmed by the pixels."""
    usable = [(v, ln) for v, ln in pairs if ln]
    if len(usable) < 3:
        return [False] * len(pairs)
    k = float(np.median([v / ln for v, ln in usable]))
    top = max(abs(v) for v, _ in pairs) or 1
    return [ln != 0 and abs(v - k * ln) <= tolerance * top for v, ln in pairs]


def bar_series(rgb: np.ndarray, read: dict, words: list[Word] | None = None) -> dict | None:
    """Bars measured and the printed labels checked. Returns {'categories': [...], 'series': [{'name', 'values' (by the
    scale, None without one), 'labels': [{'x', 'value', 'pixels', 'tesseract'}]}], 'scale': bool} or None if no bars.
    A label is confirmed by the pixels (its value fits the bar's length, or the scale, for one k across the chart) and/or
    by Tesseract (it read the same number on the crop)."""
    words = words if words is not None else chart_words(rgb)
    if any('накоп' in str(s.get('style') or '') for s in read.get('series') or []):
        return stacked_series(rgb, read, words)
    bars, base = find_bars(rgb, words)
    if not bars or base is None:
        return None
    cats = categories(bars, words, base)
    series_read = [s for s in read.get('series') or [] if 'столб' in str(s.get('style') or '')] or (read.get('series') or [])[:1]
    n = max(1, len(series_read))
    scale = any_scale(rgb, read, words)
    to_value = (lambda y: (y - scale[0][0]) / scale[0][1]) if scale else None
    labels_x = [str(t) for t in (read.get('x_axis') or {}).get('ticks') or []]
    names = [c if len(labels_x) == len(cats) else None for c in labels_x] if len(labels_x) == len(cats) else \
        [None] * len(cats)
    out = {'categories': [nm or f'{i + 1}' for i, nm in enumerate(names)], 'series': [], 'scale': bool(scale)}
    per = [[] for _ in range(n)]
    full = [sorted(c, key=lambda b: b.cx) for c in cats if len(c) == n]
    tone = [np.median([c[i].rgb for c in full], axis=0) if full else None for i in range(n)]  # each series' colour
    for cat in cats:
        ground = sorted(cat, key=lambda b: b.cx)
        if n == 1 and len(ground) > 1:
            ground = [max(ground, key=lambda b: b.height)]  # one series: a stray sliver is not a second bar
        if len(ground) == n:
            for i in range(n):
                per[i].append(ground[i])
            continue
        slots = [None] * n  # fewer bars than series (a year with no data yet): each bar to the series of its colour
        for b in ground:
            if all(t is not None for t in tone):
                i = min(range(n), key=lambda i: float(np.abs(np.array(b.rgb) - tone[i]).sum()))
                if slots[i] is None and float(np.abs(np.array(b.rgb) - tone[i]).max()) <= 40:
                    slots[i] = b
        for i in range(n):
            per[i].append(slots[i])
    tess_values = [w.value for w in words if w.value is not None]
    for i, s in enumerate(series_read or [{'name': ''}]):
        lengths = [None if b is None else (base - b.y0 if b.y1 >= base - 4 else -(b.y1 - base)) for b in per[i]]
        values = [None if b is None or not to_value else round(float(to_value(b.y0 if b.y1 >= base - 4 else b.y1) - to_value(base)), 2)
                  for b in per[i]]
        printed = [(p.get('x'), number(str(p.get('value')))) for p in s.get('printed_values') or []]
        printed = [(x, v) for x, v in printed if v is not None]
        pairs = [(v, lengths[j]) for j, (x, v) in enumerate(printed) if j < len(lengths) and lengths[j] is not None]
        fits = labels_fit(pairs) if len(printed) == len([ln for ln in lengths if ln is not None]) else [False] * len(pairs)
        labels = []
        for j, (x, v) in enumerate(printed):
            labels.append({'x': x, 'value': v, 'pixels': bool(fits[j]) if j < len(fits) else False,
                           'tesseract': any(abs(v - t) < 1e-6 for t in tess_values)})
        out['series'].append({'name': s.get('name', ''), 'values': values, 'labels': labels})
    return out


def swatches(rgb: np.ndarray) -> list[Bar]:
    """Legend swatches: small solid squares of one colour (6-40 px, about as wide as high)."""
    from scipy import ndimage
    smooth = np.stack([ndimage.median_filter(rgb[..., c], size=3) for c in range(3)], axis=-1)
    paint = (smooth.min(axis=2) < 232) & (smooth.mean(axis=2) > 25)
    labels, _ = ndimage.label(ndimage.binary_opening(paint, structure=np.ones((3, 3))))
    out = []
    for i, sl in enumerate(ndimage.find_objects(labels), 1):
        if sl is None:
            continue
        ys, xs = sl
        h, w = ys.stop - ys.start, xs.stop - xs.start
        part = labels[sl] == i
        if 6 <= w <= 40 and 6 <= h <= 40 and 0.5 <= w / h <= 2.5 and part.mean() > 0.85:
            out.append(Bar(xs.start, xs.stop - 1, ys.start, ys.stop - 1, tuple(int(c) for c in np.median(rgb[sl][part], axis=0))))
    return out


def legend_map(rgb: np.ndarray, words: list[Word], known: list[str]) -> list[tuple[tuple, str]]:
    """(swatch colour, series name) of the legend: the words right of each swatch up to the next one on its row, matched
    to the model's series names one to one, the most similar pairs first (Tesseract spells Russian with Latin
    look-alikes, so both spellings are compared). A swatch whose words match no name (similarity < 0.4) gets none."""
    from stage5_insight_detection.bank_analytics.digitize import similar
    cyr = str.maketrans('ABEHKMOPCTXaeopcyx3', 'АВЕНКМОРСТХаеорсух' + 'з')
    sim = lambda k, said: max(similar(k, said), similar(k.translate(cyr), said.translate(cyr)))
    every = swatches(rgb)
    texts = []
    for sw in every:
        cy = (sw.y0 + sw.y1) / 2
        nxt = min([o.x0 for o in every if o.x0 > sw.x1 and abs((o.y0 + o.y1) / 2 - cy) <= max(6, sw.height)], default=1e9)
        row = sorted([w for w in words if sw.x0 - 2 < w.x0 < nxt and abs(w.cy - cy) <= max(6, sw.height)], key=lambda w: w.x0)
        text, last = [], sw.x1
        for w in row:
            if w.x0 - last > 4 * max(sw.height, 8):
                break
            # Tesseract can read the swatch as a glyph glued to the first word («#Отозван»): that glyph is not the name
            t = re.sub(r'^\W+', '', w.text) if w.x0 < sw.x1 - 2 else w.text
            if t:
                text.append(t)
            last = max(last, w.x1)
        if text:
            texts.append((sw, ' '.join(text)))
    pairs = sorted(((sim(k, t), i, j) for i, (sw, t) in enumerate(texts) for j, k in enumerate(known)), reverse=True)
    used_s, used_k, out = set(), set(), []
    for score, i, j in pairs:
        if score < 0.4 or i in used_s or j in used_k:
            continue
        used_s.add(i)
        used_k.add(j)
        out.append((texts[i][0].rgb, known[j]))
    return out


def legend_names(colours: list, legend: list[tuple[tuple, str]], limit: float = 75) -> list[str | None]:
    """Each colour's series from the legend, one to one, the closest colour pairs first (a small swatch is blurred, so the
    limit is loose; one to one keeps a grey bar from taking the name of a greyish-green swatch a green bar is nearer to)."""
    pairs = sorted((float(np.abs(np.array(c, float) - np.array(sw, float)).max()), i, j)
                   for i, c in enumerate(colours) for j, (sw, _) in enumerate(legend))
    out: list[str | None] = [None] * len(colours)
    used = set()
    for d, i, j in pairs:
        if d <= limit and out[i] is None and j not in used:
            out[i] = legend[j][1]
            used.add(j)
    return out


def stroke_left_of(rgb: np.ndarray, w: Word):
    """The coloured stroke right before a word (a legend sample), read leftwards; a gap up to 8 px is a dashed one.
    (colour, y, x) or None."""
    y = int(w.cy)
    band = rgb[max(0, y - 3):y + 4]
    coloured = band.min(axis=2) < 225
    x, cols, gap = int(w.x0) - 3, [], 0
    while x > max(0, int(w.x0) - 90):
        if coloured[:, x].any():
            cols.append(x)
            gap = 0
        elif cols:
            gap += 1
            if gap > 8:
                break
        x -= 1
    if len(cols) < 6:
        return None
    px = band[:, cols][coloured[:, cols]]
    return tuple(int(c) for c in np.median(px, axis=0)), y, int(np.mean(cols))


def legend_samples(rgb: np.ndarray, words: list[Word], names: list[str]) -> dict[int, tuple[tuple, int, int]]:
    """The colour of each series from its legend entry: the words that spell its name (both spellings, ≥ 0.5) with a
    stroke right before them; among several (a year is also an axis label), the one on the row most series' entries share.
    Returns {series index: (colour, y, x of the stroke)}."""
    from stage5_insight_detection.bank_analytics.digitize import similar
    cyr = str.maketrans('ABEHKMOPCTXaeopcyx3', 'АВЕНКМОРСТХаеорсух' + 'з')
    rows = sorted(words, key=lambda w: (round(w.cy / 8), w.x0))
    cands: dict[int, list] = {}
    for i, name in enumerate(names):
        if not name.strip():
            continue
        for k, w in enumerate(rows):
            run = [w] + [o for o in rows[k + 1:k + 7] if abs(o.cy - w.cy) <= 6]
            sc = max(max(similar(name, ' '.join(o.text for o in run[:n])),
                         similar(name.translate(cyr), ' '.join(o.text for o in run[:n]).translate(cyr))) for n in range(1, len(run) + 1))
            if sc >= 0.5:
                stroke = stroke_left_of(rgb, w)
                if stroke:
                    cands.setdefault(i, []).append((sc, stroke))
    row_votes: dict[int, int] = {}
    for lst in cands.values():
        for yrow in {round(st[1] / 10) for _, st in lst}:
            row_votes[yrow] = row_votes.get(yrow, 0) + 1
    out = {}
    for i, lst in cands.items():
        out[i] = max(lst, key=lambda c: (row_votes.get(round(c[1][1] / 10), 0), c[0]))[1]
    return out


def any_scale(rgb: np.ndarray, read: dict, words: list[Word]):
    """The scale from Tesseract's tick labels, else from the model's ticks found on the pixels (digitize.y_scale)."""
    from stage5_insight_detection.bank_analytics.digitize import even_ticks, y_scale
    sc = scale_from_words(words, rgb.shape[:2])
    if sc:
        return sc
    ticks = even_ticks((read.get('y_axis') or {}).get('ticks') or [])
    if not ticks:
        return None
    dark = rgb.max(axis=2) < 120
    found = y_scale(dark, ticks, (0, rgb.shape[0]))
    return (found[0], ticks, found[1]) if not isinstance(found, str) else None


def stacked_series(rgb: np.ndarray, read: dict, words: list[Word]) -> dict | None:
    """Stacked bars: each column cut into its segments bottom to top; segment colours clustered and the clusters ordered
    by their usual place in the stack, which is the legend order the model gives the series in. A printed label is
    matched to the segment of its category whose height it fits (one value-per-pixel for the chart, from the scale or
    from the labels themselves), so it is confirmed with the right series even when the model put it under another."""
    bars, base = find_bars(rgb, words)
    if not bars or base is None:
        return None
    columns: list[list[Bar]] = []
    for b in sorted(bars, key=lambda b: b.cx):
        col = next((c for c in columns if abs(c[0].x0 - b.x0) <= 3 and abs(c[0].x1 - b.x1) <= 3), None)
        (col.append(b) if col is not None else columns.append([b]))
    columns = [sorted(c, key=lambda b: -b.y1) for c in columns if any(abs(b.y1 - base) <= 4 for b in c)]  # bottom first
    if len(columns) < 2:
        return None
    clusters: list[np.ndarray] = []
    seg_cluster = {}
    for c in columns:
        for b in c:
            i = next((k for k, cc in enumerate(clusters) if float(np.abs(np.array(b.rgb) - cc).max()) <= 40), None)
            if i is None:
                clusters.append(np.array(b.rgb, float))
                i = len(clusters) - 1
            seg_cluster[id(b)] = i
    rank = {i: np.mean([pos for c in columns for pos, b in enumerate(c) if seg_cluster[id(b)] == i]) for i in range(len(clusters))}
    order = sorted(rank, key=rank.get)  # cluster ids from the bottom of the stacks up
    series_read = read.get('series') or []
    known = [srs.get('name', '') for srs in series_read]
    legend = legend_map(rgb, words, known)
    by_legend = legend_names([clusters[i] for i in order], legend)  # a name from the legend swatch of that colour
    rest = [srs.get('name', '') for srs in series_read if srs.get('name', '') not in by_legend]  # names no swatch took, in order
    names = [by_legend[k] or (rest.pop(0) if rest else f'ряд {k + 1}') for k in range(len(order))]
    if len(set(names)) < len(names):  # two colours took one name: trust the order instead
        names = [series_read[k].get('name', '') if k < len(series_read) else f'ряд {k + 1}' for k in range(len(order))]
    labels_x = [str(t) for t in (read.get('x_axis') or {}).get('ticks') or []]
    cats = labels_x if len(labels_x) == len(columns) else [str(i + 1) for i in range(len(columns))]
    scale = any_scale(rgb, read, words)
    per_px = 1 / abs(scale[0][1]) if scale else None
    printed = [(str(p.get('x')), number(str(p.get('value')))) for srs in series_read for p in srs.get('printed_values') or []]
    printed = [(x, v) for x, v in printed if v is not None and x in cats]
    all_r = sorted(v / b.height for x, v in printed for b in columns[cats.index(x)] if b.height > 3 and v > 0)
    support = lambda r: sum(abs(o / r - 1) <= 0.03 for o in all_r)
    by_labels = max(all_r, key=support) if all_r else None  # value per pixel most label-segment pairs share
    if by_labels and support(by_labels) < 3:
        by_labels = None
    if per_px and by_labels and abs(per_px / by_labels - 1) > 0.1:  # the scale disagrees with the printed labels: a wrong
        scale, per_px = None, None                                  # grid (numbers inside the bars look like ticks too)
    if per_px is None:
        per_px = by_labels
    series = [{'name': nm, 'values': [None] * len(columns), 'labels': []} for nm in names]
    for ci, c in enumerate(columns):
        for b in c:
            k = order.index(seg_cluster[id(b)])
            if per_px and k < len(series):
                series[k]['values'][ci] = round(float(b.height * per_px), 2) if scale else None
    tess_values = [w.value for w in words if w.value is not None]
    top = max((abs(v) for _, v in printed), default=1) or 1
    for x, v in printed:
        segs = columns[cats.index(x)]
        b = min(segs, key=lambda b: abs(v - b.height * per_px)) if per_px else None
        fits = b is not None and abs(v - b.height * per_px) <= 0.03 * max(top, v)
        k = order.index(seg_cluster[id(b)]) if b is not None else None
        if k is not None and k < len(series):
            series[k]['labels'].append({'x': x, 'value': v, 'pixels': bool(fits), 'tesseract': any(abs(v - t) < 1e-6 for t in tess_values)})
    return {'categories': cats, 'series': series, 'scale': bool(scale), 'stacked': True}


def find_ring(rgb: np.ndarray):
    """The pie or donut of a chart: the largest round painted blob once text strokes are opened away. (cy, cx, inner,
    outer radius) or None. Round means a bbox about as wide as high, at least 120 px, and paint all around its centre."""
    from scipy import ndimage
    smooth = np.stack([ndimage.median_filter(rgb[..., c], size=3) for c in range(3)], axis=-1)
    paint = (smooth.min(axis=2) < 235) & (smooth.mean(axis=2) > 45)
    solid = ndimage.binary_dilation(ndimage.binary_opening(paint, structure=np.ones((9, 9))), structure=np.ones((5, 5)))
    labels, _ = ndimage.label(solid)
    best = None
    for i, sl in enumerate(ndimage.find_objects(labels), 1):
        ys, xs = sl
        h, w = ys.stop - ys.start, xs.stop - xs.start
        if min(h, w) < 120 or not 0.85 <= w / h <= 1.18 or ys.start == 0 or xs.start == 0 or \
                ys.stop == rgb.shape[0] or xs.stop == rgb.shape[1]:
            continue
        cy, cx = (ys.start + ys.stop - 1) / 2, (xs.start + xs.stop - 1) / 2
        yy, xx = np.nonzero(labels[sl] == i)
        dist = np.hypot(yy + ys.start - cy, xx + xs.start - cx)
        outer, inner = float(np.percentile(dist, 99)) - 2, float(np.percentile(dist, 1)) + 2
        angles = np.degrees(np.arctan2(yy + ys.start - cy, xx + xs.start - cx)).astype(int) % 360
        if len(np.unique(angles)) < 340 or (dist > outer + 6).mean() > 0.02:  # paint all around, nothing outside the circle
            continue
        if best is None or outer > best[3]:
            best = (cy, cx, max(inner, 0.0), outer)
    return best


def pale_names(colours: np.ndarray, legend: list[tuple[tuple, str]]) -> list[str | None]:
    """Ring colours named by legend swatches, one to one, the best fits first. A small swatch is the colour blended with
    white: swatch ≈ 255 + a·(colour − 255) with a in 0.4–1.1, the fit left within 30 per channel."""
    fits = []
    for i, c in enumerate(colours):
        for j, (sw, _) in enumerate(legend):
            d, e = np.array(c, float) - 255, np.array(sw, float) - 255
            a = float(d @ e / (d @ d)) if d @ d > 0 else 0
            miss = float(np.abs(e - a * d).max())
            if 0.4 <= a <= 1.1 and miss <= 30:
                fits.append((miss, i, j))
    out: list[str | None] = [None] * len(colours)
    used = set()
    for _, i, j in sorted(fits):
        if out[i] is None and j not in used:
            out[i] = legend[j][1]
            used.add(j)
    return out


def said_names(colours: np.ndarray, read: dict) -> list[str | None]:
    """No legend swatch read (one touching its text): a ring colour takes the series whose colour the model named, when
    that colour family has one ring colour and one series."""
    from stage5_insight_detection.bank_analytics.digitize import COLORS, RU_COLORS
    family = lambda c: [k for k, f in COLORS.items() if f(*(np.array([v]) for v in np.asarray(c, int)))[0]]
    said = {}
    for s in read.get('series') or []:
        fam = next((v for k, v in RU_COLORS.items() if k in str(s.get('color') or '').lower()), None)
        said.setdefault(fam, []).append(s.get('name'))
    fams = [family(c) for c in colours]
    out: list[str | None] = []
    for f in fams:
        one = f[0] if len(f) == 1 else None
        ok = one and sum(one in g for g in fams) == 1 and len(said.get(one, [])) == 1
        out.append(said[one][0] if ok else None)
    return out


def ring_colours(samples: np.ndarray, least: float = 0.005) -> np.ndarray:
    """Solid colours of the ring: samples grouped by colour (a group takes every sample within 40 of its seed, the most
    common colour first), groups of at least `least` of the samples kept; their median colours."""
    keys, counts = np.unique((samples // 16).astype(int), axis=0, return_counts=True)
    left, out = np.ones(len(samples), bool), []
    for k in keys[np.argsort(-counts)]:
        seed = k * 16 + 8
        near = left & (np.abs(samples - seed).max(axis=1) <= 40)
        if near.sum() >= least * len(samples):
            out.append(np.median(samples[near], axis=0))
        left &= ~near
    return np.array(out)


def pie_shares(rgb: np.ndarray, read: dict, words: list[Word]) -> dict | None:
    """Shares of a pie or donut by angle: around the ring at five radii (printed labels sit on it, so each angle takes
    the colour most radii agree on); the ring's colours named by the legend swatches (legend_names, one to one; a swatch
    is blurred and pale, so the ring's own colours are the classes, not the swatches), a single colour left without a
    swatch takes the single series left without one. White separators are left out and the shares scaled to 100. A
    printed «N%» of a slice the angle confirms within 1.5 pp is checked by the pixels."""
    ring = find_ring(rgb)
    names = [s.get('name', '') for s in read.get('series') or []]
    if not ring:
        return None
    legend = legend_map(rgb, words, names)
    legend = legend if len(legend) >= 2 else []
    from scipy import ndimage
    smooth = np.stack([ndimage.median_filter(rgb[..., c], size=3) for c in range(3)], axis=-1).astype(float)
    cy, cx, inner, outer = ring
    inner = inner if inner > 0.3 * outer else 0.3 * outer
    angles = np.radians(np.arange(1440) / 4)
    grid = np.stack([smooth[np.round(cy + (inner + f * (outer - inner)) * np.sin(angles)).astype(int),
                            np.round(cx + (inner + f * (outer - inner)) * np.cos(angles)).astype(int)]
                     for f in (0.15, 0.3, 0.5, 0.7, 0.85)], axis=1)  # angle × radius × rgb
    painted = grid.min(axis=2) < 235
    if painted.any(axis=1).mean() < 0.9:
        return None
    colours = ring_colours(grid[painted])
    dist = np.abs(grid[:, :, None, :] - colours[None, None]).max(axis=3)
    cls = np.where(painted & (dist.min(axis=2) <= 40), dist.argmin(axis=2), -1)
    votes = np.array([np.bincount(c[c >= 0]).argmax() if (c >= 0).any() else -1 for c in cls])
    counts = np.bincount(votes[votes >= 0], minlength=len(colours))
    named = pale_names(colours, legend) if legend else said_names(colours, read)
    free_names = [n for n in names if n not in named]
    unnamed = [i for i, n in enumerate(named) if n is None and counts[i] >= 0.02 * counts.sum()]  # blends at edges are fewer
    if len(unnamed) == 1 and len(free_names) == 1:
        named[unnamed[0]] = free_names[0]
    total = sum(n for i, n in enumerate(counts) if named[i])
    if total < 0.95 * counts.sum():  # a slice without a series would swell the others
        return None
    printed = {s.get('name'): number(str(pv.get('value'))) for s in read.get('series') or []
               for pv in (s.get('printed_values') or [])[:1]}
    slices = []
    for i, n in enumerate(counts):
        if named[i]:
            share, label = round(100 * n / total, 1), printed.get(named[i])
            slices.append({'name': named[i], 'share': share, 'label': label,
                           'pixels': bool(label is not None and abs(label - share) <= 1.5)})
    labels = [s['label'] for s in slices if s['label'] is not None]
    return {'slices': sorted(slices, key=lambda s: -s['share']),
            'labels_sum': round(sum(labels), 1) if len(labels) == len(slices) else None}


def measure(rgb: np.ndarray, read: dict) -> dict:
    """Everything the code takes off one crop, for charts.py: Tesseract's numbers, the scale from its tick labels, the bars
    (when the model saw bars), the pie shares, and the lines (digitize). Free; `charts.py --recode` redoes it for every
    stored reading after a change of the code."""
    from dataclasses import asdict
    from stage5_insight_detection.bank_analytics.digitize import digitize
    if 'error' in read:
        return {'code': None, 'digitized': None}
    words = chart_words(rgb)
    sc = scale_from_words(words, rgb.shape[:2])
    scale = {'a': float(sc[0][1]), 'b': float(sc[0][0]), 'ticks': sc[1], 'band': [float(sc[2][0]), float(sc[2][1])]} if sc else None
    styles = ' '.join(str(s.get('style')) for s in read.get('series') or [])
    bars = bar_series(rgb, read, words) if 'столб' in styles else None
    if bars and (bars.get('stacked') or bars['categories'] == [str(i + 1) for i in range(len(bars['categories']))]):
        # the columns matched neither the axis labels nor the words under them: a measured «column 7» tells the agent
        # nothing and may be a hatched segment or a gap read as a column (Inflation_tracker p2: 31 columns for 26 dates);
        # stacked segments measured wrong on both charts checked by eye (FI p16: 2028 ≈ 429 for ~830), so they keep
        # only the printed labels their heights confirm
        for srs in bars['series']:
            srs['values'] = [None] * len(srs['values'])
        bars['unnamed'] = True
    for srs in (bars or {}).get('series') or []:  # a measured column the printed label it carries contradicts
        cats = bars['categories']
        sure = [(lb['x'], lb['value']) for lb in srs.get('labels') or [] if lb.get('pixels') or lb.get('tesseract')]
        top = max((abs(v) for _, v in sure), default=0)
        if any(x in cats and cats.index(x) < len(srs['values']) and srs['values'][cats.index(x)] is not None and
               abs(srs['values'][cats.index(x)] - v) > 0.05 * top for x, v in sure):
            # a line drawn across the bars cuts them, and the cut part is measured (Gold p7: 39 measured as 8,3)
            srs['values'] = [None] * len(srs['values'])
            srs['dropped'] = 'замер разошёлся с подписью'
    code = {'numbers': [w.text for w in words if w.value is not None], 'scale': scale, 'bars': bars,
            'pie': pie_shares(rgb, read, words)}
    return {'code': code, 'digitized': asdict(digitize(rgb, read, scale, words))}


def chart_points(record: dict) -> list[tuple[str, str, float, str]]:
    """Every number the agent gets from a chart, with its status: (series, x or 'last', value, verified | approx), and the
    printed labels it does not get (unverified)."""
    from stage5_insight_detection.bank_analytics.digitize import checked_labels
    out = []
    sure, unsure = checked_labels(record)
    for name, x, v, _ in sure:
        if number(str(v)) is not None:
            out.append((name, str(x), number(str(v)), 'verified'))
    for name, x, v, _ in unsure:
        if number(str(v)) is not None:
            out.append((name, str(x), number(str(v)), 'unverified'))
    dg = record.get('digitized') or {}
    if dg.get('ok'):
        for s in dg.get('series') or []:
            for x, v in zip(dg.get('x_labels') or [], s.get('values') or []):
                if v is not None:
                    out.append((s['name'], str(x), v, 'approx'))
            for key in ('last', 'min', 'max'):
                if s.get(key) is not None:
                    out.append((s['name'], key, s[key], 'approx'))
    for sl in ((record.get('code') or {}).get('pie') or {}).get('slices') or []:
        out.append((sl['name'], 'доля', sl['label'] if sl['pixels'] else sl['share'], 'verified' if sl['pixels'] else 'approx'))
    bars = (record.get('code') or {}).get('bars') or {}
    if bars.get('scale'):
        for s in bars.get('series') or []:
            for x, v in zip(bars.get('categories') or [], s.get('values') or []):
                if v is not None:
                    out.append((s['name'], str(x), v, 'approx'))
    return out


def evaluate(records: dict[str, dict], reference: list[dict]) -> dict:
    """Reference points against what the agent gets: a point is covered by a verified or approx number of the same
    series (name contains the reference's) at the same x, right if within its tolerance. Unverified labels counted
    apart (the agent does not see them)."""
    res = {'points': len(reference), 'covered': 0, 'right': 0, 'verified': [0, 0], 'approx': [0, 0], 'unverified': [0, 0],
           'err_steps': []}
    for ref in reference:
        rec = records.get(ref['crop'])
        if not rec:
            continue
        pts = [p for p in chart_points(rec) if norm(p[1]) == norm(ref['x'])]
        exact = [p for p in pts if norm(str(p[0]).split('(')[0]) == norm(ref['series'])]  # «A» is not «AA» or «AAA»
        pts = exact or [p for p in pts if ref['series'].lower() in str(p[0]).lower()]
        seen = {}
        for p in pts:
            seen.setdefault(p[3], p)
        tol = lambda status: max(ref['tol'], 0.2 * ref['step']) if status == 'approx' else ref['tol']  # «≈»: ±0.2 step
        for status in ('verified', 'approx', 'unverified'):
            if status in seen:
                ok = abs(seen[status][2] - ref['value']) <= tol(status)
                res[status][0] += 1
                res[status][1] += ok
        best = seen.get('verified') or seen.get('approx')
        if best:
            res['covered'] += 1
            res['right'] += abs(best[2] - ref['value']) <= tol(best[3])
            res['err_steps'].append(abs(best[2] - ref['value']) / ref['step'])
    res['mean_err_steps'] = round(float(np.mean(res.pop('err_steps'))), 3) if res['covered'] else None
    return res


if __name__ == '__main__':
    crop = Path(sys.argv[1])
    rgb = to_rgb(crop.read_bytes())
    words = chart_words(rgb)
    print('words', len(words), [w.text for w in words][:40])
    print('scale', scale_from_words(words, rgb.shape[:2]))
    bars, base = find_bars(rgb, words)
    print('bars', len(bars), 'base', base, [(b.x0, b.y0, b.y1, b.rgb, b.stacked_on) for b in bars][:12])
    print('categories', [len(c) for c in categories(bars, words, base)])
    import json
    rec = next((r for f in Path('data/analytics/charts').glob('*.gemma.json') for r in json.loads(f.read_text(encoding='utf-8'))
                if r['crop'] == crop.name), None)
    if rec:
        res = bar_series(rgb, rec['read'], words)
        print(json.dumps(res, ensure_ascii=False)[:1500])
