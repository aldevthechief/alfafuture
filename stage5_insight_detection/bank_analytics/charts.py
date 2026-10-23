"""Pilot: read the charts of a page one by one instead of with the whole page (whole-page OCR made up most chart values).

1. The vision model (Qwen3-VL grounding, prompts/chart_boxes.md, page at 120 dpi) returns the boxes of the page's charts
   in 0-1000 page coordinates; each box is widened (title above, legend and source below) and rendered by PyMuPDF from
   the PDF itself (clip) at 250 dpi, at most 1500 px on the long side -> data/analytics/charts/<stem>_p<page>_<n>.png.
2. Every crop is read by one call (prompts/ocr_chart.md): printed values only, with series and point; series without
   labels get directions and the last point between ticks only if unambiguous, else «не определить»; axis ticks; JSON.
   `--model gemma` reads the same crops with the cheap model for comparison.
3. chart_reading.measure (code, free): Tesseract's numbers and scale, bars, pie shares, and digitize.py's lines (values
   at the dates, last point); stored with the model's reading, and library/export_text put both into the chart block.
   `--recode` redoes this part for every stored reading after a change of the code.
Cost (proxy `usage.cost`, ₽) and time per call are recorded; `--budget` stops before the spend would pass it; pages and
charts already done are skipped, so a stopped run goes on where it stopped.
python -X utf8 stage5_insight_detection/bank_analytics/charts.py Inflation_tracker_02.pdf:2 Бонд_Бриф_14092026.pdf:4 [--model qwen|gemma] [--budget 1.5]
python -X utf8 stage5_insight_detection/bank_analytics/charts.py --all --ground qwen --model gemma --budget 8   (every page with a [График] block)
python -X utf8 stage5_insight_detection/bank_analytics/charts.py --recode --model gemma   (the code part again, free)
`--charts selective` (or env BANK_CHARTS) pays only for charts that may say what the text does not
(chart_novelty.block_plan: the page OCR's block against the caption, the «см Рис.N» references and the paragraphs about
the same indicator): pages with a forecast go first, then new values, then indicators the text never names, so a budget
stop drops the least important; a page with nothing new gets no box call, a crop whose title matches a block judged not
new is not read. `--charts all` (default) reads every chart, `--charts off` none.
"""
import argparse, base64, json, re, sys, time
from pathlib import Path

import numpy as np
import pymupdf
from langchain_core.messages import HumanMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.llm import ChatModels
from stage5_insight_detection.bank_analytics.digest import BudgetExceeded, CostMeter

from stage5_insight_detection.bank_analytics import chart_novelty
from stage5_insight_detection.bank_analytics.chart_reading import measure
from stage5_insight_detection.bank_analytics.digitize import groups, snap_box, to_png, to_rgb
from stage5_insight_detection.bank_analytics.ocr import PDF_DIR, PROMPTS

CHART_DIR = PDF_DIR / 'charts'
GEMMA = 'google/gemma-4-26b-a4b-it'
DPI, GROUND_DPI, MAX_SIDE = 250, 120, 1500
PAD = (0.02, 0.035, 0.02, 0.03)  # left, top, right, bottom as a share of the page: title above, source below


REACH = 0.15
MARGIN = 0.02
CROP_VERSION = "white-band-monthly-v2"


def render(pdf: Path, page: int, dpi: int, bbox: list[float] | None = None, sides: tuple[float, float] | None = None) -> bytes:
    """PNG of the page, or of a 0-1000 box of it, at `dpi` but at most MAX_SIDE px on the long side. The box is grown
    by up to PAD only until a white band (digitize.snap_box): the title and legend come in, the neighbour chart does not.
    `sides` (0-1 of the page width, from chart_sides) are the chart's left and right edges found on the whole row, used
    instead of the grown ones; the top and bottom are grown as before."""
    with pymupdf.open(pdf) as doc:
        p = doc[page - 1]
        if not bbox:
            return p.get_pixmap(dpi=dpi).tobytes('png')
        w, h = p.rect.width, p.rect.height
        x1, y1, x2, y2 = [v / 1000 for v in bbox]
        l, t, r, b = PAD
        clip = pymupdf.Rect(max(0, (x1 - l) * w), max(0, (y1 - t) * h), min(w, (x2 + r) * w), min(h, (y2 + b) * h))
        if sides:
            clip.x0, clip.x1 = min(clip.x0, sides[0] * w), max(clip.x1, sides[1] * w)
        dpi = min(dpi, int(MAX_SIDE / (max(clip.width, clip.height) / 72)))
        rgb = to_rgb(p.get_pixmap(dpi=dpi, clip=clip).tobytes('png'))
    k = dpi / 72
    inner = (int((x1 * w - clip.x0) * k), int((y1 * h - clip.y0) * k), int((x2 * w - clip.x0) * k), int((y2 * h - clip.y0) * k))
    if sides:
        inner = (int((sides[0] * w - clip.x0) * k), inner[1], int((sides[1] * w - clip.x0) * k), inner[3])
    X1, Y1, X2, Y2 = snap_box(rgb, inner)
    if sides:
        X1, X2 = inner[0], inner[2]
    return to_png(rgb[Y1:Y2 + 1, X1:X2 + 1])


def chart_sides(pdf: Path, page: int, boxes: list[dict], dpi: int = 200, gap: int = 6) -> list[tuple[float, float]]:
    """Left and right edges of each chart (0-1 of the page width), never inside the model's box + PAD. The model's
    boxes are approximate: three charts in a row came back as three equal thirds of the row (FI_Strategy p13), cutting
    the middle chart at its last months and starting the right one on its tail. An edge is the first white band of `gap`
    columns beyond the box, at most REACH away, in the box's own rows without the rows of long rules and banners (they
    cross the white band between charts). Charts side by side are then kept apart: a chart starts after the band that
    ends its left neighbour; a chart that ran deep into its neighbour's box (no white band) keeps its old edge. Each side
    then gets MARGIN more, up to half the gap to a chart beside it."""
    if not boxes:
        return []
    with pymupdf.open(pdf) as doc:
        rgb = to_rgb(doc[page - 1].get_pixmap(dpi=dpi).tobytes('png'))
    H, W = rgb.shape[:2]
    ink = rgb.min(axis=2) < 235
    l, _, r, _ = PAD
    edges = []
    for bx in boxes:
        x1, y1, x2, y2 = [v / 1000 for v in bx['bbox']]
        X0, X3 = int(max(0, x1 - REACH) * W), int(min(1, x2 + REACH) * W)
        sub = ink[int(y1 * H):int(y2 * H), X0:X3]
        longest = np.array([max((q - p + 1 for p, q in groups(row.astype(int), gap=0)), default=0) for row in sub])
        cols = sub[longest < 0.3 * (X3 - X0)].any(0)  # a rule or a banner is not the chart's ink

        def edge(pos, step):
            blank = 0
            while 0 <= pos + step < len(cols):
                pos += step
                blank = blank + 1 if not cols[pos] else 0
                if blank >= gap:
                    return pos - step * gap
            return pos
        edges.append([(X0 + edge(int(x1 * W) - X0, -1)) / W, (X0 + edge(int(x2 * W) - X0 - 1, 1) + 1) / W])
    trimmed = set()
    for i, a in enumerate(edges):
        for j, c in enumerate(edges):
            ya, yc = boxes[i]['bbox'][1::2], boxes[j]['bbox'][1::2]
            rows = min(ya[1], yc[1]) - max(ya[0], yc[0])
            if i == j or boxes[i]['bbox'][0] >= boxes[j]['bbox'][0] or rows < 0.5 * min(ya[1] - ya[0], yc[1] - yc[0]):
                continue  # a is the left one of two charts in one row
            box_c = [v / 1000 for v in boxes[j]['bbox']]
            if a[1] > box_c[0] + 0.15 * (box_c[2] - box_c[0]):  # a ran deep into c's box: no white band between them,
                a[1] = max(boxes[i]['bbox'][2] / 1000 + r, min(a[1], box_c[0]))  # a keeps its old right edge
            if c[0] < a[1]:  # c starts on a's tail: at the first ink after the band that ends a
                y_a, y_b = int(max(ya[0], yc[0]) / 1000 * H), int(min(ya[1], yc[1]) / 1000 * H)
                ink_c = np.nonzero(ink[y_a:y_b, int(a[1] * W):int(c[1] * W)].any(0))[0]
                c[0] = (int(a[1] * W) + (ink_c[0] if len(ink_c) else 0)) / W
                trimmed.add(j)
    out = []
    for i, (e, bx) in enumerate(zip(edges, boxes)):  # a margin past the white band: a word of text in the crop costs
        # nothing, a cut date label or the last bar costs the chart; never into a chart beside it
        lo, hi = e[0] - MARGIN, e[1] + MARGIN
        for j, f in enumerate(edges):
            ya, yf = bx['bbox'][1::2], boxes[j]['bbox'][1::2]
            if j != i and min(ya[1], yf[1]) - max(ya[0], yf[0]) > 0.5 * min(ya[1] - ya[0], yf[1] - yf[0]):
                if f[1] <= e[0]:
                    lo = max(lo, (f[1] + e[0]) / 2)
                elif f[0] >= e[1]:
                    hi = min(hi, (f[0] + e[1]) / 2)
        out.append((max(0.0, lo), min(1.0, hi)))
    return out


def parse_json(text: str):
    text = re.sub(r'^```(?:json)?|```$', '', str(text).strip(), flags=re.M).strip()
    start = min((i for i in (text.find('['), text.find('{')) if i >= 0), default=-1)
    return json.loads(text[start:]) if start >= 0 else None


class ChartReadError(ValueError):
    pass


def valid_answer(prompt: str, data) -> bool:
    if prompt in ('chart_boxes', 'table_boxes'):
        if not isinstance(data, list):
            return False
        for b in data:
            box = b.get('bbox') if isinstance(b, dict) else None
            if isinstance(b, dict) and not isinstance(b.get('title', ''), str):
                return False
            if not isinstance(box, list) or len(box) != 4 or any(type(v) not in (int, float) or not 0 <= v <= 1000 for v in box):
                return False
            if box[0] >= box[2] or box[1] >= box[3]:
                return False
        return True
    if prompt == 'ocr_chart':
        if not (isinstance(data, dict) and isinstance(data.get('series'), list) and all(
                isinstance(data.get(k), dict) for k in ('x_axis', 'y_axis'))):
            return False
        return all(isinstance(s, dict) and isinstance(s.get('name', ''), str) and
                   isinstance(s.get('printed_values', []), list) and
                   all(isinstance(p, dict) and isinstance(p.get('value'), (str, int, float))
                       for p in s.get('printed_values', [])) for s in data['series'])
    if prompt == 'ocr_table':
        return isinstance(data, dict) and isinstance(data.get('header'), list) and isinstance(data.get('rows'), list) and all(
            isinstance(r, list) for r in data['rows'])
    return data is not None


class ChartReader:
    def __init__(self, model: str = 'qwen', budget: float | None = None, ground: str | None = None, callbacks: list | None = None):
        self.meter, self.budget, self.max_call = CostMeter(), budget, 0.0
        models = ChatModels(callbacks=[*(callbacks or []), self.meter])
        make = lambda name: models.vision(max_tokens=2500) if name == 'qwen' else models._make(GEMMA, 0.0, 2500)
        self.model = make(model)
        self.ground = make(ground or model)  # Qwen3-VL finds the boxes (grounding), Gemma reads the crops
        self.calls: list[dict] = []

    def ask(self, prompt: str, png: bytes, what: str):
        image = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(png).decode()}}
        for attempt in range(2):
            if self.budget is not None and self.meter.cost + max(self.max_call, 0.05) > self.budget:
                raise BudgetExceeded(f'budget: spent {self.meter.cost:.2f} of {self.budget} ₽ before {what}')
            before, t0 = self.meter.cost, time.time()
            retry = '\nПовтори: верни полный JSON в указанной схеме, без пояснений.' if attempt else ''
            response = self.model.invoke([HumanMessage(content=[{'type': 'text', 'text': PROMPTS.render(prompt) + retry}, image])])
            cost, sec = self.meter.cost - before, round(time.time() - t0, 1)
            self.max_call = max(self.max_call, cost)
            self.calls.append({'what': what, 'cost': round(cost, 4), 'sec': sec})
            if response.response_metadata.get('finish_reason') in ('length', 'max_tokens'):
                continue
            try:
                data = parse_json(response.content)
                if valid_answer(prompt, data):
                    return data, response.content
            except ValueError:
                pass
        raise ChartReadError(f'{what}: invalid {prompt} response after 2 attempts; page not cached')

    def boxes(self, pdf: Path, page: int) -> list[dict]:
        model, self.model = self.model, self.ground
        try:
            found, _ = self.ask('chart_boxes', render(pdf, page, GROUND_DPI), f'{pdf.stem} p{page} boxes')
        finally:
            self.model = model
        return [b for b in found or [] if isinstance(b, dict) and len(b.get('bbox') or []) == 4]

    def read(self, crop: bytes, what: str) -> dict:
        data, raw = self.ask('ocr_chart', crop, what)
        return data if isinstance(data, dict) else {'error': 'not JSON', 'raw': str(raw)[:2000]}


def crops(pdf: Path, page: int, boxes: list[dict], out_dir: Path = CHART_DIR) -> list[Path]:
    paths = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for n, (b, sides) in enumerate(zip(boxes, chart_sides(pdf, page, boxes)), 1):
        path = out_dir / f'{pdf.stem}_p{page}_{n}.png'
        path.write_bytes(render(pdf, page, DPI, b['bbox'], sides))
        paths.append(path)
    return paths


def read_pages(pages: list[tuple[str, int]], *, pdf_dir: Path = PDF_DIR, out_dir: Path = CHART_DIR,
               model: str = 'gemma', ground: str = 'qwen', budget: float | None = None,
               callbacks: list | None = None, reader: ChartReader | None = None,
               boxes_only: bool = False, only: list[str] | None = None, charts: str | None = None) -> dict:
    """Reusable chart stage for the standalone command and pipeline; successful crops are cached immediately.
    `charts` (chart_novelty.MODES, default env BANK_CHARTS or all): selective reads only what may be new, forecast
    pages first; off reads nothing."""
    mode = chart_novelty.mode_of(charts)
    result = {'pages': len(pages), 'charts': 0, 'cached_charts': 0, 'errors': [], 'skipped_charts': 0}
    if not pages or mode == 'off':
        return result
    pdf_of = lambda stem: pdf_dir / (stem if stem.endswith('.pdf') else f'{stem}.pdf')
    plans = {}
    if mode == 'selective':  # forecasts first: a stop on the budget drops the least important pages
        for stem in {s for s, _ in pages}:
            plans[stem] = chart_novelty.block_plan(pdf_of(stem))
        pages = sorted(pages, key=lambda sp: chart_novelty.page_priority(plans[sp[0]].get(sp[1])))
    reader = reader or ChartReader(model, budget, ground, callbacks=callbacks)
    from stage5_insight_detection.bank_analytics.assets import prepare
    for stem, page in pages:
        pdf = pdf_of(stem)
        blocks = plans[stem].get(page) if mode == 'selective' else None
        if blocks and not any(b['new'] for b in blocks) and not (out_dir / f'{pdf.stem}_p{page}.boxes.json').exists():
            result['skipped_charts'] += len(blocks)  # selective: the text has what its charts hold, no box call
            continue
        prepare(pdf, out_dir, PROMPTS.render('chart_boxes') + PROMPTS.render('ocr_chart') + CROP_VERSION)
        box_file = out_dir / f'{pdf.stem}_p{page}.boxes.json'
        boxes = json.loads(box_file.read_text(encoding='utf-8')) if box_file.exists() else None
        if not valid_answer('chart_boxes', boxes):
            try:
                boxes = reader.boxes(pdf, page)
            except ChartReadError as error:
                result['errors'].append(str(error))
                continue
            box_file.write_text(json.dumps(boxes, ensure_ascii=False, indent=1), encoding='utf-8')
        if boxes_only:
            crops(pdf, page, boxes, out_dir)
            continue
        out = out_dir / f'{pdf.stem}_p{page}.{model}.json'
        records = {r['crop']: r for r in json.loads(out.read_text(encoding='utf-8'))} if out.exists() else {}
        sides = chart_sides(pdf, page, boxes)
        for n, box in enumerate(boxes, 1):
            if only is not None and str(n) not in only:
                continue
            name = f'{pdf.stem}_p{page}_{n}.png'
            if only is None and name in records and records[name].get('crop_version') == CROP_VERSION and valid_answer('ocr_chart', records[name].get('read')):
                result['cached_charts'] += 1
                continue
            if mode == 'selective' and only is None and not chart_novelty.worth_reading(box.get('title') or '', blocks):
                result['skipped_charts'] += 1  # its title matches a block whose numbers the text already gives
                continue
            path = out_dir / name
            png = render(pdf, page, DPI, box['bbox'], sides=sides[n - 1])
            try:
                read = reader.read(png, path.stem)
            except ChartReadError as error:
                result['errors'].append(str(error))
                continue
            measured = measure(to_rgb(png), read)
            path.write_bytes(png)  # keep the previous crop until a replacement reading has succeeded
            records[name] = {'crop': name, 'title': box.get('title'), 'bbox': box['bbox'], 'read': read,
                             'crop_version': CROP_VERSION, **reader.calls[-1], **measured}
            out.write_text(json.dumps(sorted(records.values(), key=lambda r: r['crop']), ensure_ascii=False, indent=1),
                           encoding='utf-8')
            result['charts'] += 1
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('pages', nargs='*', help='file.pdf:page')
    ap.add_argument('--all', action='store_true', help='every page with a [График] block in the OCR')
    ap.add_argument('--model', choices=['qwen', 'gemma'], default='qwen', help='reads the crops')
    ap.add_argument('--ground', choices=['qwen', 'gemma'], default=None, help='finds the charts (default: --model)')
    ap.add_argument('--budget', type=float, default=None)
    ap.add_argument('--boxes-only', action='store_true', help='find and crop the charts, do not read them')
    ap.add_argument('--only', nargs='*', default=None, help='chart numbers on the page to read, e.g. 2 6')
    ap.add_argument('--recode', action='store_true', help='redo the code part (Tesseract, scale, bars, pies, lines) of every '
                                                         'stored reading of --model; no model calls')
    ap.add_argument('--charts', choices=chart_novelty.MODES, default=None,
                    help='all: read every chart; selective: only charts with what the text lacks, forecasts first; '
                         'off: none (default env BANK_CHARTS or all)')
    a = ap.parse_args()
    if chart_novelty.mode_of(a.charts) == 'off' and not a.recode:
        print('charts: mode off, nothing is read')
        sys.exit()
    if a.recode:
        for out in sorted(CHART_DIR.glob(f'*.{a.model}.json')):
            records = json.loads(out.read_text(encoding='utf-8'))
            for r in records:
                r.update(measure(to_rgb((CHART_DIR / r['crop']).read_bytes()), r['read']))
            out.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding='utf-8')
            print(out.name, len(records), 'charts', flush=True)
        sys.exit()
    reader = ChartReader(a.model, a.budget, a.ground)
    pages = a.pages
    if a.all:
        from stage5_insight_detection.bank_analytics.library import BankLibrary
        lib = BankLibrary.refresh()
        pages = sorted({f'{f.file}:{f.page}' for f in lib.fragments if f.kind == 'chart'})
    selected = [(name, int(n)) for name, n in (spec.rsplit(':', 1) for spec in pages)]
    try:
        result = read_pages(selected, reader=reader, model=a.model, boxes_only=a.boxes_only, only=a.only, charts=a.charts)
    except BudgetExceeded as error:
        print(error)
        result = {}
    print(f'spent {reader.meter.cost:.2f} ₽, {len(reader.calls)} calls, {sum(c["sec"] for c in reader.calls):.0f} s'
          + (f', {result["skipped_charts"]} charts skipped (selective)' if result.get('skipped_charts') else ''))
