"""Bank analytics PDFs -> one readable Markdown text per report: data/analytics/text/<stem>.md.

Page texts come from `library.page_texts`: the PDF's text layer, or the cached OCR of ocr.py (no model is called here),
cleaned by pages.clean_page (model loops dropped, made-up chart values flagged ⚠, with `--fallback-ocr-dir` the older
OCR appended under pages that look broken). Per file it prints pages, characters, tables (blocks of "|" lines), chart
descriptions ([График] blocks of the ocr_page_full prompt, or paragraphs that talk about a chart), values read off the
scale, flagged charts, pages with the fallback, empty pages and removed repeats. Where charts.py ran, every chart block
gets the labels the pixels confirmed and the values digitize.py measured (digitize.annotate).
python -X utf8 stage5_insight_detection/bank_analytics/export_text.py [--ocr-dir data/analytics/ocr] [--out data/analytics/text] [file.pdf ...]
python -X utf8 stage5_insight_detection/bank_analytics/export_text.py --ocr-dir data/analytics/ocr_full --fallback-ocr-dir data/analytics/ocr
"""
import argparse, sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stage5_insight_detection.bank_analytics.library import page_texts, text_layer, prepare_page
from stage5_insight_detection.bank_analytics.ocr import OCR_DIR, PDF_DIR
from stage5_insight_detection.bank_analytics import tables as table_reading
from stage5_insight_detection.bank_analytics.digitize import annotate
from stage5_insight_detection.bank_analytics.pages import PageStats, clean_page, dedup  # noqa: F401 (dedup: used by tests)
from stage5_insight_detection.bank_analytics.search import REPORTS

TEXT_DIR = PDF_DIR / 'text'
EMPTY_PAGE = '_(на странице нет текста)_'


def render_page(n: int, raw: str, fallback: str = '', report_year: int | None = None, charts: list | None = None,
                tables: dict | None = None, grounded: bool = False) -> tuple[str, PageStats]:
    text, stats = clean_page(raw, fallback, report_year)
    text = annotate(text, charts or [], grounded)  # charts read crop by crop (charts.py); invented blocks marked ⚠
    text = table_reading.replace_tables(text, tables)  # tables read three times and reconciled (tables.py)
    return f'## Страница {n}\n\n{text or EMPTY_PAGE}\n', stats


def export(pdf: Path, ocr_dir: Path, out_dir: Path, fallback_dir: Path | None = None) -> tuple[int, PageStats]:
    meta = next((r for r in REPORTS if r['file'] == pdf.name), {})
    pages = page_texts(pdf, ocr_dir)
    backup = page_texts(pdf, fallback_dir) if fallback_dir else [''] * len(pages)
    _, has_layer = text_layer(pdf)
    cache = ocr_dir / f'{pdf.stem}.jsonl'
    source = 'текстовый слой' if has_layer else f'OCR Qwen3-VL (папка {ocr_dir})' if cache.exists() else \
        'нет (ни текстового слоя, ни OCR)'
    when = date.fromisoformat(meta['date']).strftime('%d.%m.%Y') if meta.get('date') else 'неизвестна'
    head = [f'# {meta.get("title") or pdf.stem}', '', f'Файл: {pdf.name}  ', f'Дата отчёта: {when}  ',
            f'Страниц: {len(pages)}  ', f'Источник текста: {source}', '']
    total, body = PageStats(), []
    for n, (raw, old) in enumerate(zip(pages, backup), 1):
        text, stats = prepare_page(pdf, n, raw, old, int(meta['date'][:4]) if meta.get('date') else None)
        md = f'## Страница {n}\n\n{text or EMPTY_PAGE}\n'
        body.append(md)
        total += stats
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'{pdf.stem}.md').write_text('\n'.join(head + body), encoding='utf-8')
    return len(pages), total


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ocr-dir', type=Path, default=OCR_DIR)
    ap.add_argument('--out', type=Path, default=TEXT_DIR)
    ap.add_argument('--fallback-ocr-dir', type=Path, default=None, help='older OCR appended under pages that look broken')
    ap.add_argument('files', nargs='*')
    a = ap.parse_args()
    pdfs = [Path(f) if Path(f).exists() else PDF_DIR / f for f in a.files] or sorted(PDF_DIR.glob('*.pdf'))
    print(f'{"file":52} {"pages":>5} {"chars":>7} {"tables":>6} {"charts":>6} {"≈":>4} {"suspect":>7} {"backup":>6} {"empty":>5} {"repeats":>7}')
    for pdf in pdfs:
        n, s = export(pdf, a.ocr_dir, a.out, a.fallback_ocr_dir)
        print(f'{pdf.name[:52]:52} {n:>5} {s.chars:>7} {s.tables:>6} {s.charts:>6} '
              f'{s.approx:>4} {s.suspect:>7} {s.fallback:>6} {s.empty:>5} {s.repeats:>7}')
    print(f'-> {a.out}')
