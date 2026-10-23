"""Does `fragments.jsonl` cover the PDFs that lie in the analytics folder? A PDF with no fragments, or with pages that have none, is
indexed again (BankLibrary: text layer or OCR, tables, charts) and only that file: the covered ones are not touched, so a new report
dropped into the folder is found by the next search start without a manual rebuild and without paying OCR for the old ones.
(The mvp library did this by file hash on every start; `BankAnalytics` reads the finished jsonl, so the check lives here.)"""
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import pymupdf


def uncovered(folder: Path, rows: list[dict]) -> list[Path]:
    """PDFs of the folder that have no fragments, or fewer pages with fragments than pages in the file."""
    pages = defaultdict(set)
    for r in rows:
        pages[r.get('file') or Path(str(r['id'])).name].add(r['page'])
    out = []
    for pdf in sorted(folder.glob('*.pdf')):
        with pymupdf.open(pdf) as doc:
            if set(range(1, len(doc) + 1)) - pages.get(pdf.name, set()):
                out.append(pdf)
    return out


def index_missing(folder: Path, ocr_dir: Path, rows: list[dict], missing: list[Path], as_of: date | None = None, log=print) -> list[dict]:
    """Rows of the whole folder: the old rows of the other files plus the freshly parsed `missing` ones; the jsonl is rewritten."""
    from dataclasses import asdict
    from stage5_insight_detection.bank_analytics.library import (REPORTS, best_pages, date_from_name, parse_report)
    as_of = as_of or date.today()
    by_file = {r['file']: r for r in REPORTS}
    names = {p.name for p in missing}
    keep = [r for r in rows if r.get('file') not in names]
    fresh = []
    for pdf in missing:
        log(f'bank analytics: {pdf.name} is not covered by fragments.jsonl, indexing it')
        meta = by_file.get(pdf.name) or {}
        when = date.fromisoformat(meta['date']) if meta.get('date') else date_from_name(pdf.name, as_of.year)
        fresh += [asdict(f) for f in parse_report(pdf.name, best_pages(pdf, folder / 'ocr_full', ocr_dir, None, when.year if when else None), meta, as_of)]
    out = keep + fresh
    path = folder / 'fragments.jsonl'
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in out), encoding='utf-8')
    return out
