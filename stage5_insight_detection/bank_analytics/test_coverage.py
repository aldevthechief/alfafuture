"""Offline test of the coverage check: a PDF dropped next to an indexed one is found and indexed alone (text layer, no OCR, no network).
python -X utf8 stage5_insight_detection/bank_analytics/test_coverage.py"""
import json, sys, tempfile
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stage5_insight_detection.bank_analytics.coverage import index_missing, uncovered


def pdf(path: Path, pages: list[str]) -> Path:
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_textbox(pymupdf.Rect(40, 40, 550, 800), text, fontsize=11)
    doc.save(path)
    return path


with tempfile.TemporaryDirectory() as tmp:
    folder = Path(tmp)
    (folder / 'ocr').mkdir()
    text = 'Банк ожидает сохранения ключевой ставки на уровне 14% до конца года. ' * 6
    old = pdf(folder / 'old_report.pdf', [text, text])
    row = lambda page: {'id': f'bank:old_report:p{page}.1', 'file': 'old_report.pdf', 'report': 'Old', 'date': '2026-09-11', 'page': page, 'kind': 'text',
                        'text': text, 'freshness': 'fresh'}
    rows = [row(1), row(2)]
    (folder / 'fragments.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    assert uncovered(folder, rows) == [], 'everything covered'

    new = pdf(folder / 'new_report_15Oct.pdf', ['New bank report: the ruble exchange rate is expected at 90 by the end of the year. ' * 6])  # Latin: the built-in PDF font has no Cyrillic
    assert [p.name for p in uncovered(folder, rows)] == ['new_report_15Oct.pdf'], 'a PDF without fragments'
    rows2 = rows[:1]  # a file with a page that has no fragments counts as not covered, and only that file
    assert sorted(p.name for p in uncovered(folder, rows2)) == ['new_report_15Oct.pdf', 'old_report.pdf']

    messages = []
    out = index_missing(folder, folder / 'ocr', rows, uncovered(folder, rows), log=messages.append)
    assert len(messages) == 1 and 'new_report_15Oct.pdf' in messages[0], messages
    assert [r for r in out if r['file'] == 'old_report.pdf'] == rows, 'the covered report is not touched'
    fresh = [r for r in out if r['file'] == 'new_report_15Oct.pdf']
    assert fresh and 'exchange rate' in fresh[0]['text'] and fresh[0]['page'] == 1, fresh
    saved = [json.loads(l) for l in (folder / 'fragments.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(saved) == len(out) and uncovered(folder, saved) == [], 'the jsonl is rewritten and now covers the folder'
print('ok: coverage: covered folder untouched, new PDF and PDF with a missing page found, only the missing file indexed, jsonl rewritten')
