"""Text-only OCR of the bank's scanned PDFs without an LLM: EasyOCR (ru + en) on pages rendered at 300 dpi, text boxes
merged into paragraphs (paragraph=True), on the GPU when available. Charts come out as their axis labels, not values.
Writes the same JSONL as ocr.py (key = page, value = text) into data/analytics/ocr/ — only for PDFs that have no OCR
there yet, so the richer Qwen3-VL output wins. `--out-dir` + a file name: a side run, e.g. to compare with Qwen.
python -X utf8 stage5_insight_detection/bank_analytics/ocr_easy.py [file.pdf ...] [--out-dir DIR]"""
import argparse, json, re, sys
from pathlib import Path

import numpy as np
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stage5_insight_detection.bank_analytics.ocr import OCR_DIR, PDF_DIR


LATIN = str.maketrans('aceopxykmtbhACEOPXYKMTBH', 'асеорхукмтвнАСЕОРХУКМТВН')  # look-alikes EasyOCR mixes into Cyrillic


def clean(text: str) -> str:
    """Latin look-alikes inside Cyrillic words -> Cyrillic; «гlг», «rlr» -> «г/г» (year on year)."""
    fix = lambda w: re.search('[а-яё]', w, re.I) or re.fullmatch('[aceopxykmtbh]{1,3}', w)  # «a», «co» are Russian words here
    text = re.sub(r'\S+', lambda m: m[0].translate(LATIN) if fix(m[0]) else m[0], text)
    return re.sub(r'(?<!\w)[гr][lI|/1][гr](?!\w)', 'г/г', text)


def pages(pdf: Path, reader, dpi: int = 300):
    for i, page in enumerate(pymupdf.open(pdf), 1):
        pix = page.get_pixmap(dpi=dpi)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3]
        yield str(i), clean('\n\n'.join(t for _, t in reader.readtext(img, paragraph=True)))


if __name__ == '__main__':
    import easyocr, torch
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='*')
    ap.add_argument('--out-dir', default=str(OCR_DIR))
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    pdfs = [PDF_DIR / f for f in a.files] if a.files else sorted(PDF_DIR.glob('*.pdf'))
    todo = [p for p in pdfs if a.files or not (out / f'{p.stem}.jsonl').exists()]
    reader = easyocr.Reader(['ru', 'en'], gpu=torch.cuda.is_available())
    for pdf in todo:
        with open(out / f'{pdf.stem}.jsonl', 'w', encoding='utf-8') as f:
            for key, text in pages(pdf, reader):
                f.write(json.dumps({'key': key, 'value': text}, ensure_ascii=False) + '\n')
                print(f'{pdf.name} p.{key}: {len(text)} chars', flush=True)
