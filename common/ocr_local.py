"""Local OCR word boxes: native pytesseract when available, otherwise PyMuPDF's embedded Tesseract.

Language data: https://github.com/tesseract-ocr/tessdata_fast (Apache-2.0).
No downloads during inference. Setup explicitly: python -m common.ocr_local --download
"""
import argparse
from io import BytesIO
import os
from pathlib import Path
import shutil
from urllib.request import urlopen

import pymupdf

DATA_DIR = Path(__file__).resolve().parents[1] / '.cache' / 'tessdata'


def tessdata() -> Path:
    folder = Path(os.environ.get('TESSDATA_PREFIX', DATA_DIR))
    if not all((folder / f'{lang}.traineddata').is_file() for lang in ('rus', 'eng')):
        try:
            folder = Path(pymupdf.get_tessdata())
        except RuntimeError:
            pass
    if not all((folder / f'{lang}.traineddata').is_file() for lang in ('rus', 'eng')):
        raise RuntimeError('Local OCR needs rus/eng language data: run python -m common.ocr_local --download '
                           'or set TESSDATA_PREFIX to your tessdata folder')
    return folder


def word_data(image, psm: int = 11) -> dict:
    """Boxes in input pixels. Embedded OCR has no word confidence; conf=1 means present, not a quality score."""
    try:
        import pytesseract
    except ImportError:
        pytesseract = None
    if pytesseract is not None:
        cmd = pytesseract.pytesseract.tesseract_cmd
        if shutil.which(cmd) or Path(cmd).is_file():
            return pytesseract.image_to_data(image, lang='rus+eng', config=f'--psm {psm}',
                                             output_type=pytesseract.Output.DICT)
    stream = BytesIO()
    image.convert('RGB').save(stream, format='PNG')
    pix = pymupdf.Pixmap(stream.getvalue())
    pix.set_dpi(300, 300)
    pdf = pix.pdfocr_tobytes(language='rus+eng', tessdata=str(tessdata()))
    out = {k: [] for k in ('text', 'left', 'top', 'width', 'height', 'conf')}
    with pymupdf.open(stream=pdf, filetype='pdf') as doc:
        page = doc[0]
        sx, sy = image.width / page.rect.width, image.height / page.rect.height
        for x0, y0, x1, y1, text, *_ in page.get_text('words'):
            for key, value in zip(out, (text, x0 * sx, y0 * sy, (x1 - x0) * sx, (y1 - y0) * sy, 1)):
                out[key].append(value)
    return out


def download() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for lang in ('eng', 'rus'):
        target = DATA_DIR / f'{lang}.traineddata'
        if target.is_file():
            continue
        temp = target.with_suffix('.part')
        try:
            with urlopen(f'https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/{lang}.traineddata',
                         timeout=60) as response, temp.open('wb') as dest:
                shutil.copyfileobj(response, dest)
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)
    print(f'Local OCR languages: {tessdata()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--download', action='store_true')
    args = ap.parse_args()
    if args.download:
        download()
    else:
        print(tessdata())
