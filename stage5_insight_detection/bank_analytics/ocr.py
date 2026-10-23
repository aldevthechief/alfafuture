"""Bank analytics PDFs are scans without a text layer: render each page with PyMuPDF and transcribe it with the
vision model (Qwen3-VL). One JSONL per PDF in data/analytics/ocr/, cached per page (needs LLM_API_KEY).
A page that times out twice is skipped (not cached, a rerun retries it); `--budget` stops before the spend (₽, the
proxy's `usage.cost`) would pass the limit. `--prompt ocr_page_full` + `--out-dir` writes a richer OCR beside the old one.
python -X utf8 stage5_insight_detection/bank_analytics/ocr.py [file.pdf ...]   (default: all PDFs in data/analytics/)
python -X utf8 stage5_insight_detection/bank_analytics/ocr.py --prompt ocr_page_full --out-dir data/analytics/ocr_full --budget 15
"""
import argparse, base64, difflib, hashlib, json, re, sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import pymupdf
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.data import DATA
from common.llm import ChatModels, JsonlCache, PromptLibrary

PROMPTS = PromptLibrary(Path(__file__).with_name('prompts'))
PDF_DIR = DATA / 'analytics'
OCR_DIR = PDF_DIR / 'ocr'
OCR_FULL_DIR = PDF_DIR / 'ocr_full'  # richer run with prompts/ocr_page_full.md: tables in full, [График] blocks


def looped(text: str) -> bool:
    lines = [ln for ln in text.splitlines() if len(ln.strip()) > 3]
    return bool(lines) and max(map(lines.count, set(lines))) >= 10


def title_score(paragraph: str, title: str) -> float:
    """How well some line of the paragraph starts the chart title (titles may wrap onto the next line)."""
    lines = paragraph.split('\n')
    return max(difflib.SequenceMatcher(None, ' '.join(lines[i:])[:len(title)], title).ratio() for i in range(len(lines)))


def merge(text: str, figures: list[tuple[str, str]]) -> str:
    """Each (title, description) goes after the paragraph that holds the title; figures without one go to the end."""
    paras, after, tail = re.split(r'\n\s*\n', text.strip()), {}, []
    for title, desc in figures:
        block = '[Рисунок] ' + re.sub(r'\n\s*\n', '\n', desc.strip())  # one paragraph = one fragment downstream
        best = max(range(len(paras)), key=lambda i: title_score(paras[i], title))
        (after.setdefault(best, []) if title_score(paras[best], title) > 0.6 else tail).append(block)
    return '\n\n'.join([x for i, p in enumerate(paras) for x in (p, *after.get(i, []))] + tail)



class BudgetExceeded(Exception):
    pass


class AnalyticsOCR:
    def __init__(self, model: BaseChatModel | None = None, dpi: int = 150, out_dir: Path = OCR_DIR, prompt: str = 'ocr_page',
                 budget: float | None = None, attempts: int = 2):
        # at temperature 0 Qwen3-VL loops on dense pages (one paragraph repeated until max_tokens and the 120 s timeout);
        # retries are ours (`attempts`), so a looping page costs two timeouts, not four
        model = model or ChatModels().vision().model_copy(update={'frequency_penalty': 0.2, 'max_retries': 0})
        self.model, self.dpi, self.out_dir, self.prompt = model, dpi, out_dir, prompt
        self.budget, self.attempts = budget, attempts
        self.cost, self.max_page_cost, self.skipped = 0.0, 0.0, []  # spend in ₽ this run; (pdf, page) given up on
        self._attempt = 0

    def transcribe(self, pdf: Path, only_pages: list[int] | None = None) -> list[str]:
        # Keep superseded OCR for inspection instead of reusing it for another PDF or prompt.
        self.out_dir.mkdir(parents=True, exist_ok=True)
        source = {'pdf_sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(),
                  'prompt_sha256': hashlib.sha256(PROMPTS.render(self.prompt).encode()).hexdigest(),
                  'dpi': self.dpi, 'model': getattr(self.model, 'model_name', type(self.model).__name__)}
        manifest = self.out_dir / f'{pdf.stem}.source.json'
        path = self.out_dir / f'{pdf.stem}.jsonl'
        previous = json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else None
        comparable = {k: v for k, v in (previous or {}).items() if k != 'legacy_unverified'}
        changed = previous is not None and comparable != source
        if changed:
            if path.exists():
                old_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
                path.replace(path.with_name(f'{pdf.stem}.{old_hash}.previous.jsonl'))
        source['legacy_unverified'] = bool(not changed and path.exists() and (
            previous is None or previous.get('legacy_unverified', False)))
        # Legacy files without provenance are still readable; newly generated caches are bound to their inputs.
        from common.news_corpus import write_json
        write_json(manifest, source)
        cache, out = JsonlCache(pdf.stem, folder=self.out_dir), []
        with pymupdf.open(pdf) as doc:
            total = len(doc)
        selected = set(only_pages) if only_pages is not None else set(range(1, total + 1))
        cached = sum(cache.get(str(n)) is not None for n in selected if 1 <= n <= total)
        print(f'OCR {pdf.name}: {len(selected)} pages selected, {cached} cached', flush=True)
        for n, png in enumerate(self.pages(pdf), 1):
            if only_pages is not None and n not in only_pages:
                out.append(cache.get(str(n)) or '')
                continue
            if cache.get(str(n)) is None:
                if self.budget is not None and self.cost + max(self.max_page_cost, 0.05) > self.budget:
                    raise BudgetExceeded(f'{pdf.name} p{n}: spent {self.cost:.2f} of {self.budget} ₽')
                label = f'OCR {pdf.name} p{n}/{total}'
                text = self.read_with_retry(png, label=label)
                if text is None:
                    self.skipped.append((pdf.name, n))
                    print(f'  {label}: skipped after {self.attempts} failed attempts; not cached', flush=True)
                    out.append('')
                    continue
                cache.put(str(n), text)
            out.append(cache.get(str(n)))
        return out

    def read_with_retry(self, png: bytes, *, label: str = 'OCR page') -> str | None:
        for attempt in range(self.attempts):
            self._attempt = attempt
            if self.budget is not None and self.cost + max(self.max_page_cost, 0.05) > self.budget:
                raise BudgetExceeded(f'{label}: spent {self.cost:.2f} of {self.budget} ₽')
            print(f'  {label}: attempt {attempt + 1}/{self.attempts} started', flush=True)
            try:
                text = self.read_page(png)
                if not isinstance(text, str):
                    raise ValueError(f'OCR response is {type(text).__name__}, expected text')
                if not text.strip():
                    raise ValueError('empty OCR response')
                if looped(text):
                    repetitions = max(Counter(ln for ln in text.splitlines() if len(ln.strip()) > 3).values())
                    raise ValueError(f'repeating OCR response: one line repeated {repetitions} times')
                print(f'  {label}: attempt {attempt + 1}/{self.attempts} OK ({len(text)} chars)', flush=True)
                return text
            except Exception as e:  # timeouts and proxy errors; the page is retried, then skipped
                next_step = 'retrying' if attempt + 1 < self.attempts else 'attempts exhausted'
                print(f'  {label}: attempt {attempt + 1}/{self.attempts} failed: '
                      f'{type(e).__name__}: {str(e)[:120]}; {next_step}', flush=True)
        return None

    def pages(self, pdf: Path) -> Iterator[bytes]:
        with pymupdf.open(pdf) as doc:
            for page in doc:
                zoom = min(self.dpi / 72, 2924 / max(page.rect.width, page.rect.height))
                yield page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes('png')

    def read_page(self, png: bytes) -> str:
        image = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(png).decode()}}
        messages = [HumanMessage(content=[{'type': 'text', 'text': PROMPTS.render(self.prompt)}, image])]
        if self._attempt:
            messages.append(HumanMessage('Повтори распознавание: верни полный текст без циклических повторов.'))
        answer = self.model.invoke(messages)
        cost = float((answer.response_metadata.get('token_usage') or {}).get('cost') or 0)
        self.cost += cost
        self.max_page_cost = max(self.max_page_cost, cost)
        if answer.response_metadata.get('finish_reason') == 'length':
            raise ValueError('OCR response truncated at token limit')
        return answer.content


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='*')
    ap.add_argument('--prompt', default='ocr_page', help='prompt name in prompts/')
    ap.add_argument('--out-dir', type=Path, default=OCR_DIR)
    ap.add_argument('--budget', type=float, default=None, help='stop before the spend of this run passes it, ₽')
    a = ap.parse_args()
    ocr = AnalyticsOCR(out_dir=a.out_dir, prompt=a.prompt, budget=a.budget)
    pdfs = [PDF_DIR / f for f in a.files] or sorted(PDF_DIR.glob('*.pdf'))
    try:
        for pdf in pdfs:
            print(pdf.name, len(ocr.transcribe(pdf)), f'pages, spent {ocr.cost:.2f} ₽', flush=True)
    except BudgetExceeded as e:
        print('budget reached:', e)
    print(f'spent {ocr.cost:.2f} ₽; skipped pages: {ocr.skipped or "none"}')
