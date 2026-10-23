"""Step 5b: search the bank's analytics (OCR output of ocr.py) for the bank's view on a topic.

The parsed reports are cut into chunks (chunks.py: recursive, ~300 e5 tokens, overlap) and searched like the news
(experiments/retrieval): e5-large-instruct over «report title + chunk» by meaning + BM25 on Snowball stems by words, fused
with weighted reciprocal rank fusion (k = 60, 0.8 meaning / 0.2 words); vectors are cached next to the OCR. The e5
instruction goes with the query only, so it can be changed per call without re-indexing. Every hit carries report, date
and page, so the insight can cite it. Self-check rules from the case: the freshest report wins among near-equal hits,
and reports older than a year are flagged `stale` (the 2021 steel report is a trap).
python -X utf8 stage5_insight_detection/bank_analytics/search.py "ключевая ставка прогноз"
"""
import argparse, hashlib, json, re, sys
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stage5_insight_detection.bank_analytics.ocr import OCR_DIR

REPORTS = json.loads(Path(__file__).with_name('reports.json').read_text(encoding='utf-8'))['reports']
STALE_DAYS = 365
E5 = 'intfloat/multilingual-e5-large-instruct'
# e5's own default instruction and an equal split: best on the parser's matching set (search_eval.md, MRR 0.59 vs 0.27
# for the Russian task instruction at 0.8); the Russian task instructions hurt the e5 side
TASK = 'Given a web search query, retrieve relevant passages that answer the query'
W_DENSE = 0.8  # share of the e5 ranking in the fusion; BM25 gets the rest. 0.5 was best on the uncleaned index (noise helps BM25);
#               on the v1 sources recall@10 peaks at 0.8 (0.78), MRR keeps rising to 1.0 (search_eval.md)
DISCLAIMER = re.compile(r'Все права защищены|Генеральная лицензия ЦБ РФ|предназначен АО «Альфа-Банк»')  # legal boilerplate on every report


from stage5_insight_detection.bank_analytics.coverage import index_missing, uncovered
from stage5_insight_detection.bank_analytics.twins import find_twins
from stage5_insight_detection.bank_analytics.chunks import CHUNK_TOKENS, OVERLAP_TOKENS, build_chunks, passages
from stage5_insight_detection.bank_analytics.fragments import (CAPTION, CHART_META, LEGAL, TABLE_CHART, prose_reason,  # noqa: F401
                                                                     repair_headers, table_has_header, use_fragment)


def paragraphs(text: str, min_chars: int = 200) -> list[str]:
    """Blank-line paragraphs, short ones glued to the next so a heading stays with its body."""
    out, buf = [], ''
    for p in re.split(r'\n\s*\n', text):
        buf = f'{buf}\n{p}'.strip()
        if len(buf) >= min_chars:
            out.append(buf)
            buf = ''
    return out + [buf] if buf else out


@lru_cache(None)
def encoder():
    import torch
    from sentence_transformers import SentenceTransformer
    cuda = torch.cuda.is_available()
    return SentenceTransformer(E5, device='cuda' if cuda else 'cpu', model_kwargs={'torch_dtype': torch.float16} if cuda else {})


def embed(texts: list[str], folder: Path) -> np.ndarray:
    """Fragment vectors, one cache file per set of texts (another selection of fragments does not evict the others)."""
    cache = folder / f'.emb_e5_{hashlib.sha1("|".join(texts).encode()).hexdigest()[:12]}.npz'
    if cache.exists():
        return np.load(cache)['emb']
    emb = encoder().encode(texts, batch_size=8, normalize_embeddings=True).astype(np.float32)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(cache, emb=emb)
    return emb


def bm25_index(texts: list[str]):
    import bm25s, Stemmer
    stem = Stemmer.Stemmer('russian')
    tok = lambda t: stem.stemWords(re.findall(r'[а-яёa-z0-9]{2,}', t.lower()))
    docs = [tok(t) for t in texts]
    vocab = {w: i for i, w in enumerate(sorted({w for d in docs for w in d}))}
    r = bm25s.BM25()
    r.index(bm25s.tokenization.Tokenized(ids=[[vocab[w] for w in d] for d in docs], vocab=vocab), show_progress=False)
    def scores(q):
        ids = [vocab[w] for w in tok(q) if w in vocab]
        return r.get_scores(ids) if ids else np.zeros(len(docs))  # no shared word: lexical side abstains
    return scores


def ocr_rows(ocr_dir: Path, reports: list[dict]) -> list[dict]:
    """Without the parser's fragments.jsonl: the OCR pages of the reports as pseudo-fragments (blank-line paragraphs)."""
    newest = max(date.fromisoformat(r['date']) for r in reports)
    rows = []
    for r in reports:
        path = ocr_dir / f'{Path(r["file"]).stem}.jsonl'
        if not path.exists():
            continue
        stale = (newest - date.fromisoformat(r['date'])).days > STALE_DAYS
        for line in filter(None, path.read_text(encoding='utf-8').splitlines()):
            page = json.loads(line)
            rows += [{'id': f'bank:{Path(r["file"]).stem}:p{page["key"]}.{n}', 'report': r['title'], 'date': r['date'], 'page': int(page['key']),
                      'freshness': 'stale' if stale else 'fresh', 'kind': 'text', 'text': p}
                     for n, p in enumerate(paragraphs(page['value']), 1)]
    return rows


class BankAnalytics:
    """Search over the bank's analytics: chunks of ~300 e5 tokens (chunks.py), e5 over «report title + chunk» fused with BM25.
    Every chunk is the bank's own analytics as of the report date: numbers about periods before that date are statistics,
    everything after it is the bank's view. `passages()` adds the neighbours (the agent's tool)."""

    def __init__(self, ocr_dir: Path = OCR_DIR, reports: list[dict] = REPORTS, keep=None, chunk_tokens: int = CHUNK_TOKENS,
                 overlap_tokens: int = OVERLAP_TOKENS, index_new: bool = True):
        """Reads data/analytics/fragments.jsonl (next to the OCR folder) when it exists, else the OCR pages. `keep` picks the
        fragments (default use_fragment = the v1 sources); chunk_tokens = 0: one chunk per fragment (no chunking)."""
        if (frag := ocr_dir.parent / 'fragments.jsonl').exists():
            rows = [json.loads(l) for l in frag.read_text(encoding='utf-8').splitlines() if l.strip()]
            if index_new and (missing := uncovered(frag.parent, rows)):  # a PDF dropped into the folder: index only the files not covered yet
                rows = index_missing(frag.parent, ocr_dir, rows, missing)
            rows = repair_headers(rows)
            keep = keep or use_fragment
        elif any(ocr_dir.parent.glob('*.pdf')):  # mvp: the PDF folder is prepared by BankLibrary (text layer / OCR, tables, derivatives)
            from dataclasses import asdict
            from stage5_insight_detection.bank_analytics.library import BankLibrary
            folder = ocr_dir.parent
            lib = BankLibrary.refresh(pdf_dir=folder, ocr_dir=ocr_dir, full_dir=folder / 'ocr_full', out=folder / 'fragments.jsonl', registry=folder / 'registry.json')
            rows, keep = repair_headers([asdict(f) | {'date': f.date or ''} for f in lib.fragments]), keep or use_fragment
        else:
            rows, keep = ocr_rows(ocr_dir, reports), keep or (lambda f: not LEGAL.search(f['text']))
        self.rows = rows  # every parsed fragment (all kinds, file order): the agent reads whole pages from here
        tok = encoder().tokenizer
        tok.model_max_length = 10 ** 6  # lengths of long streams are measured, never fed to the model
        self.chunks, self.segments = build_chunks(rows, lambda t: len(tok.encode(t, add_special_tokens=False)), chunk_tokens,
                                                  overlap_tokens, keep)
        if not self.chunks:
            raise FileNotFoundError(f'no OCR output in {ocr_dir}; run stage5_insight_detection/bank_analytics/ocr.py first')
        have = {c['report'] for c in self.chunks}
        self.available = [f'{r["title"]} ({r["date"]})' for r in reports if r['title'] in have]
        self.missing = [f'{r["title"]} ({r["date"]})' for r in reports if r['title'] not in have]
        self.bm25 = bm25_index([c['text'] for c in self.chunks])
        self.D = embed([f'{c["report"]}\n{c["text"]}' for c in self.chunks], ocr_dir)  # the title opens every indexed text
        self.twins = find_twins(self.chunks, self.D)  # table chunk -> the same table in other reports (twins.py)
        self.index = {c['id']: i for i, c in enumerate(self.chunks)}

    def _top(self, query: str, k: int, task: str, w_dense: float) -> list[tuple[int, float]]:
        q = encoder().encode([f'Instruct: {task}\nQuery: {query}'], normalize_embeddings=True)[0].astype(np.float32)
        rank = lambda s: np.argsort(np.argsort(-s))
        fused = w_dense / (60 + rank(self.D @ q)) + (1 - w_dense) / (60 + rank(self.bm25(query)))
        top = [(int(i), round(float(fused[i]), 4)) for i in np.argsort(-fused)[:3 * k]]
        top.sort(key=lambda x: (not self.chunks[x[0]]['stale'], x[1], self.chunks[x[0]]['date']), reverse=True)  # stale last; equal: fresher first
        return top[:k]

    def search(self, query: str, k: int = 5, task: str = TASK, w_dense: float = W_DENSE) -> list[dict]:
        return [dict(self.chunks[i], score=sc, pos=i) for i, sc in self._top(query, k, task, w_dense)]

    def passages(self, query: str, k: int = 3, context: int = 2, task: str = TASK, w_dense: float = W_DENSE) -> list[dict]:
        """Top-k chunks with `context` chunks above and below each; overlapping windows merged (chunks.passages)."""
        top = self._top(query, k, task, w_dense)
        score = {self.chunks[i]['id']: sc for i, sc in top}
        out = passages(self.chunks, self.segments, [i for i, _ in top], context)
        for p in out:
            p['score'] = max(score[i] for i in p['ids'])
        return sorted(out, key=lambda p: (not p['stale'], p['score']), reverse=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('query')
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--context', type=int, default=2)
    a = ap.parse_args()
    for p in BankAnalytics().passages(a.query, a.k, a.context):
        print('--- {} «{}» {} p.{}{}  hits {}'.format(p['score'], p['report'], p['date'], p['pages'], ' STALE' if p['stale'] else '', p['ids']))
        print(p['text'] + chr(10))
