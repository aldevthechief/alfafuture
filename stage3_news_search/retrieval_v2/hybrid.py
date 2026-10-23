"""Hybrid event search of retrieval v2, chosen in experiments/retrieval (R@800 0.93, best of 6 lexical + 8 encoders):
e5-large-instruct + qwen3-emb-0.6b (`Instruct: <instruction>\\nQuery: <query>`) + BM25 over pymorphy3 lemmas, RRF (k=60)
over every (method, query) ranking; an event scores as its best news.
Document vectors are cached by text hash in data/retrieval_v2/<model>.npz, so news ids may change (url-hash migration)
and only new texts are encoded. A first full encode is slow on a small GPU: seed the cache from the Kaggle vectors
python -X utf8 stage3_news_search/retrieval_v2/hybrid.py --seed experiments/retrieval/out"""
import argparse, hashlib, re, sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.data import DATA

CACHE = DATA / 'retrieval_v2'
MODELS = {'e5-large-instruct': ('intfloat/multilingual-e5-large-instruct', 'Query: ', None),
          'qwen3-emb-0.6b': ('Qwen/Qwen3-Embedding-0.6B', 'Query:', 'left')}  # slug: (HF name, query separator, padding)
RRF_K = 60


def doc_texts(news: pd.DataFrame) -> list[str]:
    """As the Kaggle kernel (experiments/retrieval/kaggle_embed.py): its vectors stay valid for the cache."""
    return (news.title.fillna('') + '. ' + news.title.fillna('') + '. ' + news.text.fillna('').str[:2000]).tolist()


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:16]


@lru_cache(None)
def _encoder(slug: str):
    import torch
    from sentence_transformers import SentenceTransformer
    name, _, pad = MODELS[slug]
    cuda = torch.cuda.is_available()
    m = SentenceTransformer(name, device='cuda' if cuda else 'cpu', model_kwargs={'torch_dtype': torch.float16} if cuda else None,
                            tokenizer_kwargs={'padding_side': pad} if pad else None)
    m.max_seq_length = 512
    return m


def encode(slug: str, texts: list[str], batch_size: int = 16) -> np.ndarray:
    return _encoder(slug).encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=len(texts) > 200).astype(np.float32)


def _load_cache(slug: str) -> dict[str, np.ndarray]:
    path = CACHE / f'{slug}.npz'
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as z:
        return dict(zip(z['hash'].tolist(), z['emb']))


def _save_cache(slug: str, cache: dict[str, np.ndarray]) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = CACHE / f'{slug}.tmp.npz'
    np.savez(tmp, hash=np.array(list(cache)), emb=np.stack(list(cache.values())).astype(np.float16))
    tmp.replace(CACHE / f'{slug}.npz')  # atomic: a crash mid-write never leaves a broken cache


def doc_vectors(slug: str, texts: list[str], encoder=encode) -> np.ndarray:
    cache = _load_cache(slug)
    hashes = [text_hash(t) for t in texts]
    missing = list(dict.fromkeys(h for h in hashes if h not in cache))
    if missing:
        by_hash = dict(zip(hashes, texts))
        for h, v in zip(missing, encoder(slug, [by_hash[h] for h in missing])):
            cache[h] = v
        _save_cache(slug, cache)
    return np.stack([cache[h] for h in hashes]).astype(np.float32)


@lru_cache(None)
def _morph():
    import pymorphy3
    return pymorphy3.MorphAnalyzer()


@lru_cache(2 ** 20)
def _lemma(word: str) -> str:
    return _morph().parse(word)[0].normal_form


def lemmas(text: str) -> list[str]:
    return [_lemma(w) for w in re.findall(r'[а-яёa-z0-9]{2,}', text.lower())]


class BM25:
    def __init__(self, texts: list[str]):
        import bm25s
        toks = [lemmas(t) for t in texts]
        self.vocab = {w: i for i, w in enumerate(sorted({w for ts in toks for w in ts}))}
        self.index = bm25s.BM25()
        self.index.index(bm25s.tokenization.Tokenized(ids=[[self.vocab[w] for w in ts] for ts in toks], vocab=self.vocab),
                         show_progress=False)

    def __call__(self, query: str) -> np.ndarray:
        return self.index.get_scores([self.vocab[w] for w in lemmas(query) if w in self.vocab] or [0])


def rank(scores: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(-scores))


class HybridSearch:
    """news: the pool rows (id, title, text); events: stage-4 clusters (event, news_ids, title, first_date).
    News outside every event are not searched. encoder(slug, texts) -> normalized vectors (tests pass a fake)."""

    def __init__(self, news: pd.DataFrame, events: pd.DataFrame, encoder=None):
        of = {str(i): int(e) for e, ids in zip(events.event, events.news_ids) for i in ids}
        event = np.array([of.get(str(i), -1) for i in news['id']])
        keep = event >= 0
        self.news, self.event = news[keep].reset_index(drop=True), event[keep]
        self.events = events.set_index('event')
        self.encoder = encoder or encode
        texts = doc_texts(self.news)
        # everything that changes a ranking: searched texts, event membership, models, fusion (the factory cache keys on it)
        self.fingerprint = hashlib.sha1('\x00'.join(
            texts + [f'{i}:{e}' for i, e in zip(self.news['id'].astype(str), self.event)]
            + [repr(sorted(MODELS.items())), str(RRF_K)]).encode('utf-8')).hexdigest()
        self.docs = {s: doc_vectors(s, texts, self.encoder) for s in MODELS}
        self.bm25 = BM25(texts)

    def scores(self, specs: list[tuple[str, list[str]]]) -> np.ndarray:
        """news × spec: sum of 1/(k + rank) over methods × the spec's queries. One encoder call per model for all
        specs: a call per query costs ~0.3 s of GPU sync."""
        owner = [j for j, (_, qs) in enumerate(specs) for _ in qs]
        fused = np.zeros((len(self.news), len(specs)))
        for slug, (_, sep, _) in MODELS.items():
            Q = self.encoder(slug, [f'Instruct: {ins}\n{sep}{q}' for ins, qs in specs for q in qs])
            ranks = np.argsort(np.argsort(-(self.docs[slug] @ Q.T), axis=0), axis=0)
            for col, j in enumerate(owner):
                fused[:, j] += 1 / (RRF_K + ranks[:, col])
        for j, q in zip(owner, [q for _, qs in specs for q in qs]):
            fused[:, j] += 1 / (RRF_K + rank(self.bm25(q)))
        return fused

    def rank_events_many(self, specs: list[tuple[str, list[str]]], k: int = 300) -> list[list[int]]:
        """specs: (encoder instruction, queries); per spec its top-k events, best first."""
        ev = pd.DataFrame(self.scores(specs)).groupby(self.event).max()
        return [[int(e) for e in ev[j].sort_values(ascending=False).index[:k]] for j in range(len(specs))]

    def rank_events(self, instruction: str, queries: list[str], k: int = 300) -> list[int]:
        return self.rank_events_many([(instruction, queries)], k)[0]


def seed(folder: Path) -> None:
    """Kaggle vectors (<slug>.npz: doc_ids, doc) of the news.parquet they were built from → the text-hash cache."""
    df = pd.read_parquet(DATA / 'news' / 'news.parquet')
    text_of = dict(zip(df['id'].astype(str), doc_texts(df)))
    for slug in MODELS:
        with np.load(folder / f'{slug}.npz', allow_pickle=False) as z:
            pairs = [(text_of[i], v) for i, v in zip(z['doc_ids'].tolist(), z['doc']) if i in text_of]
        cache = _load_cache(slug) | {text_hash(t): v for t, v in pairs}
        _save_cache(slug, cache)
        print(f'{slug}: {len(pairs)} vectors seeded, cache {len(cache)}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=Path, required=True, help='folder with the Kaggle <slug>.npz files')
    seed(ap.parse_args().seed)
