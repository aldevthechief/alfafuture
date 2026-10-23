"""Compare retrieval methods for the candidate generators on the eval companies.
Positives: real (company -> news) pairs (stage3 golden_set/real_pairs.json) and synthetic needles (direct / chain).
For every method: each company's generators rank events; the generators' quotas (weights share of the budget, as in
candidate_generators) build the candidate set. Metrics: recall of positives at budget 200 / 800 events, the best rank
of a positive over the company's generators, hard negatives that got in.
python -X utf8 experiments/retrieval/evaluate.py [--emb experiments/retrieval/out] [--only bm25 frida ...]"""
import argparse, json, re, sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from common.data import DATA
from common.text import stems
from common.news_corpus import assert_experiment_corpus, load_pairs
from common.search import news_version
from stage4_news_relevance.plausibility.events import ensure_events

CORPUS_BINDING = HERE / 'corpus.sources.json'
assert_experiment_corpus(CORPUS_BINDING)

QS = json.loads((HERE / 'queries.json').read_text(encoding='utf-8'))['queries']
UQ = json.loads((HERE / 'query_index.json').read_text(encoding='utf-8'))
UQ_POS = {(q['reason'], q['query'], q['text']): i for i, q in enumerate(UQ)}
PAIRS = load_pairs(ROOT / 'stage3_news_search/golden_set/real_pairs.json',
                   pd.read_parquet(DATA / 'news' / 'news.parquet'))
NEEDLES = json.loads((ROOT / 'stage3_news_search/golden_set/synthetic_needles.json').read_text(encoding='utf-8'))
LABELLED: dict[str, list[int]] = {}  # filled by --labels from judge.py output


def load_labels(path: Path, min_grade: int = 2):
    for line in filter(None, path.read_text(encoding='utf-8').splitlines()):
        d = json.loads(line)
        u, e = d['key'].split('|')
        if d['value'] >= min_grade:
            LABELLED.setdefault(u, []).append(int(e))


def corpus():
    assert_experiment_corpus(CORPUS_BINDING)
    return _corpus(news_version())


@lru_cache(1)
def _corpus(version):
    """Same docs, same order as the Kaggle kernel: news.parquet rows, then needles and hard negatives."""
    df = pd.read_parquet(DATA / 'news' / 'news.parquet')
    ids = df['id'].astype(str).tolist()
    texts = (df.title.fillna('') + '. ' + df.title.fillna('') + '. ' + df.text.fillna('').str[:2000]).tolist()
    for u, ns in NEEDLES['needles'].items():
        for n in ns:
            ids.append(f'needle|{u}|{n["kind"]}'); texts.append(f"{n['title']}. {n['title']}. {n['text']}")
    for k, n in enumerate(NEEDLES['hard_negatives']):
        ids.append(f'neg|{k}'); texts.append(f"{n['title']}. {n['title']}. {n['text']}")
    ev = ensure_events()
    of = {str(i): e for e, ids_ in zip(ev.event, ev.news_ids) for i in ids_}
    nxt = ev.event.max() + 1
    event = []
    for i in ids:  # news outside events.parquet (filtered tariff tables) and needles: an event of their own
        event.append(of.get(i, nxt)); nxt += i not in of
    return ids, texts, np.array(event)


def company_queries():
    by = {}
    for q in QS.values():
        by.setdefault(q['company'], []).append((UQ_POS[(q['reason'], q['query'], q['text'])], q['weight']))
    return by


# ---------- lexical methods (local) ----------
def _snowball():
    import Stemmer
    st = Stemmer.Stemmer('russian')
    return lambda t: st.stemWords(re.findall(r'[а-яёa-z0-9]{2,}', t.lower()))


@lru_cache(None)
def _lemmatizer():
    import pymorphy3
    m = pymorphy3.MorphAnalyzer()
    cache = {}

    def lem(t):
        out = []
        for w in re.findall(r'[а-яёa-z0-9]{2,}', t.lower()):
            if w not in cache:
                cache[w] = m.parse(w)[0].normal_form
            out.append(cache[w])
        return out
    return lem


def tfidf(tok, field):
    from sklearn.feature_extraction.text import TfidfVectorizer
    _, texts, _ = corpus()
    v = TfidfVectorizer(analyzer=tok, min_df=2, max_df=0.3, sublinear_tf=True)
    X = v.fit_transform(texts)
    return (X @ v.transform([q[field] for q in UQ]).T).toarray()


def bm25(tok, field):
    import bm25s
    _, texts, _ = corpus()
    toks = [tok(t) for t in texts]
    vocab = {w: i for i, w in enumerate(sorted({w for ts in toks for w in ts}))}
    r = bm25s.BM25()
    r.index(bm25s.tokenization.Tokenized(ids=[[vocab[w] for w in ts] for ts in toks], vocab=vocab), show_progress=False)
    S = np.zeros((len(texts), len(UQ)), np.float32)
    for j, q in enumerate(UQ):
        qt = [vocab[w] for w in tok(q[field]) if w in vocab]
        if qt:
            S[:, j] = r.get_scores(qt)
    return S


LEXICAL = {
    'tfidf-stem6/query': lambda: tfidf(stems, 'query'),
    'tfidf-snowball/query': lambda: tfidf(_snowball(), 'query'),
    'tfidf-snowball/text': lambda: tfidf(_snowball(), 'text'),
    'bm25-snowball/query': lambda: bm25(_snowball(), 'query'),
    'bm25-snowball/text': lambda: bm25(_snowball(), 'text'),
    'bm25-lemma/query': lambda: bm25(_lemmatizer(), 'query'),
}


# ---------- encoder outputs from Kaggle ----------
def kaggle_methods(folder: Path):
    ids, _, _ = corpus()
    out = {}
    for f in sorted(folder.glob('*.npz')):
        if f.name.endswith('_doc.npz') or '_q_' in f.name:
            continue
        z = np.load(f, allow_pickle=False)
        assert list(z['doc_ids']) == ids, f'{f.name}: doc order differs from the local corpus'
        D = z['doc'].astype(np.float32)
        for k in z.files:
            if k.startswith('q_'):
                out[f'{f.stem}/{k[2:]}'] = (lambda D=D, Q=z[k].astype(np.float32): D @ Q.T)
    for f in sorted(folder.glob('*_doc.npz')):
        slug = f.name[:-len('_doc.npz')]
        assert list(np.load(folder / f'{slug}_doc_ids.npy')) == ids
        D = sp.load_npz(f)
        for q in sorted(folder.glob(f'{slug}_q_*.npz')):
            v = q.stem[len(slug) + 3:]
            out[f'{slug}/{v}'] = (lambda D=D, q=q: (D @ sp.load_npz(q).T).toarray())
    return out


# ---------- evaluation ----------
def event_ranks(S, event):
    """S: docs x queries -> per query an array of event ids sorted by max doc score."""
    df = pd.DataFrame(S)
    df['e'] = event
    E = df.groupby('e').max()
    return E.index.to_numpy(), E.to_numpy()


def evaluate(S, budgets=(200, 800)):
    ids, _, event = corpus()
    pos_of_id = {i: k for k, i in enumerate(ids)}
    evid, E = event_ranks(S, event)
    order = np.argsort(-E, axis=0)  # rows of evid per query, best first
    rank_of = np.empty_like(order)
    rank_of[order, np.arange(E.shape[1])] = np.arange(E.shape[0])[:, None]
    row_of_event = {e: r for r, e in enumerate(evid)}
    targets = {u: [str(i) for i in v] for u, v in PAIRS.items()}
    for u, ns in NEEDLES['needles'].items():
        for n in ns:
            targets.setdefault(f'{u}#{n["kind"]}', []).append(f'needle|{u}|{n["kind"]}')
    negs = [row_of_event[event[pos_of_id[f'neg|{k}']]] for k in range(len(NEEDLES['hard_negatives']))]
    cq = company_queries()
    rows = []
    for u, evs in LABELLED.items():  # pooled LLM labels: grade 2 events are positives (kind 'pooled')
        targets[f'{u}#pooled'] = [('event', e) for e in evs]
    for key, tids in targets.items():
        u, kind = (key.split('#') + ['real'])[:2]
        if u not in cq:
            continue
        qi = np.array([q for q, _ in cq[u]]); w = np.array([w for _, w in cq[u]])
        prow = [row_of_event[t[1]] if isinstance(t, tuple) else row_of_event[event[pos_of_id[t]]]
                for t in tids if isinstance(t, tuple) or t in pos_of_id]
        if not prow:
            continue
        best = rank_of[np.ix_(prow, qi)].min(1)
        r = {'company': u, 'kind': kind, 'n_pos': len(prow), 'best_rank_median': float(np.median(best)),
             'best_rank_le50': float((best < 50).mean())}
        for b in budgets:
            hi = np.maximum(1, np.round(b * w / w.sum())).astype(int)
            chosen = set()
            for q, h in zip(qi, hi):
                chosen.update(order[:h, q].tolist())
            r[f'recall@{b}'] = float(np.mean([p in chosen for p in prow]))
            r[f'size@{b}'] = len(chosen)
            if kind == 'direct':
                r[f'neg_in@{b}'] = float(np.mean([n in chosen for n in negs]))
        rows.append(r)
    return pd.DataFrame(rows)


def summarize(name, res):
    g = res.groupby('kind')
    s = {'method': name}
    for kind in ('pooled', 'real', 'direct', 'chain'):
        if kind in g.groups:
            x = g.get_group(kind)
            s |= {f'{kind} R@200': x['recall@200'].mean(), f'{kind} R@800': x['recall@800'].mean(),
                  f'{kind} best≤50': x['best_rank_le50'].mean()}
    s['size@800'] = res['size@800'].mean()
    return s


def rrf(*Ss, k=60):
    out = 0
    for S in Ss:
        r = np.argsort(np.argsort(-S, axis=0), axis=0)
        out = out + 1 / (k + r)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--emb', default=str(HERE / 'out'))
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--hybrid', nargs='*', default=[], help='pairs "a+b" of method names to fuse with RRF')
    ap.add_argument('--labels', help='labels.jsonl from judge.py: grade-2 events become the "pooled" positives')
    a = ap.parse_args()
    if a.labels:
        load_labels(Path(a.labels))
    every = dict(LEXICAL) | (kaggle_methods(Path(a.emb)) if Path(a.emb).exists() else {})
    methods = {k: v for k, v in every.items() if any(o in k for o in a.only)} if a.only is not None else every
    cache, rows = {}, []
    for name, f in methods.items():
        cache[name] = f()
        rows.append(summarize(name, evaluate(cache[name])))
        print(pd.DataFrame(rows[-1:]).round(2).to_string(index=False, header=len(rows) == 1), flush=True)
    for h in a.hybrid:
        parts = h.split('+')
        S = rrf(*[cache[p] if p in cache else every[p]() for p in parts])
        rows.append(summarize(h, evaluate(S)))
    res = pd.DataFrame(rows).round(3)
    old = HERE / ('results_pooled.csv' if a.labels else 'results.csv')  # merge with earlier runs, newest row per method wins
    if old.exists():
        res = pd.concat([pd.read_csv(old), res]).drop_duplicates('method', keep='last')
    res.to_csv(old, index=False)
    print(res.sort_values('real R@800', ascending=False).to_string(index=False))
