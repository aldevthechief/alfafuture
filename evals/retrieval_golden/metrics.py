"""Ranking metrics for one retriever on one golden set. A retriever returns an ordered list of news ids; the labelled items
(positives = must be found, negatives = must not) are hidden among thousands of real, unlabelled articles.
- recall@k            positives inside the first k of the whole list: «does the reader's top contain what he must see»
- fpr@k               negatives inside the first k / all negatives: «does the top carry noise that looks relevant»
- auc                 P(a random positive is ranked above a random negative); an item outside the list ranks last, ties = 0.5
- ndcg@k / map / mrr  on the labelled items only, in the order the retriever put them: unlabelled real news cannot be graded
  (it may be relevant), so they do not take part; this measures the ORDER of what we know, recall@k the COVERAGE
- <stratum>@k         share of a stratum inside the first k (positives that name the industry vs not, hard vs easy negatives):
  a lexical retriever should win the first and lose the second
Everything is computed from the rank of each labelled item, so the bootstrap over items (a set of 10 + 10 per company is small:
the interval is part of the number) is a resample of (rank, label) pairs."""
import math

import numpy as np


def ranks_of(ranking: list, ids) -> dict:
    """Rank (1-based) of every labelled id in the returned list; an id the retriever did not return ranks last (len + 1)."""
    pos = {n: r for r, n in enumerate(ranking, 1)}
    return {i: pos.get(i, len(ranking) + 1) for i in ids}


def dcg(gains: list[float], k: int) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def evaluate_ranks(rank: dict, labels: dict, ks: tuple = (15, 30, 100), strata: dict | None = None) -> dict:
    pos, neg = [i for i in labels if labels[i] == 1], [i for i in labels if labels[i] == 0]
    out = dict(n_pos=len(pos), n_neg=len(neg))
    out['auc'] = (sum((rank[n] > rank[p]) + 0.5 * (rank[n] == rank[p]) for p in pos for n in neg) / (len(pos) * len(neg))
                  if pos and neg else None)
    for k in ks:
        out[f'recall@{k}'] = sum(rank[p] <= k for p in pos) / len(pos) if pos else None
        out[f'fpr@{k}'] = sum(rank[n] <= k for n in neg) / len(neg) if neg else None
    order = [labels[i] for i in sorted(labels, key=lambda i: rank[i])]  # the labelled items in the retriever's order
    top = ks[0]
    ideal = sorted(order, reverse=True)
    out[f'ndcg@{top}'] = dcg(order, top) / dcg(ideal, top) if dcg(ideal, top) else None
    hits = [i + 1 for i, y in enumerate(order) if y == 1]
    out['map'] = float(np.mean([(j + 1) / h for j, h in enumerate(hits)])) if hits else None
    out['mrr'] = 1 / hits[0] if hits else None
    for name in sorted(set((strata or {}).values())):
        ids = [i for i in labels if (strata or {}).get(i) == name]
        out[f'{name}@{top}'] = sum(rank[i] <= top for i in ids) / len(ids) if ids else None
    return out


def evaluate(ranking: list, labels: dict, ks: tuple = (15, 30, 100), strata: dict | None = None) -> dict:
    """labels: news id -> 1 / 0. strata: news id -> a name."""
    return dict(evaluate_ranks(ranks_of(ranking, labels), labels, ks, strata), returned=len(ranking))


def bootstrap(rank: dict, labels: dict, metric: str = 'auc', n: int = 1000, seed: int = 0, **kw) -> tuple[float, float]:
    """95% interval of one metric when the labelled items are resampled (positives and negatives apart)."""
    rng = np.random.default_rng(seed)
    pos, neg = [i for i in labels if labels[i] == 1], [i for i in labels if labels[i] == 0]
    values = []
    for _ in range(n):
        picks = list(rng.choice(pos, len(pos))) + list(rng.choice(neg, len(neg)))
        keys = {f'{i}#{k}': i for k, i in enumerate(picks)}  # duplicates need distinct keys
        v = evaluate_ranks({key: rank[i] for key, i in keys.items()}, {key: labels[i] for key, i in keys.items()}, **kw).get(metric)
        if v is not None:
            values.append(v)
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))
