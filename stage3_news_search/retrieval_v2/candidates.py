"""Retrieval v2 = standard generators + generator factory over one hybrid search (experiments/candidate_sets «v1»):
- standard: stage-3 `candidate_generators.generators(facts)` (one per stage-2 fact reason, reason weights), each ranked by
  the hybrid with its reason's encoder instruction; quota union ∝ weight at the smallest budget giving >= 200 events;
- factory: top-13 events of every generator the factory agent picked (factory.py, cached per company);
- pool = union (~330–380 events); an event keeps every generator that found it with its rank there.
Order (what a top-K consumer such as pipeline.py --candidates sees): priority = Σ over the generators that found the event of
weight / (1 + rank); the factory's generators share 70% of the weight equally, the standard ones 30% by reason weight.
On judge-v2 labels this beats ordering by best rank (positives in the top 30 of 4 companies: 17 vs 7; eval_labels.py).
python -X utf8 stage3_news_search/retrieval_v2/candidates.py U_106783 [--role cfo] [--no-factory] [--show 30]"""
import argparse, hashlib, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stage3_news_search.candidate_generators.generators import generators as standard_generators
from stage3_news_search.retrieval_v2 import factory
from stage3_news_search.retrieval_v2.hybrid import HybridSearch, doc_texts

STANDARD_BUDGET = 200
FACTORY_SHARE = 0.5  # of the ordering weight and of the judge seats (quota_pick); was 0.7 tuned on 4 companies, 0.5 since 24.10
REASON_INSTRUCTIONS = {  # as experiments/retrieval/push.py, where they won over generic instructions
    'own_industry': 'Найди новости о регулировании, ценах, спросе и конкуренции в этой отрасли',
    'supplier_industry': 'Найди новости о событиях, сильно влияющих на эту отрасль поставщиков компании: цены, дефицит, регулирование, сбои',
    'customer_industry': 'Найди новости о событиях, сильно влияющих на эту отрасль покупателей компании: спрос, финансы, регулирование',
    'second_hop': 'Найди новости о событиях в этой отрасли, которые могут пройти по цепочке поставок до компании',
    'group_industry': 'Найди новости о событиях в этой отрасли, важных для группы компаний',
    'bank_product': 'Найди новости об условиях и изменениях по этому банковскому продукту или фактору для бизнеса',
    'macro': 'Найди новости о макроэкономическом факторе, влияющем на бизнес компаний',
    'trade_country': 'Найди новости о торговле, пошлинах, санкциях и логистике с этой страной',
    'region': 'Найди новости об экономике и бизнесе этого региона',
    'hypothesis': 'Найди новости, важные для бизнеса с такой характеристикой',
}
_SEARCH: dict[str, HybridSearch] = {}


def hybrid_search(news: pd.DataFrame, events: pd.DataFrame) -> HybridSearch:
    """One search per corpus (texts + event membership) and process: the BM25 index takes a while to build."""
    key = hashlib.sha1('\x00'.join(doc_texts(news) + list(news['id'].astype(str))
                                   + [str(list(ids)) for ids in events.news_ids]).encode('utf-8')).hexdigest()
    if key not in _SEARCH:
        _SEARCH.clear()
        _SEARCH[key] = HybridSearch(news, events)
    return _SEARCH[key]


def quotas(weights: list[float], tops: list[list[int]], budget: int) -> list[int]:
    """Per generator: events to take ∝ weight, at the smallest total whose union holds >= budget events."""
    w = np.array(weights)
    for b in range(budget, 20 * budget, 5):
        h = np.maximum(1, np.round(b * w / w.sum())).astype(int)
        if len(set().union(*(t[:k] for t, k in zip(tops, h)))) >= budget:
            break
    return h.tolist()


def candidates(company_id: str, facts: list[dict], role: str | None = None, *, news: pd.DataFrame | None = None,
               events: pd.DataFrame | None = None, model=None, budget: int = STANDARD_BUDGET, quota: int = factory.QUOTA,
               use_factory: bool = True, frozen_factory: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    """events rows (event, news_ids, n_sources, title, …) + `generators` [{source, reason, query, rank, …}] + `best_rank`,
    and the generators used. news / events default to the pool and its stage-4 events (built if missing or stale).
    frozen_factory: use the stored factory spec as is, never run the agent (evaluation on old labels)."""
    if news is None:
        from common.search import news_pool
        news = news_pool()
    if events is None:
        from stage4_news_relevance.plausibility.events import ensure_events
        events = ensure_events(news)
    search = hybrid_search(news, events)
    std = standard_generators(facts, role=role)
    tops = search.rank_events_many([(REASON_INSTRUCTIONS[g.reason], [g.query]) for g in std]) if std else []
    gens = [{'source': 'standard', 'reason': g.reason, 'query': g.query[:80], 'weight': g.weight, 'top': t} for g, t in zip(std, tops)]
    picked: dict[int, list[dict]] = {}
    for g, k in zip(gens, quotas([g['weight'] for g in gens], [g['top'] for g in gens], budget) if gens else []):
        for r, e in enumerate(g['top'][:k]):
            picked.setdefault(e, []).append({'source': 'standard', 'reason': g['reason'], 'query': g['query'], 'rank': r})
    spec = factory.factory_spec(company_id, facts, search, model, frozen=frozen_factory) if use_factory else None
    if use_factory and spec is None:
        print(f'warning: generator factory gave no finish for {company_id}; standard generators only', file=sys.stderr)
    fgens = factory.generators(spec) if spec else []
    ftops = search.rank_events_many([(g['instruction'], g['queries']) for g in fgens], quota) if fgens else []
    for g, top in zip(fgens, ftops):
        g |= {'source': 'factory', 'query': ' ‖ '.join(g['queries'])[:80], 'top': top}
        gens.append(g)
        for r, e in enumerate(g['top']):
            picked.setdefault(e, []).append({'source': 'factory', 'reason': g['reason'], 'channel': g['channel'],
                                             'query': g['query'], 'rank': r})
    return order(search.events, picked, gens, SHARES), gens


SHARES = {'factory': FACTORY_SHARE, 'standard': 1 - FACTORY_SHARE}


def generator_weights(gens: list[dict], shares: dict[str, float]) -> dict[tuple[str, str], float]:
    """(source, query) -> share of the ordering weight: a source's share is split among its generators by `weight` (equal
    when absent, as the factory's); the shares of the sources present are renormalised (no factory -> standard takes all)."""
    present = {g['source'] for g in gens}
    total = sum(shares[s] for s in present)
    w_src = {s: sum(g.get('weight', 1.0) for g in gens if g['source'] == s) for s in present}
    return {(g['source'], g['query']): shares[g['source']] / total * g.get('weight', 1.0) / w_src[g['source']] for g in gens}


def quota_pick(pool: pd.DataFrame, gens: list[dict], n: int, shares: dict[str, float] = SHARES) -> pd.DataFrame:
    """n distinct events for the insight attempt with a quota per candidate generator: seats ∝ generator weight (largest
    remainder), each generator gives its best-ranked events not taken yet; seats a generator cannot fill go to the pool
    order. The pool is already one row per event (a news found by several generators is one candidate). Pool order kept."""
    if pool.empty or n >= len(pool):
        return pool.head(n)
    weight = generator_weights(gens, shares)
    keys = sorted(weight, key=lambda k: -weight[k])
    raw = {k: n * weight[k] / sum(weight.values()) for k in keys}
    seats = {k: int(raw[k]) for k in keys}
    for k in sorted(keys, key=lambda k: -(raw[k] - seats[k]))[:n - sum(seats.values())]:
        seats[k] += 1
    ranked = {k: [] for k in keys}  # generator -> pool rows it found, best rank first
    for i, gs in enumerate(pool.generators):
        for g in gs:
            ranked.setdefault((g['source'], g['query']), []).append((g['rank'], i))
    taken: list[int] = []
    for k in keys:
        free = [i for _, i in sorted(ranked.get(k, [])) if i not in taken]
        taken += free[:seats[k]]
    taken += [i for i in range(len(pool)) if i not in taken][:n - len(taken)]
    return pool.iloc[sorted(taken[:n])]


def order(events: pd.DataFrame, picked: dict[int, list[dict]], gens: list[dict], shares: dict[str, float]) -> pd.DataFrame:
    """Pool rows sorted by priority = Σ over the generators that found the event of weight / (1 + rank)
    (weights: generator_weights)."""
    weight = generator_weights(gens, shares)
    res = events.loc[list(picked)].reset_index()
    res = res.assign(generators=[picked[e] for e in res.event], best_rank=[min(x['rank'] for x in picked[e]) for e in res.event],
                     priority=[sum(weight[x['source'], x['query']] / (1 + x['rank']) for x in picked[e]) for e in res.event])
    return res.sort_values('priority', ascending=False, kind='stable').reset_index(drop=True)


if __name__ == '__main__':
    from stage2_company_facts.deterministic_facts.facts import build
    ap = argparse.ArgumentParser()
    ap.add_argument('company')
    ap.add_argument('--role')
    ap.add_argument('--no-factory', action='store_true', help='standard generators only (no LLM)')
    ap.add_argument('--show', type=int, default=30)
    a = ap.parse_args()
    res, gens = candidates(a.company, build([a.company])[a.company], a.role, use_factory=not a.no_factory)
    src = pd.Series([x['source'] for xs in res.generators for x in xs]).value_counts().to_dict()
    print(f'{len(gens)} generators -> {len(res)} events (picks by source: {src})')
    for r in res.head(a.show).itertuples():
        g = r.generators[0]
        print(f'{r.priority:.3f} {str(r.first_date)[:10]} {g["source"][:4]} {g["reason"][:18]:18} | {r.title[:100]}')
