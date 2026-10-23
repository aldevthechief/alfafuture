"""Retrieval v3 = retrieval v2 (standard generators + generator factory over the e5 + qwen + BM25 hybrid) + news-first
channel generators («keys of influence» × company exposure, channels.py), on the same search and the same e5 vectors.
- channel source: quota union ∝ facet weight at the smallest budget giving >= CHANNEL_BUDGET events;
- pool = v2 pool ∪ channel events; an event keeps every generator that found it;
- order = v2's priority (retrieval_v2.candidates.order) with the channel source taking CHANNEL_SHARE of the weight and
  the v2 sources scaled down proportionally.
python -X utf8 stage3_news_search/retrieval_v3/candidates.py U_106783 [--role cfo] [--no-factory] [--show 30]"""
import argparse, sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stage3_news_search.retrieval_v2 import candidates as v2
from stage3_news_search.retrieval_v3 import channels

CHANNEL_BUDGET = 200
CHANNEL_SHARE = 0.2  # of the ordering weight; chosen on the labels of 2 tuning companies, reported on 2 held-out ones (README)
SHARES = {s: w * (1 - CHANNEL_SHARE) for s, w in v2.SHARES.items()} | {'channel': CHANNEL_SHARE}


def candidates(company_id: str, facts: list[dict], role: str | None = None, *, news: pd.DataFrame | None = None,
               events: pd.DataFrame | None = None, model=None, budget: int = v2.STANDARD_BUDGET, quota: int = v2.factory.QUOTA,
               use_factory: bool = True, frozen_factory: bool = False, channel_budget: int = CHANNEL_BUDGET,
               shares: dict[str, float] = SHARES) -> tuple[pd.DataFrame, list[dict]]:
    """Same contract as retrieval_v2.candidates; generators of the channel source have source='channel'."""
    if news is None:
        from common.search import news_pool
        news = news_pool()
    if events is None:
        from stage4_news_relevance.plausibility.events import ensure_events
        events = ensure_events(news)
    res, gens = v2.candidates(company_id, facts, role, news=news, events=events, model=model, budget=budget, quota=quota,
                              use_factory=use_factory, frozen_factory=frozen_factory)
    search = v2.hybrid_search(news, events)  # the instance v2 just used (cached per corpus)
    picked = dict(zip(res.event, res.generators))
    cgens = channels.generators(facts, search)
    for g, k in zip(cgens, v2.quotas([g['weight'] for g in cgens], [g['top'] for g in cgens], channel_budget)):
        for r, e in enumerate(g['top'][:k]):
            picked.setdefault(e, []).append({'source': 'channel', 'reason': g['reason'], 'channel': g['channel'],
                                             'query': g['query'], 'rank': r})
    gens = gens + cgens
    return v2.order(search.events, picked, gens, shares), gens


if __name__ == '__main__':
    from stage2_company_facts.deterministic_facts.facts import build
    ap = argparse.ArgumentParser()
    ap.add_argument('company')
    ap.add_argument('--role')
    ap.add_argument('--no-factory', action='store_true', help='standard + channel generators only (no LLM)')
    ap.add_argument('--show', type=int, default=30)
    a = ap.parse_args()
    res, gens = candidates(a.company, build([a.company])[a.company], a.role, use_factory=not a.no_factory)
    src = pd.Series([x['source'] for xs in res.generators for x in xs]).value_counts().to_dict()
    only = sum(all(x['source'] == 'channel' for x in xs) for xs in res.generators)
    print(f'{len(gens)} generators -> {len(res)} events (picks by source: {src}; {only} events only from channels)')
    for r in res.head(a.show).itertuples():
        g = r.generators[0]
        print(f'{r.priority:.3f} {str(r.first_date)[:10]} {g["source"][:4]} {g["reason"][:18]:18} | {r.title[:100]}')
