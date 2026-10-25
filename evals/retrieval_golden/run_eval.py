"""Run the current news retrieval (v2 = standard + factory generators, v3 = v2 + channel generators) on the frozen golden set.
The synthetic news are injected into the real corpus (one extra event each, ids from 9 000 000), so a generator has to find them
among ~9k real articles; labels exist only for the injected ones. No LLM calls: the factory runs with its stored spec
(frozen_factory), only the local encoders embed the injected texts.
Rankings scored per pair (company, role): the pools of v2 / v3, the same pool ordered by one source only (standard / factory /
channel), the quota_pick(30) head the next stage receives, and every single generator's own top list.
ALFA_DATA=<data dir> python -X utf8 -m evals.retrieval_golden.run_eval [--out report.json]"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evals.retrieval_golden.metrics import MISSED, evaluate_ranks, ranks_of
GOLDEN = Path(__file__).with_name('golden')
BASE_ID = 9_000_000
KS = (15, 30, 100)
OUTLETS = ['rbc', 'kommersant', 'interfax', 'vedomosti', 'tass', 'bfm', 'banki']  # plausible names; replaced by the corpus' own below


def load_golden() -> list[dict]:
    return [json.loads(p.read_text(encoding='utf-8')) for p in sorted(GOLDEN.glob('U_*.json'))]


def stratum(item: dict) -> str:
    return ('direct' if item['names_industry'] else 'indirect') if item['label'] else ('hard' if item['kind'] == 'hard_neg' else 'easy')


def inject(news: pd.DataFrame, events: pd.DataFrame, golden: list[dict]):
    """Append every golden item as one news + one event. Returns news, events and per pair {news id: (label, stratum)}."""
    outlets = news[news.tier == 1].source.value_counts().index.tolist()[:8] or OUTLETS
    rows, evs, marks = [], [], []
    first = int(events.event.max()) + 1
    for p, pair in enumerate(golden):
        mark = {}
        for k, it in enumerate(pair['items']):
            i = BASE_ID + p * 100 + k
            rows.append(dict(id=i, tier=1, topic='economy', source=outlets[(p + k) % len(outlets)], date=pd.Timestamp(it['date']).date(),
                             url=f'https://synthetic.invalid/{i}', title=it['title'], summary='', text=it['text'], text_src='page'))
            evs.append(dict(event=first + len(evs), title=it['title'], first_date=pd.Timestamp(it['date']).date(), n_reports=1, n_sources=1,
                            sources=np.array([rows[-1]['source']], dtype=object), news_ids=np.array([i]), plausibility=0.5,
                            numbers_disagree=False))
            mark[i] = (it['label'], stratum(it))
        marks.append(mark)
    news2 = pd.concat([news, pd.DataFrame(rows)], ignore_index=True)
    events2 = pd.concat([events, pd.DataFrame(evs)], ignore_index=True)
    return news2, events2, marks


def source_order(res: pd.DataFrame, gens: list[dict], source: str, shares: dict) -> list[int]:
    """Events of the pool ordered by the priority that only the generators of `source` give them (others: not returned)."""
    from stage3_news_search.retrieval_v2 import candidates as v2
    if source not in {g['source'] for g in gens}:
        return []
    weight = v2.generator_weights(gens, {s: float(s == source) for s in shares})
    pri = {e: sum(weight[x['source'], x['query']] / (1 + x['rank']) for x in gs if x['source'] == source)
           for e, gs in zip(res.event, res.generators)}
    return [e for e, v in sorted(pri.items(), key=lambda kv: -kv[1]) if v > 0]


def rankings(company: str, role: str, facts: list[dict], news, events, news_of_event: dict) -> tuple[dict, list[dict]]:
    """name -> ordered news ids of the injected items (the real articles stay in the list: they fill the top)."""
    from stage3_news_search.retrieval_v2 import candidates as v2
    from stage3_news_search.retrieval_v3 import candidates as v3
    r2, g2 = v2.candidates(company, facts, role, news=news, events=events, frozen_factory=True)
    r3, g3 = v3.candidates(company, facts, role, news=news, events=events, frozen_factory=True)
    ids = lambda es: [news_of_event.get(e, -1 - n) for n, e in enumerate(es)]  # a real event: a unique negative placeholder
    out = {'v2': ids(r2.event), 'v3': ids(r3.event), 'v2 head 30 (quota)': ids(v2.quota_pick(r2, g2, 30).event),
           'v3 head 30 (quota)': ids(v2.quota_pick(r3, g3, 30, v3.SHARES).event)}
    for s in ('standard', 'factory', 'channel'):
        out[f'v3 only {s}'] = ids(source_order(r3, g3, s, v3.SHARES))
    return out, g3


def per_generator(gens: list[dict], news_of_event: dict, labels: dict) -> list[dict]:
    rows = []
    for g in gens:
        rank = ranks_of([news_of_event.get(e, -1 - n) for n, e in enumerate(g['top'])], labels)
        m = evaluate_ranks(rank, labels, KS)
        found = [i for i in labels if rank[i] != MISSED]
        rows.append(dict(source=g['source'], reason=g['reason'], channel=g.get('channel', ''), query=g['query'][:60], returned=len(g['top']),
                         pos_found=sum(labels[i] for i in found), neg_found=sum(1 - labels[i] for i in found), auc=m['auc']))
    return rows


def macro(per_pair: list[dict], metric: str, n: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Mean over pairs and a 95% interval when the labelled items of every pair are resampled (positives / negatives apart)."""
    point = float(np.mean([p[metric] for p in per_pair if p[metric] is not None]))
    return point, *_interval(per_pair, metric, n, seed)


def _interval(per_pair, metric, n, seed):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        vs = []
        for p in per_pair:
            pos = [i for i, y in p['labels'].items() if y == 1]
            neg = [i for i, y in p['labels'].items() if y == 0]
            picks = list(rng.choice(pos, len(pos))) + list(rng.choice(neg, len(neg)))
            keys = {f'{i}#{k}': i for k, i in enumerate(picks)}
            v = evaluate_ranks({key: p['rank'][i] for key, i in keys.items()}, {key: p['labels'][i] for key, i in keys.items()}, KS).get(metric)
            if v is not None:
                vs.append(v)
        values.append(np.mean(vs))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path(__file__).with_name('report.json'))
    ap.add_argument('--rescore', action='store_true', help='recompute metrics from the ranks saved in --out, no search')
    a = ap.parse_args()
    if a.rescore:
        return rescore(a.out)
    from common.search import news_pool
    from stage2_company_facts.deterministic_facts.facts import build
    from stage4_news_relevance.plausibility.events import ensure_events
    golden = load_golden()
    news = news_pool()
    news2, events2, marks = inject(news, ensure_events(news), golden)
    news_of_event = {int(e): int(ids[0]) for e, ids in zip(events2.event, events2.news_ids) if int(ids[0]) >= BASE_ID}
    systems: dict[str, list] = {}
    gen_rows = []
    for pair, mark in zip(golden, marks):
        company, role = pair['company'], pair['role']
        facts = build([company], use_profiles=True)[company]
        labels = {i: y for i, (y, _) in mark.items()}
        strata = {i: s for i, (_, s) in mark.items()}
        ranked, gens = rankings(company, role, facts, news2, events2, news_of_event)
        # injected items of the OTHER pairs are real-looking unlabelled news here: they stay in the list as filler
        for name, ranking in ranked.items():
            rank = ranks_of(ranking, labels)
            systems.setdefault(name, []).append(dict(company=company, role=role, labels=labels, rank=rank, returned=len(ranking),
                                                     **evaluate_ranks(rank, labels, KS, strata)))
        gen_rows += [dict(company=company, role=role, **r) for r in per_generator(gens, news_of_event, labels)]
        print(f'{company}_{role}: {len(gens)} generators, v3 pool {len(ranked["v3"])} events', file=sys.stderr)
    write(systems, gen_rows, a.out)


def summarize(systems: dict[str, list]) -> dict:
    report = {}
    for name, pairs in systems.items():
        row = {}
        for metric in ('auc', f'recall@{KS[0]}', f'recall@{KS[1]}', f'recall@{KS[2]}', f'fpr@{KS[0]}', f'fpr@{KS[1]}', f'fpr@{KS[2]}',
                       f'ndcg@{KS[0]}', 'map', 'mrr'):
            if all(p.get(metric) is not None for p in pairs):
                row[metric] = dict(zip(('mean', 'lo', 'hi'), (round(v, 3) for v in macro(pairs, metric))))
        for metric in (f'direct@{KS[0]}', f'indirect@{KS[0]}', f'hard@{KS[0]}', f'easy@{KS[0]}'):  # point estimates: strata are small
            row[metric] = dict(mean=round(float(np.mean([p[metric] for p in pairs])), 3))
        row['returned_mean'] = float(np.mean([p['returned'] for p in pairs]))
        row['per_pair_auc'] = {f"{p['company']}_{p['role']}": round(p['auc'], 3) for p in pairs}
        report[name] = row
    return report


def write(systems: dict[str, list], gen_rows: list[dict], path: Path) -> None:
    report = summarize(systems)
    out = dict(systems=report, generators=gen_rows,
               ranks={n: [dict(company=p['company'], role=p['role'], rank=p['rank'], labels=p['labels'], returned=p['returned']) for p in ps]
                      for n, ps in systems.items()})
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: {m: v['mean'] for m, v in r.items() if isinstance(v, dict) and 'mean' in v} for k, r in report.items()},
                     ensure_ascii=False, indent=1))
    print(path)


def rescore(path: Path) -> None:
    """Recompute the report from the saved ranks (no search): after a metric change."""
    saved = json.loads(path.read_text(encoding='utf-8'))
    strata = {BASE_ID + p * 100 + k: stratum(it) for p, pair in enumerate(load_golden()) for k, it in enumerate(pair['items'])}
    systems = {}
    for name, ps in saved['ranks'].items():
        for p in ps:
            rank, labels = ({int(i): v for i, v in d.items()} for d in (p['rank'], p['labels']))
            systems.setdefault(name, []).append(dict(company=p['company'], role=p['role'], labels=labels, rank=rank, returned=p['returned'],
                                                     **evaluate_ranks(rank, labels, KS, {i: strata[i] for i in labels})))
    write(systems, saved['generators'], path)


if __name__ == '__main__':
    main()
