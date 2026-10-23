"""Old vs new stage-3 retrieval on the judge-v2 labels (experiments/judge_v2/out/labels_v2_<u>.jsonl; positive = insight
potential, significance and mechanism all >= 2), no LLM calls (factory specs must be cached: factory.py).
- old: stage3 candidate_generators.candidates (standard generators, TF-IDF + FRIDA, what pipeline.py --retrieval hybrid
  runs) at n = l = K, as the pipeline calls it; FRIDA document vectors are taken from the Kaggle run if not cached;
- standard v2: the same generators over the v2 hybrid (e5 + qwen + BM25), head K of the v2 order;
- v2: standard + factory (candidates.py), head K and the whole pool.
Labels' event ids belong to the events.parquet of the 20.10 corpus snapshot (--events), not to a rebuilt one.
Caveat: labels were collected on standard-v2 / factory / hand-written / random events, never on the old method's,
so the old method's unlabelled events may hide positives; the «unlabelled» column shows how many.
python -X utf8 stage3_news_search/retrieval_v2/eval_labels.py [--companies U_106783 …] [--k 30 100 200]"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.data import DATA
from stage2_company_facts.deterministic_facts.facts import build

COMPANIES = ['U_106783', 'U_150736', 'U_212593', 'U_9250']
LABELS = ROOT / 'experiments/judge_v2/out'
USEFUL = ROOT / 'experiments/judge_v2/useful_negatives.json'  # U_106783 events read by hand, not by the judge


def labels(u: str) -> pd.DataFrame:
    L = pd.DataFrame(json.loads(l)['value'] for l in open(LABELS / f'labels_v2_{u}.jsonl', encoding='utf-8'))
    L = L[(L.input == 'gemma_summary') & (L.refusal == 'short')].drop_duplicates('event', keep='last').set_index('event')
    L['pos'] = (L.insight_potential >= 2) & (L.significance >= 2) & (L.mechanism_strength >= 2)
    return L


def seed_frida():
    """Old method's cache (common/embed.py format: news_pool ids + vectors) from the Kaggle FRIDA run of the same pool."""
    from common.search import news_pool
    path = DATA / 'news' / 'emb_ai-forever__FRIDA.npz'
    if path.exists():
        return
    z = np.load(ROOT / 'experiments/retrieval/out/frida.npz')
    row = {i: k for k, i in enumerate(z['doc_ids'].tolist())}
    ids = news_pool()['id'].astype(str).to_numpy(dtype=str)
    np.savez(path, ids=ids, emb=z['doc'][[row[i] for i in ids]].astype(np.float32))


def old_sets(facts: dict, ks: list[int]) -> dict:
    import torch
    from common import embed
    from stage3_news_search.candidate_generators.generators import candidates
    seed_frida()
    out = {(u, k): list(candidates(f, n=k, l=k)[0].event.head(k)) for u, f in facts.items() for k in ks}
    embed._model.cache_clear(); torch.cuda.empty_cache()  # FRIDA off the 4 GB GPU before e5 + qwen
    return out


if __name__ == '__main__':
    from stage3_news_search.retrieval_v2.candidates import candidates
    ap = argparse.ArgumentParser()
    ap.add_argument('--companies', nargs='+', default=COMPANIES)
    ap.add_argument('--k', nargs='+', type=int, default=[30, 100, 200])
    ap.add_argument('--events', type=Path, default=DATA / 'news' / 'events.parquet', help='events the labels refer to')
    a = ap.parse_args()
    facts = {u: build([u])[u] for u in a.companies}
    events = pd.read_parquet(a.events)
    sets = {(u, f'old @{k}'): s for (u, k), s in old_sets(facts, a.k).items()}
    for u, f in facts.items():
        std, _ = candidates(u, f, events=events, use_factory=False)
        v2, _ = candidates(u, f, events=events, frozen_factory=True)
        sets |= {(u, f'standard v2 @{k}'): list(std.event.head(k)) for k in a.k}
        sets |= {(u, f'v2 @{k}'): list(v2.event.head(k)) for k in a.k}
        sets[(u, f'v2 pool ({len(v2)})')] = list(v2.event)
    useful = {x['event'] for x in json.loads(USEFUL.read_text(encoding='utf-8'))['items']}
    rows = []
    for (u, name), s in sets.items():
        L = labels(u)
        lab = [e for e in s if e in L.index]
        rows.append({'company': u, 'set': name, 'events': len(s), 'positives': f'{int(L.loc[lab].pos.sum())}/{int(L.pos.sum())}',
                     'pos/labelled': round(L.loc[lab].pos.mean(), 2) if lab else None, 'unlabelled': len(s) - len(lab),
                     'manual useful': f'{len(useful & set(s))}/{len(useful)}' if u == 'U_106783' else ''})
    R = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print(R.to_string(index=False))
    out = DATA / 'retrieval_v2' / 'eval_labels.csv'
    R.to_csv(out, index=False)
    print(out)
