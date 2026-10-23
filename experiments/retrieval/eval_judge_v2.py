"""Retrieval methods vs judge v2 majority labels (experiments/judge_v2, U_106783 only). Prints, writes nothing.
python -X utf8 experiments/retrieval/eval_judge_v2.py"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, 'experiments/retrieval'); sys.path.insert(0, '.')
import evaluate as ev
O = 'experiments/judge_v2/out/labels_v2_U_106783.jsonl'
L = pd.DataFrame([json.loads(l)['value'] for l in open(O, encoding='utf8')])
L = L[L.refusal == 'short']
L['pos'] = (L.insight_potential >= 2) & (L.significance >= 2) & (L.mechanism_strength >= 2)
maj = L.groupby('event').pos.agg(lambda s: s.sum() >= 2 if len(s) >= 2 else s.iloc[0])
src = L.groupby('event').source.first()
pos = [int(e) for e in maj[maj].index]
print('v2 positives:', len(pos), 'of which random:', int((src[maj[maj].index] == 'random').sum()))
ev.LABELLED['U_106783'] = pos
every = dict(ev.LEXICAL) | ev.kaggle_methods(ev.HERE / 'out')
names = ['e5-large-instruct/reason', 'qwen3-emb-0.6b/reason', 'bm25-lemma/query', 'bge-m3/query', 'frida/query', 'user-bge-m3/query',
         'giga-emb-10b/reason', 'splade-os-multi/query', 'bm25-snowball/query', 'tfidf-stem6/query', 'e5-large-instruct/web', 'e5-large-instruct/none']
S = {n: every[n]() for n in names}
hyb = {'HYB e5+qwen+bm25lemma (candidate source)': ev.rrf(S['e5-large-instruct/reason'], S['qwen3-emb-0.6b/reason'], S['bm25-lemma/query']),
       'HYB e5+bm25lemma': ev.rrf(S['e5-large-instruct/reason'], S['bm25-lemma/query'])}
_, _, event = ev.corpus()
rows = []
for n, M in list(S.items()) + list(hyb.items()):
    r = ev.evaluate(M)
    r = r[(r.company == 'U_106783') & (r.kind == 'pooled')].iloc[0]
    evid, E = ev.event_ranks(M, event)
    qi = [q for q, _ in ev.company_queries()['U_106783']]
    order = np.argsort(-E[:, qi], axis=0); rank = np.empty_like(order); rank[order, np.arange(len(qi))[None, :]] = np.arange(len(evid))[:, None]
    row = {e: i for i, e in enumerate(evid)}
    best = {e: int(rank[row[e]].min()) for e in pos}
    rows.append({'method': n, 'R@200': r['recall@200'], 'R@800': r['recall@800'], 'best_rank_median': np.median(list(best.values())),
                 'random_pos_best_ranks': [best[e] for e in pos if src[e] == 'random']})
print(pd.DataFrame(rows).round(2).to_string(index=False))
# v1 labels of this company vs v2
v1 = {}
for l in open('experiments/retrieval/labels_cheap.jsonl', encoding='utf8'):
    d = json.loads(l); u, e = d['key'].split('|')
    if u == 'U_106783': v1[int(e)] = d['value']
print('v1 labels for U_106783:', len(v1), 'grade2:', sum(v == 2 for v in v1.values()), '| v2 positives inside v1 pool:', sum(e in v1 for e in pos))
