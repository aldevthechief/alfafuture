"""Does the bank-analytics search need charts and tables? The parser's matching set (21 news × company pairs with `must` /
`ok` fragment ids) against the same hybrid (e5 default instruction + BM25, w = 0.5) over subsets of fragments:
all kinds, text only, prose only (search.prose_reason), prose + tables. Also the ceiling: how many `must` fragments exist
in a subset at all. Embeddings are computed once for all fragments, subsets reuse them. Local, no LLM.
python -X utf8 stage5_insight_detection/bank_analytics/eval_kinds.py"""
import collections, copy, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.data import DATA
from stage5_insight_detection.bank_analytics.search import DISCLAIMER, BankAnalytics, bm25_index, prose_reason, use_fragment

SUBSETS = {
    'all kinds (current)': lambda c: True,
    'text only': lambda c: c['kind'] == 'text',
    'text minus legal / notes / captions': lambda c: c['kind'] == 'text' and prose_reason(c) not in ('legal', 'parser note', 'caption / source'),
    'soft prose (+ short, few words)': lambda c: c['kind'] == 'text' and prose_reason(c) in ('', 'short', 'few words', 'no sentence end'),
    'prose only': lambda c: prose_reason(c) == '',
    'prose + tables': lambda c: prose_reason(c) == '' or c['kind'] == 'table',
    'v1 sources (search.use_fragment)': lambda c: use_fragment(c),
}


def subset(bank, keep):
    idx = [i for i, c in enumerate(bank.chunks) if keep(c)]
    b = copy.copy(bank)
    b.chunks, b.D = [bank.chunks[i] for i in idx], bank.D[idx]
    b.bm25 = bm25_index([c['text'] for c in b.chunks])
    return b


if __name__ == '__main__':
    pairs = [json.loads(l) for l in open(DATA / 'analytics/eval/matching.jsonl', encoding='utf-8') if l.strip()]
    news = pd.read_parquet(DATA / 'news' / 'news.parquet').set_index('id')
    news.index = news.index.astype(int)
    pairs = [p for p in pairs if not p['none'] and p['must'] and p['news'] in news.index]
    bank = BankAnalytics(chunk_tokens=0, keep=lambda f: not f.get('unverified') and not DISCLAIMER.search(f['text']))  # the pre-v1 superset
    kind = {c['id']: c['kind'] for c in bank.chunks}
    prose = {c['id']: prose_reason(c) == '' for c in bank.chunks}
    print('distinct news in the pairs:', len({p['news'] for p in pairs}), 'of', len(pairs), 'pairs (same news × different companies repeat the query)')
    must = [m for p in pairs for m in p['must']]
    print(f'{len(pairs)} pairs, {len(must)} must fragments; kinds of must: {dict(collections.Counter(kind.get(m, "not searchable") for m in must))}; '
          f'prose among must: {sum(prose.get(m, False) for m in must)}')
    ok = [m for p in pairs for m in p['ok']]
    print(f'ok fragments: {len(ok)}; kinds: {dict(collections.Counter(kind.get(m, "not searchable") for m in ok))}')
    rows = []
    for name, keep in SUBSETS.items():
        b = subset(bank, keep)
        ids = {c['id'] for c in b.chunks}
        rec5, rec10, hit, rr = [], [], [], []
        for p in pairs:
            n = news.loc[p['news']]
            top = [h['id'] for h in b.search(f"{n.title}. {str(n.text)[:400]}", 10)]
            mu, good = set(p['must']), set(p['must']) | set(p['ok'])
            rec5.append(len(mu & set(top[:5])) / len(mu)); rec10.append(len(mu & set(top)) / len(mu))
            hit.append(bool(good & set(top[:5]))); rr.append(next((1 / (i + 1) for i, x in enumerate(top) if x in mu), 0))
        rows.append({'subset': name, 'fragments': len(b.chunks), 'must in subset': f'{sum(m in ids for m in must)}/{len(must)}',
                     'pairs with a must': f'{sum(bool(set(p["must"]) & ids) for p in pairs)}/{len(pairs)}',
                     'must recall@5': np.mean(rec5), 'must recall@10': np.mean(rec10), 'any hit@5': np.mean(hit), 'MRR must@10': np.mean(rr)})
    print(pd.DataFrame(rows).round(2).to_string(index=False))
