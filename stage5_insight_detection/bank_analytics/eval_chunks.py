"""Chunk-level evaluation of analytic-retrieve v1 on the parser's matching set (21 news × company pairs with `must` / `ok`
fragment ids): chunk size and overlap (chunks.py) against the pre-chunking index (one chunk per fragment). A chunk counts
for a fragment when it contains text of it (`frag_ids`). Also the context window of the agent's tool: must fragments
covered by the top-3 chunks ± 2 neighbours, and the size of what the tool returns. Local, no LLM.
python -X utf8 stage5_insight_detection/bank_analytics/eval_chunks.py"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.data import DATA
from stage5_insight_detection.bank_analytics.search import BankAnalytics, encoder

CONFIGS = [('fragments (no chunking)', 0, 0), ('chunk 200 / overlap 30', 200, 30), ('chunk 300 / overlap 0', 300, 0),
           ('chunk 300 / overlap 50 (default)', 300, 50), ('chunk 400 / overlap 60', 400, 60)]

if __name__ == '__main__':
    pairs = [json.loads(l) for l in open(DATA / 'analytics/eval/matching.jsonl', encoding='utf-8') if l.strip()]
    news = pd.read_parquet(DATA / 'news' / 'news.parquet').set_index('id')
    news.index = news.index.astype(int)
    pairs = [p for p in pairs if not p['none'] and p['must'] and p['news'] in news.index]
    tok = encoder().tokenizer
    rows = []
    for name, size, overlap in CONFIGS:
        b = BankAnalytics(chunk_tokens=size, overlap_tokens=overlap)
        ids = {f for c in b.chunks for f in c['frag_ids']}
        rec5, rec10, hit, rr, win, wtok = [], [], [], [], [], []
        for p in pairs:
            n = news.loc[p['news']]
            top = b.search(f"{n.title}. {str(n.text)[:400]}", 10)
            frags = [set(h['frag_ids']) for h in top]
            mu, good = set(p['must']), set(p['must']) | set(p['ok'])
            rec5.append(len(mu & set().union(*frags[:5])) / len(mu)); rec10.append(len(mu & set().union(*frags)) / len(mu))
            hit.append(bool(good & set().union(*frags[:5]))); rr.append(next((1 / (i + 1) for i, f in enumerate(frags) if f & mu), 0))
            near = set()
            for h in top[:3]:  # window: ±2 chunks of the same report around each of the top-3
                for j in range(h['pos'] - 2, h['pos'] + 3):
                    if 0 <= j < len(b.chunks) and b.chunks[j]['report'] == h['report']:
                        near |= set(b.chunks[j]['frag_ids'])
            win.append(len(mu & near) / len(mu))
            wtok.append(sum(len(tok.encode(q['text'], add_special_tokens=False)) for q in b.passages(f"{n.title}. {str(n.text)[:400]}", 3, 2)))
        rows.append({'index': name, 'chunks': len(b.chunks), 'must recall@5': np.mean(rec5), 'must recall@10': np.mean(rec10),
                     'any hit@5': np.mean(hit), 'MRR must': np.mean(rr), 'must in top-3 ± 2': np.mean(win), 'tool tokens (k=3, ±2)': int(np.mean(wtok))})
        print(pd.DataFrame(rows[-1:]).round(2).to_string(index=False, header=len(rows) == 1), flush=True)
    print('\n' + pd.DataFrame(rows).round(2).to_string(index=False))
