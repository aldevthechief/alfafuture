"""Bank-analytics search on the parser's matching set (data/analytics/eval/matching.jsonl, 30 news × company pairs with
`must` / `ok` fragment ids): hybrid weight × encoder instruction. Query = news title + first 400 chars (the agent writes
its own query; this is the invariant baseline). Metrics: share of `must` in the top 5 / 10, any must/ok hit in the top 5,
MRR of the first must in the top 10; paired bootstrap of the best weight vs the others. Pairs marked `none` (no analytics fits) are skipped. Local, no LLM.
python -X utf8 stage5_insight_detection/bank_analytics/eval_search_matching.py"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.data import DATA
from stage5_insight_detection.bank_analytics.search import BankAnalytics

TASKS = {  # invariant: no company, no specific news
    'T0 Russian task': 'Найди фрагменты аналитики банка (оценки, прогнозы, данные), относящиеся к теме запроса',
    'T1 same factors': 'Дана новость. Найди фрагменты аналитики банка, где банк оценивает те же экономические факторы, '
                       'на которые влияет это событие',
    'T2 bank view': 'Найди позицию и прогнозы банка по макроэкономическим факторам (ставка, инфляция, курс, спрос, '
                    'бюджет, рынки), связанным с новостью',
    'T3 e5 default': 'Given a web search query, retrieve relevant passages that answer the query',
}
WEIGHTS = [round(0.1 * i, 1) for i in range(11)]

if __name__ == '__main__':
    pairs = [json.loads(l) for l in open(DATA / 'analytics/eval/matching.jsonl', encoding='utf-8') if l.strip()]
    pairs = [p for p in pairs if not p['none'] and p['must']]
    news = pd.read_parquet(DATA / 'news' / 'news.parquet').set_index('id')
    news.index = news.index.astype(int)
    pairs = [p for p in pairs if p['news'] in news.index]
    bank = BankAnalytics(chunk_tokens=0)
    ids = {c['id'] for c in bank.chunks}
    print(f'{len(pairs)} pairs with news in the pool; fragments {len(bank.chunks)}; '
          f'must ids not searchable (unverified / boilerplate): {sum(m not in ids for p in pairs for m in p["must"])}')
    rows = []
    for tname, task in TASKS.items():
        for w in WEIGHTS:
            if w == 0.0 and tname != 'T3 e5 default':
                continue  # BM25 alone ignores the instruction
            rec, hit, rr, rec10 = [], [], [], []
            for p in pairs:
                n = news.loc[p['news']]
                q = f"{n.title}. {str(n.text)[:400]}"
                top10 = [h['id'] for h in bank.search(q, 10, task=task, w_dense=w)]
                top = top10[:5]
                must, good = set(p['must']), set(p['must']) | set(p['ok'])
                rec.append(len(must & set(top)) / len(must))
                hit.append(bool(good & set(top)))
                rr.append(next((1 / (i + 1) for i, x in enumerate(top10) if x in must), 0))
                rec10.append(len(must & set(top10)) / len(must))
            rows.append({'instruction': tname, 'w_dense': w, 'must recall@5': np.mean(rec), 'must recall@10': np.mean(rec10),
                         'any hit@5': np.mean(hit), 'MRR must@10': np.mean(rr), 'per_pair_rr': rr})
    df = pd.DataFrame(rows)
    for m in ('MRR must@10', 'must recall@5', 'must recall@10', 'any hit@5'):
        print(f'\n{m} (rows: instruction, columns: w_dense)')
        print(df.pivot(index='instruction', columns='w_dense', values=m).round(2).to_string())
    best = df.loc[df['MRR must@10'].idxmax()]
    rng = np.random.default_rng(0)  # paired bootstrap over pairs: is the best really better than the others?
    b = np.array(best.per_pair_rr)
    print(f"\nbest by MRR: {best.instruction} w={best.w_dense}; paired bootstrap share of resamples where it beats:")
    for _, r in df[df.instruction == best.instruction].iterrows():
        o = np.array(r.per_pair_rr); idx = rng.integers(0, len(b), (2000, len(b)))
        print(f"  w={r.w_dense}: {(b[idx].mean(1) > o[idx].mean(1)).mean():.2f}")
