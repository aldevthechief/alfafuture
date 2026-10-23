# Bank-analytics search: hybrid weight × instruction (22.10.2026, `eval_search_matching.py`)

Set: the parser's `data/analytics/eval/matching.jsonl` — 30 news × company pairs, 21 with `must` fragments (6 «no
analytics fits», 3 without must). Index: 364 fragments of `fragments.jsonl` (unverified charts excluded). Query =
news title + first 400 chars, no company. Fusion: `w · 1/(60 + rank_e5) + (1 − w) · 1/(60 + rank_BM25)`.
Instructions (query side only, invariant): T0 current «Найди фрагменты аналитики банка (оценки, прогнозы, данные)…»,
T1 «…те же экономические факторы, на которые влияет событие», T2 «позиция и прогнозы банка по макрофакторам…»,
T3 e5 default «Given a web search query, retrieve relevant passages that answer the query».

MRR of the first `must` in the top 10:

| w (e5 share) | 0.0 BM25 | 0.1 | 0.2 | 0.3 | 0.4 | **0.5** | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 e5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 current | 0.48 | 0.45 | 0.49 | 0.50 | 0.43 | 0.43 | 0.37 | 0.33 | 0.27 | 0.24 | 0.22 |
| T1 same factors | | 0.45 | 0.39 | 0.36 | 0.35 | 0.34 | 0.30 | 0.25 | 0.22 | 0.18 | 0.12 |
| T2 bank view | | 0.47 | 0.48 | 0.50 | 0.55 | 0.51 | 0.49 | 0.47 | 0.44 | 0.42 | 0.22 |
| **T3 e5 default** | | 0.58 | 0.51 | 0.58 | 0.58 | **0.59** | 0.54 | 0.51 | 0.51 | 0.48 | 0.53 |

`must` recall@10, T3: 0.65 (w 0.1–0.3), 0.69 (0.4), **0.71 (0.5)**, 0.57–0.66 (0.6–0.8), 0.62 (0.9–1.0).
Any must/ok in the top 5: ≥ 0.95 for every setting with w ≥ 0.3.

- The Russian task instructions hurt the e5 side: with T0 / T1 the more weight e5 gets, the worse (T0 0.50 → 0.22);
  the English default instruction keeps e5 useful on its own (0.53 at w = 1.0).
- Best: T3 at w = 0.3–0.5 (MRR 0.58–0.59). Paired bootstrap over the 21 pairs: T3 w = 0.5 beats w = 0.8 in 93% of
  resamples, w = 0.2 in 99%, w = 0.1 only in 63% — the 0.1–0.5 plateau is one level, 0.8 is a bit worse.
- Caveats: 21 pairs; the set was built by the parser's author with a topic-keyword matcher in the loop, which may
  favour the lexical side; the agent sends its own queries, not news text.

## Re-measured on the v1 sources (22.10, after the audit: `metadata_audit.md`)
Index = `search.use_fragment`: prose of the reports without legal text / parser notes / captions / digit lists, plus tables
with a header row, no charts (180 fragments instead of 364). The optimum moved to the semantic side:

| w (e5 share) | 0.0 | 0.3 | 0.5 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|
| MRR must@10 (T3) | 0.34 | 0.44 | 0.45 | 0.48 | 0.49 | 0.56 | 0.61 |
| must recall@10 (T3) | 0.57 | 0.61 | 0.66 | 0.75 | **0.78** | 0.67 | 0.67 |
| any must/ok in the top 5 (T3) | 0.86 | 0.90 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 |

On the old index the noise (legal text, captions, chart readings) made BM25 useful (best 0.3–0.5); with it removed BM25
only costs rank. Default set to **0.8** (the original idea): best recall@10, MRR 0.49; 1.0 gives the best MRR but loses
recall. The English e5 default instruction still beats the Russian task instructions at every weight (T0 0.41, T1 0.27,
T2 0.40 at w = 0.5; T3 0.45). 21 pairs / 14 news: differences below ~0.1 are noise.

## Chunking (analytic-retrieve v1, `chunks.py`, `eval_chunks.py`)
Recursive splitter (langchain, e5-token length, paragraph → line → sentence → clause → word), tables cut by rows with the
header repeated. Metrics by fragment ids a chunk contains; "window" = share of `must` fragments inside top-3 chunks ± 2 neighbours.

| index | chunks | must recall@5 | recall@10 | any hit@5 | MRR | window | tool tokens |
|---|---|---|---|---|---|---|---|
| fragments (no chunking) | 184 | 0.52 | 0.67 | 0.95 | 0.42 | 0.52 | 2603 |
| 200 / overlap 30 | 263 | 0.44 | 0.63 | 0.95 | 0.41 | 0.47 | 1702 |
| 300 / overlap 0 | 197 | 0.49 | 0.64 | 1.00 | 0.59 | 0.47 | 2580 |
| **300 / overlap 50 (default)** | 197 | 0.48 | 0.60 | 1.00 | 0.50 | 0.35 | 2275 |
| 400 / overlap 60 | 157 | 0.54 | 0.62 | 1.00 | 0.44 | 0.40 | 2894 |

Differences are inside the noise of 21 pairs (~0.1 MRR): chunking is not measurably better or worse than whole fragments
here; it is kept for the 512-token limit of e5 (long macro tables, long paragraphs) and for the ±2 neighbour window.

Report title in front of the indexed text (ablation, same set): fragments — no title MRR 0.63 / "title\n" 0.42 / "title. " 0.49;
chunks 300/50 — no title 0.32 / "title\n" 0.50 / "title. " 0.53. The effect flips sign between indexes: noise. Kept (title
helps to tell reports apart, e.g. the 2021 steel report).
