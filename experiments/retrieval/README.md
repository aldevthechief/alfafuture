# Retrieval for candidate generators: lexical vs sparse vs dense encoders (20.10.2026)

Question: which search should the stage 3 candidate generators use, so that a company's generators bring as many
useful news as possible into a candidate set of 200–800 events?

## Setup
- **Corpus:** 9 147 news (`news.parquet`) + 21 synthetic needles / hard negatives; ranked at event level (`events.parquet`).
- **Queries:** generators of 27 companies (9 + 10 golden-set companies + 11 from the worked insight examples, overlapping):
  633 generators, 226 unique texts (`queries.json`, `query_index.json`). Built by `candidate_generators.generators()` from stage-2 facts.
- **Candidate set:** each generator takes its quota of the budget (by weight) of top events; budget 200 and 800.
- **Positives, three sets:**
  1. **pooled** (main): top-5 events of every generator under 11 methods → 12 126 (company, event) pairs graded 0/1/2 by
     Gemma 4 (`judge.py`, prompt `prompts/judge.md`); grade 2 = positive, ~38 per company. Gemma agrees with GLM-5.3-Flash
     on «grade 2 or not» in 94% of 137 checked pairs. Cost ≈ 15 ₽.
  2. **real**: 33 news for 11 companies from the 10 worked insight examples + the U_31505 demo (`real_pairs.json`).
  3. **needles**: 9 direct + 9 chain hand-written planted news (`synthetic_needles.json`) — saturated, every method finds ~90–100%.
- **Methods:**
  - lexical (local): TF-IDF over 6-letter stems (current), TF-IDF / BM25 (`bm25s`) with Snowball stemmer, BM25 with pymorphy3 lemmas;
  - learned sparse (Kaggle): SPLADE `opensearch-neural-sparse-encoding-multilingual-v1`, BGE-M3 sparse head;
  - dense (Kaggle, 2×T4): FRIDA, USER-bge-m3, bge-m3, multilingual-e5-large-instruct, Qwen3-Embedding-0.6B,
    Giga-Embeddings-instruct-10B-A1.8B; Qwen3-Embedding-8B pending.
  - query variants: `query` = fact text only, `text` = «why: fact»; instruct models also get task instructions
    `web` (model default), `en_generic`, `ru_generic`, `ru_insight`, `reason` (a Russian instruction per generator reason + fact) and `none`.
  - hybrids: reciprocal rank fusion (RRF, k = 60).

## Results (pooled labels; full table: `results_pooled.csv`)
| method | pooled R@200 | pooled R@800 | real R@800 | candidates |
|---|---|---|---|---|
| **e5-large-instruct/reason + qwen3-0.6b/reason + bm25-lemma** (RRF) | **0.65** | **0.93** | 0.90 | 618 |
| e5-large-instruct/reason + bm25-lemma | 0.61 | 0.91 | 0.91 | 624 |
| e5-large-instruct/reason (best single) | 0.62 | 0.88 | 0.90 | 580 |
| qwen3-emb-0.6b/reason | 0.58 | 0.86 | 0.86 | 618 |
| bge-m3/query | 0.54 | 0.82 | 0.79 | 637 |
| frida/query | 0.54 | 0.81 | 0.82 | 640 |
| giga-emb-10b/reason | 0.55 | 0.81 | 0.86 | 540 |
| splade-os-multi/query | 0.48 | 0.79 | 0.74 | 659 |
| bm25-snowball/query (best lexical) | 0.43 | 0.75 | 0.73 | 661 |
| tfidf-stem6/query (current) | 0.40 | 0.70 | 0.60 | 655 |

## Findings
1. **Dense encoders beat lexical search** by ~10–18 p.p. of recall at the same budget; the best one is the mid-size
   `multilingual-e5-large-instruct` (560M), not the 10B Giga-Embeddings.
2. **The instruction matters as much as the model.** Per-reason Russian instructions (`reason`) are best for every instruct
   model; no instruction or the default web-search instruction lose 10–15 p.p. (e5: 0.88 vs 0.73 / 0.76).
   For non-instruct models the fact alone (`query`) beats «why: fact» (`text`).
3. **Hybrids add a few points:** dense + BM25 (+ a second dense) → 0.93 at 800, 0.65 at 200.
4. **Learned sparse (SPLADE, BGE-M3 sparse) is not better than BM25** with a proper Russian stemmer / lemmas.
5. **Our current TF-IDF on 6-letter stems is the weakest** — a library BM25 with Snowball / pymorphy3 is +5 p.p. for free.

## Re-check on judge v2 labels (21.10, `eval_judge_v2.py`)
The pooled Gemma v1 labels disagree with judge v2 (multi-dimensional insight potential, experiments/judge_v2): of 18
v1 grade-2 events of U_106783 only 3 are v2 positives. v2 labels exist for one company: 13 positives among 196 events
(11 from the candidate set of the hybrid below at budget ~100, 2 from 94 random events) — biased toward the hybrid
and its parts, R@200 is mostly «is it in the hybrid's top-100».

| method | R@200 | R@800 | median best rank | ranks of the 2 random positives |
|---|---|---|---|---|
| e5-large-instruct/reason | 0.85 | 0.85 | 9 | 46, 544 |
| qwen3-emb-0.6b/reason | 0.85 | 0.85 | 7 | 43, 358 |
| e5 + qwen + bm25-lemma (RRF, candidate source) | 0.85 | 0.85 | 9 | 111, 664 |
| bge-m3/query | 0.62 | **0.92** | 11 | 48, 147 |
| user-bge-m3/query | 0.69 | **0.92** | 14 | 249, 196 |
| giga-emb-10b/reason | 0.54 | 0.85 | 42 | 14, 254 |
| splade-os-multi/query | 0.46 | 0.77 | 36 | 224, 129 |
| frida/query | 0.38 | 0.62 | 72 | 143, 612 |
| tfidf-stem6/query | 0.46 | 0.62 | 54 | 471, 685 |
| bm25-lemma/query | 0.15 | 0.54 | 99 | 99, 919 |
| bm25-snowball/query | 0.15 | 0.54 | 119 | 340, 537 |

Holds under v2 (despite the bias): dense ≫ lexical — BM25 is the worst although it is part of the candidate source;
bge-m3 / user-bge-m3 are not in the candidate source yet reach 0.92 at 800. Does not hold / unknown: «BM25 adds a few
points in the hybrid», the ranking among dense models, per-reason instruction gains (one company, 13 positives).
A proper re-run needs pooled v2 labels over several companies: pool (9 methods, top-3 per generator) = 7.4k pairs
for 27 companies; GLM ≈ 0.2 ₽/pair; Gemma ≈ 0.07 ₽/pair but it does not call the tool with the current text-format
few-shots (3/3 smoke answers in plain text).

## Caveats
- Pool bias: positives exist only among events some pooled method retrieved; Qwen3-Embedding-8B is not in the pool yet.
- Labels by one LLM (Gemma) from title + lead; spot-checked against GLM only on 137 pairs.
- Giga ran in fp16 on T4 (the card uses bf16), which may hurt it; its instructions were not tuned beyond the shared set.
- `real` has 33 positives: 1 news ≈ 3 p.p.

## Files
`queries.json` / `query_index.json` — eval queries; `kaggle_embed.py` + `push.py` — Kaggle kernels (one per account,
models on GPU lanes, OOM-safe batching); `evaluate.py` — metrics (`--labels`, `--hybrid a+b`); `judge.py` — pooled LLM labels
(parallel `--workers`); `labels_cheap.jsonl` (Gemma), `labels_agent.jsonl` (GLM check set); `results*.csv`.
Embeddings (`out/`) are not committed: rebuild with `push.py` on Kaggle.
