# Versioned candidate-generator sets

`sets.py`: `VERSIONS` = which generator families feed the candidate pool and each one's limit; `pool(u, version)` →
event × source table (rank inside the source, NaN = not picked). Pool = union, an event keeps every source that found it.
Outputs (`out/`, gitignored: cached standard tops quote company facts): `standard_<u>.json`, `versions_eval.csv`.

| version | sources and limits |
|---|---|
| v1 | standard @200 + factory v2-B (top-13 of every selected generator) |
| v2 | v1 + news-first channel generator @200 (`experiments/news_channels/channel_gen.py`, ensemble, alpha 1) |

## How limits and merging work
- **Standard** (`stage3_news_search/candidate_generators/generators.py`): one generator per stage-2 fact reason.
  Reason weights: own industry 1.0, supplier / customer industry and hypotheses 0.7, bank product 0.5, macro and FTS
  country 0.4, second hop, region, group 0.3; a reason's weight is split equally among its generators (ten countries
  do not outweigh the own industry). In stage 3 itself the budget is [n, l] = [200, 800]: a generator gets
  lo = n·w/Σw guaranteed events and up to hi = l·w/Σw if dense similarity ≥ 0.55. Here (as in judge_v2 «current»)
  the same generators are ranked by the retrieval hybrid and cut by quotas ∝ weight at the smallest budget that gives
  ≥ 200 distinct events.
- **Factory**: a fixed 13 events per selected generator (`QUOTA`), no weights; 14–23 generators → 170–250 events.
- **Channel generator**: its facets' quotas ∝ facet weight (own OKVED 1, region / credit / foreign owner / buyer
  concentration 0.5, FTS / staff 0.3, taxes 0.2, counterparties by link strength × visible flow), ≥ 200 events.
- **Between sources**: plain union with dedup, no cross-source cap → v1 ≈ 330–380, v2 ≈ 455–485 events.
- Note: `pipeline.py` does not use any of these yet — it takes 30 candidates from `stage3_news_search/facet_retrieval`.

## v1 vs v2 vs judge v2 (21.10.2026, no LLM calls)

| company | version | events | labelled | positives / known | pos / labelled | only-channel events: labelled / positives |
|---|---|---|---|---|---|---|
| U_106783 | v1 | 382 | 337 | 45 / 74 | 0.13 | — |
| U_106783 | v2 | 484 | 373 | **49 / 74** | 0.13 | 102: 36 / **4** |
| U_150736 | v1 → v2 | 342 → 463 | 96 → 104 | 9 / 10 → 9 / 10 | 0.09 | 121: 8 / 0 |
| U_212593 | v1 → v2 | 332 → 456 | 89 → 91 | 14 / 16 → 14 / 16 | 0.16 | 124: 2 / 0 |
| U_9250 | v1 → v2 | 373 → 480 | 126 → 126 | 25 / 27 → 25 / 27 | 0.20 | 107: 0 / 0 |

Manual useful set of U_106783 (`judge_v2/useful_negatives.json`, read by hand, not by the judge): v1 18 / 28, v2 19 / 28.
- U_106783: v2 adds 4 positives (+9%) at +27% events; its new events' labelled positive rate (4 / 36 = 0.11) is close
  to the pool's 0.13.
- Other companies: labels exist only for factory events + 50 random, so ~95% of v2's new events are unlabelled →
  gain unknown; labelling them ≈ 410 judge calls ≈ 80–90 ₽ (backlog item 14).
- Decisions (user, 21.10): the ensemble beats e5 alone; the channel generator is valid → v2.

## Caveat: shared channel frame
The A1–H1 checklist was derived from the judge-v2 channel list (17 P&L channels: demand, selling price, market access,
input prices, logistics, staff, taxes, working capital, financing, FX, …), and the factory v2 prompt contains A1–H1
(+ I1–I3) verbatim. Generators and judge look through similar taxonomies, so agreement may partly be agreement with the
frame. The judge's positive rule (potential, significance, mechanism) does not use the channel name, and the judge
never sees channel scores. Checks proposed in backlog item 17 (manual labels, judge without the channel list, accepted
insights as the end metric).

## After labelling the only-channel events (22.10.2026, `label_v2.sh`, 405 of 408 judged, ~118 ₽)
Corpus snapshot of 20.10 (`ALFA_DATA` = the retrieval-v2 worktree's data; the main checkout's news.parquet was rebuilt).

| company | v1 positives | v2 positives | only-channel events: labelled / positives |
|---|---|---|---|
| U_106783 | 45 / 77 | **52 / 77** | 102 / 7 |
| U_150736 | 9 / 13 | **12 / 13** | 121 / 3 |
| U_212593 | 14 / 25 | **23 / 25** | 121 / 9 |
| U_9250 | 25 / 31 | **29 / 31** | 107 / 4 |

- v2 adds +23 positives over 4 companies (+25%) at +27–37% events; only-channel precision 0.025–0.07 vs 0.09–0.20 for
  the v1 pool: the channel generator brings depth, not precision. Some new positives may be judge leniency (see
  generator_factory/README.md), not checked by hand.
- Production port: `stage3_news_search/retrieval_v3` (branch retrieval-v3) — one e5 instead of the ensemble, checklist
  v2: +0 / +3 / +5 / +2 positives so far, 345 of its only-channel events still unlabelled.
