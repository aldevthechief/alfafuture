# Judge v2: multi-dimensional «insight potential» (pilot 21.10.2026)

`judge.py` grades (company, event) pairs with GLM-5.3-Flash on `prompts/judge_v2.md`: explanation first, then
event type, significance 0–3, channel, exposure fact, direction, mechanism strength 0–3, horizon, insight potential 0–3;
6 few-shots from fixed `synthetic_insights` cases + 2 opportunity cases; company = deterministic brief (d).
Positive = potential ≥ 2 & significance ≥ 2 & mechanism ≥ 2 (preliminary).

## Pilot: U_106783, 5 candidate + 5 random events, 35 judge calls, 12.8 ₽
Variants: news input = Gemma summary / GLM summary / raw text (title + first 5000 chars); refusal = short
(«Механизма нет.») vs explained (5 random events, Gemma summary). Langfuse session `judge-20261021-002420`
(cache replay with named traces), paid run `judge-20261021-001904`.

| | Gemma summary | GLM summary | raw text |
|---|---|---|---|
| summary: chars / ₽ / sec | 726 / 0.012 / 3 | 1272 / 0.088 / 13 | — |
| numbers in summary not in the text | 0.1 | 0.6 | — |
| judge: ₽ / sec / out tokens (mean) | 0.32 / 40 / 3.4k | 0.37 / 49 / 4.2k | 0.34 / 41 / 3.6k |
| truncated at 8k tokens | 1/10 | 1/10 | 0/10 |
| agreement with raw text: potential / mechanism | 0.89 / 0.89 | 0.67 / 0.56 | — |

- Significance and event type: identical in all 3 inputs; the positive flag too (only the KRUE localisation decree,
  potential 3 everywhere). Disagreements are only 0↔1 on weak links (DOM.RF leasing bonds, T1 university lab).
- Short refusal: 9/10 «no mechanism» verdicts start with «Механизма нет.» but still explain (490 vs 650 chars):
  the explanation field is where the model reasons. Explained vs short on 5 events: 3/5 equal, no systematic shift.
- FTS rule holds: the model writes «направление неизвестно — допущение»; no «China mention = high grade».
- Reasoning length: median 2.8k completion tokens, p75 5.3k, p90 7.1k; 2/35 hit `max_tokens=8000` while still
  reasoning (empty answer, still billed ~0.6 ₽). Raw replay of one with 16k: finished at 6.0k (5.4k reasoning) —
  long deliberation on a borderline weak link, length varies between runs.

## Full run: U_106783, 103 candidate + 100 random events × 3 inputs, max_tokens 32k (21.10, 133 ₽, 26 min)
Langfuse session / local log run `judge-20261021-011426`. Positive by majority of the 3 inputs.

| | candidates (103) | random (94 judged) |
|---|---|---|
| majority positive | 11 (10.8%) | 2 (2.1%) |
| mean potential | 0.55–0.59 | 0.19–0.23 |

- **Inputs are interchangeable for the judge:** positive flag agrees in 97.5–98% of events for every pair, potential /
  mechanism / significance exact agreement 86–89% for every pair (the pilot's GLM-summary divergence was noise).
  Summaries cost: Gemma 2.2 ₽ (3 s), GLM 15.2 ₽ (11 s) per 193 news.
- **Generators (budget ~100):** positives only from `own_industry` (9/37) and `region` (2/11); `bank_product` 0/18,
  `macro` 0/15, `second_hop` 0/12, `trade_country` 0/10. Ranks of positives within their generator: 0–31.
- **Recall is low:** 2/94 random events are positive (missed by the generators: «щитовое оборудование» in the Egypt
  industrial zone list, Yandex data centres knocked out by drones) → ~170 positives outside the candidate set
  (95% CI ~20–600) vs 11 inside. One company, a generous positive definition — treat as a signal, not a number.
- **Typical positive:** «energy / grid capex → demand for switchgear» via the OKVED + rank-1 fact with one stated
  assumption (buyers unknown). Plausible for this company, but the rubric lets OKVED alone carry mechanism 2.
- **v1 labels vs v2:** of 18 overlapping events v1 (Gemma, 0/1/2) graded 2, only 3 are v2 positives → retrieval
  results on v1 labels (experiments/retrieval) need re-evaluation.
- **Tokens:** judge output median 1.8k, p90 5.2k, p99 9.4k; 9/582 > 8k (would be cut at the old cap), 1 > 16k
  (a reasoning loop to 32k, still empty). 16k is the right default; 32k only buys longer loops.
- **Unlabelled 9/582 (1.5%):** 7 plain-text answers without the tool call on obviously irrelevant news (elections,
  a monument, crime) — non-empty, so cached and the retry repeats them; 1 loop at 32k; 1 tool call missing
  `impact_explanation`. One GLM summary was cut at 16k and stored empty.

Per-field analysis, positives read-out, failed generators, judge as a ranker, insight-agent smoke: [analysis.md](analysis.md).

## Pack vs single judge (21.10, `pack.py`, 5.7 ₽)
U_106783, all 13 majority positives + 23 random negatives, 7 random packs of 5–6 news (title + Gemma summary), GLM,
max_tokens 128k; reference = single-judge gemma_summary verdicts. Run `pack-20261021-095107`.

| | pack | single |
|---|---|---|
| vs majority label: precision / recall | 1.00 / 0.77 (10 of 13) | 1.00 / 0.92 (12 of 13)* |
| ₽ per event | 0.16 | 0.20 |
| output tokens per call | 7.4–13.2k (≈2k per news) | 2.5k mean |
| explanation, chars | 466 | 628 |

\* the majority label includes the single gemma_summary verdict itself, so single is favoured.
- Agreement pack vs single: potential 0.81, mechanism 0.81, significance 0.94 — vs 0.83 / 0.81–0.83 between two
  single inputs on the same 36 events: packs add little disagreement, but are **stricter** (potential −0.22 on
  average vs −0.06 between single inputs) and dropped 3 borderline positives (Yandex digest, Moscow forecast, FAS
  grid-protection tariff: «protection costs, not switchgear» vs single «restoration = switchgear purchases»).
- Packs save only ~20%: the per-news reasoning (output) dominates the cost, the shared prompt (input) is small.
- One pack in 7 returned an item without `channel` → retried without cache, fine. Max output 13.2k: packs of 5 need
  a 32k cap, the 16k default is borderline.

## Known issues
- Fixed 21.10: FileLLMCache no longer stores cut / empty answers. Still open: a text answer instead of the tool call
  is cached (retry should bypass the cache); an empty summary is stored in summaries_glm.
- LangChain drops GLM `reasoning_content`; Langfuse shows only the final answer and no token usage/cost.
