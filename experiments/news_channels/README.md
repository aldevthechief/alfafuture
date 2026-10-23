# News-first candidate generation: channel checklist × news (pilot, 21.10.2026)

Idea: score every event once against a general checklist of channels through which news can change a company's
money ([checklist.md](checklist.md), 21 direction-neutral items A1–H1). The per-news channel vector is later matched
against each company's exposure to the same channels → candidates for all companies at the price of one pass over news.

## Files
- `channels.py` — checklist `ITEMS`, news sets (judge-labelled events of U_106783; 398 random events frozen in
  `out/random_events.json`), `news_text` = title + first 2000 chars of the event's first report (same input for all models).
- `glm_score.py` + `prompts/news_channels.md` — GLM-5.3-Flash, one call per event: event summary, significance 0–3,
  hits (item, change, winners, losers, assumption), scores 0–3 for all 21 items, industries / goods / regions /
  countries / organisations. Out: `out/glm_scores.jsonl` (key = event).
- `kaggle_rank.py` + `push.py` — Kaggle kernels (accounts kirpich682: giga-10B on 2×T4; alexandrzolotukhin: two lanes),
  score every (news, item, query variant) pair: cross-encoders read query = item and document = news in one context →
  sigmoid(logit); instruct bi-encoders put the item into the `Instruct: …\nQuery: <item>` query → cosine. Variants:
  `item` (item text) and `question` («Может ли эта новость через канал «…» заметно изменить …»). Texts are rebuilt
  in the kernel from the `alfa-hack-ru-business-news` dataset (same file as `data/news/news.parquet`; checked identical);
  only event / news ids go into the config, no company data. Out: `out/rank/<slug>.npz` (score [news, item, variant]).
- `evaluate.py` → `out/eval.csv`.

## GLM (398 random events, 87 ₽ incl. 50-event pilot = 0.215 ₽ / call, 2 events unscored)
- 15% of random events get any item ≥ 2 (59), 30% all zeros; chronicle / politics / quotes → 0, FX quote → D3 = 1.
- Hits per hit-news: 1 item 36, 2 items 14, 3+ 9. Most frequent: B1 16, F1 13, A4 11, C3 8, A2 7. **No positives at
  all for C4, B2, D4, D1** in this sample → those items are not evaluated below.
- Item score ≈ gated by significance: max item ≥ 2 in 55/63 events with significance ≥ 2 and in 4/335 with ≤ 1. The
  item score mixes «is there a channel» with «is the event big» (by the prompt's design).
- Spot check of all hits: mostly plausible (oil / Hormuz → B1, Ineos → B1 + A4, MROT → B4, FAS fine → C3); a few
  overreaches (school AI course → C3 = 2, a night digest → C5 = 2).

## Rankers vs GLM and vs judge v2
GLM side: positive pair = GLM item score ≥ 2 (12 items with ≥ 3 positives; small → ±0.03 is noise).
Judge side: 575 labelled events of U_106783 (the labels file grows with another session), 60 judge positives.

| model / variant | pair AUC | item AUC centred | any AUC | top-1 | recall@10% | judge pos AUC | judge channel top-3 |
|---|---|---|---|---|---|---|---|
| bge-reranker-v2-m3 / item | 0.81 | 0.73 | 0.66 | 0.36 | 0.56 | 0.61 | 0.79 |
| e5-large-instruct / item | **0.88** | **0.86** | 0.78 | 0.37 | **0.60** | 0.60 | 0.74 |
| e5-large-instruct / question | 0.86 | 0.87 | 0.78 | 0.44 | 0.54 | 0.59 | 0.77 |
| giga-emb-10b / item | 0.83 | 0.82 | 0.77 | 0.41 | 0.52 | 0.62 | 0.71 |
| giga-emb-480m / item | 0.87 | 0.82 | **0.80** | 0.29 | 0.58 | 0.58 | 0.74 |
| ru-reranker-edge-150m / item | 0.78 | 0.82 | 0.70 | 0.29 | 0.52 | 0.60 | **0.81** |
| random | 0.50 | 0.52 | 0.51 | 0.05 | 0.05 | 0.50 | 0.38 |
| **ensemble, mean pct rank of 5 (item)** | **0.90** | **0.90** | **0.81** | **0.48** | **0.70** | 0.63 | 0.81 |

pair AUC = all (news, item) pairs vs GLM ≥ 2; centred = score minus the news' mean over items (only the item choice
within a news counts); any = max over items vs «GLM gave any hit»; top-1 = ranker's top item equals GLM's top item on
hit news; recall@10% = share of GLM-positive pairs inside the top 10% of news per item; judge channel top-3 = the
judge's channel (mapped onto items) is among the ranker's top-3 items. Full table: `out/eval.csv`.

Findings:
1. **Rankers reproduce GLM's channel assignment well**: pair AUC 0.78–0.88 per model, 0.90 for the ensemble;
   the ensemble's top-10% per item holds 70% of GLM-positive pairs. Item choice within a news is real (centred AUC
   0.73–0.87; top-1 0.29–0.48 vs 0.05 random).
2. **Rankers do not see magnitude**: «does this news matter at all» AUC 0.66–0.81. They measure the topic; a minor
   statement about duties scores like a decided duty. Significance has to come from elsewhere.
3. **Against the company-specific judge, channel scores alone are weak** (positive AUC 0.58–0.64), as expected without
   the company side; but the judge's channel lands in the rankers' top-3 items for 71–81% of events (random 38%).
4. Bi-encoder `item` variant ≥ `question`; the instruction-free bge-reranker is the weakest on item choice (0.73
   centred). giga-10B is not better than giga-480M or e5-large-instruct. Best single model = e5-large-instruct / item.

## Full corpus, done (21.10.2026)
`push.py --all` → kernels `alfa-rank-{giga,small}-all`: all 9,147 news × 21 items, `item` variant; T4 time: e5 3 min,
giga-480M 14 min, giga-10B 25 min (2×T4), ru-reranker-edge 33 min, bge-reranker 54 min. `ensemble.py` →
`out/channels_news.parquet` (news × A1..H1, mean over the 5 models of the per-item percentile rank over all news) and
`out/channels_events.parquet` (event = max over its reports, 8,037 events). Check vs pilot on the same 912 news:
identical for 4 models; e5 Spearman 0.992 (fp16 batch-shape noise, max |Δcos| 0.013). Top news per item are on topic
even for items without GLM positives (D1: CBR rate signals, deposits; C4: subsidised agro credit; B2: utility tariff
indexation; D4: working-capital loans, payment deferrals).
Caveats: a percentile treats every item alike (top 10% = ~915 news per item, too many for rare channels like D1) and
carries no magnitude → per-item calibration (needs GLM positives for every item) and a significance signal are next.

## Proposal: full corpus (8,037 events)
- **Rankers, all events × 21 items**: free; measured 912 news in 1.5–12 min per model on T4 → ~8k events ≈ 15 min (e5,
  giga-480M) to ~1.7 h (bge-reranker). Ensemble of e5 + giga-480M + ru-reranker-edge covers most of the gain.
- **GLM on everything**: 8,037 × 0.215 ≈ **1,730 ₽** = 3.5 daily quotas → not worth it.
- **Recommended: distil GLM into a cross-encoder.** GLM teacher on ~1,000 more events chosen by the ensemble (top
  per item, so every item gets ≥ 20 positives, incl. C4/B2/D4/D1) ≈ **215 ₽**; fine-tune ru-reranker-edge-150m or
  bge-reranker-v2-m3 on (item, news) → GLM score 0–3, plus a significance head (target = GLM significance), on Kaggle;
  then score the full corpus for free. Cheaper first step if needed: GLM only on events whose ensemble max is in the
  top 20% (~1,600 events ≈ 345 ₽ — too much for one day).

## Proposal: company-side matching (no LLM)
score(company, news) = Σ_item channel(news, item) × exposure(company, item) × affinity(news, company) × significance(news)
- exposure from facts: D1/D2 ← credit / leasing products; B4 ← headcount; C5 ← foreign or state owner; D3 / E1 / A4 ←
  FTS presence (weak, capped: no direction); D4 ← customer concentration; the rest ← industry (own OKVED = seller
  side, supplier OKVEDs = buyer side for B1/B3/C2, customer OKVEDs = demand side for A*).
- affinity: embed GLM's `industries` / `winners` / `losers` strings (or the news itself on the ranker path) against
  OKVED names of the company, its suppliers and buyers (e5 / giga, ~1.6k OKVED names embedded once) + region match.
- check on U_106783: recall of the judge positives at budget 100–200 vs the company-first generators (11–32 of 38).

## Channel generator «channel × company exposure» vs judge v2 (21.10.2026, `channel_gen.py`, no LLM)
Facets from stage-2 facts: own OKVED (13 industry items), counterparty OKVEDs (weight = link strength × visible flow
share of revenue, off below 5%), region, attribute channels (credit/leasing → D1 D2, foreign owner → C5, FTS → D3 E1,
staff > 250 → B4, buyer concentration → D4, C1 for all). Facet score = max item percentile × e5 affinity^alpha;
quota union at a budget. alpha = ∞ = same facets without channels (≈ current facet generators), alpha = 0 = channels
without affinity. Out: `out/channel_gen_eval.csv`, `out/channel_gen_union.csv`.

U_106783 (735 labelled events, 74 positives; the only company with dense labels):

| set | events | labelled | positives | share of 74 | pos / labelled |
|---|---|---|---|---|---|
| current @200 | 200 | 156 | 12 | 16% | 0.08 |
| hand-written @200 | 203 | 202 | 32 | 43% | 0.16 |
| factory v1 (16 × 13) | 183 | 183 | 44 | 59% | 0.24 |
| factory v2-B (21 × 13) | 250 | 248 | 45 | 61% | 0.18 |
| channels only (alpha 0) @200 | 200 | 109 | 15 | 20% | 0.14 |
| affinity only (alpha ∞) @200 | 202 | 133 | 21 | 28% | 0.16 |
| **channel × affinity (alpha 1), ensemble @200** | 206 | 137 | **29** | 39% | **0.21** |
| channel × affinity (alpha 1), e5 only @200 | 205 | 151 | 24 | 32% | 0.16 |

- Channels add to the same facets: 21 → 29 positives at 200 (+38%), alpha 1 ≈ 2. Above current (12), near
  hand-written (32), below the agentic factory (44–45) — with zero LLM calls and 1/3 of its events unlabelled.
- e5 alone (prod constraint): 24 vs 29 on U_106783; on the other companies mixed (U_150736 e5 better, U_212593 worse).
- Overlap with factory v2-B: 37% of the channel pool; among the rest 4 labelled positives, but 68 of its 129 new
  events are unlabelled → the real gain is unknown until they are judged (~15 ₽).
- Per facet at 200 (labelled positives / labelled): own 13/48, foreign owner 5/14, region 4/16, credit 3/16; FTS,
  staff, taxes, buyer concentration 1 each — as in the factory: generic attribute channels yield little, except owner.
- Judge agreement (AUC of the facet score over labelled events): 0.64 (U_106783), 0.72–0.76 on the other three (their
  pools include 50 random easy negatives).
- Other companies (U_150736, U_212593, U_9250): labels exist only for factory v2-B events + 50 random, so only 22–42
  of the channel pool's 200 events are labelled and no new positives can be counted. On the labelled part the channel
  pool picks the better factory events (pos / labelled 0.29 vs 0.16 restaurants, 0.35 vs 0.20 construction, 0.08 vs
  0.10 packaging). Counterparty facets are off or tiny there: visible payment flows are < 1–8% of revenue.
- Decisions (user, 21.10): ensemble > e5 alone; the channel generator is valid → included as a source of candidate set
  v2 (`experiments/candidate_sets`). Item expansion / decomposition was not discussed yet.

## Query-side ablation: instructions and few-shot exemplars (22.10.2026, `instructions.py`, no LLM)
398 GLM-scored random events; e5 on CPU over the retrieval doc embeddings (title twice + text), cross-encoders on
Kaggle (`alfa-rank-ablation`, raw logits). Exemplars = 3 typical headlines per item written by Claude (`EXAMPLES`).

| model / query | pair AUC | item AUC centred | any AUC | top-1 | recall@10% |
|---|---|---|---|---|---|
| e5 / no instruction | 0.853 | 0.848 | 0.742 | 0.29 | 0.56 |
| e5 / generic web instruction | 0.862 | 0.856 | 0.747 | 0.39 | 0.58 |
| e5 / current instruction | 0.883 | 0.860 | 0.793 | 0.36 | 0.62 |
| e5 / detailed task instruction | 0.893 | 0.863 | 0.797 | 0.34 | 0.62 |
| e5 / current + item and examples in one query | **0.895** | 0.849 | **0.823** | **0.46** | **0.64** |
| e5 / current + max over item and 3 examples | 0.891 | 0.824 | 0.823 | 0.37 | 0.63 |
| bge-reranker / item | 0.829 | 0.848 | 0.694 | 0.36 | 0.56 |
| bge-reranker / item + examples in one query | 0.779 | 0.822 | 0.672 | 0.32 | 0.50 |
| bge-reranker / max over 3 examples | **0.850** | 0.837 | **0.753** | **0.44** | **0.60** |
| ru-reranker-edge / item | 0.826 | **0.873** | 0.735 | 0.29 | 0.52 |
| ru-reranker-edge / item + examples in one query | 0.823 | 0.870 | 0.716 | 0.20 | 0.50 |
| ru-reranker-edge / max over 3 examples | 0.849 | 0.847 | 0.738 | 0.36 | 0.56 |

- The instruction matters for e5 (none 0.853 → current 0.883 → detailed 0.893); differences ≤ 0.03 are within noise
  (59 hit news).
- Exemplars help: e5 with examples in the query gains on «any» (0.79 → 0.82) and top-1 (0.36 → 0.46); cross-encoders
  only gain when every exemplar is a separate query (max): bge «any» 0.69 → 0.75, top-1 0.36 → 0.44; examples packed
  into one long query hurt them (trained on short queries).
- With raw logits the cross-encoders' item choice (centred AUC 0.85–0.87) is on par with e5; the pilot's 0.73 for
  bge came from the sigmoid saturating near 0. The ensemble (per-item percentiles) is unaffected.
- Caveat: the exemplars were written after seeing the pilot's GLM hits (e.g. a MROT headline) → mild leakage; for
  checklist v2 write exemplars blind and validate on another labelled set (judge-labelled events).
