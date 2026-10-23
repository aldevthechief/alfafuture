# Judge v2 full run: per-field analysis (U_106783, 21.10.2026)

Data: `labels_v2_U_106783.jsonl`, 601 verdicts = 203 events × 3 news inputs (Gemma summary / GLM summary / raw text),
103 candidates (generators, budget ~100) + 100 random events. «Consensus» = mean of numeric fields / mode of
categorical fields over the 3 inputs.

## 1. Fields one by one
| field | candidates | random | reading |
|---|---|---|---|
| significance (mean) | 1.27 | 0.92 | 0: 11 / 26; ≥2: 39 / 14 events. Correctly orders event types: quote 0.06 < «other» 0.46 < statement 1.07 < macro stat 1.13 < corporate 1.16 < regulator decision 1.59 < draft 1.62 < incident 1.80 < market shift 1.88 < macro decision 2.27 |
| mechanism (mean) | 0.72 | 0.22 | 0: 46 / 83; 1: 42 / 14; ≥2: 15 / 3 events. The field that separates candidates from random most |
| insight potential (mean) | 0.57 | 0.20 | 0: 61 / 85; 1: 27 / 10; ≥2: 15 / 5 |
| channel | demand 32, FX 14, staff 5, other 6, none 46 | demand 8, none 83 | FX channel: 45 verdicts, mean significance 0.56 — almost always a daily quote |
| direction | opportunity 29, mixed 22, risk 6 | opp. 7, mixed 4, risk 6 | positives are mostly «opportunity» (grid / energy capex → demand) |
| horizon | year+ 33, now 15, quarter 9 | year+ 14 | — |
| event type | statement 36, corporate 14, macro stat 13, decision 11, quote 8 | statement 36, «other» 26 | random = chronicle («other»), candidates = economic news |

## 2. Correlations and what decides a positive
- Spearman: mechanism ~ potential **0.85**, significance ~ potential 0.48, significance ~ mechanism 0.34.
- Potential is almost a function of mechanism: mechanism 0 → potential 0 (374/374); mechanism 1 → potential 1 (107/172)
  or 0 (55); mechanism 2 → potential 2 (42/49). The «insight» axis adds little beyond the mechanism; significance is the
  independent axis.
- Gates (per verdict): potential ≥ 2 in 59 verdicts; blocked by significance < 2 in 7, by mechanism < 2 in 10.
  No verdict has significance ≥ 2 & mechanism ≥ 2 & potential < 2.
- Facts the judge links through: OKVED 137 times (with «rank 1» 63), FTS countries 52 (mechanism 1.0, potential 0.35 —
  the FTS cap works), staff 22, Alfa products 19, region 19, ownership 13, buyers / concentration **6**. The verdicts
  rest on «industry + №1 in the industry»; the payment facts of this company are too weak (named buyer industries cover
  14% of the flow, all «слабая») to carry anything.

## 3. Positives read one by one (13, majority of 3 inputs)
All but two follow one template: «energy / grid capex → demand for switchgear», assumption «the company sells to grids»
(buyers are not visible). My grade as a feed item / as an insight:
- strong: KRUE localisation decree (product-level regulation, 3/3); «На пределе прочности» (Lenenergo capex +20%,
  22 substations); grid-sector regulation at RЭН (TSO consolidation under Rosseti, 41 bn ₽ «excess» capex cut);
  Egypt industrial zone lists «щитовое оборудование» (names the product).
- plausible: FAS — grid protection costs into the tariff (2 reports of one event; the pack judge read it as
  «protection, not switchgear»); distributed generation integration (1.9 GW south); EV fast-charger subsidies;
  storage systems (250–600 MW); Yandex data centres knocked out (replacement orders).
- weak / generic: Moscow 2027–29 forecast; Technopolis Moscow (250 companies, battery plants); special power tariffs
  for miners. Relevant topic, but the «insight» is the same generic demand chain.
So precision as «relevant to this company» is high (13/13 on-topic), as «non-obvious insight» ~4–6/13.

## 4. Failed generators: is the reason, the search, the judge or our notion wrong?
| generator (n) | what it retrieved | verdict |
|---|---|---|
| `macro` FX (15) | 12 daily FX quotes / analyst one-liners, 2 real FX shifts (Minfin purchases ×6.3, USD to ~85) | **search**: the query «курс рубля юань доллар …» matches quote notes; **reason** weak: FTS without direction caps FX at mechanism 1. The 2 FX shifts are fair mid-rank news; the judge gives significance 1.7, mechanism 1 — right for an insight, low for a feed |
| `trade_country` (10, 1 per country) | Abkhazia tourism, Kazakhstan–Italy trade, Baltic grain transit, US–India talks, Trump pausing «hell sanctions» | **reason**: 11 FTS countries (incl. counterparties') with unknown direction, one event each; **search**: «<country> торговля поставки пошлины санкции логистика» returns any country news. Only the sanctions pause matters (significance 3, mechanism 1) |
| `second_hop` (12) | soft drinks, advertising market, rail grain, consumer appliances | **reason + search**: OKVED fragments («напитками», «рекламных агентств») of counterparties' counterparties with tiny flows. Nothing relevant; one hit («Yandex data centres attacked → advertisers pause») is the same event the random positive covers, judged from the advertising angle → 0 |
| `bank_product` (18) | leasing: GTLK helicopters, Rosagroleasing, leasing-company searches; payroll: labour-market notes | **reason mapping is wrong**: the company is a *lessee*; leasing-industry news is not about its leasing terms. Labour-market news get mechanism 1 (staff channel, generic). One positive came from here by accident (CBR risk weights for infrastructure-protection loans, rank 8) |
The judge is right on almost all of them. The failures come from the generators: reasons built on facts without
measured strength (FTS countries, second hop, product flags) and queries that are topic words, not «what event changes
this company's conditions».

## 5. Judge as a ranker
Tiers (consensus): A positive 11 / 2 (candidates / random), B mechanism ≥ 1 & significance ≥ 2: 5 / 7,
C mechanism ≥ 1 & significance < 2: 34 / 7, D mechanism 0: 53 / 84.
Tier C holds real feed items that the positive gate drops: «Башкирэнерго» smart grids (mechanism 2, potential 2,
significance 1.3), electricity demand forecast (mechanism 2), competitor IEK GROUP forum, «saved 9 mn ₽ on
switchboards», Rosseti Tyumen reliability works — next to 12 FX quotes (mechanism 1, significance 0).
So for ranking: the gate «significance ≥ 2» is too strict for own-industry news, and «potential» duplicates
«mechanism». A ranking score should lead with mechanism (relevance to this company), then significance, not require
an insight. Candidate formula to test: `mechanism + 0.5·significance` (FX quotes 1.0, Bashkirenergo 2.7, KRUE 4.0).

## 6. Insight agent (stage 5 InsightJudge → stage 6 InsightWriter), smoke on 2 events, 3.4 ₽
Blind (no judge v2 verdict as input), no bank analytics (OCR not done).
- Yuan quote 12.59: 7 tool calls (same event, follow-ups, company facts, bank search), `is_insight=false`, score 1,
  «рядовая дневная котировка … направление ВЭД неизвестно» — agrees with judge v2.
- KRUE decree: `is_insight=true`, score 2; writer produced a full insight for CFO; the checker flagged claims without
  a source («допуск к закупкам с национальным режимом», «вытесняет импорт») and the FTS-direction assumption stays
  hedged («вероятные источники комплектующих»). Review after one revision: still not ok.

## 7. Useful negatives and 10 proposed generators (21.10)
`useful_negatives.json`: 28 judge-v2 negatives of U_106783 that are useful for the feed (7 strong, 12 medium, 9 weak),
with event ids, my note and the generator that should find them. Found by: own_industry 12, random sample 10 (missed by
every generator), bank_product 2, macro 2, region 1, trade_country 1.

How the evaluated hybrid searches now: per generator one query; dense e5-large-instruct / qwen3-emb-0.6b get
`Instruct: <reason instruction>\nQuery: <fact>`, BM25 (pymorphy3 lemmas) gets the fact only; documents =
`title. title. text[:2000]`; RRF (k=60) of the 3 rankings, event = max of its news, then the generator's quota.
No HyDE. Note: `stage3_news_search/candidate_generators/generators.candidates()` still uses TF-IDF + FRIDA (+ optional
HyDE), not this hybrid.

Proposed generators (each a reason + 2–3 facet queries fused by RRF inside the generator):
| # | generator | fact it hangs on | facet queries (examples) | would catch |
|---|---|---|---|---|
| 1 | ownership_foreign | ownership: foreign | dividends to non-residents, «С» accounts, exit fee; temporary / external management; foreign-capital restrictions | 994, 226 |
| 2 | buyer_capex | OKVED → buyer sectors (LLM map per OKVED, cached) | grid companies' investment programmes, substations; data centres / plants power supply; energy storage, distributed generation | 2564, 6762, 676, 270, 480, 2576, 4163, 66 + 8 positives |
| 3 | product_regulation | OKVED / OKPD of products | localisation points, PP 719, industrial registry; technical regulations, certification; technology fee | KRUE, 5411 |
| 4 | competition_import | OKVED | import substitution of the product, foreign brands leaving; industry players' news; prices / dumping | 2130, 6478 |
| 5 | input_costs | OKVED → main inputs (LLM map) | electricity tariffs / wholesale prices; metals (copper, steel) prices; components shortage | 4578, 2465 |
| 6 | sales_regions | share of counterparties outside the home region | buyer-sector projects in any region instead of «<home city> экономика» | 676, 2564 |
| 7 | trade_regime | FTS flag (country-agnostic, replaces 11 per-country generators) | sanctions / secondary sanctions, duties, parallel import, payments with foreign partners | 6908, 2446 |
| 8 | fx_shift | FTS flag (replaces `macro` FX) | Minfin FX purchases, budget rule, CBR on the rouble, devaluation forecasts; routine quote events filtered by a deterministic title rule | 3048, 2814 |
| 9 | financing_terms | Alfa products (lessee / borrower, replaces «leasing industry») | lending rates for business, subsidised rates, CBR risk weights, leasing for lessees, key rate | 1746, 5808 |
| 10 | state_budget | scale / buyer sectors with state demand | federal / regional budget, national projects, procurement cuts | 994, 2391, 641 |
To test offline before any LLM run: encode the new queries locally (e5 / qwen, documents already on Kaggle) and measure
recall of the 13 positives + 28 useful negatives at budget 100 / 200 against the current generators.

## 8. Offline check of the proposed generators (`experiments/retrieval/new_generators.py`, no LLM)
Queries encoded locally (e5-large-instruct, qwen3-emb-0.6b; cosine with the Kaggle query vectors ≥ 0.9997), documents
from Kaggle, BM25 lemmas, RRF over every (method, facet) ranking, budget = smallest quota union with ≥ 100 / 200 events.
Targets: 13 positives + 28 useful negatives = 41; **29 of them come from the current candidate set (budget ~100)**,
so the current set finds them by construction; the new facet queries were written after reading the targets.

| set | budget | all targets /41 | random-origin targets /12 | labelled events | targets among labelled |
|---|---|---|---|---|---|
| current, original weights | 100 / 200 | 29 / 29 | 0 / 0 | 100 / 104 | 0.29 / 0.28 |
| current, equal weights | 100 / 200 | 9 / 20 | 0 / 0 | 53 / 82 | 0.17 / 0.24 |
| new, 1st facet only | 100 / 200 | 21 / 28 | 2 / 3 | 37 / 52 | **0.57 / 0.54** |
| new, 3 facets RRF | 100 / 200 | 19 / 28 | 1 / 3 | 38 / 59 | 0.50 / 0.47 |
| current + new | 100 / 200 | 18 / 27 | 1 / 1 | 50 / 74 | 0.36 / 0.36 |

- Where labels exist, the new generators' picks are useful about twice as often (0.5 vs 0.28), and they reach 3 of 12
  targets the current set cannot (Egypt zone, budget 2027–29, healthcare cut) — but 70% of their events are unlabelled,
  so their real precision / recall is unknown; `ownership_foreign` has 0 labelled events (and misses Auchan).
- 3 facet queries fused by RRF give nothing over the first facet alone (slightly worse at 100): one good query wins.
- Weights matter most: equal weights sink the current set (11 FTS countries take 11 shares) — 9/41 vs 29/41 at 100.
- A fair comparison needs judge-v2 labels on the unlabelled picks of the new set (~150 events).

## 9. New generators, labelled (21.10, 199 events, 49.6 ₽, run `judge-20261021-105342`)
Every event of the two new sets at budget 200 got a judge-v2 verdict (GLM, Gemma summary); old events use their
gemma_summary verdict, so one verdict per event for all 398 labelled events (`compare_sets.py`).

| set | budget | labelled | positives /38 | relevant (mech ≥ 2) /44 | positives per labelled |
|---|---|---|---|---|---|
| current, original weights | 100 | 100/100 | 11 | 15 | 0.11 |
| current, original weights | 200 | 139/200 | 11 | 15 | 0.08 |
| new, 1st facet only | 100 | 108/108 | 15 | 19 | 0.14 |
| new, 1st facet only | 200 | 205/206 | 27 | 32 | 0.13 |
| **new, 3 facets RRF** | 100 | 108/108 | **20** | **24** | **0.19** |
| **new, 3 facets RRF** | 200 | 202/203 | **32** | **38** | 0.16 |
| current + new, 3 facets | 100 / 200 | full | 18 / 31 | 22 / 37 | 0.18 / 0.15 |

- The new set finds ~2× the positives of the current one at budget 100 and ~3× at 200; 26 positives appear only in the
  new sets, all of them events nobody had seen when the queries were written (so not just the leakage of §8).
  The current-only events (105 at 200, 45 labelled) hold 1 positive (Moscow 2027–29 forecast).
- With full labels, **3 facets RRF beat the single facet** (20 vs 15 at 100, 32 vs 27 at 200) — §8's «no gain» was an
  artefact of missing labels.
- New positives by topic: foreign-owner regime (temporary management of Tarkett / Auchan / Nestle, Kremlin and RSPP on
  criteria, asset-seizure risk — 7), energy capex (RusHydro, Tsivilev «new investment cycle», 4.6 GW, Amur, GEH Moscow
  stations, Samara wind, RZD capex — ~10), product regulation («Честный знак» marking of relays / sockets, «второй
  лишний» extended to 35 items), state budget (2027–29 guidelines, tech-leadership money, new regions), costs (housing
  & utility tariffs, aluminium prices), EU 22nd sanctions package.
- Caveats: one company; the current set at 200 still has 61 unlabelled events; recall is relative to the labelled
  pool, which now leans toward the new sets; the facet queries are hand-written for OKVED 27.12 (in production: an LLM
  map OKVED → buyer sectors / inputs / product regulation, cached per OKVED); several foreign-owner positives are the
  same trend in different news (dedup by topic needed in the feed).

## 10. Who finds what, and do generators overlap (`overlap.py`)
- Inside a set the generators almost never share events: at budget 200, 182 of 203 events of the new set and 191 of
  200 of the current set come from a single generator. Current ∩ new = 36 events at 100, 90 at 200 (own_industry and
  labour/payroll are the same queries in both).
- Deeper retrieval (top-50 per generator): 287 of 496 generator pairs share nothing. Redundant: the 11 per-country
  `trade_country` generators among themselves (Jaccard 0.2–0.5: the template words «торговля поставки пошлины санкции
  логистика» outweigh the country), buyer_capex ~ sales_regions 0.39, macro FX ~ fx_shift 0.39, competition_import ~
  input_costs 0.23.
- Positives per generator, new set at 200 (quota 19 each): buyer_capex 8, own_industry 7, ownership_foreign 6 (all
  unique), input_costs 5, sales_regions 5, product_regulation 4, state_budget 3, trade_regime 2, labour 1,
  competition_import 0 (2 relevant), financing_terms 0, fx_shift 0. Current own_industry: 7 positives in its top-19,
  9 in its top-72 — diminishing returns from a big quota.
