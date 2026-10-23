# Agentic factory of candidate generators

`factory.py`: GLM-5.3-Flash gets the company profile (deterministic brief from stage-2 facts) and a checklist of 10
exposure channels (`prompts/factory.md`), probes its own queries on the news pool with `probe` (top-10 events + overlap
with earlier probes, same hybrid as experiments/retrieval: e5-large-instruct + qwen3-emb-0.6b + BM25 lemmas, RRF) and
returns 10–18 custom generators (reason, fact, encoder instruction, 1–3 event-style queries). Pool = equal-weight
quota union at budget 100 / 200, alone or with the standard own_industry + labour generators. The agent never sees
judge labels. Specs and unlabelled events go to `out/` (gitignored: they quote case facts).

## First run, U_106783 (21.10, run `factory-20261021-113733`, 3.5 ₽, 7 min, 7 GLM calls, 43 probes, 16 generators)
Generators: grid capex (Rosseti, generation, substations), grid reform & tariffs, capacity shortage / regional schemes,
data centres, construction & new plants, oil-gas & mining projects, transport (RZD, metro, electrification),
KRU / KRUE localisation & «второй лишний», foreign owner (government commission, external management), competition
(ex-Schneider / ABB plants, switchgear market), metals (copper, aluminium, electrical steel), key rate & leasing,
staff & wages, taxes, electrical-equipment market, FTS sanctions & logistics (one generator).

| set | budget | events | labelled | positives /38 | relevant /44 |
|---|---|---|---|---|---|
| current | 100 / 200 | 100 / 200 | 100 / 139 | 11 / 11 | 15 / 15 |
| hand-written (analysis.md §7, 3 facets) | 100 / 200 | 108 / 203 | 108 / 202 | 20 / 32 | 24 / 38 |
| factory only | 100 / 200 | 106 / 210 | 58 / 92 | **21** / 24 | 26 / 29 |
| factory + own_industry + labour | 100 / 200 | 111 / 208 | 63 / 98 | 21 / 25 | 26 / 31 |

- With no leakage the factory matches the hand-written set at budget 100 (21 vs 20 known positives) with half of its
  events unlabelled; at 200 it trails (25 vs 32) while 118 of its events are unlabelled — real recall unknown.
- Missed known positives at 200 (13): state budget (2027–29 guidelines, tech-leadership money, new regions, deficit,
  Minfin floaters — 5; the agent built «taxes» but no budget generator), housing-utility tariffs, aluminium court case
  (via EV cost), «Честный знак» marking of relays, EU 22nd package, a foreign-investor lawsuit, Tsivilev «new investment
  cycle», Moscow forecast, the Yandex digest.
- Next: label the 118 unlabelled events (judge v2, ~30 ₽); prompt v2 — one generator per checklist channel unless no
  fact supports it (budget was skipped), marking / fees explicitly in channel 2; check run-to-run variance (2–3 runs).

## Finish vs dropped probes, labelled (21.10, run `judge-20261021-122455`, 178 events, 43 ₽)
Every event in the top-13 of each of the 16 finish generators and of the 31 probes the agent did not keep got a
judge-v2 verdict (Gemma summary). Pool now: 575 labelled events, 60 positives, 72 relevant (mechanism ≥ 2).

| set | events | positives /60 | relevant /72 | positives per event |
|---|---|---|---|---|
| current generators @200 | 200 | 11 (18%) | 16 | 0.07 |
| hand-written (analysis.md §7) @200 | 203 | 32 (53%) | 38 | 0.16 |
| **factory finish** (16 × 13) | 183 | **44 (73%)** | 54 | **0.24** |
| factory finish + dropped probes | 292 | 49 (82%) | 61 | 0.17 |
| hand-written ∪ factory finish | 313 | 53 (88%) | 64 | 0.17 |
| all sets | 486 | 57 (95%) | 69 | 0.13 |

- The factory's own selection was right: the 109 events only dropped probes bring hold 5 positives (0.05 per event).
- Best generators: capacity shortage / regional schemes (9 of 13), new power plants (dropped draft, 8), grid reform,
  foreign owner, grid capex, data centres, construction.
- Factory misses that the hand-written set has: state budget, «Честный знак» marking, EU sanctions package, the
  «new investment cycle» statement — channels absent from its checklist (budget, fees / marking, sanctions).
- 4 of the 48 finish queries were never probed (rewritten at `finish`).
- Recall is relative to a pool built from these sets; the random sample (2 positives in 94 events) says many
  positives exist outside every set.

## v2: generic prompt (no OKVED hints), 24 channels, business-model card, finish by probe id (21.10)
Prompt `prompts/factory.md` (v1 kept as `factory_v1.md`): checklist A1–H1 of experiments/news_channels + I1 region,
I2 group, I3 strong counterparties' industries; `note_business_model` first; `probe` numbered, reports new events vs
earlier probes; `finish(selected=[probe_id…], coverage=[per channel])` — queries come from the probe log.
Run A = generic, run B = generic + few-shots from other industries (`prompts/factory_fewshots.md`: dairy producer,
freight carrier). Both: ~3.5 ₽, no invalid probe ids, card written, every channel covered or explained.
Their 162 unlabelled events judged (run `judge-20261021-161650`, 34 ₽). Pool: 735 labelled, 74 positives, 86 relevant.

| set | events | positives /74 | relevant /86 | positives per event |
|---|---|---|---|---|
| current @200 | 200 | 12 (16%) | 17 | 0.08 |
| hand-written @200 | 203 | 32 (43%) | 38 | 0.16 |
| v1, OKVED hints in the prompt (16 × 13) | 183 | 44 (59%) | 54 | **0.24** |
| v2-A generic (23 × 13) | 271 | 42 (57%) | 50 | 0.15 |
| v2-B generic + few-shots (21 × 13) | 250 | **45 (61%)** | 53 | 0.18 |
| A ∪ B | — | 55 (74%) | — | — |
| v1 ∪ A ∪ B | — | 66 (89%) | — | — |

- Removing the OKVED-specific hints costs no recall (A −2, B +1 positives) but precision: the generic channels
  (taxes, key rate, FX, logistics) each take a quota and bring 0–1 positives. Few-shots from other industries help
  (B vs A: +3 positives, 0.18 vs 0.15).
- v2 closes v1's gaps (budget, «Честный знак» marking, EU sanctions, utility tariffs, FRP loans) and loses v1's
  knowledge-heavy energy / transport items (BAM electrification, HSR, Amur energy projects, construction permits).
- Runs are complementary: A ∪ B finds 74%, v1 ∪ A ∪ B 89% of known positives → several runs / an ensemble is the
  cheapest lever. Some new «positives» look like judge leniency (a burning apartment block, tarpaulin prices).

## v2-B on 3 new companies, first stratified half judged (21.10, 157 ₽; `eval_companies.py`)
Companies: clean (no DQ flag), CLIENT, ≥ 15 bn, unseen before, different OKVED — U_150736 corrugated board 17.21
(Moscow, 15.8 bn), U_212593 restaurants 56.10.1 (Moscow, 26.9 bn, foreign owner), U_9250 engineering structures
42.99 (Irkutsk, 62.3 bn). Factory v2-B: 14 / 15 / 22 generators (~4.5 ₽ each; one hung GLM call took 72 min → timeout
fix). Judge input per company: events of the selected generators (top-13), events only dropped probes found, 100
random events outside both; each stratum split in half (seed 1), half 1 judged (runs `judge-20261021-1839…/1844…/1849…`).

| company | selected: pos / labelled | dropped only | random | est. recall of selected |
|---|---|---|---|---|
| U_150736 packaging | 9 / 90 (0.10) | 1 / 55 | 0 / 50 | 90%* |
| U_212593 restaurants | 14 / 86 (0.16) | 2 / 47 | 0 / 50 | 88%* |
| U_9250 construction | 24 / 123 (0.20) | 1 / 12 | 2 / 50 | 13%* |
\* recall over the corpus via the random stratum's positive rate; 0 / 50 only bounds the rate below ~6% (rule of
three), 2 / 50 extrapolates to ~300 positives outside — all three numbers are very uncertain. The second half (560
events, ~150 ₽) waits for the quota reset.
- Generic prompt transfers: every company gets sensible industry generators (restaurants: visitors / average bill,
  food input prices, alcohol excise, delivery aggregators; packaging: e-commerce demand, freight; construction:
  national projects, Irkutsk transfers, contractor regulation).
- Productive channels differ by company: packaging B3 / A1 / D4, restaurants B2 / A1 / H1 / C5, construction A3 /
  B4 / I1; the generic channels (D3 FX, E1 sanctions, G1 incidents, C1 taxes) give ~0 almost everywhere → quota
  weights per channel are the next lever.
- Judge v2 is lenient for big contractors / B2C: FAS grid-protection tariff as a positive for a restaurant chain, bank
  deposit rates and Brent > $100 for a construction company — calibration needed before trusting recall numbers.

## E2E smoke: insight agent on one factory candidate per new company (21.10, `judge_v2/agent_run.py`, 14.4 ₽)
Stage 5 `InsightJudge` (ToolLoop: same-event reports, follow-ups, news search, bank analytics, company facts) →
stage 6 `InsightWriter` (write → check → one revision), role CFO, blind to the judge-v2 verdict; bank analytics = the
one OCR'd report (key rate 11.09). Inputs: a judge-v2 positive from the factory's selected generators per company.

| company | news | agent decision | review | my read |
|---|---|---|---|---|
| U_212593 restaurants | duty on strong alcohol from 2027 | insight, score 2, medium confidence | ok | strong: pulls excise +6.6%, MRP talks, a €5/l scenario; keeps «proposal, not law»; concrete budget-2027 actions |
| U_9250 construction | 37.5 bn ₽ federal transfers to Irkutsk region | insight, score 2, medium | 1 issue (an unrelated Krasnoyarsk paragraph) | good: region + OKVED link stated as unconfirmed by payments; Q4 timing of national-project payments; guarantees (a product the company has) as advice |
| U_150736 packaging | independent e-commerce grows faster than marketplaces | insight, score 2, low | not ok after revision | medium: generic «check your revenue mix»; stage 5 overstated a weak fact (2% spice producers → «clients are food producers») and «№1 in Moscow» — the checker caught both, the one revision did not fix them |
- All three judge-v2 positives survive the agent → for these cases the judge and the agent agree; negatives not run yet.
- Problems: unsupported company claims can survive the single revision; bank view is generic while only one report
  is OCR'd; the checker once answered without the tool call (now tolerated); ~4–5 ₽ and up to ~3 min per news.
