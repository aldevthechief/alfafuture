# evals — offline quality evaluation

Strategy, thresholds and caveats: [../docs/QUALITY_AND_MONITORING.md](../docs/QUALITY_AND_MONITORING.md). Live checks: `../monitoring/`.
Data (runs, caches) goes to `$ALFA_DATA/evals/` (gitignored). All commands: `python -X utf8 -m <module>` from the repo root.

## insight_judge — LLM judge calibrated on the human labelling task
Axes: relevance, significance, clarity, grounding + winner (the labelling TZ). The TZ folder (`insight_judge_calibration`:
`U_*.md` pairs + `marking_sheet.csv`) is NOT in the repo (case-derived exposures); every command takes its path.
```
python -X utf8 -m evals.insight_judge.run --pairs <TZ dir> --pointwise            # ~6 ₽: 42 pairs x 2 orders + 34 cards (GLM low)
python -X utf8 -m evals.insight_judge.run --pairs <TZ dir> --pointwise --model deepseek/deepseek-v4.1-flash   # self-preference check
python -X utf8 -m evals.insight_judge.calibrate --pairs <TZ dir> --run <run.json> [--sheet <marking_sheet.csv>]
```
Without `--sheet`: position bias, ties, factual-error flags, mean card scores. With it: kappa per axis (strict / pointwise /
hybrid verdicts), Bradley–Terry vs the pointwise score, release-threshold table.

## retrieval_golden — blind synthetic golden set + ranking metrics
`golden/` is FROZEN (sha256 in `golden/MANIFEST.json`, committed before any generator was run). Rebuild only on purpose:
```
python -X utf8 -m evals.retrieval_golden.build --pairs U_14366:cfo,...            # ~8 ₽ per pair (generation + blind audit)
python -X utf8 -m evals.retrieval_golden.leakage                                  # can bag-of-words / length separate the labels?
ALFA_DATA=<data> python -X utf8 -m evals.retrieval_golden.run_eval                # no LLM: factory runs from its stored spec
```
`run_eval` injects the 120 items into the real corpus (ids 9 000 000+), runs v2 / v3 (all standard, factory and channel
generators) and writes `report.json` (systems with 95% intervals, every generator's own ranking, raw ranks).
