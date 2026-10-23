# News id migration (row number -> stable url hash)

`migrate_ids.py` rewrites saved artifacts from the old ids (row in `news.parquet`) to `build.stable_id(url)`.
Dry run by default; `--apply` backs up to `data/_id_migration_backup/`, refuses a second run (`data/_id_migration_done.json`)
and refuses to drop entries without `--drop-unmapped`. Needs the new `build.py` (the old one has no `stable_id`).

## Two id spaces
- **news id** — exact, `stable_id(url)`; needs only the old `news.parquet`. 9147 ids, all map.
- **event id** — row of `events.parquet`, positional, changes with every corpus rebuild. Mapped through members:
  old event -> new event holding most of its (remapped) members. Needs `--new-events` (events.py on the new corpus).

## Dry-run on the real raw data (`articles.jsonl` as of 2026-10-20 00:32, new `build.py` `--days 32`, nothing written to the repo)
- Repo corpus (9147) is an older, smaller fetch; the same raw data now gives 32167 rows with the old build.py and 31855 with the new one
  (common 31789; only-old 378 = 314 on the cut-off day, new build cuts by timestamp not by date).
- Events on the 31855-row corpus: 24441. Old events: 7807 whole, 101 split (best holds >= 50%), 23 split (< 50%), 106 lost (all members past the cut-off).
- 80 artifact files, 77.9k event refs, 94.8k news refs; 1330 event refs unmapped (lost events). 0 of 2109 (event, news) pairs inconsistent in the old tables.
- Applied on a scratch copy: every migrated news id exists in the new corpus with the same title (2054/2054; 2 lost); (event, news) pairs consistent in the new tables 1658/~1663.
- `real_pairs.json` is rewritten with a different layout (cosmetic); 2 of 33 pairs fall outside the new window.

## Not migrated
- `experiments/retrieval/labels_{cheap,agent}.jsonl`: keyed by an older events numbering (events 8037–8187 do not exist now) — already misaligned, superseded by judge v2.
- Regenerate: `data/news/{news,events}.parquet`, `data/llm_runs/*` (`experiments/llm_runs/prepare.py`), embeddings/ranks for new rows
  (retrieval `*.npz` cover only the old 9147 docs; their `doc_ids` are rewritten but the corpus is now larger).
- `data/llm_cache`, `data/logs`, Langfuse traces, ids quoted in prose in `.md` reports.
