# Bank analytics: parser metadata and text-only retrieval (22.10.2026)

Parser: `library.py` of the `mvp` worktree (`D:\Projects\hacks\alfa\wt-mvp`), output `data/analytics/fragments.jsonl`
(408 fragments: 267 text, 102 chart, 39 table). Scripts here: `check_parse.py` (cleanliness), `check_metadata.py` (the
audit below), `eval_kinds.py` (text-only retrieval), `search.py::prose_reason` (deterministic prose filter).

## 1. Text vs charts: is there a deterministic split?
Yes. One Qwen3-VL pass writes the page text and wraps every chart in `[График]…[/График]` (`prompts/ocr_page_full.md`);
`structure()` in `library.py` splits on these markers and on Markdown tables, so `kind` = text | chart | table is exact.
What still leaks into `kind == text` (checked by eye on 267 fragments): legal disclaimers (32 fragments, **25% of the
text characters**; the parser's own filter misses the «Данный материал предназначен…» and digit-garbled variants), 6
parser notes «Резервное распознавание (старый OCR)» that carry old chart descriptions («Ось Y левая…»), 60 captions /
sources / headers < 120 chars, 13 digit-only blocks and 7 axis-label lists. `search.prose_reason(fragment)` returns the
reason or '' for a prose paragraph: 143 fragments, 65% of text characters, 88% of the non-legal ones.

## 2. Is «text only» enough for retrieval? (`eval_kinds.py`, parser's matching set: 21 pairs, 14 distinct news)
Of the 37 `must` fragments 33 are text (89%), 2 chart, 2 table; of 322 `ok`: 207 text (64%), 72 chart, 29 table.

| index | fragments | must in index | must recall@10 | any hit@5 | MRR must |
|---|---|---|---|---|---|
| all kinds (current) | 364 | 37/37 | 0.71 | 0.95 | 0.59 |
| text only (= minus legal / notes / captions = soft prose) | 244 / 226 / 206 | 33/37 | 0.73 | 0.90 | 0.51 |
| prose only (strict filter) | 143 | 32/37 | 0.66 | 0.86 | 0.39 |
| prose + tables | 182 | 34/37 | 0.64 | 0.90 | 0.42 |
- Text-only loses the 4 non-text musts and ~0.05 of any-hit@5; recall@10 is unchanged. The MRR gaps (0.51 vs 0.39 vs
  0.59) are tie-order noise inside the same report (three pairs share one news): the top-3 are the same fragments in
  another order. The strict filter also cut one real paragraph that the paragraph splitter had broken mid-sentence at a
  page break (`1H26 p2.4`, «…и с учетом обозначившихся») — a chunking defect, not a filter one.
- **What text-only cannot answer: the forecast table.** The 2027 USD/RUB (93,0 in the 11.09 report, 95,0 in 7M26 and
  1H26), GDP, rate and inflation forecasts for 2026П / 2027П are printed only in the macro table; no prose fragment
  mentions the 2027 exchange rate. One table kind is worth keeping for v1 (the macro table, three readings reconciled).
- **v1 recommendation:** index `text` minus legal / parser notes / captions + tables whose header row exists; charts out
  (title + «Что видно» could come later). Do not use the strict prose filter (no gain, loses paragraphs).

## 3. Metadata, one by one (`check_metadata.py`)
| field | how it is made | what I saw | verdict |
|---|---|---|---|
| `topics` | 16 macro topics, word-prefix dictionary (`library.py:43`), every fragment | 86 of 267 text fragments have none, 58 have ≥ 3; by eye ~70% right: «инвестиций в ОФЗ» → инвестиции, captions and chart text get topics, «облигации» fires on 75 fragments (28%); sector topics nearly absent (строительство 2, металлургия 8) | not needed for retrieval (dense + BM25 cover it); keep as an optional facet / for grouping statements; do not rank by it |
| `figures` | regex `NUMBER` + unit; indicator = the sentence's first topic; period = a period within 40 chars after, else the last before | 759 in text, period on 27%; of 12 random ones 7 were junk («10 лет» tenor, «топ-10», «1К25» → 1, axis «0 – 35», «Май 25» → 25, chart ≈ values). Key sentence: «заседание 23 октября» → figure 23 (rate forecast); 1H26 p3.6 «13,5% … на конец 2026 г.; … 2027 г. … 12%» gets 13,5% with period **2027** | unreliable for comparing or verifying numbers; ignore in v1; table cells with status `verified` are the trustworthy numbers |
| `own_view` | regex «мы / наш + ожидаем / считаем / полагаем / прогноз…» not preceded by a quote verb | 59 of 267 (22%); 8/8 sampled right; 16 text fragments with мы / наш missed, ≥ 8 are real own views («по нашему мнению ограничивает пространство для снижения ставки», «мы расцениваем», «мы отмечаем») | precision high, recall ~80%: use as a label «позиция Банка» / a boost, never as a filter; add «по нашему мнению», «расцениваем», «отмечаем», «относим» |
| `stance` | up / down / flat by regexes, one direction for every topic of the sentence | 1H26 p3.6 «считаем возможным заложить еще два снижения ключевой ставки» → **flat** («сохраняющегося» matches `сохран`); a sentence with 4 topics gets 4 identical directions; by topic and date the stances are incoherent (ставка 22.07: flat 2 / down 2 / up 2) | drop from v1; conflicts between reports are better found by comparing the numeric forecast of the same indicator and period |
| `freshness` / `age_days` | report date vs `as_of` at build: ≤ 60 days fresh, ≤ 365 aging, else stale | frozen at build time (as_of 20–21.10); buckets are arbitrary (Gold 51 d fresh; consumption 77, 1H26 82, FI strategy 91 aging); only the 2021 steel report is stale; it is per report, so the July «two more cuts to 13,5%» and the September «pause at 14%» both stay «aging» / «fresh» without a link | compute from `date` at query time; show the date in every hit (done) and let the newer report win on the same topic; statement-level supersession is the real need |

Other facts found on the way: `fragments.jsonl` already includes a correct reading of the key numbers; the 11.09 report
says USD/RUB 93,0 for end-2027 (CLAUDE.md and analytics-notes.md fixed).

## 4. Decisions (approved 22.10)
- **analytic-retrieve v1 sources** (`search.use_fragment`): prose of the reports without legal text, parser notes, captions and
  digit / axis lists; tables that have a header row; no charts. 408 → 180 fragments.
- **No `own_view` label and no stance**: every fragment of the bank's documents is the bank's analytics as of the report
  date; numbers about periods before that date are statistics, everything after it is the bank's view. The agent's tool
  output and system prompt say so; hits show report, date, page.
- **Figures** are ignored in v1; trustworthy numbers = table cells with status `verified`. **Topics** stay as an optional
  facet, not used for ranking. **Freshness**: the date is shown in every hit; a newer report wins on the same topic (the
  agent names both dates when reports differ); the `stale` flag only sorts the 2021 report last.
- Hybrid weight re-measured on the v1 index: 0.8 (see `search_eval.md`).

## 5. What the filters drop, read by eye (22.10)
`search.use_fragment` keeps 184 of 408 fragments (147 text, 36 tables, 1 chart-table). Dropped, with reasons:

| dropped | n (chars) | what is in it | verdict |
|---|---|---|---|
| charts | 81 + 21 unverified (141k, 46% of all chars) | titles, model-written one-line trends («Что видно», empty in half), code descriptions of values; 1H26 p1.9 is a real table («Таблица 1: Ключевые макроиндикаторы и прогнозы на июль 2026») that the parser wrapped as a chart — now kept; chart titles are echoed by the page text in half of the cases | out of v1; later: title + «Что видно» as short anchors, numbers never |
| legal | 32 (32k) | ownership, redistribution, «not investment consulting», analysts' conflicts of interest; 0 mention a rate / inflation / forecast; the old `DISCLAIMER` regex missed 9 of them («Настоящий отчёт … исключительной собственностью») | pure noise for retrieval; keep one copy of the «not investment advice» wording for the UI disclaimer |
| short / captions | 63 (3.5k) | sources, page footers, «Центр макроэкономического анализа», legends; ~5 section headlines carry a thesis («Макроэкономический контекст: инфляционные риски растут, неопределенность высока», «ОФЗ – неиссякаемый источник…», «Ралли развивалось в три этапа…») | noise except headlines: they belong in the chunk of the paragraph below (task 4), not as stand-alone fragments |
| chart text / label lists / digits | 19 (8k) | axis ticks, legends, flattened appendix columns, old-OCR chart descriptions, the 2021 steel tables | noise; the slide-style text of the gold digest (Gold p5.1, p7.1) was wrongly dropped by the «many short lines» rule — fixed: chart metadata words decide, slides with sentences stay |
| parser notes | 6 (1.4k) | «Резервное распознавание: основное на странице зациклилось / неполное» + a footnote | no content, but a signal: the chart on that page was not read (11.09 p2, p3; 1H26 p4) |
| tables without header | 5 | 1H26 p4.4 / p4.5 are the 2026П / 2027П continuation of the appendix table (rate, wages, budget, FX 95,0, imports, credit) — real content; FI p21–23 small rating tables | repaired: a headless piece takes the header of the previous piece on the same page (`repair_headers`) |

Legal detection is a keyword regex (`search.LEGAL`: «Все права защищены», «Генеральная лицензия», «исключительной
собственностью», «несанкционированное», «не делают заявлений», «предназначен»); on this corpus no fragment is hit by
«предназначен» alone, so no content is lost; on other documents that word could be a false hit.
After the fixes (weight 0.8, `eval_kinds.py`): v1 sources 184 fragments, must in index 35/37, every pair has a must
fragment (21/21), must recall@10 **0.78** (all fragments: 0.66), any hit@5 0.95, MRR 0.49 (0.51).
