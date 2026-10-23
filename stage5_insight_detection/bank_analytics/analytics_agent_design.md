# Analytics insight agent — design (task 5; built in `analytics_insights/`, not run with a real model yet)

Idea (user, 22.10): bank analytics is an insight source as strong as the news. In news 90% is noise, hence general candidate
generators + a judge; in analytics everything is useful to a business, the job is to find what matters for *this* company
and role. So: a separate agent that reads the reports, not a filter. Candidates are presented as «аналитика и прогнозы
экспертов Альфа-Банка» (bank-view layer; no news facts). Bank products (credit, leasing, deposits, guarantees) are first-class candidates.

Case check (case-notes.md, analytics-usage.md): consistent — analytics must visibly drive conclusions (criteria 02, 03, 06), the UI
splits facts / bank view / AI conclusions / hypotheses. An analytics-only insight has no news fact: the «что произошло» is the
bank's own forecast or its change vs the previous report; the company link stays a fact from stage 2; products are advice, not sales.

## Loop (common/llm/tool_loop.py ToolLoop; same shape as insight_judge)
- Input: company facts (stage 2), role, today's date; limits: 30 steps, stops at 5 candidates or when the model calls `finish`.
- Tools:
  - `search_bank_analytics(query)` — existing, parsed-data note + top-3 passages with ±2 chunks;
  - `read_page(report, page)` — full parsed text of a page (the model reads the report itself, knowing OCR errs);
  - `look_at_page(report, page, question)` — renders the PDF page (pymupdf, as ocr.py does) and asks Qwen3-VL (`ChatModels.vision()`),
    e.g. «какой тренд безработицы прогнозируется на графике?»; answer is marked as a vision reading;
  - `compare_reports(topic)` — the same topic in the newest vs the previous report: change of the bank's view (the freshest wins, the change is shown);
  - `company_facts(category)` — existing; `bank_products(company)` — products the company already uses + what is relevant;
  - `calendar()` — dated events from the reports (CB meetings 23.10, 18.12, …);
  - `add_candidate(...)` — records an `AnalyticsCandidate`; returns the count, at 5 says «call finish»; `finish(reason)`.
- `AnalyticsCandidate`: kind (forecast | view_change | recommendation | calendar | product), title, bank_view (quote), sources
  [(report, date, page)], channel, company_link (fact id), mechanism and direction, role_fit, product (optional), dated (optional), confidence.
  Goes to the same judge/writer as news candidates: stage 6 gets `bank_view` filled and `what_happened` = the forecast / its change.

## Decisions (user, 22.10)
1. Model: GLM-5.3-Flash (`ChatModels.agent()`); pictures only through `look_at_page` → Qwen3-VL (max_tokens 8000, the page's parsed text goes
   into the prompt: without it the same question answered «there is such a table» at 150 dpi and «none» at 250 dpi).
2. Candidates join the news ones in one ranked list with a quota (wiring into the pipeline/stage 6: next step).
3. Products: known from the bank's table flags → `company_facts('bank')`, no separate `bank_products` tool; the prompt tells the agent to use them.
4. Cost per (company, role): not measured; a run needs the user's approval of that exact run.
`calendar()` tool dropped: the agent finds dated events (CB meetings) through search.

## Integration into the MVP pipeline (proposal, 23.10; nothing of this is built)

News need a wide search through a lot of data and a 90% filter; the analytics are few and everything in them is useful, so the
unit of work differs: the news path is «event → company», the analytics path is «company × role → what the reports say that matters».

### Variants
1. **Parallel source (recommended).** `run_analytics(company, role)` in `pipeline.py` next to `run` / `run_loop`; the agent
   (`analytics_insights/agent.py`) returns candidates, a deterministic mapper turns each into the same `Insight` object (+ `origin`),
   `audit()` and the card renderer are reused. `--source news|analytics|both`; with `both` the two lists meet in one ranked feed
   with a quota. The analytics result does not depend on a news item, so it is **batch, cached per (company facts hash, role, bank
   library version)** and recomputed only when a PDF or the company data changes (≈ 2–4 ₽ per pair), not on every request.
2. **Agent per news (what exists now in the judge).** The judge already searches the analytics for a news; it cannot find an insight
   that has no news (a forecast change, the 23.10 meeting, a product under the bank's rate view). Keep it as is; it is the news side.
3. **One agent with both toolboxes.** Too wide: it would spend steps on news noise; maybe later as a final «editor» that picks the
   top N from the already built news and analytics insights (cheap, one call).

### Contract: candidate -> `Insight` (stage6 `insight.py`), nothing is dropped
| analytics candidate | Insight |
|---|---|
| title | title |
| bank_view (quote), sources | `bank_view` Claim kind `bank`, `sources` = `bank:<report>:<page>` (+ date in the text); also `what_happened` (origin = bank_analytics: «Что говорит Банк») |
| company_link (the quoted fact only) | `why_it_matters` Claim kind `fact`; link_inference -> Claim kind `ai`, `sources` = `fact:<n>` |
| mechanism | `why_it_matters` Claim kind `ai` |
| assumptions (hypotheses) | `hypotheses` Claim kind `hypothesis` + a HITL question «Верно ли, что…?» (profile of the demo UI) |
| action | `actions` |
| product | `bank_product` |
| kind, event_date, channel | new `Insight.origin = 'bank_analytics'`, `analytics_kind`, `event_date`, `channel` (card badge, calendar) |
| role_fit, confidence | `confidence`; role_fit is the ranking key in the feed |
| warnings | audit warnings of the card (numbers checked against the cited pages, calendar date) |
The mapper is code, not an LLM: the writer rewriting a checked text would be the only place where numbers can change again.
An optional polish by the stage-6 writer stays off by default and goes through the same `audit()`.

### What the agent hands over in `finish` (today only `reason`, so the work around the candidates is lost)
`Finish` becomes: `summary` (what was read and concluded), `checked_channels` (channel -> found / nothing for this company),
`rejected` (topics seen and dropped, with the reason), `open_questions` (data from the client that would sharpen candidates: from
`assumptions.check`, in the shape of `DataRequest` / `open_requests` of the analyst loop). `assumptions` become structured
`{claim, check}` instead of the string «Гипотеза: …; подтвердит …», so the UI can ask the client and the answer can raise confidence.
All of it is stored with the result (`data/insights/analytics/<company>_<role>.json`), shown in the trace and the card, never only in logs.

### Merge into the feed
- One list ranked by a common score: news insight score (importance × sensitivity × confirmation, `impact.py`) and analytics
  `role_fit` (0–3) × confidence; **quota**: at least 1 and at most 40% of the slots are analytics (few, high-value items must not
  drown in the news list, and must not push all the news out).
- Duplicates: an analytics candidate on the same channel as a news insight already in the feed becomes that card's «Позиция Банка»
  (attached, not a second card); otherwise it is a standalone card labelled «Аналитика и прогнозы экспертов Альфа-Банка».
- Case checks stay: facts / bank view / AI conclusions / hypotheses separated; a forecast is the bank's position, not a guarantee;
  «не инвестиционная рекомендация»; products as advice; the freshest report wins and a change of the bank's view is shown.

### Merge risk on the code side
`mvp` has its own `bank_analytics/search.py` (paragraph TF-IDF) and `library.py` (Fragment with own_view / stance / topics); `maxon`
replaced the search (chunks.py, fragments.py, twins.py, number_check.py, tools.py) and dropped own_view by decision. Land these as
new modules over `library.Fragment` rows and keep the `BankAnalytics` interface (`search`, `passages`, `rows`), then the judge, the analyst
loop and the new agent share one retrieval; conflicts are expected in `search.py` and `insight_judge/judge.py`.

Update 23.10: `Finish` is structured (summary, covered, rejected, open_questions) and kept: `AnalyticsInsightAgent.finish`, written by `run.py` into `runs.json`. Number check advises (the candidate is always recorded, a corrected call with the same title replaces it). `assumptions` are still strings.
