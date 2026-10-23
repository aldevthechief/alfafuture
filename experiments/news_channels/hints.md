# Channel hints for the insight agent: design and test plan (22.10.2026, not built yet)

Goal 2 of the news classification: give the stage-5 insight agent (`InsightJudge`, ToolLoop) an idea through which
channel exactly an event may hit this company — as a hypothesis to check, never as a fact.

## What the hint is
Per (event, company): up to 3 lines, built in code (no LLM), from checklist v2 leaf scores of the event and the
company's exposure (`channel_gen.facets`, v1 items → v2 parents via `checklist_v2.V1_PARENT`).
1. Event side: leaves with per-leaf percentile ≥ 0.9 that are among the event's top-3 after centring (score minus the
   news' mean over leaves). If the sink X0 is the event's top leaf → no hints (and the noise filter flag).
2. Company side: the fact that makes the company exposed to the leaf's parent (own OKVED, a counterparty OKVED with its
   share, Alfa product, foreign owner, region, FTS) or «в профиле прямой связи не найдено».
3. Order: leaves with a company link first.

Wording in the agent's context:
```
Подсказки классификатора каналов (гипотезы, могут быть ошибочными; используй, только если подтверждается текстом
новости и фактами компании; если не подтверждается — так и отметь):
- Регулируемые тарифы (сетевые тарифы) — связь с компанией: ОКВЭД 27.12 «Производство электрической распределительной
  аппаратуры». Проверь: меняет ли событие спрос или цены компании через этот канал?
- Капвложения госкомпаний и монополий — связь: та же отрасль. Проверь: …
```
Later: the bank-analytics pointer per channel (backlog 13: rate → rate forecast, …) and role weights (backlog 16).

## Where it goes
- Test: through the existing `stage4` slot of `judge_task.md` («Оценка этапа 4: {stage4}») as a key
  `подсказки_классификатора` — no change to stage-5 code.
- After a positive test: a dedicated block in `judge_task.md` and a decision field `hint_used` (confirmed / rejected
  hint ids) for logging and for the UI («почему важно»).

## Risks
Anchoring: the agent forces an insight through a suggested channel → false insights (and mis-selling, a critical error
in the case). Mitigations: hypothesis wording, explicit «can be rejected», a placebo arm in the test, hints only from
leaves with a company link at all.

## Test plan (stage-5 decision only, no writer; LLM run → needs approval)
- Events: per company (4 companies) 10 events = 4 judge-v2 positives + 3 judge negatives with high channel scores
  (hints that can mislead) + 3 random negatives with hints → 40 events.
- Arms: A no hints, B real hints, C placebo hints (same format, random leaves among those the company is exposed to)
  on 20 of the events. 100 decisions.
- Smoke first: 6 events × A / B (~12 calls) to measure ₽ per decision (full agent with writer ≈ 4–5 ₽; stage 5 alone
  expected ≈ 2–3 ₽) → full run ≈ 200–300 ₽, a separate approval with the measured price.
- Metrics: insight rate on judge positives (A vs B); false-insight rate on negatives (A vs B vs C — anchoring if C > A);
  hint uptake (B's `mechanism` matches a hinted leaf; C's should be lower — the agent discriminates); concreteness
  (`company_link` cites a profile fact); ₽ and tool steps per decision; manual read of every A/B disagreement.
- Keep the hints if: recall B ≥ A, false insights B ≤ A + 1, placebo uptake clearly below real uptake. Otherwise use the
  classification only for retrieval.

## Prerequisite
Checklist v2 leaf scores for all news (Kaggle, free): e5-large-instruct and Giga-480M with `INSTRUCTION` + leaf +
examples in the query; bge-reranker-v2-m3, ru-reranker-edge-150m and the zero-shot NLI cross-encoder
`MoritzLaurer/bge-m3-zeroshot-v2.0` (hypothesis = the leaf) on the bi-encoders' top-2000 news per leaf, max over the
3 examples; `deepvk/GeRaCl-USER2-base` (Russian zero-shot classifier) as one more cheap try. Giga-10B dropped: not
better than the 480M and not hostable in prod.
