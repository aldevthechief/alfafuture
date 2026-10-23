# Trace review of the analytics agent, run v4 (Langfuse session `run-20261023-014635`, 11.94 ₽)

Read from the local call log (`data/logs/calls.jsonl`: full prompts, answers and tool outputs of the 3 agents: U_14366 cfo, U_57706 treasury,
U_177581 ceo; 43 model calls, 21 searches, 21 add_candidate, 12 read_page, 7 compare_reports, 4 look_at_page) and checked against the
report pages (parsed text and the page images) and the company facts.

## What works
- The number check works inside the loop: it sent U_14366 from page 2/3 to page 4 for «0,0%» and U_177581 from page 2 to page 3 for «4%»;
  it stopped an unverified «1,0% за май» (text of the report and the chart say ≈0,5).
- Freshness: all three took «13,5% → 14,0%» from the 11.09 report and named both dates; the FX 2027 value is 93,0 (not 95).
- Checked true against the sources: budget spending +14% against the plan +3% (Тактический взгляд p.9), deficit 5,8 трлн (11.09 p.1),
  wages 12,7% for 1П26 and the 10% forecast, construction −5% for 1П26, mortgage +1,2% (June) / ≈0,5% (May) on the chart of «Потребление на пике» p.3,
  payment concentration 67% / 75% (company facts, U_57706), ВЭД with Kazakhstan / Kyrgyzstan without direction.

## Problems
1. `finish(reason)` carries the summary (what was covered, which topics were dropped and why) and `AnalyticsInsightAgent.run` throws it away.
2. A wrong page goes unnoticed when the quote has no numbers: «эффект домино» and «ставки денежного рынка выше ключевой» are cited as
   Тактический взгляд p.21–22, they stand on p.23.
3. The number check blocks too much: «found only in the chart reading» is a note, but it refuses the call; U_177581 lost 4 of 9 add_candidate calls
   on it, and the agent misread one of them («10» was «~10% г/г» in the text, not a page number).
4. Parsed text defect: «Потребление на пике» p.2 has digits turned into Cyrillic letters in 9 places («ЗОЗЗ г. до З,%», «вырос на З,% м/м»); the
   numbers of that page can only be taken from the picture. Other reports: none found by the pattern.
5. `look_at_page` contradicts itself («не указаны в тексте» and «в тексте говорится 1,2%»); the values themselves were right on the image.
6. `kind` drifts: a «calendar» candidate without a date (U_57706 #2, a tactic about short placements).
7. The agent writes «все цифры сверены» before the check ran (U_14366); it submits four candidates in one step.
8. Two of three runs stop at the cap of 5; whether there were more strong candidates is unknown. U_14366 stopped at 4 and said why (no bank view on deposits / taxes).
