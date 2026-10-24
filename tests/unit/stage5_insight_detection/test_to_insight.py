"""Offline test of contract v2: the old news insight loses nothing (v1 -> v2 -> v1 on the demo feed), the analytics candidate becomes an
Insight by code only, hypotheses keep their matching. python -X utf8 stage5_insight_detection/analytics_insights/test_to_insight.py"""
import json, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from stage5_insight_detection.analytics_insights.agent import AnalyticsCandidate, AnalyticsTools, Finish
from stage5_insight_detection.analytics_insights.to_insight import candidate_to_insight, run_to_analysis
from stage5_insight_detection.bank_analytics.search import BankAnalytics
from stage6_insight_generation.insight_writer import insight as v1
from stage6_insight_generation.insight_writer import insight_v2 as v2

# 1. the news contract: every v1 insight of the demo feed survives v1 -> v2 -> v1 unchanged, and v1 JSON loads into v2 as it is
feed = json.loads((Path(__file__).resolve().parents[3] / 'frontend/public/feed.json').read_text(encoding='utf-8'))
found = []
def walk(o):
    if isinstance(o, dict):
        if 'what_happened' in o and 'why_it_matters' in o:
            found.append(o)
        for x in o.values():
            walk(x)
    elif isinstance(o, list):
        for x in o:
            walk(x)
walk(feed)
assert found, 'the feed has insights'
for raw in found:
    old = v1.Insight.model_validate({k: raw[k] for k in v1.Insight.model_fields if k in raw})
    new = v2.from_v1(old, company_id='U_1', news_ids=[1, 2])
    assert v2.to_v1(new).model_dump() == old.model_dump(), 'v1 -> v2 -> v1 must be lossless'
    assert v2.Insight.model_validate(old.model_dump()).title == old.title  # v1 JSON is a valid v2
    assert new.origin == 'news' and new.kind == 'event' and new.trigger.news_ids == [1, 2] and [a.text for a in new.action_details] == old.actions
    assert all(cl.id for blk, _ in v2.BLOCKS for cl in getattr(new, blk)), 'ids by position'

# 2. the analytics candidate -> Insight v2
tmp = Path(tempfile.mkdtemp())
(tmp / 'ALFA-Macro_Insights-Policy_Rate_11Sep.jsonl').write_text(json.dumps(
    {'key': '1', 'value': 'Ключевая ставка ЦБ сохранена на уровне 14%.\n\nБанк ожидает паузу по ключевой ставке до конца 2026 года '
                          'и снижение ставки до 12% к концу 2027 года; кредиты для бизнеса останутся дорогими.'}, ensure_ascii=False))
bank = BankAnalytics(ocr_dir=tmp)
facts = [{'cat': 'bank', 'text': 'Использует кредитные продукты банка', 'depth': 1}, {'cat': 'scale', 'text': 'Годовая выручка 105.6 млрд ₽', 'depth': 1}]
REPORT = 'Ключевая ставка ЦБ РФ: без изменения на 14,0%'
quote = 'Банк ожидает паузу по ключевой ставке до конца 2026 года и снижение ставки до 12% к концу 2027 года'
cand = dict(kind='view_change', title='Банк отложил снижение ставки', conclusion='Деньги дороже дольше.', channel='ставка',
            bank_claims=[dict(text='пауза 14% до конца 2026, 12% к концу 2027', evidence=[dict(report=REPORT, date='2026-09-11', page=1, quote=quote)]),
                         dict(text='значение по картинке ≈14%', evidence=[dict(report=REPORT, date='2026-09-11', page=1, quote='по картинке ≈14%', kind='picture')])],
            company_facts=[1, 2], link_inference='кредиты дороже дольше', mechanism='расходы на проценты выше', effect='risk',
            actions=[dict(text='Пересчитать бюджет', because='Банк ждёт 14%'), dict(text='Проверить ковенанты', because='ставка выше плана')],
            assumptions=[dict(claim='больше половины долга — плавающая ставка', check='справка о долге', supports=['mechanism', 'action:1', 'bank_claim:1']),
                         dict(claim='ковенанты привязаны к процентному покрытию', check='кредитные договоры', supports=['action:2', 'impact', 'link'])],
            impact=dict(level='high', criteria='бьёт по стоимости долга', estimate='на каждые 100 млн ₽ долга 1 млн ₽ в год при выручке 105.6 млрд ₽', basis=['fact:2', 'bank_claim:1']),
            product='кредитные продукты', role_fit=3, confidence='high', event_date='')
tools = AnalyticsTools(facts, bank, None, 5)
reply = tools.add_candidate(**cand)
c = tools.candidates[0]
ins = candidate_to_insight(c, 'U_1', 'cfo', bank.rows, facts)
assert ins.origin == 'bank' and ins.kind == 'view_change' and ins.what_happened == [] and ins.company_id == 'U_1' and ins.id
assert ins.trigger.reports == [{'title': REPORT, 'date': '2026-09-11'}] and ins.ai_conclusion == 'Деньги дороже дольше.' and ins.role_fit == 3

# evidence: the id of the fragment the quote stands in, offsets inside that fragment, the picture reading on the page id
ev = ins.bank_view[0].evidence[0]
frag = next(f for f in bank.rows if f['id'] == ev.source)
assert ev.source.startswith('bank:ALFA-Macro_Insights-Policy_Rate_11Sep:p1.') and frag['text'][ev.start:ev.end].lower().startswith('банк ожидает паузу'), (ev, frag['text'][:80])
assert ev.status == 'exact' and ev.title == REPORT and ev.page == 1 and ev.kind == 'text'
pic = ins.bank_view[1].evidence[0]
assert pic.kind == 'picture' and pic.start is None and pic.source == 'bank:ALFA-Macro_Insights-Policy_Rate_11Sep:p1' and ins.bank_view[1].sources == [pic.source]
assert ins.review.verification == {'exact': 1, 'picture': 1}

# why it matters: the facts (kind=fact, with their text) and the inference; the mechanism goes to risks by `effect`
assert [(x.id, x.kind, x.sources) for x in ins.why_it_matters] == [('w1', 'fact', ['fact:1']), ('w2', 'fact', ['fact:2']), ('w3', 'ai', ['fact:1', 'fact:2', ev.source, pic.source])]
assert ins.why_it_matters[0].text == 'Использует кредитные продукты банка' and [x.id for x in ins.risks] == ['r1'] and ins.opportunities == []

# actions: strings as in v1 + structured; the hypotheses keep their matching in both directions
assert ins.actions == ['Пересчитать бюджет — потому что Банк ждёт 14%', 'Проверить ковенанты — потому что ставка выше плана'] and [a.id for a in ins.action_details] == ['a1', 'a2']
h1, h2 = ins.hypotheses
assert (h1.id, h1.kind, h1.check, h1.supports) == ('h1', 'hypothesis', 'справка о долге', ['r1', 'a1', 'b1'])
assert h2.supports == ['a2', 'imp', 'w3']
assert ins.risks[0].depends_on == ['h1'] and ins.action_details[0].depends_on == ['h1'] and ins.bank_view[0].depends_on == ['h1']
assert ins.action_details[1].depends_on == ['h2'] and ins.impact.depends_on == ['h2'] and ins.why_it_matters[2].depends_on == ['h2']
assert ins.impact.level == 'high' and ins.impact.basis == ['fact:2', 'bank_claim:1'] and ins.impact.sources == ['fact:2', ev.source], ins.impact

# `mixed` keeps the mechanism in why_it_matters; `opportunity` puts it to opportunities
mixed = AnalyticsCandidate.model_validate({**c.model_dump(), 'effect': 'mixed'})
im = candidate_to_insight(mixed, 'U_1', 'cfo', bank.rows, facts)
assert [x.id for x in im.why_it_matters] == ['w1', 'w2', 'w3', 'w4'] and im.risks == [] and im.hypotheses[0].supports[0] == 'w4'
op = candidate_to_insight(AnalyticsCandidate.model_validate({**c.model_dump(), 'effect': 'opportunity'}), 'U_1', 'cfo', bank.rows, facts)
assert [x.id for x in op.opportunities] == ['o1'] and op.hypotheses[0].supports[0] == 'o1'

# the v1 projection is valid, the run container carries the closing note
assert v2.to_v1(ins).title == ins.title and len(v2.to_v1(ins).risks) == 1
run = run_to_analysis([ins], 'U_1', 'cfo', '2026-10-23', Finish(summary='s', covered=['ставка'], rejected=['золото'], open_questions=['долг']), model='glm', cost_rub=1.5)
assert run.engine == 'bank_analytics' and run.covered == ['ставка'] and run.open_requests == ['долг'] and v2.AnalysisRun.model_validate_json(run.model_dump_json()).insights[0].id == ins.id
print('ok: contract v2: v1 lossless round trip on the feed, candidate -> Insight by code, evidence ids and offsets, hypotheses matched both ways, effect routing, run container')
