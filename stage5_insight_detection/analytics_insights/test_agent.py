"""Offline test of the analytics insight agent (format v2: claims with evidence, company_facts, assumptions with supports, impact): scripted
models, synthetic bank OCR in a temp folder (no network, data/ untouched). python -X utf8 stage5_insight_detection/analytics_insights/test_agent.py"""
import json, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import common.llm.models as llm_models
llm_models.LANGFUSE = []
from common.llm import PromptLibrary
from common.llm.testing import ScriptedChatModel, calls, tool_call as c
from stage5_insight_detection.analytics_insights.agent import AnalyticsInsightAgent, AnalyticsTools, progress
from stage5_insight_detection.bank_analytics.search import BankAnalytics

for name in PromptLibrary(Path(__file__).with_name('prompts')).names():
    prompt = PromptLibrary(Path(__file__).with_name('prompts')).chat(name)
    assert prompt.format_messages(**{v: 'x' for v in prompt.input_variables}), name

tmp = Path(tempfile.mkdtemp())
(tmp / 'ALFA-Macro_Insights-Policy_Rate_11Sep.jsonl').write_text(json.dumps(
    {'key': '1', 'value': 'Ключевая ставка ЦБ сохранена на уровне 14%.\n\nБанк ожидает паузу по ключевой ставке до конца 2026 года '
                          'и снижение ставки до 12% к концу 2027 года; кредиты для бизнеса останутся дорогими.'}, ensure_ascii=False))
(tmp / 'ALFA-7M26.jsonl').write_text(json.dumps(
    {'key': '2', 'value': 'Мы ожидаем ключевую ставку на уровне 13,5% на конец 2026 года и ещё два снижения до конца года.'}, ensure_ascii=False))
bank = BankAnalytics(ocr_dir=tmp)
facts = [{'cat': 'bank', 'text': 'Использует кредитные продукты банка', 'depth': 1}, {'cat': 'profile', 'text': 'Оптовая торговля', 'depth': 1},
         {'cat': 'scale', 'text': 'Годовая выручка 105.6 млрд ₽', 'depth': 1}]
REPORT = 'Ключевая ставка ЦБ РФ: без изменения на 14,0%'
QUOTE = 'Банк ожидает паузу по ключевой ставке до конца 2026 года и снижение ставки до 12% к концу 2027 года'
claim = dict(text='пауза 14% до конца 2026, 12% к концу 2027', evidence=[dict(report=REPORT, date='2026-09-11', page=1, quote=QUOTE)])
cand = dict(kind='view_change', title='Банк отложил снижение ставки', conclusion='Деньги дороже дольше: бюджет считать по 14% до конца года.', channel='ставка',
            bank_claims=[claim], company_facts=[1], link_inference='кредиты дороже дольше, поэтому пауза по ставке важна', mechanism='кредиты дороже дольше → процентные расходы выше',
            effect='risk', actions=[dict(text='Пересчитать бюджет по ставке 14%', because='Банк ждёт 14% до конца 2026')],
            assumptions=[dict(claim='больше половины долга компании — по плавающей ставке', check='справка о структуре долга', supports=['mechanism', 'action:1'])],
            impact=dict(level='mid', criteria='ставка влияет на стоимость обслуживания долга', estimate='на каждые 100 млн ₽ долга пауза ставки на 14% стоит 1 млн ₽ в год', basis=['bank_claim:1']),
            product='кредитные продукты', role_fit=3, confidence='high')
model = ScriptedChatModel(responses=[
    calls(c('search_bank_analytics', query='ключевая ставка'), c('company_facts', category='bank')),
    calls(c('compare_reports', topic='ключевая ставка'), c('read_page', report='2026-09-11', page=1), c('read_page', report='нет такого', page=1)),
    calls(c('look_at_page', report='нет такого', page=1, question='тренд?')),
    calls(c('add_candidate', **cand)),
    calls(c('add_candidate', **{**cand, 'kind': 'product', 'title': 'Продукт под пониженную ставку'})),
    calls(c('finish', summary='ставка и продукт записаны', covered=['ставка — смена взгляда'], rejected=['золото — не касается компании'], open_questions=['структура долга — повлияет на вывод про плавающую ставку'])),
])
vision = ScriptedChatModel(responses=[])  # must not be called: the report is unknown
agent = AnalyticsInsightAgent(model, bank, vision)
found = agent.run('U_1', facts, 'cfo', '2026-10-22')
assert agent.finish and agent.finish.rejected == ['золото — не касается компании'] and agent.finish.open_questions and agent.error is None, agent.finish  # the closing note is kept
assert [x.kind for x in found] == ['view_change', 'product'] and found[0].role_fit == 3, found
assert found[0].bank_claims[0].evidence[0].status == 'exact' and found[0].fact_quotes == ['fact:1 [bank] Использует кредитные продукты банка'] and not found[0].warnings, found[0]
assert model.i == 0 and vision.i == 0  # i wraps to 0 once the script is consumed

capped = ScriptedChatModel(responses=[calls(c('add_candidate', **cand)), calls(c('add_candidate', **{**cand, 'title': 'Второй'})), calls(c('finish', summary='лимит'))])
assert len(AnalyticsInsightAgent(capped, bank, vision, max_candidates=1).run('U_1', facts, 'cfo', '2026-10-22')) == 1

# the checks advise and never refuse; a corrected call with the same title replaces the record
tools = AnalyticsTools(facts, bank, vision, 9)
wrong = {**cand, 'bank_claims': [{**claim, 'text': 'пауза 14%, снижение до 9,5% к концу 2027 года'}]}  # 9,5 is not on the cited page
reply = tools.add_candidate(**wrong)
assert reply.startswith('recorded #1') and 'Замечания проверки' in reply and 'W1:' in reply and '9,5' in reply and len(tools.candidates) == 1, reply
again = tools.add_candidate(**wrong)
assert again.startswith('replaced #1') and 'Замечания проверки' not in again, again  # each warning is shown once
done = tools.add_candidate(**{**wrong, 'accepted_warnings': [{'id': 'W1', 'reason': 'число прочитано по картинке'}, {'id': 'W9', 'reason': 'нет такого'}]})
assert [a.id for a in tools.candidates[0].accepted_warnings] == ['W1'] and 'W1' not in done, tools.candidates[0].accepted_warnings
fixed = tools.add_candidate(**cand)
assert not tools.candidates[0].warnings and not tools.candidates[0].accepted_warnings and 'Замечания проверки' not in fixed

t = AnalyticsTools(facts, bank, vision, 9)
# quotes: a wrong page is corrected, an invented quote is flagged, no claim / no quote is advised on
r = t.add_candidate(**{**cand, 'title': 'Страница', 'bank_claims': [{**claim, 'evidence': [{**claim['evidence'][0], 'page': 2}]}]})
e = t.candidates[-1].bank_claims[0].evidence[0]
assert 'стр. 1' in r and e.page == 1 and e.cited_page == 2 and e.status == 'relocated', r
r = t.add_candidate(**{**cand, 'title': 'Выдумка', 'bank_claims': [{**claim, 'evidence': [{**claim['evidence'][0], 'quote': 'Банк ожидает снижения ставки до 8% уже в ноябре из-за притока капитала'}]}]})
assert 'цитаты нет' in r and 'утверждение 1, доказательство 1' in r and t.candidates[-1].bank_claims[0].evidence[0].status == 'not_found', r
assert 'нет ни одного утверждения' in t.add_candidate(**{**cand, 'title': 'Без утверждений', 'bank_claims': []})
assert 'у утверждения 1 нет цитаты' in t.add_candidate(**{**cand, 'title': 'Без цитат', 'bank_claims': [{**claim, 'evidence': []}]})
r = t.add_candidate(**{**cand, 'title': 'Картинка', 'bank_claims': [{'text': 'значение по картинке ≈14%', 'evidence': [{**claim['evidence'][0], 'kind': 'picture', 'quote': 'по картинке ≈14%'}]}]})
assert 'цитаты' not in r and t.candidates[-1].bank_claims[0].evidence[0].status == 'picture', r

# company facts by id: the text is put in by the code; unknown ids, no ids and figures of the company in the inference are advised on
assert t.company_facts('bank') == '[fact:1][bank] Использует кредитные продукты банка'
t = AnalyticsTools(facts, bank, vision, 9)
assert 'company_facts пуст' in t.add_candidate(**{**cand, 'title': 'Без фактов', 'company_facts': []})
assert 'нет среди фактов' in t.add_candidate(**{**cand, 'title': 'Нет такого факта', 'company_facts': [99]})
assert 'число 5 из link_inference' in t.add_candidate(**{**cand, 'title': 'Число в выводе', 'link_inference': 'долг компании около 5 млрд ₽, поэтому ставка важна'})
assert 'link_inference' not in t.add_candidate(**{**cand, 'title': 'Число из позиции Банка', 'link_inference': 'ставка 14% до конца 2026 удерживает стоимость кредитов'})
assert 'link_inference' not in t.add_candidate(**{**cand, 'title': 'Выручка из фактов', 'company_facts': [1, 3], 'link_inference': 'при выручке 105.6 млрд ₽ ставка важна'})

# assumptions keep their matching: supports must name something that exists; actions are needed
bad = {**cand, 'title': 'Привязка', 'assumptions': [dict(claim='гипотеза', check='данные', supports=['bank_claim:7', 'action:1', 'что-то'])]}
r = t.add_candidate(**bad)
assert 'гипотеза 1: supports «bank_claim:7»' in r and 'supports «что-то»' in r and 'action:1»' not in r, r
assert 'нет действий' in t.add_candidate(**{**cand, 'title': 'Без действий', 'actions': []})

t = AnalyticsTools(facts, bank, vision, 9)
# impact: high needs an estimate and a basis (otherwise mid, advised), figures of the estimate come from the facts and the bank, a per-unit example is free
r = t.add_candidate(**{**cand, 'title': 'High без оценки', 'impact': dict(level='high', criteria='бьёт по деньгам')})
assert 'понижена до mid' in r and t.candidates[-1].impact.level == 'mid', r
r = t.add_candidate(**{**cand, 'title': 'Выдуманная сумма', 'impact': dict(level='mid', criteria='x', estimate='удорожание долга на 420 млн ₽ в год', basis=['fact:3'])})
assert 'число 420 из estimate' in r, r
r = t.add_candidate(**{**cand, 'title': 'На единицу', 'impact': dict(level='high', criteria='x', estimate='на каждые 100 млн ₽ долга 1 п.п. ставки стоит 1 млн ₽ в год; ставка 14% по прогнозу Банка', basis=['fact:3', 'bank_claim:1'])})
assert 'estimate' not in r and t.candidates[-1].impact.level == 'high', r

# fields that used to make a call fail validation are advised on instead: no impact, no role_fit; the date of the report is filled by the code
t = AnalyticsTools(facts, bank, vision, 9)
no_date = {**cand, 'title': 'Без даты отчёта', 'bank_claims': [{**claim, 'evidence': [{k: v for k, v in claim['evidence'][0].items() if k != 'date'}]}]}
assert 'W' not in t.add_candidate(**no_date).split(chr(10))[0] and t.candidates[-1].bank_claims[0].evidence[0].date == '2026-09-11'
r = t.add_candidate(**{**{k: v for k, v in cand.items() if k not in ('impact', 'role_fit', 'confidence')}, 'title': 'Без оценки'})
assert 'нет оценки влияния' in r and 'role_fit' in r and t.candidates[-1].impact is None and t.candidates[-1].confidence == 'medium', r
# calendar needs a date; a bad date is advised on for any type
assert 'нужна дата события' in t.add_candidate(**{**cand, 'title': 'Заседание ЦБ', 'kind': 'calendar'})
assert 'нужна дата события' not in t.add_candidate(**{**cand, 'title': 'Заседание ЦБ', 'kind': 'calendar', 'event_date': '2026-10-23'})
assert 'YYYY-MM-DD' in t.add_candidate(**{**cand, 'title': 'Другое', 'kind': 'recommendation', 'event_date': '23.10.2026'})

# a loop that repeats the same call: stopped after STUCK_LIMIT, the candidates stay, the closing note is made by one tool-free call
stuck = ScriptedChatModel(responses=[*(calls(c('add_candidate', **cand)) for _ in range(4)), calls(c('finish', summary='Остановлено из-за повторов вызовов', covered=['ставка']))])
agent = AnalyticsInsightAgent(stuck, bank, vision)
found = agent.run('U_1', facts, 'cfo', '2026-10-22')
assert len(found) == 1 and agent.error.startswith('LoopStuck') and agent.finish and agent.finish.covered == ['ставка'], (found, agent.error, agent.finish)
assert 'add_candidate сейчас' in progress(8, 0, 5, 30) and 'вызови finish' in progress(21, 2, 5, 30) and 'сейчас' not in progress(3, 0, 5, 30)
print('ok: analytics agent v2: claims with evidence, company_facts by id, assumptions with supports, impact rules, warnings once with ids, accepted reasons, calendar, loop stop')
