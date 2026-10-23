"""Offline test of the analyst loop (tasks 1-8) and the FinRobot techniques on synthetic data: no network, no case
tables, nothing written to data/. [CARD_OUT=card.html] python -X utf8 stage5_insight_detection/analyst_loop/test_analyst_loop.py"""
import ast, json, os, sys, tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.dedup import dedup_news
from common.llm.testing import ScriptedChatModel, calls, tool_call as c
from common.safety import clean_block, clean_line, render_untrusted, suspicious
from common.trace import AgentTrace, redact
from stage4_news_relevance.news_classifier.classifier import NewsClassifier, fresh
from stage5_insight_detection.analyst_loop.audit import audit
from stage5_insight_detection.analyst_loop.loop import AnalystLoop
from stage5_insight_detection.analyst_loop.pool import ListNewsPool
from stage5_insight_detection.analyst_loop.profile import CompanyProfile
from stage5_insight_detection.bank_analytics.library import BankLibrary, date_from_name, date_from_text
from stage5_insight_detection.bank_analytics.match import BankMatcher
from stage6_insight_generation.insight_writer.insight import Insight
from stage6_insight_generation.insight_writer.writer import InsightWriter

HERE = Path(__file__).parent
SYN = HERE / 'synthetic'
AS_OF = date(2026, 10, 20)
pages = json.loads((SYN / 'bank_pages.json').read_text(encoding='utf-8'))['reports']
news = json.loads((SYN / 'news.json').read_text(encoding='utf-8'))['news']
company = {k: v for k, v in json.loads((SYN / 'company.json').read_text(encoding='utf-8')).items() if not k.startswith('_')}

# --- compute layer never imports an LLM client ("numbers by code, text by the model") ---
COMPUTE = [HERE / f for f in ('facts.py', 'profile.py', 'event.py', 'impact.py', 'risks.py', 'grounds.py', 'verdict.py',
                              'enrich.py', 'dossier.py')] + [ROOT / 'stage5_insight_detection/bank_analytics' / f for f in ('library.py', 'match.py')]
for path in COMPUTE:
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ''] if isinstance(node, ast.ImportFrom) else []
        assert not any(n.startswith(('langchain', 'langgraph', 'openai', 'common.llm')) for n in names), (path.name, names)

# --- task 1: parsing the bank's analytics ---
lib = BankLibrary.from_pages(pages, AS_OF)
by_file = {f.file: f for f in lib.fragments}
assert by_file['SYNTH_Steels_15Apr2021.pdf'].freshness == 'stale'
assert by_file['SYNTH-Policy_Rate_12Sep.pdf'].date == '2026-09-12'
own = lib.by_id['bank:SYNTH-Policy_Rate_12Sep:p1.2']
assert own.own_view and own.stance['ставка'] == 'flat' and 'ставка' in own.topics
fc = [(f['value'], f['period']) for f in own.figures if f['kind'] == 'forecast' and f['indicator'] == 'ставка']
assert (12.0, '2027 г') in fc, fc
quote = lib.by_id['bank:SYNTH-Policy_Rate_12Sep:p1.1']
assert not quote.own_view and any(f['value'] == 14.0 and f['kind'] == 'actual' for f in quote.figures)
assert date_from_name('Gold_digest_August26.pdf', 2026) == date(2026, 8, 31)
assert date_from_name('FI_Strategy_3Q26_22072026.pdf', 2026) == date(2026, 7, 22)
assert date_from_text('Обзор\n11 сентября 2026 г.') == date(2026, 9, 11)
tmp = Path(tempfile.mkdtemp())
lib.save(tmp / 'fragments.jsonl')
later = BankLibrary.load(tmp / 'fragments.jsonl', as_of=date(2027, 10, 1))  # freshness is recomputed on load
assert later.by_id[own.id].freshness == 'stale'

# --- task 2: bank analytics for the event and the profile ---
profile = CompanyProfile(**company)
assert profile.exposure['ставка'] == 1.0 and profile.exposure['строительство'] == 1.0 and 'fact:product_loan' in profile.reasons['ставка']
matches = BankMatcher(lib).match('ЦБ повысил ключевую ставку до 15%', ['ставка', 'кредит'], profile.exposure)
assert matches[0].fragment.file == 'SYNTH-Policy_Rate_12Sep.pdf' and matches[-1].fragment.stale, [m.fragment.id for m in matches]
assert not any(m.fragment.file.startswith('SYNTH-Consumption') for m in matches)
# select(): topics on word starts, foreign rates apart, no topics for a disclaimer, directions of falls and of the rouble
from stage5_insight_detection.bank_analytics.library import direction, topics_of
from stage5_insight_detection.bank_analytics.match import BankMatch, brief_bank
from stage5_insight_detection.analyst_loop.event import EventCard
assert topics_of('Ставка ФРС не менялась') == ['ставка_мир'] and 'ставка' in topics_of('Банк России сохранил ключевую ставку')
assert topics_of('Несанкционированное копирование запрещено. Генеральная лицензия ЦБ РФ') == [] and topics_of('Россияне стали экономить') == []
assert [direction(t) for t in ('Число компаний сократилось', 'Спрос упал на 40%', 'ЦБ повысил ставку', 'Инфляция не снизилась',
                               'Рубль на подъеме', 'ослабление курса рубля')] == ['down', 'down', 'up', 'flat', 'down', 'up']
event101 = EventCard.from_news(news[0])
picked = BankMatcher(lib).select(event101, profile)
assert picked and picked[0].fragment.file == 'SYNTH-Policy_Rate_12Sep.pdf' and picked[-1].fragment.stale, [m.fragment.id for m in picked]
assert all(m.score >= 0.4 * 0.3 for m in picked) and any(m.parts['direction'] for m in picked)
software = EventCard.from_news({'id': 900, 'title': 'Облачный софт привяжут к одной доверенной операционной системе',
                                'text': 'Требования к совместимости программного обеспечения смягчат.', 'date': '2026-10-16'})
assert BankMatcher(lib).select(software, profile) == []  # no bank analytics on it: the verdict is «нет»
assert len(brief_bank(picked * 3, cap=2)) == min(2, len(picked))
# the bank analyst agent over the code's wider pick: keeps what it says fits, with the channel; ignores ids it was not
# given; «none» empties the pick; the digests' own views of the picked reports go to the brief with the report date
from stage5_insight_detection.bank_analytics.review import BankReviewer, bank_positions
wide = BankMatcher(lib).select(event101, profile, k=20)
assert len(wide) >= 2
reviewer = BankReviewer(ScriptedChatModel(responses=[
    calls(c('BankReview', items=[{'id': wide[1].fragment.id, 'fits': True, 'channel': 'ставка → стоимость кредита', 'agrees': 'расходится'},
                                 {'id': wide[0].fragment.id, 'fits': False, 'why': 'не о том'},
                                 {'id': 'bank:made-up:p1.1', 'fits': True}], none=False)),
    calls(c('BankReview', items=[{'id': wide[0].fragment.id, 'fits': True}], none=True))]))
kept, rec = reviewer.review(event101, profile, wide)
assert kept[0].fragment.id == wide[1].fragment.id and kept[0].parts['review']['agrees'] == 'расходится'
assert all(m.parts['review'].get('doubt') for m in kept[1:])  # wide[0] only as doubtful, if the code was sure of it
assert 'оговорка' in kept[0].reason and rec['by'] == 'agent' and all(d['id'] != 'bank:made-up:p1.1' for d in rec['decisions'])
weak = [BankMatch(m.fragment, m.score, dict(m.parts, event=0.5), m.reason) for m in wide[:2]]
assert reviewer.review(event101, profile, weak)[0] == [] and reviewer.review(event101, profile, [])[1]['verdict'] == 'аналитики Банка нет'
# the agent filters but cannot empty the pick: a strong candidate (event >= 0.8, fresh own view) it rejected or a bare «нет»
# keeps 1–2 such fragments marked as doubtful, with its reason; weak ones it rejected stay out
from dataclasses import replace as dc_replace
strong_m = lambda: BankMatch(dc_replace(wide[0].fragment, freshness='fresh', own_view=True), 0.9, {'event': 0.9}, 'код')
weak_m = BankMatch(wide[1].fragment, 0.5, {'event': 0.5}, 'код')
doubt = BankReviewer(ScriptedChatModel(responses=[
    calls(c('BankReview', items=[{'id': wide[0].fragment.id, 'fits': False, 'why': 'про другой показатель'},
                                 {'id': wide[1].fragment.id, 'fits': True, 'channel': 'ставка → кредит'}])),
    calls(c('BankReview', items=[], none=True)),
    calls(c('BankReview', items=[{'id': wide[1].fragment.id, 'fits': False, 'why': 'не о том'}], none=True))]))
got, rec = doubt.review(event101, profile, [strong_m(), weak_m])
assert [m.fragment.id for m in got] == [wide[1].fragment.id, wide[0].fragment.id] and got[1].parts['review']['doubt']
assert 'агент сомневается: про другой показатель' in got[1].reason and rec['doubtful'] == [wide[0].fragment.id] and 'под сомнением' in rec['verdict']
assert 'про другой показатель' in got[1].prompt_item() and 'ставка → кредит' in got[0].prompt_item()  # the brief shows both
got, rec = doubt.review(event101, profile, [strong_m(), weak_m])
assert [m.fragment.id for m in got] == [wide[0].fragment.id] and 'без разбора' in got[0].reason  # a bare «нет» is doubtful
got, rec = doubt.review(event101, profile, [weak_m])
assert got == [] and rec['verdict'] == 'аналитики Банка нет'  # a reasoned «нет» on weak candidates stands
(tmp / 'SYNTH-Policy_Rate_12Sep.md').write_text('# Ставка\n\n## Страница 1\nМы ожидаем сохранения ключевой ставки на 14% до конца 2026 года.', encoding='utf-8')
(tmp / 'SYNTH-Policy_Rate_12Sep.digest.json').write_text(json.dumps({'summary': 'Прогноз ставки', 'bank_views': [
    {'indicator': 'Ключевая ставка', 'value': '14%', 'horizon': 'конец 2026', 'page': 1, 'unverified': False},
    {'indicator': 'Инфляция', 'value': '9%', 'page': 1, 'unverified': True}]}, ensure_ascii=False), encoding='utf-8')
pos = bank_positions(kept + [m for m in wide if m.fragment.file == 'SYNTH-Policy_Rate_12Sep.pdf'], folder=tmp)
assert any('Ключевая ставка — 14%, конец 2026 (стр. 1)' in p for p in pos) and not any('Инфляция' in p for p in pos), pos

# --- FinRobot 3: injection guard and dedup ---
dirty = news[4]
assert suspicious(dirty['title']) and suspicious(dirty['text'])
wrapped = render_untrusted('news:105', dirty['title'], dirty['text'])
assert '<system>' not in wrapped and 'ignore previous' not in wrapped.lower() and wrapped.count('<untrusted_item>') == 1
assert '\n' not in clean_line('a\nb ### c') and '</untrusted_item>' not in clean_block('x </untrusted_item> y')
stories = dedup_news(news[:3])
merged = next(s for s in stories if 101 in s['duplicate_ids'])
assert set(merged['duplicate_ids']) == {101, 102} and merged['n_sources'] == 2, stories
assert len(dedup_news([news[0], dict(news[0], id=999, url=news[0]['url'].replace('?utm_source=tg', ''))])) == 1

# --- FinRobot 2: classifier returns scores only, missing indices are asked again ---
cls_model = ScriptedChatModel(responses=[
    calls(c('scores', items=[{'index': 0, 'category': 'ставка_и_кредит', 'sentiment': 'negative', 'importance': 4,
                              'affected_roles': ['cfo']}])),  # skipped index 1 and 2
    calls(c('scores', items=[{'index': 1, 'category': 'ставка_и_кредит', 'sentiment': 'negative', 'importance': 4},
                             {'index': 0, 'category': 'прочее', 'sentiment': 'neutral', 'importance': 1}])),  # 0 must not be overwritten
    calls(c('scores', items=[])),  # third try: index 2 still unscored -> dropped
])
trace = AgentTrace(folder=None)
rows = NewsClassifier(cls_model, trace=trace).classify(news[:2] + [news[5]], profile.card())
assert [r['id'] for r in rows] == [101, 102] and rows[0]['importance'] == 4 and rows[0]['title'] == news[0]['title']
assert rows[0]['url'] == news[0]['url'] and trace.steps('classify_dropped')[0]['data']['news_id'] == 106
assert fresh([{'date': None}], AS_OF) == [] and cls_model.i == 0

# --- tasks 3-8: the loop with scripted models ---
pool = ListNewsPool(news)
claim = lambda text, *refs: {'text': text, 'sources': list(refs)}
insight = {'title': 'Ставка 15%: долг дороже, чем закладывал Банк', 'role': 'cfo',
           'what_happened': [dict(claim('ЦБ повысил ключевую ставку до 15%', 'news:101'),  # the writer's check needs a verbatim quote
                                  evidence=[{'source': 'news:101', 'quote': 'повысил ключевую ставку на 1 п.п., до 15% годовых'}])],
           'why_it_matters': [claim('Компания пользуется кредитными продуктами, рост ставки на 1 п.п. даёт около 80 млн ₽ '
                                    'процентов в год', 'fact:product_loan', 'fact:estimate_1')],
           'risks': [claim('Рост стоимости проектного финансирования на 25%', 'news:103')],  # news:103 was not given to the writer
           'opportunities': [], 'actions': ['Проверить ковенанты и график рефинансирования до конца ноября'],
           'bank_view': [claim('Банк ожидал паузу на 14,0% до конца 2026 года', 'bank:SYNTH-Policy_Rate_12Sep:p1.2'),
                         claim('Банки повышают ставки по кредитам', 'news:103')],  # not bank analytics -> moves to AI
           'ai_conclusion': 'Стоимость долга вырастет раньше, чем ожидалось.', 'confidence': 'high',
           'bank_product': 'Депозит для свободной ликвидности'}  # the company has no deposit upside here -> dropped
agent = ScriptedChatModel(responses=[
    calls(c('assess', explanations=[{'consequence_id': 'C1', 'text': 'Долг подорожает: +1 п.п. к ставке даёт 80 млн ₽ в год.',
                                     'refs': ['fact:estimate_1', 'fact:product_loan', 'news:999']}],
            missing=[{'kind': 'bank_analytics', 'query': 'ипотека строительство спрос', 'reason': 'взгляд Банка на отрасль'},
                     {'kind': 'company_facts', 'query': 'строительство', 'reason': 'уточнить деятельность компании'},
                     {'kind': 'computation', 'query': 'долговая нагрузка компании', 'reason': 'нет данных о долге'}])),
    calls(c('Insight', **insight)),
    calls(c('Review', ok=True)),
])
trace = AgentTrace('test-run', folder=tmp)
loop = AnalystLoop(lib, pool, agent, InsightWriter(agent), trace=trace,
                   company_facts=[{'cat': 'activity', 'text': 'Основной вид деятельности: строительство жилых зданий'}])
res = loop.run(pool.get(101), profile, 'cfo', scores={'category': 'ставка_и_кредит', 'importance': 4})
assert res['status'] == 'audit_rejected' and res['verdict']['status'] == 'reject', res['verdict']
assert 'insight' not in res and any(w['code'] == 'unknown_ref' for w in res['audit'])
# the loop with the bank analyst agent: its decisions and verdict go to the trace, its pick becomes task 2's matches
wide101 = BankMatcher(lib).select(EventCard.from_news(pool.get(101)), profile, k=20)
rtrace = AgentTrace(folder=None)
rloop = AnalystLoop(lib, pool, None, None, trace=rtrace, explain=False, reviewer=BankReviewer(ScriptedChatModel(responses=[
    calls(c('BankReview', items=[{'id': wide101[0].fragment.id, 'fits': True, 'channel': 'ставка → кредит'}]))])))
rres = rloop.run(pool.get(101), profile, 'cfo')
step = rtrace.steps('bank_match')[0]['data']
assert step['review']['by'] == 'agent' and [m['id'] for m in step['matches']] == [wide101[0].fragment.id], step
assert rres['bank'][0]['id'] == wide101[0].fragment.id
cons = {x['topic']: x for x in res['consequences']}
rate = cons['ставка']
assert rate['sign'] == -1 and rate['estimate']['value'] == 80.0 and rate['estimate']['kind'] == 'hypothesis'  # 40 bn × 20% × 1 п.п.
assert any(f.startswith('fact:delta') for f in rate['facts'])
risk = next(r for r in res['risks'] if r['topic'] == 'ставка')
assert risk['type'] == 'risk' and risk['bank_disagrees'] and risk['basis'] == 'ai'  # the bank expected a pause
codes = {i['code'] for i in res['grounds']['issues']}
assert {'bank_disagrees', 'newer_data'} <= codes, codes
assert 102 in [r['id'] for r in loop.enricher.pool.same_event(101)]
assert any(o['kind'] == 'computation' for o in res['open_requests'])  # no tool -> handed back to the caller
assert any(e['data'].get('origin') == 'model' and e['data']['ok'] for e in trace.steps('request'))
brief = res['verdict']['brief']
assert brief['hypotheses'] and brief['impact']['level'] in ('mid', 'high') and 'score' in brief['impact']['criteria']
out = Insight(**res['draft'])
assert len(out.bank_view) == 1 and out.bank_view[0].kind == 'bank'
assert any(cl.text.startswith('Банки повышают') and cl.kind == 'ai' for cl in out.why_it_matters)
assert next(cl for cl in out.risks if '25%' in cl.text).unconfirmed  # unknown ref
assert not out.what_happened[0].unconfirmed and out.what_happened[0].kind == 'fact'
assert out.hypotheses and all(h.kind == 'hypothesis' for h in out.hypotheses)
assert out.impact is not None and out.disclaimer and out.bank_product == '' and out.confidence == 'medium'
assert agent.i == 0, 'scripts must be consumed exactly'

# reject: not the company's business; no date
res_gold = loop.run(pool.get(104), profile, 'cfo')
assert res_gold['verdict']['status'] == 'reject' and res_gold['verdict']['reasons']
res_nodate = loop.run(pool.get(106), profile, 'cfo')
assert res_nodate['verdict']['status'] == 'reject' and any('дата' in r for r in res_nodate['verdict']['reasons'])

# --- FinRobot 4 on its own: bank product without a matching risk is dropped ---
lonely = Insight(**dict(insight, bank_product='Факторинг', bank_view=[]))
from stage5_insight_detection.analyst_loop.verdict import InsightBrief
b = InsightBrief(**brief)
fixed, warnings = audit(lonely, b, {})
assert fixed.bank_product == '' and any(w['code'] == 'mis_selling_guard' for w in warnings) and fixed.not_bank_position

# dates and their parts are not numbers; «66 350» is one number in the claim and in the source
from stage5_insight_detection.analyst_loop.audit import numbers
assert numbers('отчёт «Инфляция 25–31 августа» от 02.09.2026, данные 2026-09-02, на 2027–2028 гг., к 31.07') == set()
assert numbers('АИ-92 до 66 350 ₽/т') == {92.0, 66.35}  # canonical price unit is ₽/kg
assert numbers('бюджет на 4К26–1К27 (FI Strategy 3Q26, «Макростатистика 1П26»): рост 0,9%') == {0.9}

# «69 тысяч» in the source confirms «69 000» in the claim, and «1,5 млрд» confirms «1 500 млн»
assert numbers('до 69 тысяч') == {69000.0} and numbers('1,5 млрд руб.') == {1.5e9}

# --- OCR page structure: a chart and a table are fragments of their own; a value read off the scale confirms nothing ---
from stage5_insight_detection.bank_analytics.match import BankMatch
from stage5_insight_detection.bank_analytics.pages import clean_page
ocr_page = clean_page('''Обзор ставки

[График] Рис. 2: Ключевая ставка, %
Тип: линия
Ряды (легенда): ставка
Значения: ставка — сен.26: 14,0; ставка — дек.26: ≈ 13,5 по шкале
Источник: ЦБ РФ
[/График]

[График] Рис. 3: Инфляция, % г/г
Значения: инфляция — янв: 4; фев: 4; мар: 4; апр: 4
[/График]

Табл. 1: Прогноз
| | 2025 | 2026П |
|---|---|---|
| Инфляция, % дек./дек. | 5,6% | 7,5% |''')[0]
from stage5_insight_detection.bank_analytics.pages import close_charts
assert close_charts('[График] A\nТип: линия\n\nИсточник: ЦБ\n\nТекст.\n[График] B\nТип: x') == \
    '[График] A\nТип: линия\n\nИсточник: ЦБ\n[/График]\n\nТекст.\n[График] B\nТип: x\n[/График]'
slib = BankLibrary.from_pages({'SYNTH-Charts_10Oct2026.pdf': [ocr_page]}, AS_OF)
rate_chart, flagged = [f for f in slib.fragments if f.kind == 'chart']
table = next(f for f in slib.fragments if f.kind == 'table')
assert rate_chart.heading.startswith('график Банка, рис. 2, стр. 1') and rate_chart.unverified and 'не проверено' in rate_chart.text
# the page OCR's chart values are never verified (a whole-page reading made most of them up)
assert [(x['value'], x['period'], x['status']) for x in rate_chart.figures] == [(14.0, 'сен.26', 'unverified'), (13.5, 'дек.26', 'unverified')]
assert rate_chart.approx() == {14.0, 13.5} and not rate_chart.own_view and rate_chart.stance == {}
assert flagged.unverified and flagged.figures and not any(x['verified'] for x in flagged.figures)
assert table.heading.endswith('Табл. 1: Прогноз') and not table.unverified and '| 2025 | 2026П |' in table.text
assert {(x['value'], x['period'], x['kind'], x['indicator']) for x in table.figures} == {(5.6, '2025', 'actual', 'инфляция'),
                                                                                       (7.5, '2026П', 'forecast', 'инфляция')}
item = BankMatch(rate_chart, 1.0, {}, '').prompt_item()  # the brief gets no unconfirmed values, one note instead
assert 'график Банка, рис. 2' in item and '14,0' not in item and '13,5' not in item and 'см. PDF, стр. 1' in item, item
flagged_item = BankMatch(flagged, 1.0, {}, '').prompt_item()
assert '⚠' not in flagged_item and 'янв: 4' not in flagged_item and 'распознаны ненадёжно' in flagged_item
assert '13,5' in rate_chart.text and '⚠' in flagged.text  # text/ and the card keep everything
from stage5_insight_detection.bank_analytics.pages import suspect_reason
assert '2019' in suspect_reason("[График] Доходность\nОсь X: дата; 01.05.2019 — 09.08.2026\nЗначения: AAA — 15,3\n[/График]", 2026)
assert '2030' in suspect_reason("[График] Курс\nЗначения: курс — авг'30: 85\n[/График]", 2026)
assert suspect_reason("[График] Курс\nОсь X: авг'23 — авг'26\nЗначения: курс — авг'26: 85\n[/График]", 2026) == ''
from stage5_insight_detection.bank_analytics.match import SHAKY_CHART
shaky = [m for m in BankMatcher(slib).match('ключевая ставка график', ['ставка'], {'ставка': 1.0}, k=10) if m.fragment.id == rate_chart.id]
assert shaky and '×0.85' in shaky[0].reason and SHAKY_CHART == 0.85
assert all('×0.85' not in m.reason for m in BankMatcher(slib).match('прогноз инфляции', ['инфляция'], {}, k=10) if m.fragment.kind == 'table')

# --- digitize: a synthetic chart with known values (drawn by PyMuPDF), code reads the pixels ---
import numpy as np
import pymupdf
from stage5_insight_detection.bank_analytics.digitize import annotate, digitize, to_rgb
from stage5_insight_detection.bank_analytics.pages import cut_phrase_loops, drop_empty_tables, clean_page as clean_ocr_page
doc = pymupdf.open()
pg = doc.new_page(width=420, height=300)
red_v, black_v = [1, 3, 2, 5, 4, 6, 5, 7], [6, 5.5, 6, 4, 3, 3.5, 2, 1]
yy = lambda v: 250 - 25 * v
xx = [70 + 40 * i for i in range(8)]
for v in range(0, 9, 2):
    pg.insert_text((20, yy(v) + 3), str(v), fontsize=9)
for i, x in enumerate(xx):
    pg.insert_text((x - 2.5, 272), str(i + 1), fontsize=9)
pg.draw_polyline([(x, yy(v)) for x, v in zip(xx, red_v)], color=(0.9, 0.1, 0.1), width=1.5)
for x, v in zip(xx, red_v):
    pg.draw_polyline([(x, yy(v) - 4), (x + 4, yy(v)), (x, yy(v) + 4), (x - 4, yy(v)), (x, yy(v) - 4)], color=(0.9, 0.1, 0.1),
                     fill=(1, 1, 1), width=1.2)
pg.draw_polyline([(x, yy(v)) for x, v in zip(xx, black_v)], color=(0, 0, 0), width=1.5, dashes='[2 2] 0')
synthetic = {'y_axis': {'ticks': ['0', '2', '4', '6', '8']}, 'x_axis': {'ticks': [str(i) for i in range(1, 9)]},
             'series': [{'name': 'Ряд A', 'color': 'красный', 'style': 'линия с маркерами'},
                        {'name': 'Ряд B', 'color': 'черный', 'style': 'пунктир'}]}
dg = digitize(to_rgb(pg.get_pixmap(dpi=150).tobytes('png')), synthetic)
assert dg.ok and dg.x_labels == synthetic['x_axis']['ticks'], dg.reason
for got, want in zip(dg.series, (red_v, black_v)):
    err = np.abs(np.array(got['values']) - np.array(want))
    assert err.mean() < 0.2 * 2 and err.max() < 0.5 * 2, (got['name'], got['values'])  # within 0.2 step on average
assert dg.series[0]['last'] == dg.series[0]['values'][-1] and abs(dg.series[1]['max'] - 6) < 0.4
assert not digitize(to_rgb(pg.get_pixmap(dpi=150).tobytes('png')), dict(synthetic, series=[
    {'name': 'Ряд A', 'color': 'красный', 'style': 'столбцы с накоплением'}])).ok  # a type the pilot did not allow
assert 'Значения не определены' in digitize(to_rgb(pg.get_pixmap(dpi=150).tobytes('png')), dict(synthetic, y_axis={'ticks': ['1']})).table()
# a grey series: grey is also gridlines and label edges, so its last point must agree with the model's «между A и B»
pg.draw_polyline([(x, yy(v)) for x, v in zip(xx, [3, 3.5, 4, 4.5, 4, 4.5, 5, 5.5])], color=(0.55, 0.55, 0.55), width=1.5)
grey_read = dict(synthetic, series=synthetic['series'] + [{'name': 'Ряд C', 'color': 'серый', 'style': 'линия', 'last_point': 'между 4 и 6'}])
png = pg.get_pixmap(dpi=150).tobytes('png')
grey = digitize(to_rgb(png), grey_read).series[2]
assert grey['last'] is not None and abs(grey['last'] - 5.5) < 0.4 and not grey.get('note'), grey
far = dict(grey_read, series=grey_read['series'][:2] + [dict(grey_read['series'][2], last_point='между 0 и 1')])
assert 'серый ряд не отличить' in digitize(to_rgb(png), far).series[2]['note']
# the code's values go into the chart block; a printed label the pixels confirm is verified, the measured ones approx
record = {'title': 'Рис. 9: Синтетика, %', 'read': dict(synthetic, series=[dict(synthetic['series'][0],
          printed_values=[{'x': '8', 'value': '7'}, {'x': '2', 'value': '-3'}]), synthetic['series'][1]]),
          'digitized': __import__('dataclasses').asdict(dg)}
chart_page = annotate('Текст.\n\n[График] Рис. 9: Синтетика, %\nТип: линия\nЗначения: Ряд A — 8: 9; Ряд B — 1: 6\n[/График]', [record])
assert 'сверены: пиксели): Ряд A — 8: 7' in chart_page and 'Описание графика (составлено кодом)' in chart_page and 'не подтверждены пикселями): Ряд A — 2: -3' in chart_page
assert 'Оцифровано кодом по графику' in chart_page and '| 8 |' in chart_page
clib = BankLibrary.from_pages({'SYNTH-Digit_10Oct2026.pdf': [chart_page]}, AS_OF)
cfr = next(f for f in clib.fragments if f.kind == 'chart')
st = {(x['value'], x['status']) for x in cfr.figures}
assert (7.0, 'verified') in st and (-3.0, 'unverified') in st and (9.0, 'unverified') in st and any(s == 'approx' for _, s in st)
assert not cfr.unverified and len(cfr.measured()) >= 8  # measured by code: a usable chart, not a shaky one
brief_chart = cfr.brief_text()
assert '≈' in brief_chart and 'оцифровано' in brief_chart and 'Ряд A — 2: -3' not in brief_chart and 'Ряд B — 1: 6' not in brief_chart
# a chart block the page OCR invented (grounding looked at the page and found no such chart): marked, not given to the agent
from stage5_insight_detection.bank_analytics.digitize import PHANTOM
invented = annotate('Текст.\n\n[График] Динамика ставки\nЗначения: 2024: 21%\n[/График]', [], grounded=True)
assert PHANTOM in invented and annotate('[График] Динамика ставки\n[/График]', [], grounded=False).count(PHANTOM) == 0
plib = BankLibrary.from_pages({'SYNTH-Phantom_10Oct2026.pdf': [invented]}, AS_OF)
assert not [f for f in plib.fragments if f.kind == 'chart'] and not any('21' in f.text for f in plib.fragments)

# a donut: shares by angle, each colour named by its pale legend swatch, a printed share the angle confirms is verified
from stage5_insight_detection.bank_analytics.chart_reading import chart_points, chart_words, pie_shares
ddoc = pymupdf.open()
dp = ddoc.new_page(width=420, height=260)
start, centre = 0.0, pymupdf.Point(130, 130)
for (name, share, rgb), y in zip((('Alpha', 50, (0.9, 0.1, 0.1)), ('Beta', 30, (0.35, 0.35, 0.35)), ('Gamma', 20, (0.1, 0.6, 0.3))),
                                 (80, 110, 140)):
    end = start + 3.6 * share
    arc = [centre + pymupdf.Point(100 * np.cos(np.radians(a)), 100 * np.sin(np.radians(a))) for a in np.linspace(start, end, 60)]
    dp.draw_polyline([centre] + arc + [centre], color=(1, 1, 1), fill=rgb, width=1.5)
    dp.draw_rect(pymupdf.Rect(280, y - 8, 288, y), color=None, fill=tuple(1 - 0.75 * (1 - c) for c in rgb))
    dp.insert_text((294, y), name, fontsize=11)
    start = end
dp.draw_circle(centre, 62, color=None, fill=(1, 1, 1))
pie_png = to_rgb(dp.get_pixmap(dpi=150).tobytes('png'))
pie_read = {'type': 'круговая', 'series': [{'name': n, 'printed_values': [{'x': n, 'value': v}]}
                                          for n, v in (('Alpha', '50%'), ('Beta', '30%'), ('Gamma', '25%'))]}
pie = pie_shares(pie_png, pie_read, chart_words(pie_png))
assert pie and {s['name']: round(s['share']) for s in pie['slices']} == {'Alpha': 50, 'Beta': 30, 'Gamma': 20}, pie
assert [s['pixels'] for s in pie['slices']] == [True, True, False] and pie['labels_sum'] == 105  # 25% is not the angle
assert ('Gamma', 'доля', 20.0, 'approx') in [(n, x, round(v), st) for n, x, v, st in chart_points({'code': {'pie': pie}})]
assert pie_shares(to_rgb(pg.get_pixmap(dpi=150).tobytes('png')), pie_read, []) is None  # a line chart has no ring
# stacked bars: series named by their legend swatches (the model gave them in the other order), a printed label goes to
# the segment whose height it fits, whatever series the model put it under
from stage5_insight_detection.bank_analytics.chart_reading import bar_series
sp = pymupdf.open().new_page(width=420, height=300)
yv = lambda v: 250 - 20 * v
for v in range(0, 11, 2):
    sp.insert_text((22, yv(v) + 3), str(v), fontsize=9)
for i, (lo, hi, cat) in enumerate(zip([3, 5, 2], [4, 2, 6], 'ABC')):
    sp.draw_rect(pymupdf.Rect(70 + 80 * i, yv(lo), 110 + 80 * i, yv(0)), color=None, fill=(0.9, 0.1, 0.1))
    sp.draw_rect(pymupdf.Rect(70 + 80 * i, yv(lo + hi), 110 + 80 * i, yv(lo)), color=None, fill=(0.4, 0.4, 0.4))
    sp.insert_text((86 + 80 * i, 266), cat, fontsize=10)
for name, rgb, y in (('Bottom', (0.9, 0.1, 0.1), 60), ('Top', (0.4, 0.4, 0.4), 80)):
    sp.draw_rect(pymupdf.Rect(320, y - 8, 328, y), color=None, fill=rgb)
    sp.insert_text((334, y), name, fontsize=10)
stack_png = to_rgb(sp.get_pixmap(dpi=150).tobytes('png'))
stack_read = {'x_axis': {'ticks': ['A', 'B', 'C']}, 'y_axis': {'ticks': [str(v) for v in range(0, 11, 2)]},
              'series': [{'name': 'Top', 'style': 'столбцы с накоплением', 'printed_values': [{'x': 'B', 'value': '5'}]},
                         {'name': 'Bottom', 'style': 'столбцы с накоплением', 'printed_values': [{'x': 'C', 'value': '6'}]}]}
stack = {s['name']: s for s in bar_series(stack_png, stack_read, chart_words(stack_png))['series']}
assert np.allclose(stack['Bottom']['values'], [3, 5, 2], atol=0.2) and np.allclose(stack['Top']['values'], [4, 2, 6], atol=0.2)
assert [(lb['x'], lb['pixels']) for lb in stack['Bottom']['labels']] == [('B', True)]
assert [(lb['x'], lb['pixels']) for lb in stack['Top']['labels']] == [('C', True)]

# a model loop and a table of dashes
looped, phrase = cut_phrase_loops('| июл-25 (прогноз) [неразборчиво] ' + '(прогноз) [неразборчиво] ' * 20 + 'конец')
assert phrase == '(прогноз) [неразборчиво]' and looped.count('[неразборчиво]') == 1 and 'конец' in looped
assert cut_phrase_loops('рост на 5%, рост на 5%. Обычный текст.')[1] == ''
assert drop_empty_tables('Рис. 2\n| Дата | A |\n|---|---|\n| янв-24 | — |\n| фев-24 | — |\nИсточник') == 'Рис. 2\nзначения не распознаны\nИсточник'
from stage5_insight_detection.bank_analytics.pages import only_new
main_ocr = 'Мы ожидаем паузы на 14% до конца года.\n\nИнфляция в августе 6,1%.'
assert only_new('Мы ожидаем паузы на 14%.\n\nПрогноз бюджета на 2027 год пересмотрен.', main_ocr) == 'Прогноз бюджета на 2027 год пересмотрен.'
doubled = clean_ocr_page(main_ocr + ' ' + '(прогноз) [неразборчиво] ' * 6, main_ocr)[0]
assert 'Резервное распознавание' not in doubled and doubled.count('Мы ожидаем паузы') == 1  # the fallback adds nothing new
assert 'Резервное распознавание' in clean_ocr_page('Текст ' + '(прогноз) [неразборчиво] ' * 6, 'старый OCR этой страницы')[0]

# --- tables: Tesseract on a synthetic table, three readings reconciled cell by cell, automatic checks ---
from stage5_insight_detection.bank_analytics.tables import auto_checks, norm, reconcile, tesseract_cells
from stage5_insight_detection.bank_analytics.tables import reconciled_markdown, split_label
assert split_label(['Ставка, % 6,3% 4,3%', '7,5%'], 4) == ['Ставка, %', '6,3%', '4,3%', '7,5%']
assert split_label(['2019 2020', '2021'], 4) == ['', '2019', '2020', '2021'] and split_label(['Итого 2025', '1'], 2) == ['Итого 2025', '1']
head = [['2025', '2026'], ['ВВП', '1,0', '1,5'], ['Russia', '', ''], ['', '', '']]
md_tab = reconciled_markdown({'tables': [{'bbox': [0, 0, 100, 100], 'title': 'T', 'readings': {
    'gemma': head[:2] + [['ИПЦ', '6,1', '5,0'], ['Ключевая', '16', '14']], 'qwen': head + [['ИПЦ', '6,1', '5,0'], ['Ключевая', '16', '14']],
    'tesseract': [['2025 2026'], ['ВВП 1,0', '1,5'], ['ИПЦ 6,1 5,0'], ['Ключевая 16 14']]}}]})[0]
assert '|  | 2025 | 2026 |' in md_tab and 'Russia' not in md_tab and '| ВВП | 1,0 | 1,5 |' in md_tab  # corner cell; prompt leak
tdoc = pymupdf.open()
tp = tdoc.new_page(width=360, height=150)
grid = [['', '2024', '2025', '2026F'], ['Rate', '21,0%', '16,0%', '13,5%'], ['FX', '102', '78,2', '87,2'], ['Share', '40%', '35%', '25%']]  # base-14 font: no Cyrillic
for i, row in enumerate(grid):
    for j, cell in enumerate(row):
        tp.insert_text((12 + 85 * j, 30 + 30 * i), cell, fontsize=11)
for i in range(5):
    tp.draw_line((8, 12 + 30 * i), (352, 12 + 30 * i), width=0.6)
tess = tesseract_cells(to_rgb(tp.get_pixmap(dpi=300).tobytes('png')))
assert len(tess) == 4 and all(len(r) == 4 for r in tess), tess
assert sum(norm(a) == norm(b) for r, g in zip(tess, grid) for a, b in zip(r, g) if b) >= 12, tess  # 12 of 15 printed cells
assert norm('1 777') == norm('1777') == '1777' and norm('I5,9%') == '15,9%' and norm('−0,8%') == norm('-0.8%') and norm('ruААА') == 'ruaaa'
three = reconcile({'gemma': [['Ставка', '21,0%', '16,0%'], ['Курс', '102', '78,2']],
                   'qwen': [['Ставка', '21,0 %', '16,0%'], ['Курс', '102', '76,2']],
                   'tesseract': [['Ставка', '21.0%', ''], ['Курс', '1O2', '78,2']]})
assert [c['status'] for c in three[0]] == ['verified', 'verified', 'two'], three[0]  # the third reading empty: «2 of 3»
assert [c['status'] for c in three[1]] == ['verified', 'verified', 'unverified'] and three[1][2]['value'] == '78,2', three[1]
checks = auto_checks(['', '2024', '2025'], [[{'value': 'Доля A'}, {'value': '60%'}], [{'value': 'Доля B'}, {'value': '30%'}],
                                            [{'value': 'Доля C'}, {'value': '10%'}]])
assert any('год или период' in c for c in checks) and any('100%' in c for c in checks), checks

# overlapping bands glued by row labels; an unsure junction keeps its rows as doubtful
from stage5_insight_detection.bank_analytics.tables import merge_bands, widen, column_style
head = ['', '2024', '2025']
glued, unsure = merge_bands([[head, ['A', '1', '2'], ['B', '3', '4'], ['C', '5', '6']],
                             [head, ['B', '3', '4'], ['C', '5', '6'], ['D', '7', '8']]])
assert [r[0] for r in glued] == ['', 'A', 'B', 'C', 'D'] and not unsure, glued
glued, unsure = merge_bands([[head, ['A', '1', '2'], ['B', '3', '4']], [head, ['X', '9', '9'], ['D', '7', '8']]])
assert [r[0] for r in glued] == ['', 'A', 'B', 'X', 'D'] and unsure == {3, 4}  # nothing lost, the junction rows marked
assert all(c['status'] == 'unverified' for c in reconcile({'gemma': glued, 'qwen': glued, 'tesseract': glued},
                                                         doubtful={'gemma': unsure, 'qwen': unsure, 'tesseract': unsure})[3])
# values matched to the row by its label: a row that carries the next row's values is not confirmed by the next row
shifted = reconcile({'gemma': [['Корп. кредиты', '22%'], ['Корп. депозиты', '22%']],
                     'qwen': [['Корп. кредиты', '22%'], ['Корп. депозиты', '22%']],
                     'tesseract': [['Корп. кредиты', '10%'], ['Корп. депозиты', '22%']]})
assert shifted[0][1]['status'] == 'unverified' and shifted[1][1]['status'] == 'verified', shifted
# same digits, another separator: the column decides (thousands here)
thousands = reconcile({'gemma': [['', 'MarCap'], ['Severstal', '17 760'], ['Gerdau', '9 215'], ['Kardemir', '992']],
                       'qwen': [['', 'MarCap'], ['Severstal', '17 760'], ['Gerdau', '9,215'], ['Kardemir', '992']],
                       'tesseract': [['', 'MarCap'], ['Severstal', '17760'], ['Gerdau', '9 215'], ['Kardemir', '992']]})
assert thousands[2][1] == {'value': '9 215', 'status': 'verified'} and column_style(['17 760', '9 215', '12 553']) == 'thousands'
assert column_style(['21,2', '33,1', '8,6']) == 'decimal' and column_style(['1', 'x']) == ''
# a row one cell short of the header placed by the x of a full Tesseract row, not by order
assert widen(['Реал.', '5,2%', '6,5%', '8,6%', '8,1%'], 6, ['Реал.', '5,2%', '6,5%', '8,6%', '8,6%', '8,1%']) == \
    ['Реал.', '5,2%', '6,5%', '8,6%', '', '8,1%']
assert widen(['A', '1', '3'], 4, ['A', '1', '2', '3']) == ['A', '1', '', '3']

# the page's broken tables are replaced by the reconciled one; a cell the readings disagree on is unverified
from stage5_insight_detection.bank_analytics.tables import replace_tables
page_rec = {'tables': [{'title': 'ММК', 'bbox': [0, 0, 100, 100], 'readings': {
    'gemma': [['US$ млн', '1К20', '4К20'], ['Выручка', '1710', '1852'], ['EBITDA', '442', '474']],
    'qwen': [['US$ млн', '1К20', '4К20'], ['Выручка', '1710', '1852'], ['EBITDA', '442', '471']],
    'tesseract': [['US$ млн', '1К20', '4К20'], ['Выручка', '1710', '1852'], ['EBITDA', '442', '']]}}]}
replaced = replace_tables('Текст отчёта.\n| Выручка | I7I0 | I852 |\nВыручка\t1710\t1852', page_rec)
assert 'I7I0' not in replaced and '\t' not in replaced and '[таблица сверена' in replaced and '| EBITDA | 442 | 474 (?) |' in replaced, replaced
tlib = BankLibrary.from_pages({'SYNTH-Tables_10Oct2026.pdf': [replaced]}, AS_OF)
tfig = {(x['value'], x['status']) for f in tlib.fragments if f.kind == 'table' for x in f.figures}
assert (1710.0, 'verified') in tfig and (474.0, 'unverified') in tfig and (442.0, 'verified') in tfig, tfig

# stage-2 facts the writer saw as fact:stage2_<n> are allowed refs, their numbers come from the label, the card shows them
s2 = {'id': 'fact:stage2_1', 'label': 'Выручка группы 2025: 40 млрд ₽', 'value': '', 'unit': '', 'formula': '',
      'source': 'stage2_company_facts', 'kind': 'fact'}
b2 = InsightBrief(**dict(brief, refs=brief['refs'] + ['fact:stage2_1'], facts=brief['facts'] + [s2]))
cited = Insight(**dict(insight, bank_view=[], what_happened=[{'text': 'Выручка группы — 40 млрд ₽', 'sources': ['fact:stage2_1']}]))
_, warnings = audit(cited, b2, {})
assert not any(w['code'] == 'unknown_ref' and w['section'] == 'what_happened' for w in warnings), warnings
from types import SimpleNamespace
from stage5_insight_detection.analyst_loop.dossier import Dossier
from stage5_insight_detection.analyst_loop.loop import BriefEvidence
from stage6_insight_generation.insight_card.render import render
d2 = Dossier(SimpleNamespace(id='news:1', news_id=1, title='t', text='', date=None, source='', url='', matched_fragments=[]), profile, 'cfo',
             extra_facts=['Выручка группы 2025: 40 млрд ₽'])
assert 'fact:stage2_1' in d2.refs() and 'fact:stage2_2' not in d2.refs()
assert BriefEvidence(d2, b2).render().count('Выручка группы 2025') == 1  # rendered once, as the stage-2 line
card = render(json.loads(json.dumps(dict(res, insight=dict(res['draft'], what_happened=[cited.what_happened[0].model_dump()]),
                                         verdict=dict(res['verdict'], brief=dict(brief, refs=b2.refs, facts=b2.facts))), default=str)))
assert 'Факт о компании (этап 2)' in card and 'Выручка группы 2025' in card

# --- FinRobot 5: JSONL journal and the card with source popovers ---
from stage6_insight_generation.insight_card.render import render
# Inspect the marked draft only in this component test; the actual rejected result never displays it as an insight.
assert 'Событие отклонено' in render(json.loads(json.dumps(res, default=str)))
assert 'Что сделать' not in render(json.loads(json.dumps(res, default=str)))
page = render(json.loads(json.dumps(dict(res, insight=res['draft']), default=str)), AgentTrace.read(tmp / 'test-run.jsonl'))
assert 'Позиция Банка' in page and 'Вывод AI' in page and 'не подтверждено' in page and 'class="pop"' in page
if os.environ.get('CARD_OUT'):  # CARD_OUT=card.html -> save the demo card to look at it in a browser
    Path(os.environ['CARD_OUT']).write_text(page, encoding='utf-8')
assert 'Гипотезы' in page and 'инвестиционной рекомендацией' in page and 'Шаги агента' in page
assert 'вывод AI, не официальная позиция Банка' in render(json.loads(json.dumps(dict(res, insight=fixed.model_dump()), default=str)))
assert 'Событие отклонено' in render(json.loads(json.dumps(res_gold, default=str)))

lines = AgentTrace.read(tmp / 'test-run.jsonl')
steps = [e['step'] for e in lines]
for s in ('start', 'bank_match', 'enrich', 'compute', 'risks', 'grounds', 'assess', 'request', 'verdict', 'write', 'audit', 'end'):
    assert s in steps, s
if os.name != 'nt':  # POSIX mode bits are not Windows ACLs.
    assert (tmp / 'test-run.jsonl').stat().st_mode & 0o777 == 0o600
assert 'sk-***' in redact('key sk-abcdefghijklmnop1234') and '***' in redact('api_key=supersecretvalue')
# --- digest of several chunks: one merge call; claims of absence and new numbers do not survive ---
from langchain_core.messages import AIMessage
from stage5_insight_detection.bank_analytics.digest import ReportDigester, drop_absent, keep_known_numbers
assert drop_absent('Банк ждёт паузы на 14%. Прогнозов Банка в этой части нет.') == 'Банк ждёт паузы на 14%.'
assert keep_known_numbers('Банк ждёт 14%. Инфляция будет 9%.', ['Банк ждёт 14%.']) == 'Банк ждёт 14%.'
two = '# Ставка\n\n## Страница 1\n\n' + 'Мы ожидаем паузы на 14%. ' * 500 + '\n\n## Страница 2\n\n' + 'Рис. 1: ставка. ' * 50
digester = ReportDigester(ScriptedChatModel(responses=[
    calls(c('ReportDigest', summary='Банк ждёт паузы на 14%.', bank_views=[{'indicator': 'ставка', 'value': '14', 'unit': '%', 'page': 1}])),
    calls(c('ReportDigest', summary='На странице 2 только график. Содержательных прогнозов нет.')),
    AIMessage(content='Банк ждёт паузы на 14%. Ставка снизится до 12%. На странице 2 — график ставки.')]))
dg2, parts2 = digester.digest(two, 'Ставка')
assert len(parts2) == 2 and dg2.summary == 'Банк ждёт паузы на 14%. На странице 2 — график ставки.', dg2.summary
assert not dg2.bank_views[0].unverified

# --- export_text: model loops are cut, a page renders to Markdown ---
from stage5_insight_detection.bank_analytics.export_text import dedup, render_page
loop = 'Ставка 14%.  \nИнфляция замедлилась до 6,1% г/г.\nИнфляция замедлилась до 6,1% г/г.\n\n| a | b |\n| 1 | 2 |\n\n| a | b |\n| 1 | 2 |'
text, removed = dedup(loop)
assert removed == 2 and text.count('Инфляция замедлилась') == 1 and text.count('| a | b |') == 1 and 'Ставка 14%.\n' in text
assert dedup('Рост цен ускорился в июле. Рост цен ускорился в июле. Мы ждём паузы.')[0] == 'Рост цен ускорился в июле. Мы ждём паузы.'
md, st = render_page(3, '[График] Ключевая ставка\nЗначения: ставка — сен.26: 14%\n[/График]\n\n| % | 2026 |\n|---|---|\n| ИПЦ | 6,1 |')
assert md.startswith('## Страница 3\n') and (st.tables, st.charts, st.empty) == (1, 1, 0)
assert render_page(5, '[График] x\nЗначения: ' + '; '.join(['0,15'] * 5) + '\n[/График]')[1].suspect == 1 and st.suspect == 0
garbled = '| РусГидро | I7,4% | IZ | I5 | I5,9% | I,O | ZOZ7 | I4,I% |'
assert render_page(6, garbled, 'РусГидро 17,4% 12 15 15,9%')[1].fallback == 1 and 'Резервное распознавание' in render_page(6, garbled, 'x')[0]
assert render_page(6, 'Ставка 14%.', 'Ставка 14%.')[1].fallback == 0 and render_page(6, 'ok ' + 'уа' * 10, 'x')[1].fallback == 1
assert render_page(4, '  \n')[0] == '## Страница 4\n\n_(на странице нет текста)_\n' and render_page(4, '')[1].empty == 1

# --- digest: every number of an item must be on the page it cites ---
from stage5_insight_detection.bank_analytics.digest import ReportDigest, ReportDigester, chunks, split_pages
report_md = ('# Ставка\n\nФайл: x.pdf\n\n## Страница 1\n\nМы ожидаем снижения ключевой ставки до 12% к концу 2026 г.\n\n'
             '## Страница 2\n\nИнфляция в августе составила 6,1% г/г.\n')
_, pages = split_pages(report_md)
from stage5_insight_detection.bank_analytics.digest import drop_scale_values
chart = '[График] Курс\nЗначения: максимум ≈110\nЧто видно: рост\n[/График]'
assert '110' not in drop_scale_values(chart) and 'Что видно: рост' in drop_scale_values(chart)
assert drop_scale_values(chart.replace(' ≈110', ' 1,62')) == chart.replace(' ≈110', ' 1,62')
assert set(pages) == {1, 2} and len(chunks({1: 'a' * 9000, 2: 'b' * 9000, 3: 'c'})) == 2
scripted = ReportDigest(summary='Банк ждёт снижения ставки.', bank_views=[
    {'indicator': 'ключевая ставка', 'value': '12', 'unit': '%', 'horizon': 'конец 2026', 'direction': 'down', 'page': 1},
    {'indicator': 'ключевая ставка', 'value': '11', 'unit': '%', 'page': 1}],
    key_figures=[{'indicator': 'инфляция', 'value': '6,1', 'unit': '% г/г', 'period': 'август', 'page': 2},
                 {'indicator': 'инфляция', 'value': '6,1', 'unit': '%'}])
dg, summaries = ReportDigester(ScriptedChatModel(responses=[calls(c('ReportDigest', **scripted.model_dump()))])).digest(report_md, 'Ставка')
assert summaries == ['Банк ждёт снижения ставки.'] and dg.summary == summaries[0]  # one chunk: no merge call
assert [v.unverified for v in dg.bank_views] == [False, True]  # 11% is not on page 1
assert [f.unverified for f in dg.key_figures] == [False, True]  # no page -> unverified

print(f'ok: {len(lib.fragments)} fragments; insight for news:101 ({len(res["consequences"])} consequences, '
      f'{len(res["open_requests"])} open request); 2 refusals; {len(lines)} journal lines')
