"""Offline test of the integration: contract v2 for the news results, the cache of analytics runs, the quota next to the news insights, the rows of
the pipeline's result list, the card. Scripted models, synthetic bank OCR, temporary cache (no network, the committed cache untouched).
python -X utf8 stage5_insight_detection/analytics_insights/test_service.py"""
import json, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import common.llm.models as llm_models
llm_models.LANGFUSE = []
from common.llm.testing import ScriptedChatModel, calls, tool_call as c
from pipeline import analytics_rows
from stage5_insight_detection.analytics_insights import service
from stage5_insight_detection.analytics_insights.agent import AnalyticsInsightAgent
from stage5_insight_detection.bank_analytics.search import BankAnalytics
from stage6_insight_generation.insight_card.render import render
from stage6_insight_generation.insight_writer import insight as v1
from stage6_insight_generation.insight_writer import insight_v2 as v2

# 1. news results migrate to v2 without losing anything; an insight already in v2 and one that does not parse are left alone
feed = json.loads((Path(__file__).resolve().parents[3] / 'tests/fixtures/feed_v1_insights.json').read_text(encoding='utf-8'))
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
raw = found[0]
old = {k: raw[k] for k in v1.Insight.model_fields if k in raw}
result = {'news_id': 17, 'title': 't', 'status': 'ready', 'insight': old, 'stage4': {'x': 1}}
new = v2.migrate_result(result, 'U_1')
assert new['stage4'] == {'x': 1} and new['insight']['origin'] == 'news' and 17 in new['insight']['trigger']['news_ids'] and new['insight']['company_id'] == 'U_1'
assert v1.Insight.model_validate({k: new['insight'][k] for k in v1.Insight.model_fields}).model_dump() == v1.Insight.model_validate(old).model_dump(), 'nothing lost'
assert v2.migrate_result(new, 'U_1')['insight'] == new['insight'], 'a v2 insight is not migrated twice'
assert 'migration_errors' in v2.migrate_result({'news_id': 1, 'insight': {'title': 'broken'}}, 'U_1') and v2.migrate_result({'news_id': 1, 'status': 'model_error'}, 'U_1') == {'news_id': 1, 'status': 'model_error'}
assert render(new), 'the old card renders a v2 insight'

# 2. the cache of analytics runs
tmp = Path(tempfile.mkdtemp())
(tmp / 'ALFA-Macro_Insights-Policy_Rate_11Sep.jsonl').write_text(json.dumps(
    {'key': '1', 'value': 'Ключевая ставка ЦБ сохранена на уровне 14%.\n\nБанк ожидает паузу по ключевой ставке до конца 2026 года '
                          'и снижение ставки до 12% к концу 2027 года; кредиты для бизнеса останутся дорогими.'}, ensure_ascii=False))
bank = BankAnalytics(ocr_dir=tmp)
pdfs = Path(tempfile.mkdtemp())
facts = [{'cat': 'bank', 'text': 'Использует кредитные продукты банка', 'depth': 1}, {'cat': 'scale', 'text': 'Годовая выручка 105.6 млрд ₽', 'depth': 1}]
REPORT = 'Ключевая ставка ЦБ РФ: без изменения на 14,0%'
quote = 'Банк ожидает паузу по ключевой ставке до конца 2026 года и снижение ставки до 12% к концу 2027 года'
cand = dict(kind='view_change', title='Банк отложил снижение ставки', conclusion='Деньги дороже дольше.', channel='ставка', effect='risk', role_fit=3, confidence='high',
            bank_claims=[dict(text='пауза 14% до конца 2026, 12% к концу 2027', evidence=[dict(report=REPORT, date='2026-09-11', page=1, quote=quote)])],
            company_facts=[1], link_inference='кредиты дороже дольше', mechanism='расходы на проценты выше',
            actions=[dict(text='Пересчитать бюджет', because='Банк ждёт 14%')], impact=dict(level='mid', criteria='ставка влияет на долг'))
def agent_factory():
    return AnalyticsInsightAgent(ScriptedChatModel(responses=[calls(c('add_candidate', **cand)), calls(c('finish', summary='ок', covered=['ставка']))]), bank, ScriptedChatModel(responses=[]))

cache = tmp / 'cache'
run = service.analytics_run('U_1', 'cfo', facts, cache_dir=cache, pdf_dir=pdfs, agent_factory=agent_factory, today='2026-10-23')
assert run and len(run.insights) == 1 and not run.cached and run.insights[0].origin == 'bank' and run.covered == ['ставка'] and run.key and (cache / 'U_1_cfo.json').exists()
hit = service.analytics_run('U_1', 'cfo', facts, cache_dir=cache, pdf_dir=pdfs, agent_factory=lambda: 1 / 0, today='2026-10-23')  # a hit never builds an agent
assert hit.cached and not hit.stale_cache and hit.insights[0].id == run.insights[0].id
assert service.analytics_run('U_2', 'cfo', facts, cache_dir=cache, pdf_dir=pdfs, cache_only=True) is None, 'cache only: nothing cached -> None, no model'
changed = facts + [{'cat': 'geo', 'text': 'Москва', 'depth': 1}]
stale = service.analytics_run('U_1', 'cfo', changed, cache_dir=cache, pdf_dir=pdfs, cache_only=True)
assert stale.cached and stale.stale_cache, 'cache only serves a stale cache, marked'
(pdfs / 'new_report.pdf').write_bytes(b'%PDF-1.4 stub')
assert service.cache_key('U_1', 'cfo', facts, pdfs) != run.key, 'a new report in the folder changes the key'
re_run = service.analytics_run('U_1', 'cfo', facts, cache_dir=cache, pdf_dir=pdfs, agent_factory=agent_factory, refresh=True, today='2026-10-24')
assert not re_run.cached and re_run.as_of == '2026-10-24'
failed = service.analytics_run('U_9', 'cfo', facts, cache_dir=cache, pdf_dir=pdfs, today='2026-10-23',
                               agent_factory=lambda: AnalyticsInsightAgent(ScriptedChatModel(responses=[]), bank, ScriptedChatModel(responses=[])))
assert failed.error and not failed.insights and not (cache / 'U_9_cfo.json').exists(), 'a failed run is returned, not cached'

# 3. the quota next to the news insights
def ins(title, fit, level, conf='medium'):
    return v2.Insight(title=title, role='cfo', role_fit=fit, confidence=conf, impact=v2.Impact(level=level, criteria='x') if level else None)
pool = [ins('a', 2, 'mid'), ins('b', 3, 'low'), ins('c', 3, 'mid'), ins('d', 1, 'high'), ins('e', 3, 'mid', 'high')]
assert [i.title for i in service.select(pool, 0)] == ['e', 'c', 'b', 'a', 'd'], 'without news all, best first (fit, impact, confidence)'
assert len(service.select(pool, 1)) == 1 and len(service.select(pool, 5)) == 3 and len(service.select(pool, 100)) == 5 and len(service.select([], 3)) == 0
assert all(len(service.select(pool, n)) / (n + len(service.select(pool, n))) <= 0.4 + 1e-9 or len(service.select(pool, n)) == 1 for n in range(1, 12))

# 4. the rows of the result list and the card
rows = analytics_rows(run, '', [{'status': 'ready', 'insight': {}}] * 3, 0.4, False)
assert [r['status'] for r in rows] == ['ready', 'analysis_run'] and rows[0]['news_id'] is None and rows[0]['source'] == 'bank_analytics' and rows[0]['insight']['origin'] == 'bank'
assert rows[1]['analysis']['covered'] == ['ставка'] and 'insights' not in rows[1]['analysis'] and render(rows[0]), 'the card of an analytics insight renders'
assert analytics_rows(None, '', [], 0.4, True) == [] and analytics_rows(None, 'FileNotFoundError: x', [], 0.4, False) == [], 'no run: no rows, the news results stay as they are'
print('ok: integration: news results in contract v2, cache hit / stale / refresh / failed, quota next to the news, rows of the result list, card renders')
