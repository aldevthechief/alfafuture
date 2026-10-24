"""Offline test of stages 2-6 wiring with scripted models (no network; real news pool and case tables; temporary
LLM cache and synthetic bank OCR, so data/ is not touched): python -X utf8 test_pipeline.py"""
import json, sys, tempfile
from pathlib import Path

from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import common.llm.models as llm_models
llm_models.LANGFUSE = []  # offline test: agent runs attach Langfuse to their root trace, scripted calls must not be sent
from common.llm import JsonlCache, PromptLibrary
from common.llm.testing import ScriptedChatModel, calls, tool_call as c
from pipeline import InsightPipeline
from stage4_news_relevance.consequence_extraction.consequences import ConsequenceExtractor
from stage5_insight_detection.bank_analytics.search import BankAnalytics
from stage2_company_facts.deterministic_facts.facts import build
from stage3_news_search.retrieval import candidates

ROOT = Path(__file__).resolve().parents[2]

# every prompt file parses and renders with dummy variables
for folder in ROOT.glob('stage*/**/prompts'):
    lib = PromptLibrary(folder)
    for name in lib.names():
        if name.startswith('ocr_'):  # sent raw via PROMPTS.text with an image, never formatted (JSON braces inside)
            continue
        prompt = lib.chat(name)
        assert prompt.format_messages(**{v: 'x' for v in prompt.input_variables}), (folder, name)

tmp = Path(tempfile.mkdtemp())
(tmp / 'ALFA-Macro_Insights-Policy_Rate_11Sep.jsonl').write_text(json.dumps(
    {'key': '1', 'value': 'Ключевая ставка ЦБ сохранена на уровне 14%.\n\nБанк ожидает паузу по ключевой ставке до конца 2026 года '
                          'и снижение ставки до 12% к концу 2027 года; кредиты для бизнеса останутся дорогими.'}, ensure_ascii=False))
(tmp / 'ALFA_Russian_Steels_1К21PreviewUpdate_15Apr2021.jsonl').write_text(json.dumps(
    {'key': '1', 'value': 'Ключевая ставка и спрос на сталь в 2021 году: прогноз ставки 5%, рост спроса на металлопрокат.'},
    ensure_ascii=False))
bank = BankAnalytics(ocr_dir=tmp)
hits = bank.search('ключевая ставка прогноз')
assert hits and not hits[0]['stale'] and any(h['stale'] for h in hits), hits

facts = build(['U_68622'])['U_68622']
first = candidates('U_68622', 'cfo', top=1, facts=facts, method='lexical')[0].iloc[0]
claim = {'text': first['title'], 'sources': [f'news:{first["id"]}']}
quoted = dict(claim, evidence=[{'source': claim['sources'][0], 'quote': str(first['text'])[:80]}])  # the writer's check wants a verbatim quote
insight = {'title': 'Дорогой долг дольше', 'role': 'cfo', 'what_happened': [quoted], 'why_it_matters': [claim],
           'actions': ['Пересмотреть график рефинансирования до 18 декабря'], 'bank_view': [claim],
           'ai_conclusion': 'Стоимость долга не снизится до 2027 года.', 'confidence': 'medium'}
cheap = ScriptedChatModel(responses=[
    calls(c('Relevance', score=2, link='кредиты', channel='ставка', direction='-', role_fit=2)),
    calls(c('NewsConsequences', economic=True, event='ЦБ сохранил ставку', status='fact', significance=2,
            affected=[{'who': 'заёмщики', 'channel': 'ставка', 'mechanism': 'дорогой кредит'}])),
])
agent = ScriptedChatModel(responses=[
    calls(c('search_bank_analytics', query='ключевая ставка'), c('same_event_reports')),
    calls(c('decide', is_insight=True, score=2, company_link='кредитные продукты', mechanism='ключевая ставка и стоимость кредитов',
            bank_view='пауза 14% до конца 2026 (11.09.2026, p.1)', related_news_ids=[])),
    calls(c('Insight', **insight)),
    calls(c('Review', ok=False, issues=['действие без срока'])),  # forces one revision
    calls(c('Insight', **insight)),
    calls(c('Review', ok=True)),
])
consequences = ConsequenceExtractor(cheap, cache=JsonlCache('consequences', folder=tmp))
results = InsightPipeline(cheap, agent, consequences=consequences, bank=bank).run(
    'U_68622', 'cfo', n_candidates=1, n_judge=1, retrieval='lexical')

assert len(results) == 1 and results[0]['decision']['is_insight'], results
assert results[0]['insight']['title'] == 'Дорогой долг дольше' and results[0]['review']['ok'], results[0]
assert results[0]['stage4']['consequences']['status'] == 'fact'
assert cheap.i == 0 and agent.i == 0, 'scripts must be consumed exactly'
print('ok: prompts render; bank search prefers fresh reports; stages 2-6 produce a reviewed insight; news:', results[0]['title'][:60])
