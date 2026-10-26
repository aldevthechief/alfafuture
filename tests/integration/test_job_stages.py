"""Stages a real InsightPipeline run reports with scripted models (no network): the checker's rejection sends the draft
back to the writer, but the job stage never goes back from «проверяем» to «формируем»."""
import json, sys, tempfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import common.llm.models as llm_models
llm_models.LANGFUSE = []
from common import progress
from common.llm import JsonlCache
from common.llm.testing import ScriptedChatModel, calls, tool_call as c
from pipeline import InsightPipeline
from stage4_news_relevance.consequence_extraction.consequences import ConsequenceExtractor
from stage5_insight_detection.bank_analytics.search import BankAnalytics
from stage2_company_facts.deterministic_facts.facts import build
from stage3_news_search.retrieval import candidates


def test_pipeline_reports_forward_stages():
    tmp = Path(tempfile.mkdtemp())
    (tmp / 'ALFA-Macro_Insights-Policy_Rate_11Sep.jsonl').write_text(json.dumps(
        {'key': '1', 'value': 'Банк ожидает паузу по ключевой ставке до конца 2026 года.'}, ensure_ascii=False))
    first = candidates('U_68622', 'cfo', top=1, facts=build(['U_68622'])['U_68622'], method='lexical')[0].iloc[0]
    claim = {'text': first['title'], 'sources': [f'news:{first["id"]}']}
    quoted = dict(claim, evidence=[{'source': claim['sources'][0], 'quote': str(first['text'])[:80]}])
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
        calls(c('decide', is_insight=True, score=2, company_link='кредитные продукты', mechanism='ключевая ставка',
                bank_view='пауза до конца 2026', related_news_ids=[])),
        calls(c('Insight', **insight)),
        calls(c('Review', ok=False, issues=['действие без срока'])),
        calls(c('Insight', **insight)),
    ])
    seen = []
    p = progress.Progress(lambda snap: seen.append((snap['code'], snap['done'], snap['total'])))
    progress.bind(p)
    try:
        results = InsightPipeline(cheap, agent, consequences=ConsequenceExtractor(cheap, cache=JsonlCache('consequences', folder=tmp)),
                                  bank=BankAnalytics(ocr_dir=tmp)).run('U_68622', 'cfo', n_candidates=1, n_judge=1, retrieval='lexical')
    finally:
        progress.bind(None)
    assert results[0]['review']['ok'] and cheap.i == 0 and agent.i == 0, results
    history = [h['code'] for h in p.snapshot()['history']]
    assert history == ['queued', 'collecting', 'analyzing', 'generating', 'verifying'], history
    assert ('analyzing', 1, 1) in seen and seen[-1] == ('verifying', 1, 1), seen


if __name__ == '__main__':
    test_pipeline_reports_forward_stages(); print('ok')
