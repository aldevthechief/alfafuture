"""Writer verdicts (24.10): a rejected draft is fixed once in the same dialog and published without a second check;
a checker with no valid verdict is retried, then the draft is published marked as not checked."""
import asyncio

from langchain_core.messages import AIMessage

from common.llm.testing import ScriptedChatModel, calls, tool_call
from stage6_insight_generation.insight_writer.insight import Insight
from stage6_insight_generation.insight_writer.writer import Evidence, InsightWriter

DRAFT = Insight(title='Синтетический инсайт', role='cfo', what_happened=[], why_it_matters=[],
                actions=['Проверить данные'], bank_view=[], ai_conclusion='Проверить условия', confidence='low')
EVIDENCE = Evidence(news=[{'id': 1, 'date': '2026-10-01', 'source': 's', 'title': 't', 'text': 'x'}], facts=['Факт'])


def write(responses):
    model = ScriptedChatModel(responses=responses)
    insight, review = asyncio.run(InsightWriter(model).awrite('U_1', 'cfo', '{}', EVIDENCE))
    assert model.i == 0, 'scripts must be consumed exactly'
    return insight, review, model


def test_rejected_draft_is_fixed_once_in_the_same_dialog_without_a_second_check():
    fixed = DRAFT.model_copy(update={'title': 'Исправленный инсайт'})
    insight, review, model = write([calls(tool_call('Insight', **DRAFT.model_dump())),
                                    calls(tool_call('Review', ok=False, issues=['нет срока'])),
                                    calls(tool_call('Insight', **fixed.model_dump()))])
    assert insight.title == 'Исправленный инсайт' and review.ok
    assert review.issues == ['исправлено по проверке: нет срока']


def test_checker_without_a_verdict_publishes_the_draft_marked_unchecked():
    empty = AIMessage(content='')  # reasoning only: no tool call, no text
    insight, review, _ = write([calls(tool_call('Insight', **DRAFT.model_dump())), empty, empty, empty, empty])
    assert insight.title == DRAFT.title and review.ok
    assert review.issues == ['проверка по материалам не дала вердикта: инсайт не проверен']
