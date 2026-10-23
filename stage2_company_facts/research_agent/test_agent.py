"""Offline test of the agent graph with a scripted model (no network, real case tables):
python -X utf8 stage2_company_facts/research_agent/test_agent.py"""
import sys
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import common.llm.models as llm_models
from common.llm import json_args
from common.llm.testing import ScriptedChatModel, calls, tool_call as c
from stage2_company_facts.research_agent.agent import ResearchAgent
from stage2_company_facts.research_agent.memory import ProposeHypothesis


class Runs(BaseCallbackHandler):
    """Stands in for the Langfuse handler: records every run with its parent, to check the trace tree."""
    def __init__(self):
        self.runs = {}  # run_id -> (kind, name, parent_run_id)

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, name=None, **kw):
        self.runs[run_id] = ('chain', name or (serialized or {}).get('name'), parent_run_id)

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kw):
        self.runs[run_id] = ('llm', 'chat', parent_run_id)

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kw):
        self.runs[run_id] = ('tool', (serialized or {}).get('name'), parent_run_id)

    def root(self, run_id):
        while (parent := self.runs.get(run_id, (None, None, None))[2]) is not None:
            run_id = parent
        return run_id


script = [
    # main loop
    calls(c('counterparties', company_id='U_44767', direction='in'), c('company_card', company_id='U_45078')),
    calls(c('propose_hypothesis', text='Специализированный застройщик одного проекта', evidence_for='выручка ~0 при обороте 1.2 млрд',
            evidence_against='не видно покупателей', test='платит ли генподрядчику и стройконтролю')),
    calls(c('finish', summary='рано')),  # must be rejected: H1 is still open
    calls(c('verify_hypothesis', hid='H1')),
    # verification sub-loop
    calls(c('counterparties', company_id='U_44767', direction='out', group_by='okved')),
    calls(c('report_verdict', status='supported', reason='платит 41.20 и 71.12.1 (obs3)')),
    # compaction after the sub-task, then the main loop again
    AIMessage('сводка: проверены входящие и исходящие'),
    # the same bad call twice: the second must be answered with a hint, not re-run (live run: 30 repeats)
    calls(c('counterparties', company_id='U_44767', direction='sideways'),
          c('counterparties', company_id='U_44767', direction='sideways')),
    # list / dict arguments as JSON strings, exactly as GLM sent them in the live run
    calls(c('table_query', table='payments', filters='[["PAYER", "==", "U_44767"]]', group_by='["BENEFICIARY"]',
            agg='{"PAY_AMT": "sum"}', sort_by='PAY_AMT', top=5)),
    calls(c('finish', summary='SPV-застройщик (H1)',
            new_facts='[{"text": "~100% входящих от одного плательщика вне базы", "evidence": "obs1"}]')),
]
runs = Runs()
llm_models.LANGFUSE = [runs]  # trace_config attaches LANGFUSE to the root run; offline, record instead of sending
model = ScriptedChatModel(responses=script)
res = ResearchAgent(model, log=lambda *_: None).run('U_44767')

assert res['summary'] == 'SPV-застройщик (H1)', res['summary']
assert res['hypotheses'][0]['status'] == 'supported', res['hypotheses']
assert res['hypotheses'][0]['notes'] == ['платит 41.20 и 71.12.1 (obs3)']
assert [f['text'] for f in res['facts']] == ['~100% входящих от одного плательщика вне базы'], res['facts']
assert res['observations'] == 5  # the repeated bad call is not archived again
assert 'compact' in res['trace'], res['trace']
assert any('уже вернул ошибку' in t for t in res['trace']), 'a repeated failing call must get the hint'
assert any('[obs5]' in t and 'tool error' not in t.split('[obs5]')[1] for t in res['trace']), 'JSON-string args must work'
assert json_args(ProposeHypothesis, {'text': '[важно] не JSON'}) == {'text': '[важно] не JSON'}  # text fields untouched
assert any('finish отклонён' in t for t in res['trace']), 'finish with an open hypothesis must be rejected'
assert model.i == 0, 'script must be consumed exactly'  # the fake model wraps its index to 0 after the last response

# one agent run = one trace: every model call, data tool and memory tool hangs off a single root run
roots = {runs.root(r) for r in runs.runs}
assert len(roots) == 1, f'{len(roots)} roots'
assert runs.runs[roots.pop()][1] == 'research_agent U_44767'
names = {n for _, n, _ in runs.runs.values()}
assert {'finish', 'propose_hypothesis', 'verify_hypothesis', 'counterparties', 'tool_loop:report_verdict'} <= names, names
print('ok:', len(res['trace']), 'graph steps;', res['steps'], 'research turns; hypothesis', res['hypotheses'][0]['status'],
      f'; one trace with {len(runs.runs)} runs')
