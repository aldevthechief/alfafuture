"""Company research agent on LangGraph (default model GLM-5.3-Flash via the organizers' LLM proxy).

Starts from the deterministic facts of `deterministic_facts/facts.py`, explores the tables with read-only tools, stores
facts and competing hypotheses in memory, verifies every hypothesis in a refutation-first sub-loop and compacts the
working transcript at sub-task boundaries or on a size budget.

    START -> research -> act ─┬─ summary set / step limit ──> END
               ^              ├─ sub-task done / over budget ─> compact ─┐
               └──────────────┴──────────────────────────────────────────┘

`act` executes tool calls: data tools (output archived as obsN), memory tools, `verify_hypothesis`
(a `ToolLoop` sub-graph that ends with `report_verdict`) and `finish` (rejected while hypotheses are open).

python -X utf8 stage2_company_facts/research_agent/agent.py U_44767 [--max-steps 40]   (needs LLM_API_KEY)
-> data/companies/agent/U_44767.json
"""
import argparse, json, sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.data import DATA
from common.llm import ChatModels, PromptLibrary, ToolLoop, call_tool, json_args, trace_config
from stage2_company_facts.deterministic_facts.facts import build
from stage2_company_facts.research_agent.data_tools import CompanyDataTools
from stage2_company_facts.research_agent.memory import (MEMORY_TOOLS, Finish, Memory, ProposeHypothesis,
                                                        RecallObservation, Verdict, VerifyHypothesis)

PROMPTS = PromptLibrary(Path(__file__).with_name('prompts'))
MEMORY_SCHEMAS = {m.model_config['title']: m for m in MEMORY_TOOLS}
OBS_CHARS = 3500  # observation shown in context; the full text stays in memory.observations


class ResearchState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    company_id: str
    facts: list[dict]
    hypotheses: list[dict]
    observations: dict[str, dict]
    summary: str | None
    steps: int
    compact_due: bool


def starting_facts(company_id: str, depth12_limit: int = 120) -> str:
    facts = build([company_id])[company_id]
    lines = [f'[{f["cat"]}] {f["text"]}' for f in facts if f['depth'] <= 2][:depth12_limit]
    return '\n'.join(lines + [f'[{f["cat"]}] {f["text"]}' for f in facts if f['depth'] == 3])


class ResearchAgent:
    def __init__(self, model: BaseChatModel, data_tools: list[BaseTool] | None = None, max_steps: int = 40,
                 verify_steps: int = 8, budget_chars: int = 120_000, log: Callable[[str], None] = print):
        self.model = model
        self.data_tools = {t.name: t for t in data_tools or CompanyDataTools().as_tools()}
        self.researcher = model.bind_tools([*self.data_tools.values(), *MEMORY_TOOLS])
        self.summarizer = PROMPTS.chat('compact') | model | StrOutputParser()
        self.max_steps, self.verify_steps, self.budget_chars, self.log = max_steps, verify_steps, budget_chars, log
        self.graph = self._build()

    def run(self, company_id: str) -> dict:
        task = PROMPTS.render('research_task', company_id=company_id, facts=starting_facts(company_id))
        start = {'messages': [HumanMessage(task)], 'company_id': company_id, 'facts': [], 'hypotheses': [],
                 'observations': {}, 'summary': None, 'steps': 0, 'compact_due': False}
        config = {'configurable': {'thread_id': company_id}, 'recursion_limit': 4 * self.max_steps + 10,
                  **trace_config(f'research_agent {company_id}')}
        trace = []
        for update in self.graph.stream(start, config, stream_mode='updates'):
            for node, delta in update.items():
                trace.append(self._describe(node, delta))
                self.log(trace[-1])
        state = self.graph.get_state(config).values
        return {'company': company_id, 'summary': state['summary'], 'facts': state['facts'],
                'hypotheses': state['hypotheses'], 'steps': state['steps'], 'observations': len(state['observations']),
                'trace': trace}

    # --- graph ---
    def _build(self):
        g = StateGraph(ResearchState)
        g.add_node('research', self._research)
        g.add_node('act', self._act)
        g.add_node('compact', self._compact)
        g.add_edge(START, 'research')
        g.add_edge('research', 'act')
        g.add_conditional_edges('act', self._after_act, ['research', 'compact', END])
        g.add_edge('compact', 'research')
        return g.compile(checkpointer=InMemorySaver())

    def _research(self, state: ResearchState) -> dict:
        reply = self.researcher.invoke([SystemMessage(PROMPTS.text('research_system')), *state['messages']])
        return {'messages': [reply], 'steps': state['steps'] + 1}

    def _act(self, state: ResearchState) -> dict:
        calls = state['messages'][-1].tool_calls
        if not calls:
            return {'messages': [HumanMessage(PROMPTS.text('nudge_use_tools'))]}
        memory, control = Memory.from_state(state), {}
        replies = [ToolMessage(self._execute(c['name'], c['args'], memory, state['company_id'], control), tool_call_id=c['id'])
                   for c in calls]
        return {'messages': replies, **memory.to_state(), **control}

    def _after_act(self, state: ResearchState) -> str:
        if state['summary'] is not None or state['steps'] >= self.max_steps:
            return END
        if state['compact_due'] or self._size(state['messages']) > self.budget_chars:
            return 'compact'
        return 'research'

    def _compact(self, state: ResearchState) -> dict:
        """Summarize the working transcript; the task message and memory survive verbatim."""
        work = state['messages'][1:]
        transcript = '\n'.join(str(m.content or json.dumps(getattr(m, 'tool_calls', []), ensure_ascii=False))[:1500]
                               for m in work if m.type in ('ai', 'tool'))
        summary = self.summarizer.invoke({'transcript': transcript[-60_000:]})
        note = PROMPTS.render('after_compaction', summary=summary, memory=Memory.from_state(state).render())
        return {'messages': [RemoveMessage(id=m.id) for m in work] + [HumanMessage(note)], 'compact_due': False}

    # --- tools ---
    def _execute(self, name: str, args: dict, memory: Memory, company_id: str, control: dict) -> str:
        if name in self.data_tools:
            # the first live runs repeated one failing call up to 30 times; answer a repeat without re-running it
            failed = next((o['text'] for o in memory.observations.values()
                           if o['tool'] == name and o['args'] == args and o['text'].startswith('tool error')), None)
            if failed:
                return PROMPTS.render('repeated_error', tool=name, error=failed[:300])
            return self._observe(memory, name, args, call_tool(self.data_tools[name], args))
        if name not in MEMORY_SCHEMAS:
            return f'unknown tool {name}'
        # a span named after the tool, so memory tools (finish included) show in the trace like the data tools
        return RunnableLambda(lambda a: self._memory_tool(name, a, memory, company_id, control), name=name).invoke(args)

    def _memory_tool(self, name: str, args: dict, memory: Memory, company_id: str, control: dict) -> str:
        try:
            a = MEMORY_SCHEMAS[name].model_validate(json_args(MEMORY_SCHEMAS[name], args))
        except ValidationError as e:
            return f'bad arguments for {name}: {e.errors()[:2]}'
        match a:
            case ProposeHypothesis():
                return memory.propose_hypothesis(a.text, a.evidence_for, a.evidence_against, a.test)
            case RecallObservation():
                return memory.recall(a.oid)
            case VerifyHypothesis():
                control['compact_due'] = True  # sub-task boundary: good moment to compact (SelfCompact-style)
                return self._verify(company_id, a.hid, memory)
            case Finish():
                if open_ids := memory.open_hypotheses():  # self-check: no finish with unverified hypotheses
                    return PROMPTS.render('reject_finish', open_ids=', '.join(open_ids))
                for f in a.new_facts:
                    memory.note_fact(f.text, f.evidence)
                control['summary'] = a.summary
                return f'ok: {len(memory.facts)} new facts'

    def _verify(self, company_id: str, hid: str, memory: Memory) -> str:
        """Chain-of-Verification sub-loop in a fresh context; its observations go to the same archive."""
        h = memory.hypothesis(hid)
        if h is None:
            return f'unknown hypothesis {hid}'
        loop = ToolLoop(self.model, list(self.data_tools.values()), Verdict, PROMPTS.text('verify_system'),
                        max_steps=self.verify_steps, observe=lambda n, a, t: self._observe(memory, n, a, t))
        try:
            verdict = loop.run(PROMPTS.render('verify_task', company_id=company_id, hid=hid, text=h['text'],
                                              evidence_for=h['for'], evidence_against=h['against'], test=h['test'],
                                              memory=memory.render()))
        except (ValidationError, OutputParserException):
            memory.set_status(hid, 'weak', 'model returned an invalid verification verdict')
            return f'{hid}: weak (invalid model verdict)'
        if verdict is None:
            memory.set_status(hid, 'weak', 'verification budget exhausted')
            return f'{hid}: weak (no verdict within {self.verify_steps} steps)'
        memory.set_status(hid, verdict.status, verdict.reason)
        return f'{hid}: {verdict.status} — {verdict.reason}'

    @staticmethod
    def _observe(memory: Memory, name: str, args: dict, text: str) -> str:
        oid = memory.observe(name, args, text)
        shown = text if len(text) <= OBS_CHARS else text[:OBS_CHARS] + f'\n...(truncated; recall_observation {oid})'
        return f'[{oid}] {shown}'

    @staticmethod
    def _size(messages: list[AnyMessage]) -> int:
        return sum(len(str(m.content)) + len(json.dumps(getattr(m, 'tool_calls', []), ensure_ascii=False)) for m in messages)

    @staticmethod
    def _describe(node: str, delta: dict) -> str:
        if node == 'research':
            names = ', '.join(c['name'] for c in delta['messages'][-1].tool_calls)
            return f'research: {names or "(text)"}'
        if node == 'act':
            return 'act: ' + ' | '.join(str(m.content)[:80] for m in delta['messages'])
        return node


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('company_id')
    ap.add_argument('--max-steps', type=int, default=40)
    a = ap.parse_args()
    usage = UsageMetadataCallbackHandler()
    res = ResearchAgent(ChatModels(callbacks=[usage]).agent(), max_steps=a.max_steps).run(a.company_id)
    out = DATA / 'companies' / 'agent' / f'{a.company_id}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'{out}: {len(res["facts"])} facts, {len(res["hypotheses"])} hypotheses; tokens {usage.usage_metadata}')
