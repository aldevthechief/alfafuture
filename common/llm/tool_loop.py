"""Reusable ReAct sub-loop on LangGraph: the model calls read-only tools until it calls the *final* tool, whose
arguments (validated by a pydantic schema) are the result. Used by hypothesis verification (stage 2) and the insight
judge (stage 5).

    think ──tool calls──> act ──> think ... ──final tool / step limit──> END
"""
import json
from collections.abc import Callable
from types import UnionType
from typing import Annotated, TypedDict, Union, get_args, get_origin

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from common.llm.log import log_tool
from common.llm.models import trace_config

Observe = Callable[[str, dict, str], str]  # (tool name, args, raw output) -> text shown to the model


def _accepts_str(annotation) -> bool:
    if annotation is str:
        return True
    return get_origin(annotation) in (Union, UnionType) and str in get_args(annotation)


def json_args(schema, args: dict) -> dict:
    """GLM via the proxy sends list/dict arguments as JSON strings ('[["PAYER", "==", "U_1"]]'); decode them where the
    schema field does not take a string. Without it the call fails validation the same way on every retry."""
    fields = getattr(schema, 'model_fields', {})
    out = dict(args)
    for k, v in args.items():
        if (isinstance(v, str) and v.lstrip().startswith(('[', '{')) and k in fields
                and not _accepts_str(fields[k].annotation)):
            try:
                out[k] = json.loads(v)
            except ValueError:
                pass  # leave it: validation reports the bad value to the model
    return out


def call_tool(tool: BaseTool, args: dict) -> str:
    """Run a tool; a bad call from the model becomes an error message for the model, never a crash of the loop."""
    try:
        out = str(tool.invoke(json_args(tool.args_schema, args)))
    except Exception as e:
        out = f'tool error: {type(e).__name__}: {e}'
    log_tool(tool.name, args, out)
    return out


class LoopState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    steps: int


class ToolLoop:
    def __init__(self, model: BaseChatModel, tools: list[BaseTool], final: type[BaseModel], system_prompt: str,
                 max_steps: int = 8, observe: Observe | None = None, nudge: str = 'Вызови инструменты или финальный инструмент.'):
        self.tools = {t.name: t for t in tools}
        self.final = final
        self.final_name = convert_to_openai_tool(final)['function']['name']
        self.model = model.bind_tools([*tools, final])
        self.system_prompt, self.max_steps, self.nudge = system_prompt, max_steps, nudge
        self.observe = observe or (lambda name, args, text: text)
        self.graph = self._build()

    def run(self, task: str) -> BaseModel | None:
        """Returns the validated final answer, or None if the step limit was hit first."""
        state = self.graph.invoke({'messages': [HumanMessage(task)], 'steps': 0},
                                  {'recursion_limit': 2 * self.max_steps + 5, **trace_config(f'tool_loop:{self.final_name}')})
        call = self._final_call(state['messages'][-1])
        if getattr(state['messages'][-1], 'response_metadata', {}).get('finish_reason') in ('length', 'max_tokens'):
            return None
        return self.final.model_validate(json_args(self.final, call['args'])) if call else None

    def _build(self):
        g = StateGraph(LoopState)
        g.add_node('think', self._think)
        g.add_node('act', self._act)
        g.add_edge(START, 'think')
        g.add_conditional_edges('think', self._route, ['act', END])
        g.add_edge('act', 'think')
        return g.compile()

    def _think(self, state: LoopState) -> dict:
        reply = self.model.invoke([SystemMessage(self.system_prompt), *state['messages']])
        return {'messages': [reply], 'steps': state['steps'] + 1}

    def _route(self, state: LoopState) -> str:
        done = self._final_call(state['messages'][-1]) or state['steps'] >= self.max_steps
        return END if done else 'act'

    def _act(self, state: LoopState) -> dict:
        calls = state['messages'][-1].tool_calls
        if not calls:
            return {'messages': [HumanMessage(self.nudge)]}
        return {'messages': [ToolMessage(self._execute(c), tool_call_id=c['id']) for c in calls]}

    def _execute(self, call: dict) -> str:
        tool = self.tools.get(call['name'])
        if tool is None:
            return f'unknown tool {call["name"]}; available: {sorted(self.tools)} and {self.final_name}'
        return self.observe(call['name'], call['args'], call_tool(tool, call['args']))

    def _final_call(self, message: AnyMessage) -> dict | None:
        if not isinstance(message, AIMessage):
            return None
        return next((c for c in message.tool_calls if c['name'] == self.final_name), None)
