"""Offline check of FileLLMCache: a repeated request is answered from disk (tool calls included), a new one goes to the model.
python -X utf8 common/llm/test_cache.py"""
import sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from langchain_core.messages import AIMessage
from common.llm.cache import FileLLMCache
from common.llm.testing import ScriptedChatModel, calls, tool_call

with tempfile.TemporaryDirectory() as tmp:
    script = [calls(tool_call('company_card', u='U_1')), AIMessage(content='second'), AIMessage(content='third')]
    model = ScriptedChatModel(responses=script, cache=FileLLMCache(Path(tmp)))
    a = model.invoke('q1')
    assert model.invoke('q1').tool_calls == a.tool_calls == script[0].tool_calls  # cached, script not advanced
    assert model.invoke('q2').content == 'second'
    fresh = ScriptedChatModel(responses=[AIMessage(content='never')], cache=FileLLMCache(Path(tmp)))  # new process: read from file
    assert fresh.invoke('q2').content == 'second' and fresh.invoke('q1').tool_calls[0]['name'] == 'company_card'
    empty = ScriptedChatModel(responses=[AIMessage(content=''), AIMessage(content='retry')], cache=FileLLMCache(Path(tmp)))
    assert empty.invoke('q3').content == '' and empty.invoke('q3').content == 'retry'  # empty answer not cached
print('ok: repeated requests served from the JSONL cache, empty answers not cached')
