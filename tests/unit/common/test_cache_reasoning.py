"""A cached answer given with more reasoning serves the same prompt asked with less; never the other way round."""
import tempfile
from pathlib import Path

from langchain_core.outputs import Generation

from common.llm.cache import FileLLMCache, stronger

BASE = '{"kwargs": {"model": "glm", "openai_api_key": {"type": "secret"}, %s"temperature": 0.0}}---[(\'stop\', None)]'
DEFAULT, LOW, HIGH = BASE % '', BASE % '"reasoning_effort": "low", ', BASE % '"reasoning_effort": "high", '


def test_stronger_lists_more_reasoning_closest_first_default_last():
    assert stronger(LOW) == [BASE % '"reasoning_effort": "medium", ', HIGH, DEFAULT]
    assert stronger(HIGH) == [DEFAULT] and stronger(DEFAULT) == []


def test_lookup_takes_an_answer_with_more_reasoning_only():
    with tempfile.TemporaryDirectory() as tmp:
        cache = FileLLMCache(Path(tmp))
        cache.update('q', DEFAULT, [Generation(text='default answer')])
        assert cache.lookup('q', LOW)[0].text == 'default answer'
        assert cache.lookup('q', HIGH)[0].text == 'default answer'
        assert cache.lookup('other prompt', LOW) is None
        cache.update('p', LOW, [Generation(text='low answer')])
        assert cache.lookup('p', HIGH) is None and cache.lookup('p', DEFAULT) is None
        cache.update('p', HIGH, [Generation(text='high answer')])
        assert cache.lookup('p', LOW)[0].text == 'low answer'  # its own entry first


def test_streamed_answer_is_cached_whole_and_replayed():
    import asyncio
    from langchain_core.messages import AIMessage, HumanMessage
    from common.llm.cache import astream_cached
    from common.llm.testing import ScriptedChatModel

    async def ask(model):
        return ''.join([c async for c in astream_cached(model, [HumanMessage('вопрос')])])

    with tempfile.TemporaryDirectory() as tmp:
        first = ScriptedChatModel(responses=[AIMessage(content='ответ по инсайту')], cache=FileLLMCache(Path(tmp)))
        assert asyncio.run(ask(first)) == 'ответ по инсайту'
        again = ScriptedChatModel(responses=[AIMessage(content='never')], cache=FileLLMCache(Path(tmp)))
        assert asyncio.run(ask(again)) == 'ответ по инсайту'  # from the cache, the script is not used


def test_tool_loop_rerun_comes_from_the_cache_at_every_step():
    # a cached answer carries usage_metadata {total_cost: 0}; the next step's prompt must still match the live run
    from pydantic import BaseModel
    from langchain_core.tools import tool
    from common.llm.testing import ScriptedChatModel, calls, tool_call
    from common.llm.tool_loop import ToolLoop

    @tool
    def lookup(q: str) -> str:
        """look something up"""
        return 'данные: ' + q

    class Done(BaseModel):
        answer: str

    with tempfile.TemporaryDirectory() as tmp:
        def run(script):
            model = ScriptedChatModel(responses=script, cache=FileLLMCache(Path(tmp)))
            return ToolLoop(model, [lookup], Done, 'system', max_steps=4).run('вопрос').answer
        assert run([calls(tool_call('lookup', q='x')), calls(tool_call('Done', answer='ok'))]) == 'ok'
        assert run([calls(tool_call('Done', answer='not cached'))]) == 'ok'
