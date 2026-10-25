"""Chat models by job. All models go through the organizers' OpenAI-compatible LiteLLM proxy, so LangChain's ChatOpenAI
is enough. Every call is traced to Langfuse (if LANGFUSE_* keys are set); one script run = one Langfuse session."""
import json
import os
import time

from langchain_core.language_models import BaseChatModel
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, convert_to_messages
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
import httpx
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from common.config import SETTINGS, Settings
from common.llm.cache import LLM_CACHE
from common.llm.log import RUN_ID, CallLog, log_tool


def langfuse_callbacks() -> list:
    if not os.environ.get('LANGFUSE_PUBLIC_KEY'):
        return []
    from langfuse import propagate_attributes
    from langfuse.langchain import CallbackHandler  # reads LANGFUSE_PUBLIC_KEY / SECRET_KEY / BASE_URL; flushes at exit
    # entered for the whole process, never exited: every trace of this run lands in one Langfuse session.
    # Kept in a global: a garbage-collected context manager closes its generator and detaches the context.
    global _RUN_CONTEXT
    _RUN_CONTEXT = propagate_attributes(session_id=RUN_ID, tags=[RUN_ID.rsplit('-', 2)[0]])
    _RUN_CONTEXT.__enter__()
    return [CallbackHandler()]


LANGFUSE = langfuse_callbacks()
CALL_LOG = CallLog()  # local data/logs/calls.jsonl, see common/llm/log.py


def trace_config(name: str) -> dict:
    """Config for a multi-step run (agent graph, tool loop): all its model and tool calls nest under one Langfuse trace
    `name`. Without it every model call is a separate root trace "ChatOpenAI". Nested inside a traced run, it only names
    the span: passing callbacks there would cut the link to the parent."""
    from langchain_core.runnables.config import var_child_runnable_config
    if var_child_runnable_config.get():
        return {'run_name': name}
    return {'run_name': name, 'callbacks': list(LANGFUSE), 'metadata': {'langfuse_session_id': RUN_ID}}


# Streamed calls send a chunk every <= 2 s, reasoning included (23.10), so a stream silent for 2 minutes is dead: that is
# the only client limit. The total duration is bounded by max_tokens and by the proxy, which closes any request at ~15 min.
TIMEOUT = httpx.Timeout(connect=30.0, read=120.0, write=60.0, pool=60.0)
import openai  # noqa: E402
PROXY_DROPS = (openai.APIConnectionError, openai.APITimeoutError, httpx.TransportError)
PROXY_CUTOFF = 600  # s: a drop this late is the proxy's ~15 min limit, not a network blip


class ProxyChatOpenAI(ChatOpenAI):
    """Keeps the proxy's billed cost (`usage.cost` of the last streamed chunk), which LangChain's usage parsing drops, and
    puts every call into the run's Langfuse session even from a thread that did not inherit the run's context."""

    def generate(self, *args, **kwargs):
        started = time.monotonic()
        try:
            if not LANGFUSE:
                return super().generate(*args, **kwargs)
            from langfuse import propagate_attributes
            with propagate_attributes(session_id=RUN_ID, tags=[RUN_ID.rsplit('-', 2)[0]]):
                return super().generate(*args, **kwargs)
        except PROXY_DROPS as error:
            if fallback := self._fallback(error, started):
                return fallback.generate(*args, **kwargs)  # the same messages: the dialog goes on as it was
            raise

    async def agenerate(self, *args, **kwargs):
        started = time.monotonic()
        try:
            if not LANGFUSE:
                return await super().agenerate(*args, **kwargs)
            from langfuse import propagate_attributes
            with propagate_attributes(session_id=RUN_ID, tags=[RUN_ID.rsplit('-', 2)[0]]):
                return await super().agenerate(*args, **kwargs)
        except PROXY_DROPS as error:
            if fallback := self._fallback(error, started):
                return await fallback.agenerate(*args, **kwargs)
            raise

    def _fallback(self, error: Exception, started: float) -> 'ProxyChatOpenAI | None':
        """The proxy closes any request at ~15 min (902 s, 24.10: 4 of 6 writers with 50k+ reasoning tokens at ~57 tok/s).
        A call dropped that late is retried once with reasoning_effort='high' (less reasoning, the same messages)."""
        if self.reasoning_effort is not None or time.monotonic() - started < PROXY_CUTOFF:
            return None
        log_tool('llm_fallback_high', {'model': self.model_name, 'after_s': round(time.monotonic() - started)},
                 f'{type(error).__name__}: {error}')
        return self.model_copy(update={'reasoning_effort': 'high'})

    def _convert_chunk_to_generation_chunk(self, chunk, default_chunk_class, base_generation_info):
        gen = super()._convert_chunk_to_generation_chunk(chunk, default_chunk_class, base_generation_info)
        cost = (chunk.get('usage') or {}).get('cost')
        if gen is not None and cost is not None:
            gen.message.response_metadata['cost'] = cost
        return gen


class ChatModels:
    """Factory: `agent()` for multi-step reasoning, `cheap()` for bulk per-news calls, `vision()` for OCR."""

    def __init__(self, settings: Settings = SETTINGS, callbacks: list | None = None):
        self.settings, self.callbacks = settings, callbacks  # e.g. [UsageMetadataCallbackHandler()] to count tokens

    # Team default max_tokens = 64000 for every model: reasoning counts toward it, and a cut answer has no text and no tool
    # call yet is billed (insight writer at 16000: 16000/16000 reasoning, empty, 3 of 5 insights lost on U_68622; it needs
    # ~15.7k reasoning + 2.6k answer). A cap, not a cost. Reasoning stays at the model's default (no reasoning_effort).
    def agent(self, temperature: float = 0.2, max_tokens: int = 64000) -> ChatOpenAI:
        return self._make(self.settings.agent_model, temperature, max_tokens)

    def cheap(self, temperature: float = 0.0, max_tokens: int = 64000) -> ChatOpenAI:
        return self._make(self.settings.cheap_model, temperature, max_tokens)

    def vision(self, max_tokens: int = 64000) -> ChatOpenAI:
        return self._make(self.settings.vision_model, 0.0, max_tokens)

    def _make(self, model: str, temperature: float, max_tokens: int) -> ChatOpenAI:
        if not self.settings.llm_api_key:
            raise RuntimeError('set LLM_API_KEY (team key, common/llm/.env)')
        return ProxyChatOpenAI(model=model, api_key=self.settings.llm_api_key, base_url=self.settings.base_url,
                          temperature=temperature, max_tokens=max_tokens, max_retries=1, timeout=TIMEOUT,
                          # the proxy drops a request that sends nothing for 300 s (long GLM reasoning); a streamed answer
                          # keeps the connection alive (writer calls of 386 and 525 s passed); usage comes in the last chunk
                          streaming=True, stream_usage=True,
                          callbacks=[*(self.callbacks or []), *LANGFUSE, CALL_LOG], cache=LLM_CACHE,
                          metadata={'langfuse_session_id': RUN_ID})  # session survives worker threads (context vars do not)


class StructuredOutputError(ValueError):
    """A model did not produce a validated result after bounded attempts (no raw data in the error)."""

    def __init__(self, schema: type[BaseModel], attempts: int):
        self.schema, self.attempts = schema.__name__, attempts
        super().__init__(f'{self.schema}: no valid structured response after {attempts} attempts')


def structured(model: BaseChatModel, schema: type[BaseModel], *, attempts: int = 2) -> Runnable:
    """Return a validated model or raise StructuredOutputError; never pass None to the next stage."""
    if attempts < 1:
        raise ValueError('Structured output attempts must be positive')
    chain = model.with_structured_output(schema, method='function_calling', include_raw=True)

    def accept(answer):
        result = answer['parsed'] if answer['parsed'] is not None else from_text(answer['raw'], schema)
        cut = answer['raw'].response_metadata.get('finish_reason') in ('length', 'max_tokens')
        return result if isinstance(result, schema) and not cut else None

    def retry_request(value, attempt: int):
        messages = value.to_messages() if isinstance(value, PromptValue) else convert_to_messages(
            [HumanMessage(value)] if isinstance(value, str) else value)
        # A changed request also avoids replaying an invalid, non-empty answer from the LLM cache.
        return [*messages, HumanMessage(
            f'Попытка {attempt + 1}: предыдущий ответ не соответствует схеме {schema.__name__}. '
            'Верни полный ответ через указанный инструмент, соблюдая типы и обязательные поля.')]

    def invoke(value, config: RunnableConfig):
        request = value
        for attempt in range(1, attempts + 1):
            try:
                if (result := accept(chain.invoke(request, config=config))) is not None:
                    return result
            except (OutputParserException, ValidationError):
                pass
            if attempt < attempts:
                request = retry_request(value, attempt)
        raise StructuredOutputError(schema, attempts)

    async def ainvoke(value, config: RunnableConfig):  # the same, awaiting the model (no thread per call)
        request = value
        for attempt in range(1, attempts + 1):
            try:
                if (result := accept(await chain.ainvoke(request, config=config))) is not None:
                    return result
            except (OutputParserException, ValidationError):
                pass
            if attempt < attempts:
                request = retry_request(value, attempt)
        raise StructuredOutputError(schema, attempts)

    return RunnableLambda(invoke, afunc=ainvoke)


def from_text(message, schema: type[BaseModel]) -> BaseModel | None:
    """Repair what some models behind the proxy send: tool arguments with a nested object as a JSON string (glm), or no
    tool call and the JSON printed in the text, often in ```json``` (gemma). None as before if nothing validates."""
    candidates = [unquote_json(c['args']) for c in getattr(message, 'tool_calls', None) or []]
    text = message.content if isinstance(message.content, str) else str(message.content)
    start, end = text.find('{'), text.rfind('}')
    if 0 <= start < end:
        try:
            candidates.append(unquote_json(json.loads(text[start:end + 1])))
        except ValueError:
            pass
    for args in candidates:
        try:
            return schema.model_validate(args)
        except ValidationError:
            continue
    return None


def unquote_json(value):
    if isinstance(value, str) and value.strip()[:1] in ('{', '['):
        try:
            value = json.loads(value)
        except ValueError:
            return value
    if isinstance(value, dict):
        return {k: unquote_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [unquote_json(v) for v in value]
    return value
