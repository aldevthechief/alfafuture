"""Chat models by job. All models go through the organizers' OpenAI-compatible LiteLLM proxy, so LangChain's ChatOpenAI
is enough. Every call is traced to Langfuse (if LANGFUSE_* keys are set); one script run = one Langfuse session."""
import json
import os

from langchain_core.language_models import BaseChatModel
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, convert_to_messages
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from common.config import SETTINGS, Settings
from common.llm.cache import LLM_CACHE
from common.llm.log import RUN_ID, CallLog


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


def timeout_for(max_tokens: int) -> float:
    """Seconds for one call: the full answer at >= 80 tokens/s + 60 s queueing; with one retry a hung request costs at most
    2 × this (64k -> 860 s); was a flat 900 s × 4 attempts."""
    return 60 + max_tokens / 80


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
        return ChatOpenAI(model=model, api_key=self.settings.llm_api_key, base_url=self.settings.base_url,
                          temperature=temperature, max_tokens=max_tokens, max_retries=1, timeout=timeout_for(max_tokens),
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

    def invoke(value, config: RunnableConfig):
        request = value
        for attempt in range(1, attempts + 1):
            try:
                answer = chain.invoke(request, config=config)
                result = answer['parsed'] if answer['parsed'] is not None else from_text(answer['raw'], schema)
                cut = answer['raw'].response_metadata.get('finish_reason') in ('length', 'max_tokens')
                if isinstance(result, schema) and not cut:
                    return result
            except (OutputParserException, ValidationError):
                pass
            if attempt < attempts:
                messages = value.to_messages() if isinstance(value, PromptValue) else convert_to_messages(
                    [HumanMessage(value)] if isinstance(value, str) else value)
                # A changed request also avoids replaying an invalid, non-empty answer from the LLM cache.
                request = [*messages, HumanMessage(
                    f'Попытка {attempt + 1}: предыдущий ответ не соответствует схеме {schema.__name__}. '
                    'Верни полный ответ через указанный инструмент, соблюдая типы и обязательные поля.')]
        raise StructuredOutputError(schema, attempts)

    return RunnableLambda(invoke)


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
