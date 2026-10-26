"""Answer caches: disk JSONL by default, one Redis for everything when ALFA_CACHE_REDIS_URL is set
(the docker compose run shares the LLM cache and the named caches across containers; the Redis volume
outlives restarts). `JsonlCache`: append-only key -> JSON value, reused by every company. `FileLLMCache`:
LangChain cache for every model call (prompt + model + params), plugged into all `ChatModels` models."""
import hashlib
import json
import os
import re
import threading

_CACHE_LOCK = threading.RLock()
_KEY_LOCKS = {}
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from langchain_core.caches import BaseCache
from langchain_core.load import dumps, loads
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, Generation

from common.data import DATA


def _redis(prefix: str):
    """A shared Redis hash for `prefix` when ALFA_CACHE_REDIS_URL is set, else None."""
    url = os.environ.get('ALFA_CACHE_REDIS_URL')
    if not url:
        return None
    import redis  # only inside docker compose / an explicit opt-in
    client = redis.Redis.from_url(url, decode_responses=True)
    return client, f'alfa:{prefix}'


def unpack(path: Path) -> Path:
    """The repo ships the LLM cache as <name>.jsonl.gz: a fresh clone unpacks it once, next to it (atomic rename)."""
    packed = path.with_name(path.name + '.gz')
    if not path.exists() and packed.exists():
        import gzip, shutil
        tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
        try:
            with gzip.open(packed, 'rb') as src, open(tmp, 'wb') as dst:
                shutil.copyfileobj(src, dst)
            os.replace(tmp, path)
        except OSError:
            tmp.unlink(missing_ok=True)
    return path


def seed_if_empty(client, key: str, path: Path) -> None:
    """An empty Redis (new volume) is warmed from the shipped JSONL; a filled one is left as is."""
    try:
        if client.exists(key) or not unpack(path).exists():
            return
        with open(path, encoding='utf-8') as f:
            entries = [json.loads(line) for line in f if line.strip()]
        pipe = client.pipeline(transaction=False)
        for d in entries:
            pipe.hsetnx(key, d['key'], d['value'])
        pipe.execute()
    except Exception:
        pass  # warm start only: a cold cache still works, just calls the model


class JsonlCache:
    def __init__(self, name: str, folder: Path = DATA / 'news' / 'llm_cache'):
        self.name = name
        self.path = Path(folder) / f'{name}.jsonl'
        self._values: dict[str, Any] | None = None

    def get(self, key: str) -> Any:
        return self._load().get(key)

    def put(self, key: str, value: Any) -> None:
        shared = _redis(self.name)
        if shared:
            client, hkey = shared
            client.hset(hkey, key, json.dumps(value, ensure_ascii=False))
            with _CACHE_LOCK:  # keep the in-process dict warm for the readers
                self._load()[key] = value
            return
        with _CACHE_LOCK:
            self._load()[key] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'key': key, 'value': value}, ensure_ascii=False) + '\n')

    def get_or_compute(self, key: str, compute: Callable[[], Any]) -> Any:
        with _CACHE_LOCK:
            lock = _KEY_LOCKS.setdefault((str(self.path.resolve()), key), threading.RLock())
        with lock:
            with _CACHE_LOCK:
                self._values = None
                values = self._load()
                if key in values:
                    return values[key]
            value = compute()
            self.put(key, value)
            return value

    def _load(self) -> dict[str, Any]:
        if self._values is not None:
            return self._values
        shared = _redis(self.name)
        if shared:
            client, key = shared
            with _CACHE_LOCK:
                self._values = {k: json.loads(v) for k, v in client.hgetall(key).items()}
            return self._values
        lines = self.path.read_text(encoding='utf-8').splitlines() if self.path.exists() else []
        self._values = {d['key']: d['value'] for d in map(json.loads, filter(None, lines))}
        return self._values


EFFORTS = ['minimal', 'low', 'medium', 'high']  # absent = the model's default; for GLM it reasons the most (~30k tokens)
_EFFORT = re.compile(r'"reasoning_effort": "(\w+)", ')


def stronger(llm_string: str) -> list[str]:
    """The same model and params with more reasoning, closest first, the model's default last: a cached answer to the
    same prompt given with more reasoning serves a request with less (24.10: Q&A moved from default to low)."""
    m = _EFFORT.search(llm_string)
    if not m or m.group(1) not in EFFORTS:
        return []
    more = EFFORTS[EFFORTS.index(m.group(1)) + 1:]
    return [llm_string.replace(m.group(0), f'"reasoning_effort": "{e}", ') for e in more] + [llm_string.replace(m.group(0), '')]


_VOLATILE = ('usage_metadata', 'response_metadata', 'id')  # differ between a live answer and its cached replay


def stable_prompt(prompt: str) -> str:
    """The prompt without per-call fields of earlier model answers. LangChain marks a cached answer with
    usage_metadata {total_cost: 0}: the next step of a tool loop then never matched the live run, and only step 1 of
    every agent came from the cache (24.10). A prompt without such fields is returned unchanged (old keys stay valid)."""
    try:
        messages = json.loads(prompt)
    except ValueError:
        return prompt
    changed = False
    for m in messages if isinstance(messages, list) else []:
        if isinstance(m, dict) and str(m.get('id', [''])[-1]).startswith('AIMessage'):
            for k in _VOLATILE:
                changed |= m.get('kwargs', {}).pop(k, None) is not None
    return json.dumps(messages) if changed else prompt


def lookup_any(get: Callable[[str], Any], prompt: str, llm_string: str) -> Sequence[Generation] | None:
    """First complete cached answer for this request, else for the same prompt with more reasoning."""
    prompt = stable_prompt(prompt)
    for candidate in (llm_string, *stronger(llm_string)):
        v = get(FileLLMCache._key(prompt, candidate))
        gens = loads(v, allowed_objects='all') if v is not None else None  # our own store, trusted
        if isinstance(gens, str):  # seeded by scripts_seed_redis.py before 24.10: the dumps string was JSON-encoded once more
            gens = loads(gens, allowed_objects='all')
        if gens and FileLLMCache.complete(gens):  # also skips cut answers cached before this check
            return gens
    return None


async def astream_cached(model, messages: list[BaseMessage]):
    """Text chunks of the model's streamed answer, through the model's cache: LangChain's `astream` neither reads nor
    writes it (24.10: no Q&A answer had ever been cached). A hit — this request or the same prompt with more reasoning —
    comes as one chunk; a fresh answer is stored whole once the stream ends."""
    import asyncio
    cache = model.cache if isinstance(getattr(model, 'cache', None), BaseCache) else None
    prompt, llm_string = dumps(messages), model._get_llm_string()
    if cache is not None and (hit := await asyncio.to_thread(cache.lookup, prompt, llm_string)):
        yield hit[0].text
        return
    parts = []
    async for chunk in model.astream(messages):
        if chunk.text:
            parts.append(chunk.text)
            yield chunk.text
    if cache is not None and parts:
        await asyncio.to_thread(cache.update, prompt, llm_string, [ChatGeneration(message=AIMessage(content=''.join(parts)))])


class FileLLMCache(BaseCache):
    """Same request -> same answer from disk, no API call: reruns of agents and chains during development are free.
    shortcut: one local JSONL, no cross-process locking; swap for langchain's RedisCache when runs go parallel."""

    def __init__(self, folder: Path = DATA / 'llm_cache'):
        self.store = JsonlCache('llm', folder)
        unpack(self.store.path)

    @staticmethod
    def _key(prompt: str, llm_string: str) -> str:
        return hashlib.sha1(f'{llm_string}\n{prompt}'.encode()).hexdigest()

    @staticmethod
    def complete(gens: Sequence[Generation]) -> bool:
        """False for an answer cut at max_tokens or empty (no text, no tool call): caching it would make every retry fail."""
        def ok(g):
            msg = getattr(g, 'message', None)
            cut = (g.generation_info or {}).get('finish_reason') in ('length', 'max_tokens') or (
                msg is not None and msg.response_metadata.get('finish_reason') in ('length', 'max_tokens'))
            return not cut and bool(g.text or (msg is not None and (msg.content or getattr(msg, 'tool_calls', None))))
        return bool(gens) and all(map(ok, gens))

    def lookup(self, prompt: str, llm_string: str) -> Sequence[Generation] | None:
        return lookup_any(self.store.get, prompt, llm_string)

    def update(self, prompt: str, llm_string: str, return_val: Sequence[Generation]) -> None:
        if self.complete(return_val):
            self.store.put(self._key(stable_prompt(prompt), llm_string), dumps(list(return_val)))

    def clear(self, **kwargs: Any) -> None:
        self.store.path.unlink(missing_ok=True)
        self.store._values = None


def _redis_llm_cache() -> BaseCache | None:
    shared = _redis('llm')
    if not shared:
        return None
    client, key = shared
    seed_if_empty(client, key, DATA / 'llm_cache' / 'llm.jsonl')

    class RedisLLMCache(BaseCache):
        """Same request -> same answer from Redis, no API call; the volume keeps them across restarts."""

        def lookup(self, prompt: str, llm_string: str) -> Sequence[Generation] | None:
            return lookup_any(lambda k: client.hget(key, k), prompt, llm_string)

        def update(self, prompt: str, llm_string: str, return_val: Sequence[Generation]) -> None:
            if FileLLMCache.complete(return_val):
                client.hset(key, self_key(stable_prompt(prompt), llm_string), dumps(list(return_val)))

        def clear(self, **kwargs: Any) -> None:
            client.delete(key)

    self_key = staticmethod(FileLLMCache._key)
    return RedisLLMCache()


LLM_CACHE = _redis_llm_cache() or FileLLMCache()
