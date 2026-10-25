"""Answer caches: disk JSONL by default, one Redis for everything when ALFA_CACHE_REDIS_URL is set
(the docker compose run shares the LLM cache and the named caches across containers; the Redis volume
outlives restarts). `JsonlCache`: append-only key -> JSON value, reused by every company. `FileLLMCache`:
LangChain cache for every model call (prompt + model + params), plugged into all `ChatModels` models."""
import asyncio
import hashlib
import json
import os
import threading

_CACHE_LOCK = threading.RLock()
_KEY_LOCKS = {}
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from langchain_core.caches import BaseCache
from langchain_core.load import dumps, loads
from langchain_core.outputs import Generation

from common.data import DATA


def _redis(prefix: str):
    """A shared Redis hash for `prefix` when ALFA_CACHE_REDIS_URL is set, else None."""
    url = os.environ.get('ALFA_CACHE_REDIS_URL')
    if not url:
        return None
    import redis  # only inside docker compose / an explicit opt-in
    client = redis.Redis.from_url(url, decode_responses=True)
    return client, f'alfa:{prefix}'


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
            lock = _KEY_LOCKS.setdefault((str(self.path.resolve()), key), threading.Lock())
        with lock:
            with _CACHE_LOCK:
                self._values = None
                values = self._load()
                if key in values:
                    return values[key]
            value = compute()
            self.put(key, value)
            return value

    async def aget_or_compute(self, key: str, compute):
        """Share sync/async cache misses across runs without a thread waiting on the model."""
        with _CACHE_LOCK:
            lock = _KEY_LOCKS.setdefault((str(self.path.resolve()), key), threading.Lock())
        while not lock.acquire(blocking=False):
            await asyncio.sleep(0.01)
        try:
            def lookup():
                with _CACHE_LOCK:
                    self._values = None
                    values = self._load()
                    return key in values, values.get(key)
            found, value = await asyncio.to_thread(lookup)
            if found:
                return value
            value = await compute()
            await asyncio.to_thread(self.put, key, value)
            return value
        finally:
            lock.release()

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


def stable_prompt(prompt: str) -> str:
    """Ignore replay-only metadata of earlier AI turns, keeping their content and tool calls intact."""
    try:
        messages = json.loads(prompt)
    except ValueError:
        return prompt
    changed = False
    for message in messages if isinstance(messages, list) else []:
        if not isinstance(message, dict):
            continue
        identifier = message.get('id')
        if not isinstance(identifier, list) or not identifier or not str(identifier[-1]).startswith('AIMessage'):
            continue
        kwargs = message.get('kwargs', {})
        for key in ('usage_metadata', 'response_metadata', 'id'):
            if key in kwargs:
                del kwargs[key]
                changed = True
    return json.dumps(messages) if changed else prompt


def lookup_prompt(get, prompt: str, llm_string: str):
    stable = stable_prompt(prompt)
    # Existing entries still work for their original prompt; no destructive cache migration.
    for candidate in dict.fromkeys((stable, prompt)):
        value = get(FileLLMCache._key(candidate, llm_string))
        gens = loads(value, allowed_objects='all') if value is not None else None
        if gens and FileLLMCache.complete(gens):
            return gens
    return None


class FileLLMCache(BaseCache):
    """Same request -> same answer from disk, no API call: reruns of agents and chains during development are free.
    shortcut: one local JSONL, no cross-process locking; swap for langchain's RedisCache when runs go parallel."""

    def __init__(self, folder: Path = DATA / 'llm_cache'):
        self.store = JsonlCache('llm', folder)

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
        return lookup_prompt(self.store.get, prompt, llm_string)

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

    class RedisLLMCache(BaseCache):
        """Same request -> same answer from Redis, no API call; the volume keeps them across restarts."""

        def lookup(self, prompt: str, llm_string: str) -> Sequence[Generation] | None:
            return lookup_prompt(lambda field: client.hget(key, field), prompt, llm_string)

        def update(self, prompt: str, llm_string: str, return_val: Sequence[Generation]) -> None:
            if FileLLMCache.complete(return_val):
                client.hset(key, self_key(stable_prompt(prompt), llm_string), dumps(list(return_val)))

        def clear(self, **kwargs: Any) -> None:
            client.delete(key)

    self_key = staticmethod(FileLLMCache._key)
    return RedisLLMCache()


LLM_CACHE = _redis_llm_cache() or FileLLMCache()
