"""Stage-result cache keyed by the *inputs* of a stage, not by the text of its prompt.

The LLM cache (FileLLMCache) keys an exact prompt, and the judge/writer prompts embed upstream outputs, so one changed upstream
answer or one re-ordered fact misses every stage after it. Here a stage result is stored under
sha256(stage version, what the stage reads): the news article and its event, the company facts, the role, the bank library, the
prompts and model of that stage. Unchanged inputs -> the stored result, no model call; one new event or one new bank report
recomputes only the pairs it touches. Backed by JsonlCache (file under data/news/llm_cache/, or Redis when ALFA_CACHE_REDIS_URL is set).
Only successful results are stored; a failed or rejected stage is retried on the next run."""
import hashlib
import json
import threading
from collections import Counter

from common.llm.cache import JsonlCache


def digest(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def prompts_version(library) -> str:
    """Hash of every prompt of a PromptLibrary: editing a prompt invalidates exactly the stages that use it."""
    return digest([(n, library.text(n)) for n in library.names()])


def model_id(model) -> str:
    # Only public generation settings: never serialize API keys or credentials into cache metadata.
    fields=('model_name','model','temperature','max_tokens','top_p','seed','frequency_penalty','presence_penalty',
            'reasoning_effort','model_kwargs','base_url','openai_api_base')
    return digest(type(model).__module__,type(model).__name__,{name:getattr(model,name,None) for name in fields})


def code_version(*paths) -> str:
    from pathlib import Path
    return digest([(str(path),hashlib.sha256(Path(path).read_bytes()).hexdigest()) for path in paths])


class StageCache:
    def __init__(self, stage: str, cache: JsonlCache | None = None):
        self.stage, self.cache, self.stats, self._lock = stage, cache or JsonlCache(f'stage_{stage}'), Counter(), threading.Lock()

    def get(self, key: str):
        value = self.cache.get(key)
        with self._lock:
            self.stats['hit' if value is not None else 'miss'] += 1
        return value

    def put(self, key: str, value) -> None:
        self.cache.put(key, json.loads(json.dumps(value, ensure_ascii=False, default=str)))  # plain JSON: models and numpy values
