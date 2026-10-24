"""Redis cache of validated discussion answers; SQLite remains the durable conversation store.

No file fallback: without Redis the discussion just runs normally. Keys contain only a context hash;
values expire and never contain an unvalidated/raw model response. Redis outages are cache misses.
"""
import hashlib
import json
import logging
import os

from pydantic import ValidationError

from feedback_api.schemas import AnalysisDetails

log = logging.getLogger(__name__)
PREFIX = 'alfa:feedback:answer:v1:'


def answer_key(context: dict) -> str:
    encoded = json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return PREFIX + hashlib.sha256(encoded.encode()).hexdigest()


class RedisAnswerCache:
    def __init__(self, client, ttl: int = 3600):
        self.client, self.ttl = client, ttl

    @classmethod
    def from_env(cls):
        url = os.environ.get('ALFA_CACHE_REDIS_URL')
        if not url:
            return None
        try:
            ttl = int(os.environ.get('FEEDBACK_LLM_CACHE_TTL', '3600'))
        except ValueError:
            log.warning('Invalid FEEDBACK_LLM_CACHE_TTL; discussion cache disabled')
            return None
        if ttl <= 0:
            return None
        import redis
        return cls(redis.Redis.from_url(url, decode_responses=True,
                                       socket_connect_timeout=0.3, socket_timeout=0.5), ttl)

    def lookup(self, key: str) -> AnalysisDetails | None:
        try:
            value = self.client.get(key)
            return AnalysisDetails.model_validate_json(value) if value is not None else None
        except (ValueError, ValidationError, TypeError):
            return None
        except Exception:  # cache availability must never prevent a new answer
            log.warning('Discussion cache read unavailable')
            return None

    def put(self, key: str, answer: AnalysisDetails):
        if self.ttl <= 0:
            return
        try:
            self.client.setex(key, self.ttl, answer.model_dump_json())
        except Exception:
            log.warning('Discussion cache write unavailable')
