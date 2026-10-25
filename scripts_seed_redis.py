"""One cache for the backend: seed the shared Redis (ALFA_CACHE_REDIS_URL) from the local JSONL caches, and export it back.
python -X utf8 scripts_seed_redis.py            # seed: LLM answers (data/llm_cache/llm.jsonl[.gz]) + every named cache
                                                #       in data/news/llm_cache/*.jsonl; existing Redis entries are kept
python -X utf8 scripts_seed_redis.py --export   # export: Redis -> the same JSONL files (union), then rebuild llm.jsonl.gz,
                                                #         so CLI runs and the committed warm start see the backend's answers"""
import gzip, json, os, sys
from pathlib import Path

url = os.environ.get('ALFA_CACHE_REDIS_URL')
if not url:
    sys.exit('set ALFA_CACHE_REDIS_URL first')
import redis
client = redis.Redis.from_url(url, decode_responses=True)
LLM = Path('data/llm_cache/llm.jsonl')
NAMED = Path('data/news/llm_cache')


def read(path: Path) -> dict:
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as f:
        return {d['key']: d['value'] for d in map(json.loads, filter(str.strip, f))}


def redis_value(name: str, value):
    # the LLM cache holds langchain's `dumps` string as is (RedisLLMCache reads it with `loads`); named caches hold JSON
    return value if name == 'llm' else json.dumps(value, ensure_ascii=False)


def seed(name: str, path: Path) -> None:
    entries = read(path)
    added = sum(client.hsetnx(f'alfa:{name}', k, redis_value(name, v)) for k, v in entries.items())
    print(f'{name}: {len(entries)} in {path}, {added} new in Redis')


def export(name: str, path: Path) -> None:
    stored = client.hgetall(f'alfa:{name}')
    entries = read(path) if path.exists() else {}
    for k, v in stored.items():
        if k not in entries:
            entries[k] = v if name == 'llm' else json.loads(v)
            if name == 'llm' and isinstance(json.loads(v), str):  # double-encoded by the seed script before 24.10
                entries[k] = json.loads(v)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(json.dumps({'key': k, 'value': v}, ensure_ascii=False) + '\n' for k, v in entries.items())
    print(f'{name}: {len(entries)} in {path}')


if '--export' in sys.argv:
    export('llm', LLM)
    for key in client.scan_iter('alfa:*'):
        if key != 'alfa:llm' and client.type(key) == 'hash':  # feedback-api keeps per-answer string keys, not named caches
            export(key.split(':', 1)[1], NAMED / f"{key.split(':', 1)[1]}.jsonl")
    with open(LLM, 'rb') as src, gzip.open(LLM.with_suffix('.jsonl.gz'), 'wb') as dst:
        dst.write(src.read())
    print(f'{LLM}.gz rebuilt')
else:
    seed('llm', LLM if LLM.exists() else LLM.with_suffix('.jsonl.gz'))
    for path in sorted(NAMED.glob('*.jsonl')):
        seed(path.stem, path)
