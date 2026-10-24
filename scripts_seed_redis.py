"""Seed the shared Redis (ALFA_CACHE_REDIS_URL) from the local JSONL caches, so a fresh container starts warm:
python -X utf8 scripts_seed_redis.py [--llm data/llm_cache/llm.jsonl] [name=path.jsonl ...]"""
import json, os, sys
from pathlib import Path

url = os.environ.get('ALFA_CACHE_REDIS_URL')
if not url:
    sys.exit('set ALFA_CACHE_REDIS_URL first')
import redis
client = redis.Redis.from_url(url, decode_responses=True)


def seed_hash(name: str, path: Path) -> None:
    n = 0
    import gzip
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as f:
        lines = f.read().splitlines()
    for line in lines:
        if not line.strip():
            continue
        d = json.loads(line)
        client.hset(f'alfa:{name}', d['key'], json.dumps(d['value'], ensure_ascii=False))
        n += 1
    print(f'{name}: {n} entries')


seed_hash('llm', Path(sys.argv[sys.argv.index('--llm') + 1] if '--llm' in sys.argv else 'data/llm_cache/llm.jsonl.gz'
       if Path('data/llm_cache/llm.jsonl.gz').exists() else 'data/llm_cache/llm.jsonl'))
for arg in sys.argv[1:]:
    if '=' in arg and not arg.startswith('--'):
        name, path = arg.split('=', 1)
        seed_hash(name, Path(path))
