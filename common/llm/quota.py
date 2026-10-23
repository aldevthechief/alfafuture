"""Today's spend of the team key and model prices.
python -X utf8 common/llm/quota.py

Spend comes from the proxy's own `/key/info` (LiteLLM): `spend` in ₽ since the last budget reset (00:00 UTC = 03:00 MSK),
`max_budget` = 500 ₽/day. The proxy routes to routerai.ru and bills its real per-call `usage.cost` (₽), so list prices
below are only a rough guide: measured cost was 1.6–5x the routerai list price and 100–600x the OpenRouter USD price."""
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import SETTINGS


def key_info() -> dict:
    r = httpx.get(SETTINGS.base_url.removesuffix('/v1') + '/key/info',
                  headers={'Authorization': f'Bearer {SETTINGS.llm_api_key}'}, timeout=60)
    r.raise_for_status()
    return r.json()['info']


def prices(models: list[str]) -> list[tuple]:
    """(model, OpenRouter $/1M in, out, routerai ₽/1M in, out); None where a source does not list the model."""
    def table(url):
        return {m['id']: m['pricing'] for m in httpx.get(url, timeout=60).json()['data']}
    orr, rr = table('https://openrouter.ai/api/v1/models'), table('https://routerai.ru/api/v1/models')
    per_m = lambda p, k: float(p[k]) * 1e6 if p and k in p else None
    return [(m, per_m(orr.get(m), 'prompt'), per_m(orr.get(m), 'completion'),
             per_m(rr.get(m), 'prompt'), per_m(rr.get(m), 'completion')) for m in models]


if __name__ == '__main__':
    info = key_info()
    spend, budget = info['spend'], info['max_budget']
    print(f"key {info['key_alias']}: spent {spend:.3f} ₽ of {budget:.0f} ₽ ({spend / budget:.2%}), "
          f"left {budget - spend:.2f} ₽, resets {info['budget_reset_at']} (00:00 UTC)")
    print(f"{'model':34} {'OR $/1M in':>11} {'out':>7} {'routerai ₽/1M in':>17} {'out':>7}")
    fmt = lambda x: f'{x:.2f}' if x is not None else '-'
    for m, *p in prices([m for m in info['models'] if '/' in m]):
        print(f'{m:34} {fmt(p[0]):>11} {fmt(p[1]):>7} {fmt(p[2]):>17} {fmt(p[3]):>7}')
