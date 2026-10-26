"""Deterministic check: can every cabinet be shown from the caches alone (no LLM, no inference)?
Reads only the read endpoints of the backend. Usage: python scripts_check_demo_cache.py [http://localhost:18000] [data/api/compact] [U_1:ceo,U_2:cfo]
Per card: ready row, every claim cited, every cited id resolves to evidence (news url / bank report+page / fact),
screen version (compact) available, stale deadlines («до 23 октября» already in the past). Exit code 1 if any card fails."""
import json
import sys
import urllib.request
from pathlib import Path

API = (sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:18000').rstrip('/')
COMPACT_DIR = Path(sys.argv[2] if len(sys.argv) > 2 else 'data/api/compact')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backend.cache_check import check_row  # the same check the backend runs in the background


def get(path):
    with urllib.request.urlopen(API + path, timeout=120) as r:
        return json.load(r)


def check_card(card, compact_ids):
    row = (get(f"/v1/jobs/{card['job_id']}/insights/{card['result_index']}").get('result') or {}).get('results', [{}])[0]
    problems = [f'{"" if level == "fail" else "warn: "}{text}' for level, text in check_row(row)]
    if card['id'] not in compact_ids:
        problems.append('warn: no compact version (UI falls back to full text)')
    return problems


def main():
    bad = 0
    extra = [dict(company_id=x.split(':')[0], role=x.split(':')[1]) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else []  # companies with finished runs but no cabinet
    for a in get('/v1/cabinet/accounts')['accounts'] + extra:
        feed = get(f"/v1/cabinet/feed?company_id={a['company_id']}&role={a['role']}")['insights']
        # read the compact files, not the endpoint: in the full build GET /compact computes missing ones with an LLM
        compact = {}
        for job in {c['job_id'] for c in feed}:
            f = COMPACT_DIR / f'{job}.json'
            compact[job] = {k for k, v in json.loads(f.read_text(encoding='utf-8')).items() if v} if f.exists() else set()
        print(f"\n{a['company_id']} {a['role']}: {len(feed)} cards")
        for c in feed:
            p = check_card(c, compact.get(c['job_id'], set()))
            hard = [x for x in p if not x.startswith('warn:')]
            bad += bool(hard)
            print(f"  {'OK ' if not p else 'FAIL' if hard else 'WARN'} {c['id']} {c['title'][:70]}")
            for x in p:
                print('       -', x)
    print('\nhard failures:', bad)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
