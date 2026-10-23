"""Coverage of golden-set facts by the deterministic extractor (completeness, not form): a golden fact counts as covered
when every one of its keys (lowercase substring) occurs in the company's script facts. python -X utf8 stage2_company_facts/golden_set/coverage.py [facts.jsonl]"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.data import DATA

gold = json.loads(Path(__file__).with_name('company_facts_golden.json').read_text(encoding='utf-8'))['companies']
path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / 'companies' / 'golden_facts.jsonl'
got = {d['id']: [f['text'].lower() for f in d['facts']] for d in map(json.loads, open(path, encoding='utf-8'))}
hit = tot = 0
for u, g in gold.items():
    for f in g['facts']:
        blob = '\n'.join(got.get(u, []))
        ok = all(k in blob for k in f['keys'])
        hit += ok; tot += 1
        if not ok:
            print(f'MISS {u}: {f["text"]}  keys={f["keys"]}')
print(f'coverage {hit}/{tot} = {hit / tot:.0%}')
