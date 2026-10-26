"""Offline check: prompts, job counts and the heuristic-only baseline (no LLM)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import run  # noqa: E402

pools = {k: json.loads((run.DATA / f'pool_{k}.json').read_text(encoding='utf-8')) for k in run.KEYS}
print(run.profile(run.KEYS[0]), '\n')
for k, cards in pools.items():
    print(k, 'L3 prompt chars', len(run.listwise_prompt(k, 'L3', cards)), 'L2', len(run.listwise_prompt(k, 'L2', cards)))
    pre = sorted(range(len(cards)), key=lambda i: -run.prescore(cards[i]))
    div = run.diversify(cards, [run.prescore(c) for c in cards])
    print(' prescore order', pre[:8], run.evaluate(k, pre[:5]))
    print(' prescore+diversity', run.evaluate(k, div))
for ph in 'AB':
    jobs = run.build_jobs(ph, pools)
    print('phase', ph, len(jobs), 'calls')
