"""Recall of the snowball expansion on the stage-3 golden set (11 companies, 33 hand-picked relevant news).
Compares seeds only (rounds=0) with the full expansion: does each round find relevant news the seeds miss, and at what cost
in volume. No LLM. python -X utf8 stage2_company_facts/expansion/eval_expansion.py [--rounds 2]
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stage2_company_facts.expansion.expand import expand
from stage2_company_facts.profile.profile import profiles

GOLD = Path(__file__).resolve().parents[2] / 'stage3_news_search' / 'golden_set' / 'real_pairs.json'


def recall(res: dict, gold: set[int], top: int | None = None) -> int:
    ids = [n['news_id'] for n in res['news']][:top]
    return len(gold & set(ids))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', type=int, default=2)
    a = ap.parse_args()
    pairs = {u: set(v) for u, v in json.loads(GOLD.read_text(encoding='utf-8'))['pairs'].items()}
    P = profiles(list(pairs))
    rows, tot = [], {'gold': 0, 'seed_all': 0, 'seed_100': 0, 'full_all': 0, 'full_100': 0, 'seed_n': 0, 'full_n': 0}
    for u, gold in pairs.items():
        if u not in P:
            print(u, 'not in tables'); continue
        s, f = expand(P[u], rounds=0), expand(P[u], rounds=a.rounds)
        r = {'gold': len(gold), 'seed_all': recall(s, gold), 'seed_100': recall(s, gold, 100), 'full_all': recall(f, gold),
             'full_100': recall(f, gold, 100), 'seed_n': len(s['news']), 'full_n': len(f['news'])}
        found_late = sorted({n['first_round'] for n in f['news'] if n['news_id'] in gold and n['first_round'] > 0})
        for k in tot:
            tot[k] += r[k]
        print(u, r, 'gold found in rounds >0:', found_late)
    g = tot['gold']
    print(f"\nrecall seeds: all {tot['seed_all']}/{g} ({tot['seed_all'] / g:.0%}), top-100 {tot['seed_100']}/{g}; "
          f"news per company {tot['seed_n'] / len(pairs):.0f}")
    print(f"recall full : all {tot['full_all']}/{g} ({tot['full_all'] / g:.0%}), top-100 {tot['full_100']}/{g}; "
          f"news per company {tot['full_n'] / len(pairs):.0f}")
