"""Agreement of the LLM judge with the human labeller (insight_judge_calibration) and the human ranking.
- kappa per axis (Cohen, A/B/=) of the pairwise judge against the human; the same on the pairs the human was sure about
- position bias of the raw judge answers (share of A in both orders) and how often both orders agreed
- Bradley–Terry rating of every card from the human winners; the pointwise 1–5 judge against it (Spearman) and as a
  pairwise predictor (score A > score B), plus a release-threshold table.
python -X utf8 -m evals.insight_judge.calibrate --pairs <dir with U_*.md> --sheet <marking_sheet.csv> --run <run.json>"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evals.insight_judge.pairs import AXES, load_human, load_pairs

LABELS = ('A', 'B', '=')


def kappa(human: list[str], judge: list[str]) -> float | None:
    """Cohen's kappa; None when chance agreement is 1 (a single label on both sides: kappa is undefined)."""
    n = len(human)
    if not n:
        return None
    po = sum(h == j for h, j in zip(human, judge)) / n
    ch, cj = Counter(human), Counter(judge)
    pe = sum(ch[label] * cj[label] for label in LABELS) / n ** 2
    return None if pe == 1 else (po - pe) / (1 - pe)


def agreement(human: dict, judged: dict) -> dict:
    """Per axis and overall: n, agreement, kappa, tie shares; `sure` = pairs the human labelled with confidence 3."""
    ids = [i for i in human if i in judged]
    report = {}
    for key in (*AXES, 'winner'):
        h, j = [human[i][key] for i in ids], [judged[i][key] for i in ids]
        sure = [k for k, i in enumerate(ids) if human[i]['confidence'] == 3]
        report[key] = dict(n=len(ids), agree=sum(a == b for a, b in zip(h, j)) / len(ids) if ids else None, kappa=kappa(h, j),
                           kappa_sure=kappa([h[k] for k in sure], [j[k] for k in sure]), n_sure=len(sure),
                           human_ties=h.count('=') / len(ids) if ids else None, judge_ties=j.count('=') / len(ids) if ids else None)
    flags = [(human[i]['factual_error'], judged[i]['factual_error']) for i in ids]
    report['factual_error'] = dict(human=sum(h for h, _ in flags), judge=sum(j for _, j in flags), both=sum(h and j for h, j in flags))
    return report


def position_bias(judged: dict) -> dict:
    """Raw answers of both orders: share of 'A' per axis over all calls (0.5 = no preference for a position) and the share
    of pairs where both orders gave the same verdict."""
    out = {}
    for key in (*AXES, 'winner'):
        raw = [r[key] for v in judged.values() for r in v['raw'] if r[key] != '=']
        out[key] = dict(share_A=raw.count('A') / len(raw) if raw else None,
                        both_orders_agree=float(np.mean([v['agree'][key] for v in judged.values()])))
    return out


def bradley_terry(games: list[tuple[str, str, float]], reg: float = 0.5, iters: int = 500) -> dict[str, float]:
    """games: (card i, card j, score of i: 1 win / 0.5 tie / 0). `reg` pseudo-games against a virtual average card keep
    the rating finite for a card that never lost (the human pairs are sparse: 14 per company). Returns log-strengths."""
    items = sorted({c for g in games for c in g[:2]})
    strength = dict.fromkeys(items, 1.0)
    for _ in range(iters):
        new = {}
        for i in items:
            wins = reg + sum(s for a, b, s in games if a == i) + sum(1 - s for a, b, s in games if b == i)
            den = 2 * reg / (strength[i] + 1) + sum(1 / (strength[a] + strength[b]) for a, b, _ in games if i in (a, b))
            new[i] = wins / den
        scale = np.exp(np.mean(np.log(list(new.values()))))
        strength = {i: v / scale for i, v in new.items()}
    return {i: float(np.log(v)) for i, v in strength.items()}


def human_ratings(pairs, human) -> dict[str, float]:
    score = {'A': 1.0, 'B': 0.0, '=': 0.5}
    return bradley_terry([(p.a, p.b, score[human[p.id]['winner']]) for p in pairs if p.id in human])


def derive_pairs(pairs, scores: dict[str, dict]) -> dict[str, dict]:
    """Pairwise verdicts implied by the pointwise scores of the two cards (higher axis score wins, equal = tie)."""
    mean = lambda card: float(np.mean([scores[card][k] for k in AXES]))
    cmp = lambda x, y: 'A' if x > y else 'B' if y > x else '='
    out = {}
    for p in pairs:
        if p.a in scores and p.b in scores:
            out[p.id] = {k: cmp(scores[p.a][k], scores[p.b][k]) for k in AXES}
            out[p.id].update(winner=cmp(mean(p.a), mean(p.b)), factual_error=False)
    return out


def hybrid(judged: dict, derived: dict) -> dict:
    """The pairwise verdict where both orders agreed, the pointwise one where the pairwise judge fell back to «=»."""
    return {i: {k: (v[k] if v[k] != '=' else derived.get(i, {}).get(k, '=')) for k in (*AXES, 'winner')}
               | {'factual_error': v['factual_error']} for i, v in judged.items()}


def pointwise_check(pairs, human, scores: dict[str, dict]) -> dict:
    """scores: card text -> {axis: 1..5} of the pointwise judge. The cards that were never labelled are ignored."""
    rating = human_ratings(pairs, human)
    mean = lambda card: float(np.mean([scores[card][k] for k in AXES]))
    cards = [c for c in rating if c in scores]
    rho = spearmanr([rating[c] for c in cards], [mean(c) for c in cards]) if len(cards) > 2 else None
    derived = {i: v for i, v in derive_pairs(pairs, scores).items() if i in human}
    median = float(np.median([rating[c] for c in cards])) if cards else 0.0
    table = []
    for t in (2.0, 2.5, 3.0, 3.5, 4.0, 4.5):
        passed = [c for c in cards if mean(c) >= t]
        table.append(dict(threshold=t, passed=len(passed) / max(1, len(cards)),
                          precision=sum(rating[c] > median for c in passed) / max(1, len(passed))))
    return dict(cards=len(cards), spearman=None if rho is None else float(rho.statistic),
                as_pairwise=agreement({i: human[i] for i in derived}, derived) if derived else None, release_thresholds=table)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', required=True, type=Path)
    ap.add_argument('--sheet', type=Path, default=None, help='marking_sheet.csv; without it only the human-free checks are printed')
    ap.add_argument('--run', required=True, type=Path, help='output of evals.insight_judge.run')
    a = ap.parse_args()
    pairs, run = load_pairs(a.pairs), json.loads(a.run.read_text(encoding='utf-8'))
    human = load_human(a.sheet) if a.sheet else {}
    out = dict(judged_pairs=len(run['pairs']), position_bias=position_bias(run['pairs']),
               judge_ties={k: sum(v[k] == '=' for v in run['pairs'].values()) / len(run['pairs']) for k in (*AXES, 'winner')},
               factual_error_flags=sum(v['factual_error'] for v in run['pairs'].values()))
    if run.get('cards'):
        out['card_score_mean'] = {k: float(np.mean([c[k] for c in run['cards'].values()])) for k in AXES}
    if human:
        out |= dict(labelled_pairs=len(human), pairwise=agreement(human, run['pairs']))
        if run.get('cards'):
            out['pointwise'] = pointwise_check(pairs, human, run['cards'])
            mixed = hybrid(run['pairs'], derive_pairs(pairs, run['cards']))
            out['hybrid'] = agreement(human, mixed)  # pick the variant with the best kappa per axis
    else:
        out['note'] = 'no filled human rows: agreement / kappa / Bradley–Terry are not computed yet'
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
