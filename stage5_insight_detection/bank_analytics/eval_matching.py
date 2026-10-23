"""Quality of the bank analytics picked for an event and a company (task 2) on the labelled pairs.

data/analytics/eval/matching.jsonl: one line per pair {id, split dev|test|test2, news, company, note, must, ok, none}; fragments
are labelled by meaning: `must` (the insight is worse without it), `ok` (fits); anything else counts as bad. `none`: the
right answer is «аналитики Банка нет». Metrics over the top 5 the matcher returns:
- recall_must: share of the pair's must fragments in the top 5 (pairs with must only);
- bad: share of the top 5 that is neither must nor ok (pairs where something was returned);
- ndcg: nDCG@5 with gain 2 for must, 1 for ok;
- none_ok: the verdict «нет аналитики» (nothing returned) is right, over all pairs.
Pure code, no LLM. `--split test` only at the very end: nothing is tuned on it.
python -X utf8 stage5_insight_detection/bank_analytics/eval_matching.py [--split dev|test|test2|all] [-v] [--review --budget 4]
"""
import argparse, json, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.data import DATA

PAIRS = DATA / 'analytics' / 'eval' / 'matching.jsonl'
K = 5


def load_pairs(split: str = 'dev') -> list[dict]:
    rows = [json.loads(line) for line in PAIRS.read_text(encoding='utf-8').splitlines() if line.strip()]
    return [r for r in rows if split == 'all' or r['split'] == split]


def score_pair(pair: dict, got: list[str]) -> dict:
    must, ok = set(pair['must']), set(pair['ok'])
    top = got[:K]
    gain = lambda i: 2 if i in must else 1 if i in ok else 0
    dcg = sum(gain(i) / math.log2(n + 2) for n, i in enumerate(top))
    ideal = sorted([2] * len(must) + [1] * len(ok), reverse=True)[:K]
    idcg = sum(g / math.log2(n + 2) for n, g in enumerate(ideal))
    return {'recall_must': len(must & set(top)) / len(must) if must else None,
            'bad': sum(gain(i) == 0 for i in top) / len(top) if top else None,
            'ndcg': dcg / idcg if idcg else (1.0 if not top else 0.0),
            'none_ok': (not top) == bool(pair.get('none'))}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def evaluate(select, split: str = 'dev', verbose: bool = False) -> dict:
    """`select(event, profile)` -> fragment ids in rank order (empty = «аналитики Банка нет»)."""
    from common.search import news_pool
    from stage5_insight_detection.analyst_loop.event import EventCard
    from stage5_insight_detection.analyst_loop.profile import CompanyProfile
    news = news_pool().set_index('id')
    profiles, rows = {}, []
    for pair in load_pairs(split):
        row = dict(news.loc[pair['news']])
        row['id'] = pair['news']
        event = EventCard.from_news(row)
        profile = profiles.setdefault(pair['company'], CompanyProfile.from_tables(pair['company']))
        got = select(event, profile)
        res = score_pair(pair, got)
        rows.append(res)
        if verbose:
            mark = lambda i: 'MUST' if i in pair['must'] else 'ok  ' if i in pair['ok'] else 'bad '
            print(f'{pair["id"]} {pair["note"]}: recall {res["recall_must"]} bad {res["bad"]} ndcg {res["ndcg"]:.2f}'
                  f' none_ok {res["none_ok"]}')
            for i in got[:K]:
                print(f'   {mark(i)} {i}')
    return {'pairs': len(rows), **{k: mean(r[k] for r in rows) for k in ('recall_must', 'bad', 'ndcg')},
            'none_ok': mean(float(r['none_ok']) for r in rows)}


def current_select(reviewer=None, library=None):
    """The code's pick (BankMatcher.select), or with `reviewer` the bank analyst agent over the code's wider pick."""
    from stage5_insight_detection.bank_analytics.library import BankLibrary
    from stage5_insight_detection.bank_analytics.match import BankMatcher
    from stage5_insight_detection.bank_analytics.review import CANDIDATES
    matcher = BankMatcher(library or BankLibrary.load())
    if reviewer is None:
        return lambda event, profile: [m.fragment.id for m in matcher.select(event, profile)]
    return lambda event, profile: [m.fragment.id for m in reviewer.review(event, profile, matcher.select(event, profile, k=CANDIDATES))[0]]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='dev', choices=['dev', 'test', 'test2', 'all'])
    ap.add_argument('-v', action='store_true')
    ap.add_argument('--review', action='store_true', help='the bank analyst agent after the code (paid: model calls)')
    ap.add_argument('--budget', type=float, default=None, help='₽ for --review; stops before a call would pass it')
    a = ap.parse_args()
    reviewer = None
    if a.review:
        from stage5_insight_detection.bank_analytics.review import BankReviewer
        reviewer = BankReviewer(budget=a.budget)
    print(json.dumps(evaluate(current_select(reviewer), a.split, a.v), ensure_ascii=False))
    if reviewer is not None:
        print(f'spent {reviewer.meter.cost:.2f} ₽')
