"""My hand-picked top-5 per company (the target the LLM methods try to reproduce) and the evaluation.
Indices are positions in data/experiments/ranking/pool_<company>_<role>.json. The picks use the manual ratings in
insight-value-assessment.md (mean of Рел и Знач where a card is rated), then diversity of mechanisms (no two cards on one theme,
risk and opportunity both present, both news and the bank's view when it is strong), then urgency."""
import math

TODAY = '2026-10-26'
GOLD = {
    # rate 14->12 (5/5, core of funding cost) > state leasing on cheap money (4/4, competitor) > paused subsidised auto leasing
    # (4/4, demand + funding) > utilisation fee from 1 Jan 2027 (4/4, procurement cost, dated) > bond window (3.5, funding alternative)
    'U_14366_cfo': [16, 0, 2, 4, 7],
    # export channel through the south narrows (5/5) > domestic prices turned up (4.5, stock decision) > ports of the Azov-Black Sea
    # basin (5/5, other mechanism: bottleneck) > rate stays 14% (4, credit for purchases) > state-backed credit for grain and rail (3.5)
    'U_42222_owner': [0, 1, 8, 3, 2],
}
# themes: cards of one theme are alternatives of each other (used for diversity and for the lenient "same theme" match)
THEMES = {
    'U_14366_cfo': {'rate': [6, 11, 12, 16, 17], 'state_leasing': [0, 3, 9], 'auto_pause': [2, 8, 14], 'util_fee': [4, 10],
                    'bonds': [7, 13, 15, 20], 'tax': [5], 'mer_forecast': [1], 'fx': [18], 'credit_quality': [19]},
    'U_42222_owner': {'export': [0, 8], 'prices': [1, 4, 5], 'state_credit': [2, 6], 'rate': [3], 'tax': [7]},
}
# manual value 1-5 (mean of Рел и Знач from the doc; cards the doc does not rate are my estimate, marked in VALUE_GUESSED)
VALUE = {
    'U_14366_cfo': {0: 4, 1: 3, 2: 4, 3: 4, 4: 4, 5: 2, 6: 3.5, 7: 3.5, 8: 4, 9: 4, 10: 4, 11: 2, 12: 3.5, 13: 3.5, 14: 4, 15: 3.5,
                    16: 5, 17: 3, 18: 3, 19: 3.5, 20: 3.5},
    'U_42222_owner': {0: 5, 1: 4.5, 2: 3.5, 3: 4, 4: 3, 5: 3.5, 6: 2.25, 7: 2.5, 8: 5},
}
VALUE_GUESSED = {'U_14366_cfo': {6, 12, 13, 14, 17, 19, 20}, 'U_42222_owner': set()}


def theme_of(key, i):
    return next((t for t, v in THEMES[key].items() if i in v), f'x{i}')


def evaluate(key, top):
    """overlap@5, ndcg@5 against GOLD order, theme coverage, theme-level match, mean manual value of the picked cards."""
    gold, top = GOLD[key], list(top)[:5]
    gain = {c: 5 - r for r, c in enumerate(gold)}
    dcg = sum(gain.get(c, 0) / math.log2(r + 2) for r, c in enumerate(top))
    ideal = sum((5 - r) / math.log2(r + 2) for r in range(5))
    gold_themes = [theme_of(key, c) for c in gold]
    themes = [theme_of(key, c) for c in top]
    return dict(overlap=len(set(top) & set(gold)), ndcg=round(dcg / ideal, 3), distinct_themes=len(set(themes)),
                theme_match=len(set(themes) & set(gold_themes)), value=round(sum(VALUE[key].get(c, 2.5) for c in top) / max(len(top), 1), 2),
                top=top)
