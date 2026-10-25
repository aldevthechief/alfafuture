import pandas as pd

from stage3_news_search.retrieval_v2.candidates import quota_pick

SHARES = {'factory': .7, 'standard': .3}


def pool(rows):
    """rows: [[(source, query, rank), ...] per event], already in pool (priority) order."""
    return pd.DataFrame({'event': range(len(rows)),
                         'generators': [[{'source': s, 'query': q, 'rank': r} for s, q, r in gs] for gs in rows]})


def test_every_generator_gets_seats_not_only_the_priority_head():
    gens = [{'source': 'factory', 'query': 'A'}, {'source': 'factory', 'query': 'B'}, {'source': 'standard', 'query': 'C', 'weight': 1.0}]
    # A dominates the head of the pool; B and C only find events further down
    p = pool([[('factory', 'A', 0)], [('factory', 'A', 1)], [('factory', 'A', 2)], [('factory', 'A', 3)],
              [('factory', 'B', 0)], [('standard', 'C', 0)], [('factory', 'B', 1)]])
    picked = quota_pick(p, gens, 4, SHARES)
    assert len(picked) == 4 and {4, 5} <= set(picked.event)  # B's and C's best events are in, not A's 3rd and 4th
    assert list(picked.event) == sorted(picked.event)          # pool order kept


def test_shared_event_counts_once_and_unfilled_seats_go_to_pool_order():
    gens = [{'source': 'factory', 'query': 'A'}, {'source': 'factory', 'query': 'B'}]
    p = pool([[('factory', 'A', 0), ('factory', 'B', 0)], [('factory', 'A', 1)], [('factory', 'A', 2)]])
    picked = quota_pick(p, gens, 2, SHARES)
    assert list(picked.event) == [0, 1]  # event 0 is one candidate; B has nothing else, its seat goes to the pool order
