"""Ranking metrics of the golden-set evaluation and the leakage diagnostic."""
from evals.retrieval_golden.leakage import leakage
from evals.retrieval_golden.metrics import MISSED, bootstrap, evaluate, ranks_of

LABELS = {'p1': 1, 'p2': 1, 'p3': 1, 'n1': 0, 'n2': 0}


def test_perfect_and_worst_rankings():
    perfect = ['p1', 'p2', 'p3'] + [f'x{i}' for i in range(10)] + ['n1', 'n2']
    m = evaluate(perfect, LABELS, ks=(3, 10))
    assert m['auc'] == 1.0 and m['recall@3'] == 1.0 and m['fpr@3'] == 0.0 and m['ndcg@3'] == 1.0 and m['mrr'] == 1.0 and m['map'] == 1.0
    assert m['recall@10'] == 1.0 and m['fpr@10'] == 0.0 and m['returned'] == 15
    worst = ['n1', 'n2'] + [f'x{i}' for i in range(10)] + ['p1', 'p2', 'p3']
    m = evaluate(worst, LABELS, ks=(3, 10))
    assert m['auc'] == 0.0 and m['recall@3'] == 0.0 and m['fpr@3'] == 1.0 and m['mrr'] == 1 / 3


def test_items_the_retriever_did_not_return_rank_last_and_tie():
    ranking = ['p1', 'x1', 'x2']  # p2, p3, n1, n2 are missing
    r = ranks_of(ranking, LABELS)
    assert r == {'p1': 1, 'p2': MISSED, 'p3': MISSED, 'n1': MISSED, 'n2': MISSED}
    m = evaluate(ranking, LABELS, ks=(3,))
    assert m['recall@3'] == 1 / 3 and m['fpr@3'] == 0.0
    assert abs(m['auc'] - (2 * 1.0 + 4 * 0.5) / 6) < 1e-9  # p1 beats both negatives; two missing positives tie with them


def test_unlabelled_news_do_not_hurt_the_order_metrics_but_push_recall_down():
    near = ['p1', 'p2', 'p3', 'n1', 'n2']
    padded = ['x1', 'x2', 'x3', 'p1', 'p2', 'p3', 'n1', 'n2']
    a, b = evaluate(near, LABELS, ks=(3,)), evaluate(padded, LABELS, ks=(3,))
    assert a['auc'] == b['auc'] == 1.0 and a['ndcg@3'] == b['ndcg@3'] == 1.0  # the order of the labelled items is the same
    assert a['recall@3'] == 1.0 and b['recall@3'] == 0.0  # but the unlabelled ones fill the top


def test_strata_are_reported_apart():
    strata = {'p1': 'direct', 'p2': 'indirect', 'p3': 'indirect'}
    m = evaluate(['p1', 'x', 'y', 'p2', 'p3'], LABELS, ks=(2,), strata=strata)
    assert m['direct@2'] == 1.0 and m['indirect@2'] == 0.0


def test_bootstrap_interval_is_narrow_for_a_perfect_ranking_and_wide_for_a_random_one():
    perfect = ranks_of(['p1', 'p2', 'p3', 'n1', 'n2'], LABELS)
    assert bootstrap(perfect, LABELS) == (1.0, 1.0)
    mixed = ranks_of(['p1', 'n1', 'p2', 'n2', 'p3'], LABELS)
    lo, hi = bootstrap(mixed, LABELS)
    assert lo < 0.6 < hi


def item(kind, text, label):
    return dict(kind=kind, label=label, title='t', text=text, names_industry=False)


def test_leakage_flags_a_set_that_a_bag_of_words_can_separate():
    separable = {f'c{c}': [item('pos', f'ставка кредит фондирование лизинг новость{c}{i}', 1) for i in range(8)]
                 + [item('easy_neg', f'музей выставка театр спорт новость{c}{i}', 0) for i in range(8)] for c in range(3)}
    assert leakage(separable)['all_negatives']['bag_of_words_auc'] > 0.9
    same = {f'c{c}': [item(k, 'ставка кредит музей спорт', y) for k, y in (('pos', 1), ('easy_neg', 0)) for _ in range(8)]
            for c in range(3)}
    assert abs(leakage(same)['all_negatives']['bag_of_words_auc'] - 0.5) < 0.2
