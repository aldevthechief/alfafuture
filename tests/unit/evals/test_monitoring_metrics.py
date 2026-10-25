"""Feed metrics and leak tripwires on small hand-made jobs."""
from monitoring.feed_metrics import feed_metrics, feed_overlap, jaccard
from monitoring.leaks import client_hits, peer_hits


def insight(title='T', facts=(), why='', bank=False, **extra):
    return dict(title=title, what_happened=[dict(text=t, sources=['news:1'] if src else [], evidence=[dict(quote='q')] if ev else [])
                                            for t, src, ev in facts],
                why_it_matters=[dict(text=why)] if why else [], bank_view=[dict(text='b')] if bank else [], actions=[], **extra)


def row(news_id, ins, source=None, date='2026-10-10', outlet='rbc', status='ready'):
    return dict(status=status, news_id=news_id, source=source, insight=ins if status == 'ready' else None,
                news_references={f'news:{news_id}': dict(date=date, source=outlet)} if news_id is not None else {})


def job(rows, company='U_1', role='cfo'):
    return dict(status='succeeded', request=dict(company_id=company, role=role), created_at='2026-10-20T10:00:00+00:00',
                finished_at='2026-10-20T10:30:00+00:00', result=dict(results=rows))


def test_leak_tripwires_separate_peer_data_from_the_clients_own_figures():
    assert peer_hits('Компания — 1-е место среди 1082 компаний группировки. Это важно.') == ['Компания — 1-е место среди 1082 компаний группировки.']
    assert peer_hits('Исходящие платежи 27% выручки против медианы 7% по отрасли.')
    assert not peer_hits('Ставка 14% до конца 2026 года, рост выручки в 2027 году.')
    assert client_hits('68% исходящих платежей уходит одному контрагенту.') and not peer_hits('68% исходящих платежей уходит одному контрагенту.')
    assert not client_hits('Проверить контрагентов до конца октября.')  # no figure next to the client's data


def test_feed_metrics_age_concentration_duplicates_and_reliability():
    rows = [row(1, insight(facts=[('a', True, True), ('b', True, False)], bank=True)),
            row(2, insight(why='Компания — 1-е место среди 5 компаний.'), date='2026-09-01', outlet='rbc'),
            row(3, insight(), outlet='kommersant', date='2026-10-19'),
            row(4, None, status='model_error'),
            row(None, insight(), source='bank_analytics')]
    rows[2]['news_references']['news:1'] = dict(date='2026-10-10', source='rbc')  # insight 3 also cites news 1: a duplicate pair with 1
    m = feed_metrics(job(rows))
    assert m['insights'] == 4 and m['news_insights'] == 3
    assert m['age_median_days'] == 10 and abs(m['share_older'] - 1 / 3) < 1e-9  # 10, 49, 1 days; only 49 is older than 14
    assert m['top_outlet_share'] == 0.75 and m['outlets'] == 2  # rbc 3 refs of 4
    assert abs(m['duplicate_pair_share'] - 1 / 3) < 1e-9  # pairs (1,2), (1,3), (2,3): only 1-3 share news:1
    assert m['claims_with_source'] == 1.0 and m['claims_with_quote'] == 0.5
    assert m['peer_leak_share'] == 0.25 and m['with_bank_position'] == 0.25 and m['from_analytics'] == 0.25
    assert abs(m['model_error_share'] - 0.25) < 1e-9 and m['wall_minutes'] == 30


def test_feeds_of_different_companies_and_roles_overlap_little():
    a = job([row(1, insight()), row(2, insight())], 'U_1', 'cfo')
    b = job([row(2, insight()), row(3, insight())], 'U_1', 'ceo')
    c = job([row(9, insight())], 'U_2', 'cfo')
    out = feed_overlap({('U_1', 'cfo'): a, ('U_1', 'ceo'): b, ('U_2', 'cfo'): c})
    assert abs(out['same_company_other_role'] - 1 / 3) < 1e-9 and out['other_company'] == 0.0 and out['pairs_other'] == 2
    assert jaccard(set(), set()) == 0.0
