"""Collector breakage alert: below half of the source's own P5 (workday / weekend apart), silent quiet sources, one alert for a
dead collector, no verdict without history."""
from datetime import date, timedelta

import pandas as pd

from monitoring.source_health import check, daily_counts

DAYS = [date(2026, 9, 1) + timedelta(days=i) for i in range(30)]  # a Tuesday .. Wednesday
BUSY = [20, 22, 19, 21, 23, 20, 8, 7] * 4  # workdays ~20, weekends ~8 (the pattern repeats by 8 days: only the level matters)


def table(**columns) -> pd.DataFrame:
    return pd.DataFrame({k: v[:len(DAYS)] for k, v in columns.items()}, index=DAYS)


def steady(level, n=30):
    return [level + (i % 3) for i in range(n)]  # 20, 21, 22, 20 ...


def test_a_source_below_half_of_its_p5_alerts_and_a_normal_day_does_not():
    ok = table(a=steady(20), b=steady(30))
    assert check(ok, DAYS[-1]) == []
    broken = ok.copy()
    broken.loc[DAYS[-1], 'a'] = 5  # P5 ~ 20 -> threshold 10
    alerts = check(broken, DAYS[-1])
    assert [(a.source, a.kind) for a in alerts] == [('a', 'low_volume')] and alerts[0].threshold > 9
    broken.loc[DAYS[-1], 'a'] = 15  # a bad but not a broken day
    assert check(broken, DAYS[-1]) == []


def test_weekend_is_judged_against_weekends():
    counts = [8 if d.weekday() >= 5 else 40 for d in DAYS]
    t = table(a=counts, b=steady(30))
    saturday = next(d for d in reversed(DAYS) if d.weekday() == 5)
    assert check(t, saturday) == []  # 8 is a normal Saturday, though it would be far below the pooled P5


def test_a_quiet_source_is_caught_by_the_silence_rule_only():
    quiet = [1 if i % 5 == 0 else 0 for i in range(30)]  # an article every 5th day: gaps of 4
    t = table(q=quiet + [], b=steady(30))
    t.loc[DAYS[-1], 'q'] = 0
    assert check(t, DAYS[-1]) == []  # one more quiet day is normal for it
    for d in DAYS[-12:]:
        t.loc[d, 'q'] = 0
    assert [(a.source, a.kind) for a in check(t, DAYS[-1])] == [('q', 'silent')]


def test_a_dead_collector_is_one_alert():
    t = table(a=steady(20), b=steady(30), c=steady(40))
    t.loc[DAYS[-1]] = [0, 0, 0]
    alerts = check(t, DAYS[-1])
    assert [(a.source, a.kind) for a in alerts] == [('*', 'collector_down')]


def test_no_verdict_without_history():
    t = table(a=[0] * 25 + [30, 29, 31, 30, 30])  # the source appeared 5 days ago
    t.loc[DAYS[-1], 'a'] = 0
    assert check(t, DAYS[-1]) == []


def test_daily_counts_fills_missing_days_with_zero():
    news = pd.DataFrame({'source': ['a', 'a', 'b'], 'date': ['2026-10-01', '2026-10-03', '2026-10-01']})
    t = daily_counts(news)
    assert list(t.index) == [date(2026, 10, 1), date(2026, 10, 2), date(2026, 10, 3)]
    assert t.loc[date(2026, 10, 2)].sum() == 0 and t.loc[date(2026, 10, 1), 'a'] == 1


def test_consecutive_daily_alerts_are_one_episode():
    from monitoring.source_health import episodes
    alerts = pd.DataFrame({'source': ['a'] * 3 + ['a', 'b'], 'kind': ['silent'] * 5,
                           'day': ['2026-10-01', '2026-10-02', '2026-10-03', '2026-10-07', '2026-10-02']})
    ep = episodes(alerts)
    assert [(r.source, r.first, r.last, r.days) for r in ep.itertuples()] == [
        ('a', '2026-10-01', '2026-10-03', 3), ('a', '2026-10-07', '2026-10-07', 1), ('b', '2026-10-02', '2026-10-02', 1)]
