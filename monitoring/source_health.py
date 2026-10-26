"""Collector breakage alert: a source publishes less than half of its own 5th percentile of daily articles.
Rule (per source, per evaluated day d): count(d) < FACTOR * P5(counts of the source's earlier days of the same kind),
kind = workday / weekend (a weekend is quiet for every outlet; a pooled P5 would be blind to a breakage on a workday).
- A source that is quiet by nature (P5 < MIN_P5: a zero day is normal for it) is watched by a second, `silent` rule: no
  article for longer than twice its longest earlier gap (and at least MIN_SILENT days).
- The same rule on the TOTAL of all sources: when the whole collector stops, ONE `collector_down` alert replaces a dozen.
History = the days before d inside WINDOW days (the corpus is ~31 days: nothing older exists); a source with fewer than
MIN_HISTORY days of its own history gets no verdict (a new source is not an alert).
Run on the corpus:  python -X utf8 -m monitoring.source_health [--as-of 2026-10-20] [--backtest]
Exit code 1 when an alert fires (cron / CI friendly). Evaluate after the day's close: a half-collected day trips the rule."""
import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FACTOR, QUANTILE, WINDOW, MIN_HISTORY, MIN_SILENT, MIN_P5 = 0.5, 5, 28, 14, 3, 3


@dataclass
class Alert:
    source: str        # '*' = the whole collector
    day: str
    kind: str          # low_volume | silent | collector_down
    count: int
    threshold: float   # low_volume: FACTOR * P5; silent: allowed days without an article
    p5: float | None
    history_days: int


def daily_counts(news: pd.DataFrame) -> pd.DataFrame:
    """days x sources table of article counts by publication date, zeros for the days without an article."""
    days = pd.to_datetime(news['date'], errors='coerce').dt.date
    table = news.assign(day=days).dropna(subset=['day']).groupby(['day', 'source']).size().unstack(fill_value=0)
    return table.reindex([d.date() for d in pd.date_range(min(table.index), max(table.index))], fill_value=0)


def _history(series: pd.Series, day: date) -> pd.Series:
    """The earlier days of the source inside the window, from its first article on (before it the source did not exist)."""
    first = series[series > 0].index.min()
    if pd.isna(first):
        return series.iloc[:0]
    return series[(series.index < day) & (series.index >= max(first, day - timedelta(days=WINDOW)))]


def _p5(history: pd.Series, day: date) -> float:
    same = history[[(d.weekday() >= 5) == (day.weekday() >= 5) for d in history.index]]
    return float(np.percentile(same if len(same) >= 6 else history, QUANTILE))


def _silence(series: pd.Series, history: pd.Series, day: date) -> tuple[int, int]:
    """(days without an article up to and including `day`, the allowed number)."""
    past = list(history)
    while past and past[-1] == 0:
        past.pop()  # the ongoing silence is not a precedent for itself
    gaps, run = [], 0
    for v in past:
        if v == 0:
            run += 1
        else:
            gaps.append(run)
            run = 0
    gaps.append(run)
    silent = 0
    for v in reversed(list(series[series.index <= day])):
        if v:
            break
        silent += 1
    return silent, max(MIN_SILENT, 2 * max(gaps))


def check(table: pd.DataFrame, day: date) -> list[Alert]:
    total = table.sum(axis=1)
    history = _history(total, day)
    if len(history) >= MIN_HISTORY and (p5 := _p5(history, day)) >= MIN_P5 and total[day] < FACTOR * p5:
        return [Alert('*', day.isoformat(), 'collector_down', int(total[day]), FACTOR * p5, p5, len(history))]
    alerts = []
    for source in table.columns:
        history = _history(table[source], day)
        if len(history) < MIN_HISTORY:
            continue
        p5, count = _p5(history, day), int(table.at[day, source])
        if p5 >= MIN_P5:
            if count < FACTOR * p5:
                alerts.append(Alert(source, day.isoformat(), 'low_volume', count, FACTOR * p5, p5, len(history)))
        elif (gap := _silence(table[source], history, day))[0] > gap[1]:
            alerts.append(Alert(source, day.isoformat(), 'silent', count, gap[1], p5, len(history)))
    return alerts


def backtest(table: pd.DataFrame) -> pd.DataFrame:
    """The rule over every day: which alerts the corpus itself would have produced."""
    rows = [asdict(a) for d in table.index for a in check(table, d)]
    return pd.DataFrame(rows, columns=list(Alert.__dataclass_fields__))


def episodes(alerts: pd.DataFrame) -> pd.DataFrame:
    """Consecutive daily alerts of one source and kind = one episode (what a person should be paged about once)."""
    rows = []
    for (source, kind), g in alerts.assign(d=pd.to_datetime(alerts['day'])).sort_values('d').groupby(['source', 'kind']):
        start = g['d'].iloc[0]
        for prev, cur in zip(g['d'], g['d'].iloc[1:].tolist() + [None]):
            if cur is None or (cur - prev).days > 1:
                rows.append(dict(source=source, kind=kind, first=start.date().isoformat(), last=prev.date().isoformat(),
                                 days=(prev - start).days + 1))
                start = cur
    return pd.DataFrame(rows, columns=['source', 'kind', 'first', 'last', 'days'])


def main():
    from common.data import DATA
    ap = argparse.ArgumentParser()
    ap.add_argument('--news', type=Path, default=DATA / 'news' / 'news.parquet')
    ap.add_argument('--as-of', type=date.fromisoformat, default=None, help='the evaluated day (default: the last day of the corpus)')
    ap.add_argument('--backtest', action='store_true')
    a = ap.parse_args()
    table = daily_counts(pd.read_parquet(a.news, columns=['source', 'date']))
    if a.backtest:
        bt = backtest(table)
        ep = episodes(bt) if len(bt) else bt
        print(f'{len(table.columns)} sources, {len(table)} days {table.index.min()}..{table.index.max()}: '
              f'{len(bt)} daily alerts = {len(ep)} episodes')
        if len(ep):
            print(ep.to_string(index=False))
        return
    day = a.as_of or table.index.max()
    alerts = check(table, day)
    print(json.dumps(dict(day=day.isoformat(), sources=len(table.columns), alerts=[asdict(x) for x in alerts]), ensure_ascii=False, indent=1))
    sys.exit(1 if alerts else 0)


if __name__ == '__main__':
    main()
