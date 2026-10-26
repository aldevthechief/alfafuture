"""Deterministic check of one ready result row: can the card be shown from the saved data alone?
Pure functions, no I/O, no models. Used by the background watcher (backend/cabinet.py) and scripts_check_demo_cache.py.
'fail' problems hide the card (a claim without a resolvable source); 'warn' ones are only reported."""
import datetime as dt
import re

BLOCKS = ('what_happened', 'why_it_matters', 'risks', 'opportunities', 'hypotheses', 'bank_view')
MONTHS = {m: n for n, m in enumerate('января февраля марта апреля мая июня июля августа сентября октября ноября декабря'.split(), 1)}
DEADLINE = re.compile(r'до (\d{1,2}) (%s)' % '|'.join(MONTHS))


def known_ids(row):
    """Every id the UI can turn into a source line, with a quality tag."""
    ev, ids = row.get('evidence') or {}, {}
    for f in (ev.get('facts') or []) + (ev.get('calculations') or []):
        ids[f['id']] = 'fact'
    for n in ev.get('news') or []:
        ids[f"news:{n['id']}"] = 'news' if n.get('url') else 'news-no-url'
    for b in (ev.get('bank') or []) + (row.get('judge_bank') or []) + (row.get('calculation_bank') or []) + (row.get('bank') or []):
        ids[b.get('id') or f"bank:{b.get('report')}:{b.get('page')}"] = 'bank' if b.get('report') and b.get('page') else 'bank-no-page'
    for k, n in (row.get('news_references') or {}).items():
        ids[k] = 'news' if n.get('url') else 'news-no-url'
    return ids


def check_row(row, today=None):
    """-> [('fail' | 'warn', message)]; empty list = fully showable."""
    today = today or dt.date.today()
    ins = row.get('insight')
    if row.get('status') != 'ready' or not ins:
        return [('fail', 'row is not ready')]
    ids, problems, stale = known_ids(row), [], set()
    for block in BLOCKS:
        for c in ins.get(block) or []:
            srcs = c.get('sources') or []
            if not srcs and block != 'hypotheses':
                problems.append(('fail', f'{block}: claim without sources'))
            for s in srcs:
                kind = ids.get(s) or next((('bank' if e['source'].startswith('bank:') else 'news')
                                           for e in c.get('evidence') or [] if e['source'] == s), None)
                if kind is None and s.startswith('fact:'):
                    kind = 'fact'  # the UI shows facts as «данные компании» from the live profile
                if kind is None and s.startswith('bank:') and any(e['source'].startswith('bank:') for e in c.get('evidence') or []):
                    kind = 'bank'  # the claim cites a page id, its evidence names the fragment: the UI maps one to the other
                if kind is None and not s.startswith(('user:', 'lookup:')):
                    problems.append(('fail', f'{block}: {s} not in evidence'))
                elif kind in ('news-no-url', 'bank-no-page'):
                    problems.append(('fail', f'{block}: {s} is {kind}'))
            for m in DEADLINE.finditer(c.get('text', '')):
                try:
                    if dt.date(today.year, MONTHS[m.group(2)], int(m.group(1))) < today:
                        stale.add(m.group(0))
                except ValueError:
                    pass
    if not ins.get('actions'):
        problems.append(('warn', 'no actions'))
    if stale:
        problems.append(('warn', 'past deadlines: ' + ', '.join(sorted(stale))))
    return problems
