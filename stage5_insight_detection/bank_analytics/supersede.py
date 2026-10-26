"""A newer bank report supersedes an older one on the same forecast.

The live demo of 24.10 showed one feed citing the July report («ставка 13,5% на конец 2026») next to the September one
(«Банк отказался от 13,5%, теперь 14%»): both are the bank's own view, the July one is only «aging», so search kept it.
Rule: for every comparable own-view forecast (indicator, period/horizon, measure, unit/currency), the newest report wins; fragments of
older reports with a different value for that key are superseded. Search ranks them last and marks them stale, so the
writer sees «УСТАРЕЛ» and does not present them as the bank's current position. Deterministic, no LLM.
"""
import re
from collections import defaultdict
from datetime import date
from math import isfinite

YEAR = re.compile(r'20\d\d')


def _period(figure):
    period=re.sub(r'\s+',' ',str(figure.get('period') or '').lower().replace('ё','е')).strip()
    if len(set(YEAR.findall(period)))!=1: return None
    text=f'{period} {figure.get("sentence", "")}'.lower().replace('ё','е')
    average=bool(re.search(r'среднегод|в среднем за год|годовая средн',text))
    end=bool(re.search(r'(?:конец|конц[аеу])\s+(?:20\d\d|года)',text))
    if average and end: return None  # two horizons in one sentence: do not guess which figure belongs to which
    cleaned=re.sub(r'\b(?:на|к|в|за|конец|конца|конце|концу|года|год|г|среднегодовая|среднегодовой)\b',' ',period)
    cleaned=re.sub(r'[\s.]','',cleaned).rstrip('пp')
    return cleaned, 'average' if average else 'end' if end else 'unspecified'


def _keys(row: dict):
    """Only checked, finite, comparable forecasts. Preserve ranges/ambiguous values instead of picking one."""
    out = {}
    if not row.get('own_view') or not row.get('date') or row.get('unverified'):
        return out
    try: date.fromisoformat(row['date'])
    except (TypeError,ValueError): return out
    for g in row.get('figures') or []:
        period=_period(g)
        if (g.get('kind')!='forecast' or not g.get('indicator') or not period or not g.get('unit') or
            g.get('verified') is False or g.get('status') in ('approx','unverified')):
            continue
        value=g.get('value')
        if not isinstance(value,(int,float)) or isinstance(value,bool) or not isfinite(value): continue
        currencies=tuple(code for code,pattern in [('USD',r'доллар|usd|\$'),('EUR',r'евро|eur|€'),('CNY',r'юан|cny|cnh')]
                         if re.search(pattern,f'{g.get("unit", "")} {g.get("sentence", "")}',re.I))
        key=(g['indicator'],period,g['unit'],g.get('measure') or 'unknown',currencies)
        out.setdefault(key,set()).add(round(float(value),6))
    return out


def superseded(rows: list[dict]) -> dict[str, str]:
    """fragment id -> 'report (date)' that supersedes it."""
    newest = {}  # key -> (date, report, {values of that report})
    for r in rows:
        for key, values in _keys(r).items():
            d = r['date']
            if key not in newest or d > newest[key][0]:
                newest[key] = (d, r.get('report', ''), set(values))
            elif d == newest[key][0]:
                newest[key][2].update(values)
    out = {}
    for r in rows:
        for key, old_values in _keys(r).items():
            d, report, values = newest[key]
            if r['date'] < d and len(values)==1 and old_values.isdisjoint(values):
                out[r['id']] = f'{report} ({d})'
                break
    return out


def mark_chunks(chunks: list[dict], rows: list[dict]) -> int:
    """Chunks holding a superseded fragment become stale with `superseded_by`; returns how many."""
    sup = superseded(rows)
    n = 0
    for row in rows:
        # Rows are part of this view; the calculator and whole-page tools must see the same mark as search.
        if row['id'] in sup: row['superseded_by']=sup[row['id']]
    for c in chunks:
        by = next((sup[f] for f in c.get('frag_ids', []) if f in sup), None)
        if by:
            c['stale'], c['freshness'], c['superseded_by'] = True, 'stale', by
            n += 1
    return n


def group_by_key(rows: list[dict]) -> dict:
    """Debug view: key -> [(date, report, value)] over own-view forecasts."""
    out = defaultdict(list)
    for r in rows:
        for key, values in _keys(r).items():
            out[key].extend((r['date'], r.get('report', ''), value) for value in values)
    return {k: sorted(v) for k, v in out.items()}
