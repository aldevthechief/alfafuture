"""Feed metrics that need no LLM judge, so they carry no self-preference of a model grading its own family:
computed by code from the saved analysis jobs (`data/api/analyses/*.json`, or any list of job dicts).
Per (company, role) feed of the latest finished job:
- freshness: age of the news behind each news insight at the end of the job (days): median, share older than 14 days
- source concentration: share of the single most used outlet among the news the feed cites; distinct outlets
- residual duplicates: insight pairs that cite a common news item (what the dedup gate was supposed to remove)
- grounding: share of fact claims with a source and with a verbatim evidence quote
- confidential data: share of insights that show other companies' data (peer ranks / medians: a leak) and the client's own
  payment / revenue figures (tracked; monitoring.leaks)
- bank part: share of insights that carry the bank's position; share of analytics-agent insights
- reliability: share of news rows that ended in a model error; wall time of the job
Across feeds: how different the feeds are (Jaccard of cited news): same company other role / other company (the case wants
a feed that changes with the client, A3).
python -X utf8 -m monitoring.feed_metrics [--jobs <dir>]"""
import argparse
import itertools
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from monitoring.leaks import client_hits, peer_hits

TEXT_BLOCKS = ('why_it_matters', 'risks', 'opportunities', 'hypotheses', 'bank_view')
OLD_DAYS = 14


def ready(job: dict) -> list[dict]:
    return [r for r in (job.get('result') or {}).get('results', []) if r.get('status') == 'ready' and r.get('insight')]


def news_ids(row: dict) -> set[str]:
    refs = {k.split(':', 1)[1] for k in (row.get('news_references') or {}) if k.startswith('news:')}
    return refs or ({str(row['news_id'])} if row.get('news_id') is not None else set())


def _day(value: str):
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00')).date()
    except (AttributeError, ValueError):
        return None


def feed_metrics(job: dict) -> dict:
    rows, end = ready(job), _day(job.get('finished_at') or '')
    news_rows = [r for r in rows if r.get('source') != 'bank_analytics']
    ages = []
    for r in news_rows:
        ref = (r.get('news_references') or {}).get(f"news:{r.get('news_id')}") or {}
        if (d := _day(ref.get('date', ''))) and end:
            ages.append((end - d).days)
    outlets = Counter(ref.get('source') for r in news_rows for ref in (r.get('news_references') or {}).values() if ref.get('source'))
    pairs = list(itertools.combinations(news_rows, 2))
    facts = [c for r in rows for c in r['insight'].get('what_happened', [])]
    texts = [' '.join([r['insight'].get('title', '')] + [c.get('text', '') for k in TEXT_BLOCKS for c in r['insight'].get(k, [])]
                      + r['insight'].get('actions', [])) for r in rows]
    started = _day(job.get('created_at') or '')
    all_news = [r for r in (job.get('result') or {}).get('results', []) if r.get('source') != 'bank_analytics' and r.get('news_id') is not None]
    return dict(
        insights=len(rows), news_insights=len(news_rows),
        age_median_days=float(np.median(ages)) if ages else None, share_older=(sum(a > OLD_DAYS for a in ages) / len(ages)) if ages else None,
        top_outlet_share=(max(outlets.values()) / sum(outlets.values())) if outlets else None, outlets=len(outlets),
        duplicate_pair_share=(sum(bool(news_ids(a) & news_ids(b)) for a, b in pairs) / len(pairs)) if pairs else None,
        claims_with_source=(sum(bool(c.get('sources')) for c in facts) / len(facts)) if facts else None,
        claims_with_quote=(sum(bool(c.get('evidence')) for c in facts) / len(facts)) if facts else None,
        peer_leak_share=sum(bool(peer_hits(t)) for t in texts) / len(texts) if texts else None,
        client_payment_data_share=sum(bool(client_hits(t)) for t in texts) / len(texts) if texts else None,
        with_bank_position=sum(bool(r['insight'].get('bank_view')) for r in rows) / len(rows) if rows else None,
        from_analytics=sum(r.get('source') == 'bank_analytics' for r in rows) / len(rows) if rows else None,
        model_error_share=(sum(r.get('status') == 'model_error' for r in all_news) / len(all_news)) if all_news else None,
        wall_minutes=(datetime.fromisoformat(job['finished_at']) - datetime.fromisoformat(job['created_at'])).total_seconds() / 60
        if job.get('finished_at') and job.get('created_at') else None,
    )


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def feed_overlap(feeds: dict[tuple[str, str], dict]) -> dict:
    """feeds: (company, role) -> job. Mean Jaccard of the cited news between two feeds, split by what differs."""
    cited = {k: set().union(*[news_ids(r) for r in ready(j)]) if ready(j) else set() for k, j in feeds.items()}
    same_company = [jaccard(cited[a], cited[b]) for a, b in itertools.combinations(cited, 2) if a[0] == b[0]]
    other_company = [jaccard(cited[a], cited[b]) for a, b in itertools.combinations(cited, 2) if a[0] != b[0]]
    return dict(same_company_other_role=float(np.mean(same_company)) if same_company else None, pairs_same=len(same_company),
                other_company=float(np.mean(other_company)) if other_company else None, pairs_other=len(other_company))


def latest_jobs(folder: Path) -> dict[tuple[str, str], dict]:
    jobs = [json.loads(p.read_text(encoding='utf-8')) for p in Path(folder).glob('*.json')]
    done = [j for j in jobs if j.get('status') in ('succeeded', 'partial') and ready(j)]
    out = {}
    for j in sorted(done, key=lambda j: j['created_at']):
        out[(j['request']['company_id'], j['request']['role'])] = j  # the newest wins
    return out


def main():
    from common.data import DATA
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=Path, default=DATA / 'api' / 'analyses')
    a = ap.parse_args()
    feeds = latest_jobs(a.jobs)
    print(json.dumps(dict(feeds={f'{c}/{r}': feed_metrics(j) for (c, r), j in feeds.items()}, overlap=feed_overlap(feeds)),
                     ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
