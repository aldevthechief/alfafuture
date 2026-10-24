"""Register the ready insights of a saved pipeline result (data/insights/<company>_<role>.json) for discussion.
Only checked insights (status ready, or an analytics insight) are registered; each gets its evidence snapshot rebuilt
from the result by code (no LLM). python -X utf8 -m feedback_api.importer data/insights/U_42222_cfo.json --user-id demo"""
import argparse, json, sys
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.evidence import EvidenceSnapshot
from common.roles import canonical_role
from feedback_api.schemas import InsightRecord, RegisterInsight
from feedback_api.store import Store


def ready(result: dict) -> bool:
    """A checked insight: the news engines keep a rejected one under `draft`; analytics rows carry their own source."""
    if not isinstance(result.get('insight'), dict):
        return False
    return result.get('status') in ('ready', None) or result.get('source') == 'bank_analytics'


def register_results(store: Store, user_id: str, company_id: str, results: list[dict], *, news_lookup=None, facts=None,
                     output=print, registration_key: str | None = None) -> list[InsightRecord]:
    records = []
    pool = None
    if news_lookup is None and any(ready(r) and not r.get('evidence') for r in results):
        from common.search import news_pool
        pool = news_pool()
        source_index = pool.set_index('id', drop=False)
        news_lookup = lambda i, source_index=source_index: source_index.loc[i].to_dict() if i in source_index.index else None
    for row_index, r in enumerate(results):
        if not ready(r):
            continue
        if facts is None and not r.get('evidence') and r.get('evidence_facts') is None:
            from stage2_company_facts.deterministic_facts.facts import build
            facts = build([company_id], use_profiles=True)[company_id]
        # Older writers put the human-readable title in role; the frozen evidence keeps the machine ID.
        # Recognize equivalent names only, so an actual company/role mismatch still fails validation.
        insight = dict(r['insight'], role=canonical_role(r['insight'].get('role', '')))
        role = insight.get('role', '')
        try:
            lookup = news_lookup
            if not r.get('evidence') and pool is not None and r.get('corpus_fingerprint'):
                from common.data import DATA
                from common.news_corpus import result_references
                from common.search import corpus_fingerprint
                fingerprint = r['corpus_fingerprint']
                if len(fingerprint) != 64 or any(c not in '0123456789abcdef' for c in fingerprint):
                    raise ValueError('Invalid original corpus fingerprint')
                original = pool
                if corpus_fingerprint(pool) != fingerprint:
                    import pandas as pd
                    path = DATA/'news'/'versions'/fingerprint/'news.parquet'
                    if not path.exists():
                        raise ValueError('Original news snapshot unavailable; run a new analysis')
                    original = pd.read_parquet(path)
                if corpus_fingerprint(original) != fingerprint or not result_references(r, original)['ok']:
                    raise ValueError('News snapshot does not match original references')
                source_index = original.set_index('id', drop=False)
                lookup = lambda i, source_index=source_index: source_index.loc[i].to_dict() if i in source_index.index else None
            evidence = EvidenceSnapshot.model_validate(r['evidence']) if r.get('evidence') else \
                EvidenceSnapshot.from_result(r, company_id, role, news_lookup=lookup, facts=facts)
            records.append(store.register(user_id, RegisterInsight(company_id=company_id, news_id=r.get('news_id'),
                insight=insight, evidence=evidence,
                registration_key=f'{registration_key}:{row_index}' if registration_key else None)))
        except (ValidationError, ValueError) as e:
            output(f'skipped news {r.get("news_id")}: {type(e).__name__}')
    return records


if __name__ == '__main__':
    from common.personalization import feedback_db
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('file', type=Path)
    ap.add_argument('--user-id', required=True)
    ap.add_argument('--company-id', help='default: from the file name <company>_<role>.json')
    a = ap.parse_args()
    results = json.loads(a.file.read_text(encoding='utf-8-sig'))
    results = results.get('results', results) if isinstance(results, dict) else results
    store = Store(feedback_db())
    store.initialize()
    company = a.company_id or a.file.stem.rsplit('_', 1)[0]
    for rec in register_results(store, a.user_id, company, results):
        print(rec.id, '|', rec.insight.title)
