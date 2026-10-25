"""Read questions from the same owned/shared results that the cabinet actually shows."""
import json
from pathlib import Path

from backend.cabinet import cabinet_feed, cached_job
from common.data import DATA


def latest_results(company_id: str, role: str, user_id: str, root=None) -> list[dict]:
    root = Path(root) if root is not None else DATA / 'api'
    jobs = []
    for path in (root / 'analyses').glob('*.json'):
        try:
            job = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        request = job.get('request') or {}
        if (request.get('company_id'), request.get('role')) == (company_id, role) \
                and request.get('user_id') in (None, user_id):
            jobs.append(job)
    feed = cabinet_feed(company_id, role, jobs, user_id, root)
    by_id = {job['id']: job for job in jobs}
    rows = []
    for card in feed['insights']:
        if card['job_id'] not in by_id:
            by_id[card['job_id']] = cached_job(card['job_id'], root)
        rows.append(by_id[card['job_id']]['result']['results'][card['result_index']])
    return rows
