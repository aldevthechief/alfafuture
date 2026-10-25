"""Read-only demo accounts backed by committed agent runs. Never initializes an LLM or encoder."""
import copy
import json
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from common.roles import roles
from common.data import DATA
from backend.insight_previews import preview_feed

CACHE = Path(__file__).resolve().parents[1] / 'stage5_insight_detection/analytics_insights/cache'
NEWS_CACHE = Path(__file__).with_name('demo_cache')
ACCOUNTS = (
    dict(id='finance-ceo', company_id='U_14366', initials='ФЛ', sector='Финансовый лизинг',
         city='Москва', role='ceo', color='blue', balance=912400150.30, credit=3000000000, deposit=0),
    dict(id='finance', company_id='U_14366', initials='ФЛ', sector='Финансовый лизинг',
         city='Москва', role='cfo', color='blue', balance=912400150.30, credit=3000000000, deposit=0),
    dict(id='development', company_id='U_177581', initials='СМ', sector='Строительно-монтажные работы',
         city='Москва', role='ceo', color='amber', balance=1786300420, credit=5000000000, deposit=0),
    dict(id='energy', company_id='U_57706', initials='ОТ', sector='Оптовая торговля топливом',
         city='Москва', role='treasury', color='green', balance=1428750900, credit=2000000000, deposit=800000000),
)


def account_job_id(account):
    return f"demo-{account['company_id']}-{account['role']}"


def catalog(jobs=(), user_id=None, root=None):
    accounts = []
    for account in ACCOUNTS:
        try:
            feed = cabinet_feed(account['company_id'], account['role'], jobs, user_id, root)
        except (KeyError, OSError, ValueError):
            continue
        if not feed['insights']:
            continue
        spec = roles()[account['role']]
        accounts.append(dict(account, name=company_name(account['company_id'], account['sector']),
                             job_id=account_job_id(account), role_title=spec['title'],
                             insight_count=len(feed['insights']), updated_at=feed['computed_at']))
    return dict(accounts=accounts, roles=[dict(id=key, title=value['title']) for key, value in roles().items()])


def cached_job(job_id, root=None):
    match = re.fullmatch(r'demo-([A-Za-z0-9_]+)-([a-z_]+)', job_id)
    if not match or match[2] not in roles():
        raise KeyError(job_id)
    company_id, role = match.groups()
    filename = f'{company_id}_{role}.json'
    saved = []
    path = NEWS_CACHE / filename
    if path.exists():
        saved.append(_read_saved_job(str(path), path.stat().st_mtime_ns, job_id, company_id, role))
    local = (Path(root) if root is not None else DATA / 'api') / 'analytics_cache'
    paths = sorted(local.glob(f'*/{filename}'), key=lambda p: p.stat().st_mtime_ns, reverse=True)
    paths.append(CACHE / filename)
    for path in paths:
        if not path.exists():
            continue
        try:
            job = _read_job(str(path), path.stat().st_mtime_ns, job_id)
        except (OSError, ValueError, KeyError):
            continue
        if (job['request']['company_id'], job['request']['role']) == (company_id, role):
            saved.append(job)
    if not saved:
        raise KeyError(job_id)
    rows, seen = [], set()
    for job in saved:
        for row in job['result']['results']:
            key = _row_key(row)
            if key not in seen:
                seen.add(key)
                rows.append(row)
    latest = max(saved, key=lambda job: _time(job.get('finished_at')))
    return dict(latest, id=job_id, request=dict(company_id=company_id, role=role, user_id=None, source='both'),
                result=dict(company_id=company_id, role=role, cached=True, results=rows))


def _time(value):
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
    except (AttributeError, ValueError):
        return 0


def _row_key(row):
    insight = row['insight']
    origin = insight.get('origin') or ('bank' if row.get('source') == 'bank_analytics' else 'news')
    return origin, insight.get('id') or row.get('news_id') or insight.get('title')


def cabinet_feed(company_id, role, jobs=(), user_id=None, root=None):
    """Merge all available checked sources for exactly this reader; never starts inference."""
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', company_id) or role not in roles():
        raise ValueError('Unknown company or role')
    mine = [job for job in jobs if job.get('request', {}).get('company_id') == company_id
            and job['request'].get('role') == role and job['request'].get('user_id') in (None, user_id)]
    mine.sort(key=lambda job: _time(job.get('created_at')), reverse=True)
    completed = [job for job in mine if job['status'] in ('succeeded', 'partial')]
    try:
        completed.append(cached_job(f'demo-{company_id}-{role}', root))
    except (KeyError, OSError, ValueError):
        pass
    cards, seen = [], set()
    for job in completed:
        rows = (job.get('result') or {}).get('results', [])
        for card in preview_feed(job)['insights']:
            key = _row_key(rows[card['result_index']])
            if key not in seen:
                seen.add(key)
                cards.append(card)
    newest = completed[0] if completed else None
    running = next((job for job in mine if job['status'] in ('queued', 'running')
                    and job['request'].get('user_id') == user_id), None)
    latest = mine[0] if mine else None
    return dict(computed_at=max((job.get('finished_at') or job['created_at'] for job in completed),
                                key=_time, default=None),
                insights=cards, status=newest['status'] if newest else 'none',
                errors=(newest.get('result') or {}).get('errors', []) if newest else [],
                running_job=dict(id=running['id'], status=running['status']) if running else None,
                latest_job=dict(id=latest['id'], status=latest['status'], error=latest.get('error')) if latest else None)


@lru_cache(maxsize=12)
def _read_saved_job(path, version, job_id, company_id, role):
    """Use checked news and bank rows with frozen sources, without recomputation."""
    job = json.loads(Path(path).read_text(encoding='utf-8'))
    request = job.get('request') or {}
    if (request.get('company_id'), request.get('role')) != (company_id, role):
        raise ValueError('Cached news run belongs to another company or role')
    if job.get('status') not in ('succeeded', 'partial'):
        raise ValueError('No completed cached news run')
    rows = [row for row in (job.get('result') or {}).get('results', [])
            if row.get('status') == 'ready' and row.get('insight') and row.get('evidence')]
    if not rows:
        raise ValueError('No checked news insights with frozen sources')
    job.update(id=job_id, request=dict(request, user_id=None))
    job['result'] = dict(company_id=company_id, role=role, cached=True, results=rows)
    return job


@lru_cache(maxsize=12)
def _read_job(path, version, job_id):
    run = json.loads(Path(path).read_text(encoding='utf-8'))
    if run.get('error') or not run.get('insights'):
        raise ValueError('No completed cached run')
    rows = []
    for original in run['insights']:
        insight = copy.deepcopy(original)
        blocks = ('what_happened', 'why_it_matters', 'risks', 'opportunities', 'hypotheses', 'bank_view')
        claims = [claim for block in blocks for claim in insight.get(block, [])]
        # The saved run contains the verbatim company facts actually cited by its claims.
        facts = {source: claim['text'] for claim in claims if claim.get('kind') == 'fact'
                 for source in claim.get('sources', []) if source.startswith('fact:')}
        ordered = sorted(facts, key=lambda source: int(source.split(':')[1]))
        mapping = {old: f'fact:{index}' for index, old in enumerate(ordered, 1)}

        def remap(value):
            if isinstance(value, dict):
                return {key: remap(item) for key, item in value.items()}
            if isinstance(value, list):
                return [remap(item) for item in value]
            return mapping.get(value, value) if isinstance(value, str) else value

        bank = {}
        for claim in claims:
            for evidence in claim.get('evidence', []):
                source = evidence.get('source', '')
                if not source.startswith('bank:'):
                    continue
                item = bank.setdefault(source, dict(id=source, report=evidence.get('title') or 'Аналитика Банка',
                    date=evidence.get('date') or '', page=evidence.get('page'), stale=False, text=''))
                quote = evidence.get('quote') or ''
                if quote and quote not in item['text']:
                    item['text'] += ('\n\n' if item['text'] else '') + quote
        rows.append(dict(status='ready', source='bank_analytics', news_id=None, insight=remap(insight),
            evidence=dict(version=2, company_id=run['company_id'], role=run['role'],
                          captured_at=f"{run['as_of']}T00:00:00Z", news=[], calculations=[],
                          facts=[dict(id=mapping[old], text=facts[old]) for old in ordered], bank=list(bank.values())),
            review=dict(ok=True), audit=[]))
    return dict(id=job_id, kind='analysis', status='succeeded', created_at=f"{run['as_of']}T00:00:00Z",
                finished_at=f"{run['as_of']}T00:00:00Z", error=None,
                request=dict(company_id=run['company_id'], role=run['role'], user_id=None, source='analytics'),
                result=dict(company_id=run['company_id'], role=run['role'], cached=True, results=rows))


@lru_cache(maxsize=1)
def _company_table():
    import pandas as pd
    from common.data import DATA
    return pd.read_csv(DATA / 'df_Uid_data_hack.csv', sep=';', dtype=str,
                       usecols=['SYNTHETIC_U', 'MAINOKVED_NAME', 'CITY']).fillna('')


def search_companies(query):
    table = _company_table()
    needle = query.strip().casefold()
    if len(needle) < 2:
        return []
    match = table['SYNTHETIC_U'].str.casefold().eq(needle)
    if not match.any():
        match = table['SYNTHETIC_U'].str.casefold().str.contains(needle, regex=False) | \
                table['MAINOKVED_NAME'].str.casefold().str.contains(needle, regex=False)
    return [dict(id=row.SYNTHETIC_U, name=company_name(row.SYNTHETIC_U, row.MAINOKVED_NAME),
                 industry=row.MAINOKVED_NAME, city=row.CITY) for row in table[match].head(8).itertuples()]


def company_name(company_id, activity):
    activity = re.sub(r'^Деятельность по\s+', '', activity or '', flags=re.I).strip()
    return f'{activity[:1].upper() + activity[1:] if activity else "Компания"} №{company_id.removeprefix("U_")}'
