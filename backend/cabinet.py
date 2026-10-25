"""Read-only demo accounts backed by committed agent runs. Never initializes an LLM or encoder."""
import copy
import json
from functools import lru_cache
from pathlib import Path

from common.roles import roles

CACHE = Path(__file__).resolve().parents[1] / 'stage5_insight_detection/analytics_insights/cache'
ACCOUNTS = (
    dict(id='finance', company_id='U_14366', name='Север Финанс', initials='СФ', sector='Финансы и лизинг',
         city='Москва', role='cfo', color='blue', balance=912400150.30, credit=3000000000, deposit=0),
    dict(id='development', company_id='U_177581', name='Горизонт Девелопмент', initials='ГД', sector='Строительство',
         city='Москва', role='ceo', color='amber', balance=1786300420, credit=5000000000, deposit=0),
    dict(id='energy', company_id='U_57706', name='СибТопливо', initials='СТ', sector='Топливо и энергетика',
         city='Москва', role='treasury', color='green', balance=1428750900, credit=2000000000, deposit=800000000),
)


def account_job_id(account):
    return f"demo-{account['company_id']}-{account['role']}"


def catalog():
    accounts = []
    for account in ACCOUNTS:
        try:
            job = cached_job(account_job_id(account))
        except (KeyError, OSError, ValueError):
            continue
        spec = roles()[account['role']]
        accounts.append(dict(account, job_id=job['id'], role_title=spec['title'],
                             insight_count=len(job['result']['results']), updated_at=job['finished_at']))
    return dict(accounts=accounts, roles=[dict(id=key, title=value['title']) for key, value in roles().items()])


def cached_job(job_id):
    account = next((a for a in ACCOUNTS if account_job_id(a) == job_id), None)
    if account is None:
        raise KeyError(job_id)
    path = CACHE / f"{account['company_id']}_{account['role']}.json"
    return _read_job(str(path), path.stat().st_mtime_ns, job_id)


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
    return [dict(id=row.SYNTHETIC_U, name=f'Компания {row.SYNTHETIC_U}',
                 industry=row.MAINOKVED_NAME, city=row.CITY) for row in table[match].head(8).itertuples()]
