"""Cached discovery keeps the current cabinet contract and never exposes private runs."""
import copy
from unittest.mock import Mock, patch

import pandas as pd
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.cabinet import cabinet_feed
from backend.service import Service


def saved_job(job_id, company='U_900001', user=None, source='news:1', date='2026-10-26'):
    return dict(id=job_id, kind='analysis', status='succeeded', created_at=f'{date}T00:00:00Z',
                finished_at=f'{date}T00:01:00Z', error=None,
                request=dict(company_id=company, role='cfo', user_id=user),
                result=dict(results=[dict(status='ready', news_id=1,
                    news_references={'news:1': dict(id=1, url='https://example.org/news', date=date)},
                    insight=dict(id='cached-card', title='Изменение ставки',
                        what_happened=[dict(text='Ставка изменилась', sources=[source])], actions=['Оценить влияние']))]))


def test_discovery_preserves_frontend_contract_excludes_private_runs_and_stops(tmp_path):
    models = Mock(side_effect=AssertionError('Discovery must not initialize an LLM'))
    service = Service(tmp_path, models_factory=models)
    public = saved_job('public')
    private = saved_job('private', company='U_900002', user='reader')
    service.agents.items.update(public=public, private=private)
    table = pd.DataFrame([dict(SYNTHETIC_U='U_900001', MAINOKVED_NAME='Финансовый лизинг',
                               CITY='г. Москва', REVENUE_COMPANY='20000000000')])
    app = create_app(service)
    with patch('backend.cabinet.CACHE', tmp_path / 'bank'), \
         patch('backend.cabinet.NEWS_CACHE', tmp_path / 'news'), \
         patch('backend.cabinet._company_table', return_value=table), TestClient(app) as client:
        app.state.precomputed.refresh()
        response = client.get('/v1/cabinet/precomputed')
        assert response.status_code == 200
        report = response.json()
        assert report['checked_at']
        assert [r['company_id'] for r in report['items']] == ['U_900001']
        account, = client.get('/v1/cabinet/accounts').json()['accounts']
        assert account['company_id'] == 'U_900001'
        assert account['role'] == 'cfo' and account['insight_count'] == 1
        assert {'id', 'name', 'sector', 'city', 'initials', 'color', 'balance', 'credit', 'deposit',
                'job_id', 'role_title', 'updated_at'} <= account.keys()
        feed = client.get('/v1/cabinet/feed?company_id=U_900001&role=cfo').json()
        card, = feed['insights']
        opened = client.get(f"/v1/jobs/{card['job_id']}/insights/{card['result_index']}")
        assert opened.status_code == 200
        assert opened.json()['result']['results'][0]['insight']['id'] == card['id']
        assert feed['hidden'] == 0
        models.assert_not_called()
    assert not app.state.precomputed._thread.is_alive()


def test_invalid_newer_card_does_not_hide_an_older_usable_version(tmp_path):
    old = saved_job('old', date='2026-10-25')
    broken = saved_job('new', source='news:missing')
    with patch('backend.cabinet.cached_job', side_effect=KeyError):
        feed = cabinet_feed('U_900001', 'cfo', [old, broken], root=tmp_path)
    assert feed['hidden'] == 1
    card, = feed['insights']
    assert card['job_id'] == 'old'


def test_invalid_cards_are_hidden_but_deadline_warnings_remain_visible(tmp_path):
    job = saved_job('saved')
    invalid = copy.deepcopy(job['result']['results'][0])
    invalid['insight']['id'] = 'broken-card'
    invalid['insight']['what_happened'][0]['sources'] = ['news:missing']
    job['result']['results'].append(invalid)
    job['result']['results'][0]['insight']['what_happened'][0]['text'] = 'Решить до 1 января'
    with patch('backend.cabinet.cached_job', side_effect=KeyError):
        feed = cabinet_feed('U_900001', 'cfo', [job], root=tmp_path)
    assert feed['hidden'] == 1
    assert [c['id'] for c in feed['insights']] == ['cached-card']
