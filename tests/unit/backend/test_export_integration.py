"""Export follows the current combined cabinet feed and its company/role/user boundaries."""
import copy
import json
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.service import Service
from test_exports import ROW, tables


@pytest.fixture
def export_api(tmp_path):
    models=Mock(side_effect=AssertionError('Export must not initialize models'))
    service=Service(tmp_path,models_factory=models)
    service.case_tables=tables
    with TestClient(create_app(service,cors_origins=['http://localhost:5173'],staff_token='test-staff')) as client:
        yield service,client,tmp_path
    models.assert_not_called()


def job(key,card,company='U_1',role='cfo',user='alice'):
    row=copy.deepcopy(ROW)
    row['insight']['id']=card
    return dict(id=key,kind='analysis',status='succeeded',created_at='2026-10-25T00:00:00Z',finished_at='2026-10-25T00:01:00Z',
                request=dict(company_id=company,role=role,user_id=user),result=dict(results=[row]),error=None)


def test_combined_feed_exports_all_owned_and_shared_cards_for_exact_company_and_role(export_api):
    service,client,_=export_api
    jobs=[job('a'*32,'own-one'),job('b'*32,'shared',user=None),job('c'*32,'own-two'),
          job('d'*32,'foreign-user',user='bob'),job('e'*32,'foreign-role',role='ceo'),job('f'*32,'foreign-company',company='U_2')]
    service.agents.items.update({j['id']:j for j in jobs})
    for user,expected in [('alice',{'own-one','shared','own-two'}),('bob',{'shared','foreign-user'})]:
        headers={'X-User-ID':user}
        feed=client.get('/v1/cabinet/feed',params=dict(company_id='U_1',role='cfo'),headers=headers).json()
        response=client.get('/v1/exports/insights',params=dict(company_id='U_1',role='cfo',format='json'),headers=headers)
        assert response.status_code==200,response.text
        body=response.json()
        assert {r['insight_id'] for r in body['tables']['insights']}==expected=={c['id'] for c in feed['insights']}
        assert {r['company_id'] for r in body['tables']['insights']}=={'U_1'}
        assert {r['role'] for r in body['tables']['insights']}=={'cfo'}


def test_job_context_cannot_be_relabelled_and_staff_does_not_bypass_its_owner(export_api):
    service,client,_=export_api
    service.agents.items['a'*32]=job('a'*32,'ceo-card',role='ceo')
    params=dict(job_id='a'*32,format='json')
    headers={'X-User-ID':'alice','Origin':'http://localhost:5173'}
    response=client.get('/v1/exports/insights',params=params,headers=headers)
    assert response.status_code==200
    assert response.json()['meta']['role']=='ceo'
    assert response.json()['tables']['insights'][0]['role']=='ceo'  # the writer's old cfo label is not authoritative
    assert 'Content-Disposition' in response.headers['access-control-expose-headers']
    for extra in (dict(company_id='U_2'),dict(role='cfo')):
        assert client.get('/v1/exports/insights',params=dict(params,**extra),headers=headers).status_code==404
    for other in ({},{'X-User-ID':'bob'},{'X-User-ID':'bob','Authorization':'Bearer test-staff'}):
        assert client.get('/v1/exports/insights',params=params,headers=other).status_code==404


def test_committed_demo_exports_match_the_visible_feed_without_an_agent_run(export_api):
    service,client,_=export_api
    params=dict(company_id='U_14366',role='cfo')
    cards=client.get('/v1/cabinet/feed',params=params).json()['insights']
    assert cards
    for target in (params,dict(job_id='demo-U_14366-cfo')):
        response=client.get('/v1/exports/insights',params=dict(target,format='json'))
        assert response.status_code==200,response.text
        assert {r['insight_id'] for r in response.json()['tables']['insights']}=={card['id'] for card in cards}
    assert service.agents.items=={}


def test_company_only_permission_does_not_expand_to_group_and_no_default_staff_override(export_api):
    _,client,root=export_api
    (root/'access.json').write_text(json.dumps({'alice':['U_1']}),encoding='utf-8')
    headers={'X-User-ID':'alice'}
    assert client.get('/v1/exports/payments',params=dict(company_id='U_1'),headers=headers).status_code==200
    assert client.get('/v1/exports/payments',params=dict(company_id='U_1',scope='group'),headers=headers).status_code==404
    assert client.get('/v1/exports/payments',params=dict(company_id='U_2'),headers=headers).status_code==404


def test_export_parameters_dates_and_pending_files_are_validated(export_api):
    service,client,_=export_api
    for bad in (dict(role='unknown'),dict(date_from='2026-02-30'),dict(date_from='2026-10-27',date_to='2026-10-26'),dict(job_id='../secret')):
        assert client.get('/v1/exports/insights',params=dict(company_id='U_1',**bad)).status_code==422
    assert client.get('/v1/exports/counterparties',params=dict(company_id='U_1',top=0)).status_code==422
    assert client.get('/v1/exports/payments',params=dict(company_id='U_1',format='pdf')).status_code==422
    pending=dict(id='a'*32,kind='export',status='queued',created_at='2026-10-26',request=dict(user_id='alice'),result=None,error=None)
    service.exports.items[pending['id']]=pending
    assert client.get(f'/v1/exports/{pending["id"]}/file',headers={'X-User-ID':'alice'}).status_code==409
    assert client.get(f'/v1/exports/{pending["id"]}',headers={'X-User-ID':'bob'}).status_code==404


def test_payment_period_remains_unspecified(export_api):
    _,client,_=export_api
    response=client.get('/v1/exports/payments',params=dict(company_id='U_1',format='json'))
    assert response.status_code==200
    assert response.json()['meta']['period'] is None
    assert 'с 1 января' not in response.json()['meta']['note']
