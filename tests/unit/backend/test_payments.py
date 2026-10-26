"""Payment metrics share the cleaned tables, account for unknown metadata and never call a model."""
from unittest.mock import Mock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.payments import PaymentAnalytics


@pytest.fixture
def tables():
    companies = pd.DataFrame([
        ['U_1','Лизинг','64.91','Москва',1,'merged_id_partner'],
        ['U_2','Ремонт автомобилей','45.20','Москва',1,''],
        ['U_3','Торговля деталями','45.32','Самара',2,''],
        ['U_4',None,None,None,None,''],
    ],columns=['SYNTHETIC_U','MAINOKVED_NAME','okved','REGION','GROUP_ID','DQ_FLAGS']).set_index('SYNTHETIC_U')
    payments = pd.DataFrame([
        ['U_1','U_2',600.,3],['U_1','U_3',300.,1],['U_1','U_9',100.,2],
        ['U_2','U_1',200.,2],['U_3','U_1',800.,8],
    ],columns=['PAYER','BENEFICIARY','PAY_AMT','PAY_CNT'])
    own = pd.DataFrame([['U_1','U_1',500.,5]],columns=payments.columns).set_index('PAYER')
    return companies,payments,own,pd.DataFrame()


def test_totals_weighted_average_concentration_unknowns_and_self_transfers(tables):
    analytics = PaymentAnalytics(loader=lambda:tables)
    report = analytics.report('U_1')
    assert report['summary']['out'] == dict(amount=1000.,count=6,average=1000/6,counterparties_count=3)
    assert report['summary']['in'] == dict(amount=1000.,count=10,average=100.,counterparties_count=2)
    assert report['self_transfers'] == dict(amount=500.,count=5,average=100.)
    assert report['flows']['out']['top1_share'] == .6
    assert report['flows']['out']['top5_share'] == pytest.approx(1)
    assert report['flows']['out']['within_group_share'] == .6
    assert report['flows']['in']['within_group_share'] == .2
    assert report['flows']['out']['known_industry_share'] == pytest.approx(.9)
    assert report['flows']['out']['known_region_share'] == pytest.approx(.9)
    unknown = next(row for row in report['counterparties'] if row['id']=='U_9')
    assert unknown['industry'] is None and unknown['region'] is None
    assert unknown['name'] == 'Компания №9'
    for dimension in ('industries','regions'):
        assert sum(item['amount'] for item in report['flows']['out'][dimension]) == 1000
        assert sum(item['share'] for item in report['flows']['out'][dimension]) == pytest.approx(1)
        assert next(item for item in report['flows']['out'][dimension] if item['unknown'])['amount'] == 100
    assert report['data_notes']
    assert report['period'] is None


def test_filter_sort_page_and_direction_use_one_load_and_preserve_share_denominator(tables):
    loader = Mock(return_value=tables)
    analytics = PaymentAnalytics(loader=loader)
    assert [row['id'] for row in analytics.report('U_1',sort='count')['counterparties']] == ['U_2','U_9','U_3']
    searched = analytics.report('U_1',query='самАРА')
    assert searched['total'] == 1 and searched['counterparties'][0]['id'] == 'U_3'
    assert searched['counterparties'][0]['share'] == .3
    assert searched['summary']['out']['amount'] == 1000
    page = analytics.report('U_1',offset=1,limit=1)
    assert page['total'] == 3 and page['counterparties'][0]['id']=='U_3'
    assert analytics.report('U_1','in')['counterparties'][0]['id']=='U_3'
    assert analytics.report('U_1',query='ничего')['total']==0
    assert analytics.report('U_4')['summary']['out']['count']==0
    loader.assert_called_once()


def test_empty_flow_and_invalid_counts_do_not_fabricate_an_average(tables):
    analytics = PaymentAnalytics(loader=lambda:tables)
    empty = analytics.report('U_4')
    assert empty['counterparties']==[] and empty['summary']['in']['average'] is None
    assert empty['flows']['in']['industries']==[] and empty['self_transfers']['count']==0
    tables[1].loc[0,'PAY_CNT']=0
    zero_count = PaymentAnalytics(loader=lambda:tables).report('U_1')['counterparties'][0]
    assert zero_count['average'] is None
    tables[1].loc[0,'PAY_CNT']=-1
    with pytest.raises(ValueError):
        PaymentAnalytics(loader=lambda:tables).report('U_1')


def test_other_buckets_preserve_all_amounts(tables):
    companies,payments,own,persons = tables
    for n in range(10,20):
        companies.loc[f'U_{n}'] = [f'Отрасль {n}',str(n),f'Регион {n}',n,'']
        payments.loc[len(payments)] = ['U_1',f'U_{n}',10.,1]
    report = PaymentAnalytics(loader=lambda:tables).report('U_1')
    for dimension in ('industries','regions'):
        items=report['flows']['out'][dimension]
        assert len(items)<=8
        assert any(item['label'].startswith('Другие') for item in items)
        assert sum(item['amount'] for item in items)==1100
        assert sum(item['share'] for item in items)==pytest.approx(1)


def test_http_validation_read_only_mode_and_missing_tables(tables,monkeypatch):
    monkeypatch.setenv('CABINET_MODE','server')
    service=Mock()
    analytics=PaymentAnalytics(loader=lambda:tables)
    with TestClient(create_app(service=service,payment_analytics=analytics)) as client:
        base='/v1/companies/U_1/payments'
        response=client.get(base,headers={'X-User-ID':'demo-user'})
        assert response.status_code==200,response.text
        assert response.json()['currency']=='RUB'
        assert client.get(base+'?direction=in&sort=count&limit=1&offset=1').json()['counterparties'][0]['id']=='U_2'
        for query in ('direction=bad','sort=bad','offset=-1','limit=101','q='+'x'*121):
            assert client.get(base+'?'+query).status_code==422
        assert client.get('/v1/companies/U_missing/payments').status_code==404
        service.models_factory.assert_not_called()
        service.analyze.assert_not_called()
    broken=PaymentAnalytics(loader=Mock(side_effect=FileNotFoundError()))
    with TestClient(create_app(service=Mock(),payment_analytics=broken)) as client:
        assert client.get('/v1/companies/U_1/payments').status_code==503


def test_counterparties_merge_directions_without_double_counting_and_keep_denominators(tables):
    loader=Mock(return_value=tables)
    analytics=PaymentAnalytics(loader=loader)
    report=analytics.counterparties('U_1')
    assert report['summary']==dict(total=3,incoming=2,outgoing=3,both=2,same_group=1)
    assert [row['id'] for row in report['items']]==['U_3','U_2','U_9']
    row=report['items'][1]
    assert row['relationship']=='both' and row['amount']==800 and row['count']==5
    assert row['in']['share']==.2 and row['out']['share']==.6
    assert row['in']['average']==100 and row['out']['average']==200
    own=analytics.counterparties('U_1',group='own')
    assert own['total']==1 and own['items'][0]['id']=='U_2'
    assert analytics.counterparties('U_1',direction='in')['total']==2
    assert analytics.counterparties('U_1',direction='both')['total']==2
    searched=analytics.counterparties('U_1',query='самАРА')
    assert searched['total']==1 and searched['items'][0]['out']['share']==.3
    assert searched['summary']==report['summary']
    assert analytics.counterparties('U_1',sort='in')['items'][0]['id']=='U_3'
    assert analytics.counterparties('U_1',offset=1,limit=1)['items'][0]['id']=='U_2'
    assert analytics.counterparties('U_4')['items']==[]
    analytics.report('U_1')
    loader.assert_called_once()


def test_counterparty_profile_unknown_metadata_and_company_scoping(tables):
    analytics=PaymentAnalytics(loader=lambda:tables)
    detail=analytics.counterparty('U_1','U_2')
    assert detail['profile']['available'] is True
    assert detail['profile']['revenue'] is None
    assert detail['counterparty']['same_group'] is True
    assert detail['counterparty']['group_known'] is True
    missing=analytics.counterparty('U_1','U_9')
    assert missing['profile']['available'] is False
    assert missing['counterparty']['name']=='Компания №9'
    assert missing['counterparty']['group_known'] is False
    assert missing['counterparty']['in']==dict(amount=0.,count=0,average=None,share=0.)
    for owner,other in [('U_2','U_9'),('U_1','U_4'),('U_1','U_1')]:
        with pytest.raises(KeyError):
            analytics.counterparty(owner,other)


def test_counterparty_api_validates_filters_and_reads_without_inference(tables,monkeypatch):
    monkeypatch.setenv('CABINET_MODE','server')
    service=Mock()
    with TestClient(create_app(service=service,payment_analytics=PaymentAnalytics(loader=lambda:tables))) as client:
        base='/v1/companies/U_1/counterparties'
        response=client.get(base+'?direction=both&group=own&sort=count&limit=1',headers={'X-User-ID':'demo-user'})
        assert response.status_code==200,response.text
        assert response.json()['total']==1
        assert client.get(base+'/U_2').json()['counterparty']['out']['amount']==600
        assert client.get(base+'/U_9').json()['profile']['available'] is False
        for query in ('direction=bad','group=bad','sort=bad','offset=-1','limit=101','q='+'x'*121):
            assert client.get(base+'?'+query).status_code==422
        assert client.get(base+'/U_4').status_code==404
        assert client.get('/v1/companies/U_2/counterparties/U_9').status_code==404
        service.models_factory.assert_not_called()
        service.analyze.assert_not_called()
