from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.company_profile import display_profile
from common.data import DQ


def sample():
    return dict(id='U_7', card=dict(okved='64.91',okved_name='Финансовый лизинг',region='Москва',
        revenue=9000000.,group_revenue=12000000.,products=['loan','rko'],fts_flag=False,fts_countries=['Китай']),
        scale=dict(class_ru='средний бизнес',group_size=3),data_quality=['fts_inconsistent'],
        external_factors=[dict(label='Ставка',weight=.9,mechanism='Стоимость фондирования')],
        facts=[dict(id='fact:profile_1',cat='profile',text='Численность по ФНС: 101–250 чел.',source='ФНС',kind='fact'),
               dict(id='fact:group_1',cat='group',text='Входит в группу из 3 компаний',source='GROUP_ID',kind='fact'),
               dict(id='fact:trade_1',cat='trade',text='Связи с Китаем, направление не указано',source='ФТС',kind='fact'),
               dict(id='hypothesis',cat='profile',text='Возможно, есть импорт',source='правила',kind='hypothesis'),
               dict(id='quality',cat='data_quality',text=DQ['fts_inconsistent'][1],kind='fact')])


def test_company_overview_keeps_legacy_contract_and_source_precision():
    source=sample()
    before=deepcopy(source)
    result=display_profile(source)
    assert source==before
    assert result['name']=='Финансовый лизинг №7'
    assert result['industry']=='Финансовый лизинг' and result['region']=='Москва'
    assert result['scale']=='Средний бизнес · выручка 9 млн ₽'
    assert result['group']=='Входит в группу из 3 компаний'
    assert result['bank_products']==['Кредитные продукты','РКО']
    assert result['channels']==[dict(name='Ставка',why='Стоимость фондирования')]
    assert result['hypotheses']==[]
    overview=result['overview']
    assert overview['revenue']==9000000. and overview['group_revenue']==12000000.
    assert overview['group_size']==3 and overview['industry_code']=='64.91'
    # Preserve contradictions rather than guessing import/export or erasing countries when FTS_FLAG is false.
    assert overview['trade']==dict(in_customs_database=False,countries=['Китай'])
    assert overview['facts'][0]['text']=='Численность по ФНС: 101–250 чел.'
    assert overview['facts'][0]['source']=='ФНС'
    assert all(f['id']!='hypothesis' for f in overview['facts'])
    assert overview['data_notes']==[DQ['fts_inconsistent'][1]]


def test_company_overview_does_not_fill_missing_values():
    result=display_profile(dict(id='U_8',card={},facts=[]))
    assert result['name']=='Компания №8'
    overview=result['overview']
    assert overview['revenue'] is None and overview['group_size'] is None
    assert overview['trade']==dict(in_customs_database=None,countries=[])
    assert overview['facts']==[] and overview['data_notes']==[]


def test_existing_company_endpoint_adds_overview_without_reloading_profile_or_calling_models(monkeypatch):
    monkeypatch.setenv('CABINET_MODE','server')
    source=sample()
    service=Mock()
    with patch('stage2_company_facts.profile.load.load_profile',return_value=source) as loader, \
         patch('stage2_company_facts.profile.integration.company_inputs',return_value=SimpleNamespace(card=source['card'],facts=source['facts'])), \
         TestClient(create_app(service=service)) as client:
        response=client.get('/v1/companies/U_7')
        assert response.status_code==200,response.text
        result=response.json()
        assert result['profile']['overview']['company_id']=='U_7'
        assert result['card']==source['card'] and result['facts']==source['facts']
        loader.assert_called_once_with('U_7')
        service.models_factory.assert_not_called()
