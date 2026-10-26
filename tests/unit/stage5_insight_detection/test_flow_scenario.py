"""Price change × the company's known payments by counterparty activity: a ruble estimate for the card, a scenario and a
lower bound, never the company's costs. Stubbed facts, no case data."""
from types import SimpleNamespace
import json
from unittest.mock import patch
import pandas as pd

from stage5_insight_detection.analyst_loop.compute import compute
from stage5_insight_detection.analyst_loop.facts import Fact


D = SimpleNamespace(profile=SimpleNamespace(facts={
    'fact:pay_out:46.71.2': Fact('fact:pay_out:46.71.2', 'топливо', 1.66e9, '₽'),
    'fact:pay_out:49.41.2': Fact('fact:pay_out:49.41.2', 'перевозчики', 5.6e9, '₽'),
    'fact:pay_in:46.38': Fact('fact:pay_in:46.38', 'покупатели продуктов', 1.5e8, '₽')}),
    event=SimpleNamespace(topics=[], date=None), bank=[])


def test_fuel_price_on_payments_to_fuel_suppliers():
    res, why = compute('рост цен на топливо на 10% на платежи поставщикам', D, 1)
    calc = res[-1]
    assert calc.kind == 'hypothesis' and calc.unit == 'млн ₽' and calc.value == 166.0
    assert '46.71.2' in calc.label and '49.41' not in calc.label and 'видимой части платежей' in calc.formula
    assert 'период не указан' in calc.label and 'в год' not in calc.label
    assert 'допущение' in calc.formula and 'это не затраты' in calc.formula


def test_transport_is_not_a_unit_conversion_and_buyers_side_works():
    res, _ = compute('рост тарифов на перевозки на 15% на платежи поставщикам', D, 2)
    assert res[-1].value == 840.0 and '49.41.2' in res[-1].label
    res, _ = compute('снижение цен на продукты на 10% на платежи от покупателей', D, 3)
    assert res[-1].value == -15.0 and 'от покупателей' in res[-1].label


def test_missing_data_gives_a_reason_not_a_number():
    for query, reason in [('рост цен на металл на 10% на платежи поставщикам', 'нет поставщиков'),
                          ('рост цен на топливо на платежи поставщикам', 'нет изменения цены'),
                          ('рост цен на 10% на платежи поставщикам', 'нет вида контрагентов')]:
        res, why = compute(query, D, 4)
        assert res == [] and reason in why, (query, why)


def test_negative_and_explicit_percentages_and_ambiguous_requests():
    for text in ('снижение цен на топливо на 10% на платежи поставщикам',
                 'цены на топливо -10% на платежи поставщикам',
                 'уменьшение цены топлива на −10% на платежи поставщикам'):
        res,_=compute(text,D)
        assert res[-1].value==-166
    for text in ('снижение цен на топливо на +10% на платежи поставщикам',
                 'рост цен на топливо на -10% на платежи поставщикам',
                 'цены на топливо 10%, объём 20% на платежи поставщикам',
                 'снижение цен на топливо на 110% на платежи поставщикам',
                 'рост цен на топливо на 10% на платежи покупателей и поставщиков'):
        res,why=compute(text,D)
        assert not res and why


def test_specific_goods_win_over_generic_tariffs_and_petroleum_is_not_food():
    extra=dict(D.profile.facts)
    extra['fact:pay_out:35.1']=Fact('fact:pay_out:35.1','электроэнергия',1e9,'₽')
    extra['fact:pay_out:10.1']=Fact('fact:pay_out:10.1','продукты',1e9,'₽')
    dossier=SimpleNamespace(profile=SimpleNamespace(facts=extra),event=D.event,bank=[])
    res,_=compute('рост тарифов на перевозки на 15% на платежи поставщикам',dossier)
    assert res[-1].value==840
    res,_=compute('рост цен на нефтепродукты на 10% на платежи поставщикам',dossier)
    assert res[-1].value==166
    res,_=compute('рост на 10% на платежи поставщикам с ОКВЭД 49',dossier)
    assert res[-1].value==560


def test_adapter_sums_all_classified_counterparties_keeps_sides_and_reuses_facts():
    from stage5_insight_detection.insight_judge.calculations import JudgeCalculations
    companies=pd.DataFrame({'okved':['46.71.2','46.71.2','49.41.2','46.38',None]},index=['fuel1','fuel2','transport','buyer','unknown'])
    payments=pd.DataFrame([['U_1','fuel1',1e8],['U_1','fuel2',2e8],['U_1','transport',5e8],
                          ['buyer','U_1',1.5e8],['U_1','unknown',9e9],['U_1','U_1',8e9],['U_other','fuel1',7e9]],
                         columns=['PAYER','BENEFICIARY','PAY_AMT'])
    calc=JudgeCalculations({'id':1,'title':'Топливо дорожает','text':'Цена топлива','date':'2026-10-26'},'U_1',None)
    with patch('common.data.load',return_value=(companies,payments,None,None)) as loader, \
         patch('stage5_insight_detection.insight_judge.calculations.load_company',return_value=SimpleNamespace(card={})):
        out=json.loads(calc.run('рост цен на топливо на 10% на платежи поставщикам'))
        assert out['ok'] and out['facts'][-1]['value']==30
        assert len(out['facts'])==len({f['id'] for f in out['facts']})==2
        assert all('за год' not in f['label'] for f in out['facts'])
        incoming=json.loads(calc.run('снижение цен на продукты на 10% на платежи от покупателей'))
        assert incoming['facts'][-1]['value']==-15
        loader.assert_called_once()
