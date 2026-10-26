"""Business value by code: confidence follows the evidence, the bank offer fits the card and skips owned products;
credibility names the outlets. No case data, no models (the company profile is stubbed)."""
import stage6_insight_generation.business_value as bv
from common.credibility import credibility
from common.credibility import from_saved

PROFILE = {'card': {'products': ['loan', 'rko'], 'fts_countries': ['Китай']},
           'links': {'buyers': {'top5_share': 0.73}}}


def card(title, why=(), hypotheses=0, **extra):
    return {'title': title, 'role': 'cfo', 'why_it_matters': [{'text': t, 'sources': []} for t in why],
            'hypotheses': [{'text': 'h', 'sources': []}] * hypotheses, 'confidence': 'medium', **extra}


def run(result, monkeypatch):
    monkeypatch.setattr(bv, '_profile', lambda company_id: PROFILE)
    return bv.enrich(result, 'U_1')


def test_bank_card_with_checked_quotes_is_high(monkeypatch):
    ins = card('Ставка 14% до конца года', review={'verification': {'exact': 2}})
    out = run({'source': 'bank_analytics', 'insight': ins}, monkeypatch)
    assert out['insight']['confidence'] == 'high' and out['business_value']['writer_confidence'] == 'medium'


def test_news_card_built_on_assumptions_is_low(monkeypatch):
    why = ['Если часть долга плавающая, расходы растут', 'Если в портфеле импорт, валюта важна', 'Факт про рынок']
    out = run({'insight': card('Курс доллара упал', why, hypotheses=5), 'stage4': {'plausibility': 0.5}}, monkeypatch)
    assert out['business_value']['confidence'] == 'low'
    assert any('«если»' in b for b in out['business_value']['confidence_basis'])


def test_corroborated_news_with_fresh_bank_view_is_high(monkeypatch):
    ins = card('ЦБ сохранил ключевую ставку', ['Кредиты дорогие'], bank_view=[{'text': 'Пауза', 'sources': ['bank:x']}])
    res = {'insight': ins, 'judge_bank': [{'id':'bank:x','stale': False}],
           'credibility': credibility(['interfax', 'vedomosti', 'rbc'], {}, 0.7)}
    assert run(res, monkeypatch)['business_value']['confidence'] == 'high'


def test_offer_follows_topic_and_skips_owned_products(monkeypatch):
    rate = run({'insight': card('Ставка ЦБ сохранена: кредиты дорогие')}, monkeypatch)['business_value']['bank_product']
    assert rate['offer'].startswith('Фиксация ставки'), 'the company has loans: fix the rate, do not sell a new line'
    fx = run({'insight': card('Курс доллара упал', ['концентрация топ-5 покупателей'])}, monkeypatch)['business_value']
    assert fx['bank_product']['offer'].startswith('Валютный'), 'the card topic (fx) wins over the receivables hint'
    cash = run({'insight': card('Сезон слабее: ликвидность под пики', ['кассовый разрыв в декабре'])}, monkeypatch)
    assert 'Факторинг' in cash['business_value']['bank_product']['offer'] and '73%' in cash['business_value']['bank_product']['why']


def test_a_specific_writer_offer_is_kept(monkeypatch):
    offer = 'Рублёвый депозит на 3–6 месяцев под свободные остатки после пиков закупок'
    out = run({'insight': card('Ставка ЦБ сохранена', bank_product=offer)}, monkeypatch)
    assert out['insight']['bank_product'] == offer and 'bank_product' not in out['business_value']


def test_credibility_labels():
    assert credibility(['interfax'], {'interfax': 1})['label'].startswith('Интерфакс — деловое издание')
    assert credibility(['mk'], {'mk': 2})['level'] == 'low'
    assert credibility(['cbr', 'rbc'])['label'].startswith('Официальный источник: Банк России')


def test_unknown_outlets_and_legal_aggregators_do_not_become_official_or_business_sources():
    for source in ('unknown','consultant','garant'):
        value=credibility([source])
        assert not value['official'] and not value['business_only'] and value['level']=='low'
    assert credibility(['rbc','rbc',None,float('nan')])['n_outlets']==1


def test_saved_credibility_only_counts_known_members_of_the_same_event():
    row={'news_id':1,'evidence':{'news':[{'id':1,'source':'rbc'},{'id':2,'source':'interfax'},{'id':3,'source':'cbr'}]},
         'retrieval':{'event_news_ids':[1,2]}}
    out=from_saved(row)
    assert out['n_outlets']==2 and not out['official']
    assert 'Банк России' not in out['label']
    del row['retrieval']
    assert from_saved(row)['n_outlets']==1


def test_unrelated_or_stale_bank_evidence_does_not_raise_confidence(monkeypatch):
    row={'insight':card('ЦБ сохранил ставку',bank_view=[{'text':'Пауза','sources':['bank:quoted']}]),
         'credibility':credibility(['rbc','interfax'],{'rbc':1,'interfax':1}),
         'judge_bank':[{'id':'bank:other','stale':False}]}
    assert run(row,monkeypatch)['insight']['confidence']=='medium'
    row['judge_bank']=[{'id':'bank:quoted','stale':True}]
    assert run(row,monkeypatch)['insight']['confidence']=='medium'


def test_enrichment_is_idempotent_and_preserves_original_model_values_and_evidence(monkeypatch):
    original={'insight':card('ЦБ сохранил ставку',bank_product='кредитные продукты'),
              'evidence':{'company_id':'U_1','role':'cfo','facts':[{'id':'fact:1','text':'Frozen fact'}]}}
    first=run(original,monkeypatch);second=run(first,monkeypatch)
    assert first==second
    assert first['evidence'] is original['evidence']
    assert original['insight']['bank_product']=='кредитные продукты' and 'business_value' not in original
    assert second['business_value']['writer_confidence']=='medium'


def test_specific_short_offer_is_kept_and_unknown_profile_or_cash_shortage_do_not_trigger_a_deposit():
    assert not bv.vague('Рефинансирование долга')
    assert bv.bank_product(card('Ключевая ставка растёт'),{}) is None
    offer=bv.bank_product(card('Кассовый разрыв: нехватка ликвидности'),PROFILE)
    assert offer is None or 'Депозит' not in offer[0]


def test_profile_refresh_changes_recommendations_without_a_model_call(tmp_path,monkeypatch):
    import json
    import stage2_company_facts.profile.load as profiles
    path=tmp_path/'profiles.jsonl'
    monkeypatch.setattr(profiles,'PROFILES',path)
    bv._cached_profile.cache_clear()
    insight=card('Свободные остатки разместить на депозит')
    path.write_text(json.dumps({'id':'U_1','card':{'products':[]}})+'\n',encoding='utf-8')
    first=bv.enrich({'insight':insight},'U_1')
    assert 'Депозит' in first['insight']['bank_product']
    path.write_text(json.dumps({'id':'U_1','card':{'products':['deposit']}})+'\n',encoding='utf-8')
    second=bv.enrich(first,'U_1')
    assert second['insight']['bank_product']==''


def test_main_channel_wins_over_incidental_rate_mentions():
    insight=card('Свободные остатки: доходность при снижении ставки ЦБ',channel='ликвидность')
    offer=bv.bank_product(insight,PROFILE)
    assert offer and 'Депозит' in offer[0]
    fx=card('Курс доллара и стоимость кредита',channel='курс')
    assert bv.bank_product(fx,{'card':{'products':['loan']}}) is None


def test_automatic_offer_does_not_contradict_the_cards_recommendation():
    insight=card('Свободные остатки — в среднесрочные бумаги',channel='ликвидность',
                 ai_conclusion='Разместить в бумаги, а не на короткий депозит.')
    assert bv.bank_product(insight,PROFILE) is None
