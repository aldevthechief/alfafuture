"""Offline checks of generator building and per-generator selection (no encoder, no news).
python -X utf8 stage3_news_search/candidate_generators/test_generators.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stage3_news_search.candidate_generators.generators import generators, select

facts = [
    {'cat': 'activity', 'text': 'Основной вид деятельности (ОКВЭД 46.71.2): Торговля оптовая моторным топливом', 'entities': ['46.71.2', 'Торговля оптовая моторным топливом']},
    {'cat': 'payments_out', 'text': 'Исходящие платежи — 8 компаний с ОКВЭД 46.71 «Торговля оптовая топливом»: 1.9 млрд ₽ (33% суммы)', 'entities': ['46.71', 'Торговля оптовая топливом']},
    {'cat': 'payments_out', 'text': 'Исходящие платежи — 1 компания с ОКВЭД 62.01 «Разработка ПО»: 264 тыс. ₽ (<1% суммы)', 'entities': ['62.01', 'Разработка ПО']},
    {'cat': 'payments_in', 'text': 'Входящие платежи — контрагенты из региона Москва: 2.1 млрд ₽ (19% суммы)', 'entities': ['Москва']},
    {'cat': 'payments_in', 'text': 'Входящие платежи — среди контрагентов участники ВЭД со связями со странами: Китай (2), ОАЭ (1)', 'entities': ['Китай', 'ОАЭ']},
    {'cat': 'geo', 'text': 'Зарегистрирована: г. Омск, Омская область', 'entities': ['г. Омск', 'Омская область']},
    {'cat': 'bank', 'text': 'Продукты Альфа-Банка: расчётно-кассовое обслуживание, кредитные продукты, лизинг', 'entities': ['расчётно-кассовое обслуживание', 'кредитные продукты', 'лизинг']},
]
gens = generators(facts, hypotheses=['межрегиональный топливный трейдер без своих НПЗ'], role='cfo')
by = {g.reason: [x for x in gens if x.reason == g.reason] for g in gens}
assert len(by['supplier_industry']) == 1, 'counterparty industries below 5% of the flow are not generators'
assert {g.query.split()[0] for g in by['region']} == {'Москва', 'Омская'}
assert len(by['trade_country']) == 2 and len(by['hypothesis']) == 1 and len(by['bank_product']) == 2
assert abs(sum(g.weight for g in by['trade_country']) - 0.4) < 1e-9, 'a reason budget is split among its generators'
assert all(g.facts for g in gens), 'every generator keeps the facts it came from'

scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
sims = np.array([0.9, 0.2, 0.7, 0.1, 0.6])
assert list(select(scores, sims, lo=2, hi=4, threshold=0.5)) == [0, 1, 2], 'first lo always, then only above threshold, at most hi'
print(f'ok: {len(gens)} generators from 7 facts; quotas and threshold selection')
