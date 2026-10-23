"""Offline checks of retrieval v3 (bag-of-words fake encoder, no factory, temp caches; no network, no data).
python -X utf8 stage3_news_search/retrieval_v3/test_retrieval_v3.py"""
import re, sys, tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stage3_news_search.retrieval_v2 import candidates as v2, hybrid
from stage3_news_search.retrieval_v3 import candidates as v3, channels

hybrid.CACHE = Path(tempfile.mkdtemp())
VOCAB = 'топливо бензин дизель цены нефть ставка кредит лизинг пошлины москва регион погода футбол кино концерт'.split()


def encoder(slug, texts):
    v = np.array([[t.lower().count(w[:5]) for w in VOCAB] + [1e-3] for t in texts], np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


hybrid.encode = encoder
titles = ['Цены на бензин и дизель выросли', 'Нефть подорожала, топливо дорожает', 'ЦБ сохранил ставка 14%',
          'Кредит для бизнеса подорожал', 'Лизинг грузовиков: ставки выросли', 'Ввели пошлины на топливо',
          'Москва: регион увеличил бюджет', 'Футбол: итоги тура', 'Кино недели', 'Концерт в Москве перенесли',
          'Погода: заморозки', 'Дизель: биржевые цены рекорд']
news = pd.DataFrame({'id': range(12), 'title': titles, 'text': titles})
events = pd.DataFrame({'event': [100 + i for i in range(11)], 'news_ids': [[0, 11]] + [[i] for i in range(1, 11)],
                       'title': titles[:11], 'first_date': ['2026-10-01'] * 11, 'n_sources': [2] + [1] * 10})
facts = [
    {'cat': 'activity', 'text': 'Основной вид деятельности (ОКВЭД 46.71.2): Торговля оптовая моторным топливом',
     'entities': ['46.71.2', 'Торговля оптовая моторным топливом']},
    {'cat': 'geo', 'text': 'Зарегистрирована: г. Москва, Москва', 'entities': ['г. Москва', 'Москва']},
    {'cat': 'bank', 'text': 'Продукты Альфа-Банка: кредитные продукты, лизинг', 'entities': ['кредитные продукты', 'лизинг']},
    {'cat': 'payments_in', 'text': 'Входящие платежи: 10 контрагентов, 1 млрд ₽, 50 платежей; это <1% годовой выручки', 'entities': []},
    {'cat': 'payments_in', 'text': 'Входящие платежи — 2 компании с ОКВЭД 47.30 «Торговля розничная моторным топливом»: 1 млн ₽ (90% суммы)',
     'entities': ['47.30', 'Торговля розничная моторным топливом']},
]

# NEUTRAL: example headlines never come from the evaluation companies' sectors (packaging, electrical / grids,
# construction, restaurants, marketplaces); leaf texts may name mechanisms and commodities, so they are not checked here
EVAL_SECTORS = re.compile(r'электр|кабел|подстанц|сетев|упаков|картон|гофр|бумаг|ресторан|кафе|общепит|доставк|маркетплейс|'
                          r'строител|стройк|стройматер|арматур|застройщ', re.I)
leaks = [(k, x) for k, xs in channels.EXAMPLES.items() for x in xs if EVAL_SECTORS.search(x)]
assert not leaks, f'example headlines from the evaluation sectors: {leaks}'

fs = channels.facets(facts)
names = [f['name'] for f in fs]
assert all(isinstance(f['spec'][1], list) and all(isinstance(q, str) and len(q) > 1 for q in f['spec'][1]) for f in fs if f['spec']), \
    'a spec is (instruction, [queries]): a bare string would be searched character by character'
assert names[0] == 'own_industry' and {'region', 'credit', 'taxes_and_rules'} <= set(names)
assert 'customer_industry' not in names, 'a counterparty link in a flow < 5% of revenue is no facet'
assert all(p in set(channels.PARENT.values()) for f in fs for p in f['parents']) and 'X0' not in {p for f in fs for p in f['parents']}

r2, _ = v2.candidates('U_TEST', facts, 'cfo', news=news, events=events, budget=4, use_factory=False)
r3, gens = v3.candidates('U_TEST', facts, 'cfo', news=news, events=events, budget=4, use_factory=False, channel_budget=4)
assert set(r2.event) <= set(r3.event), 'v3 keeps the whole v2 pool'
assert {g['source'] for g in gens} == {'standard', 'channel'} and r3.event.is_unique and r3.priority.is_monotonic_decreasing
assert set(r3.columns) >= {'event', 'news_ids', 'n_sources', 'generators', 'best_rank', 'priority'}, 'stage-3 event contract'
credit = next(g for g in gens if g['reason'] == 'credit')
assert credit['top'][0] in (102, 103, 104), 'the credit facet ranks rate / credit / leasing news first'
assert any(x['source'] == 'channel' for x in r3.generators[r3.event == credit['top'][0]].iloc[0]), 'its event carries the channel pick'
no_ch = v2.order(r3.drop(columns=['generators', 'best_rank', 'priority']).set_index('event'), dict(zip(r3.event, r3.generators)), gens,
                 {'factory': 0.7, 'standard': 0.3, 'channel': 0.0})
v2_head = list(r2.event)
assert list(no_ch.event)[:len(v2_head)] == v2_head, 'with a zero channel share the v2 events keep the v2 order'
print(f'ok: {len(gens)} generators -> v2 {len(r2)} / v3 {len(r3)} events; facets {names}')
