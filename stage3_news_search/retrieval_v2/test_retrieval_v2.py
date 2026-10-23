"""Offline checks of retrieval v2 (bag-of-words fake encoder, scripted factory agent, temp caches; no network, no data).
python -X utf8 stage3_news_search/retrieval_v2/test_retrieval_v2.py"""
import re, sys, tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.llm.testing import ScriptedChatModel, calls, tool_call as c
from stage3_news_search.retrieval_v2 import candidates as cand, factory, hybrid

tmp = Path(tempfile.mkdtemp())
hybrid.CACHE, factory.SPECS = tmp, tmp / 'factory'
factory.deterministic = lambda facts: 'Отрасль: торговля топливом'  # the real brief parses full stage-2 fact texts

VOCAB = 'топливо бензин дизель цены нефть ставка кредит лизинг китай пошлины москва регион погода футбол кино концерт'.split()
doc_calls = []


def encoder(slug, texts):
    if not texts[0].startswith('Instruct:'):
        doc_calls.append(len(texts))
    v = np.array([[t.lower().count(w[:5]) for w in VOCAB] + [1e-3] for t in texts], np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


hybrid.encode = encoder

titles = ['Цены на бензин и дизель выросли', 'Нефть подорожала, топливо дорожает', 'ЦБ сохранил ставка 14%',
          'Кредит для бизнеса подорожал', 'Лизинг грузовиков: ставки выросли', 'Китай ввёл пошлины на топливо',
          'Москва: регион увеличил бюджет', 'Футбол: итоги тура', 'Кино недели', 'Концерт в Москве перенесли',
          'Погода: заморозки', 'Дизель: биржевые цены рекорд']
news = pd.DataFrame({'id': range(12), 'title': titles, 'text': titles})
events = pd.DataFrame({'event': [100 + i for i in range(11)], 'news_ids': [[0, 11]] + [[i] for i in range(1, 11)],
                       'title': titles[:11], 'first_date': ['2026-10-01'] * 11, 'n_sources': [2] + [1] * 10})
facts = [
    {'cat': 'activity', 'text': 'Основной вид деятельности (ОКВЭД 46.71.2): Торговля оптовая моторным топливом',
     'entities': ['46.71.2', 'Торговля оптовая моторным топливом']},
    {'cat': 'bank', 'text': 'Продукты Альфа-Банка: кредитные продукты, лизинг', 'entities': ['кредитные продукты', 'лизинг']},
]
script = [
    calls(c('note_business_model', products='топливо', buyers='АЗС', costs_and_suppliers='НПЗ', regulation='пошлины',
            ownership_and_group='—', hypotheses='—')),
    calls(c('probe', instruction='Найди новости о ценах на топливо', queries=['цены на бензин и дизель выросли'])),
    calls(c('probe', instruction='Найди новости о пошлинах', queries=['Китай пошлины на топливо'])),
    calls(c('finish', selected=[{'probe_id': 1, 'channel': 'C2', 'reason': 'fuel_duty', 'fact': 'отрасль 46.71'},
                                {'probe_id': 9, 'channel': 'B1', 'reason': 'bad', 'fact': '—'}],
            coverage=[{'channel': 'B1', 'probe_ids': [0]}])),
]

res, gens = cand.candidates('U_TEST', facts, 'cfo', news=news, events=events, model=ScriptedChatModel(responses=script),
                            budget=4, quota=3)
by_src = {s: [g for g in gens if g['source'] == s] for s in ('standard', 'factory')}
assert len(by_src['factory']) == 1 and by_src['factory'][0]['reason'] == 'fuel_duty', 'invalid probe ids are dropped'
assert res.priority.is_monotonic_decreasing and res.event[0] == 105, 'the event found first by the factory leads'
assert set(res.columns) >= {'event', 'news_ids', 'n_sources', 'generators', 'best_rank', 'priority'}, 'stage-3 event contract'
assert 105 in set(res.event), 'factory generator found the duty event'
assert {x['source'] for xs in res.generators for x in xs} == {'standard', 'factory'}
assert res.event.is_unique and all(isinstance(x['rank'], int) for xs in res.generators for x in xs)
first_docs = sum(doc_calls)
assert first_docs == 2 * len(news), 'every news encoded once per model'

# same input again: the cached factory spec is used, no model at all
spec = factory.factory_spec('U_TEST', facts, cand.hybrid_search(news, events), ScriptedChatModel(responses=script))
assert spec['input']['pool'] == cand.hybrid_search(news, events).fingerprint and spec['selected'][0]['reason'] == 'fuel_duty'
cached = factory.factory_spec('U_TEST', facts, cand.hybrid_search(news, events), ScriptedChatModel(responses=[]))
assert cached == spec, 'identical input -> cache hit (an empty scripted model would fail if called)'

# strict key: changed facts or a changed pool -> the agent runs again
facts_b = facts + [{'cat': 'geo', 'text': 'Зарегистрирована: г. Омск, Омская область', 'entities': ['г. Омск', 'Омская область']}]
try:
    factory.factory_spec('U_TEST', facts_b, cand.hybrid_search(news, events), ScriptedChatModel(responses=[]))
    raise AssertionError('changed facts must not hit the cache')
except IndexError:  # the empty scripted model was called
    pass

# id migration + one new news: vectors reused by text; frozen mode keeps the stored spec without running the agent
cand._SEARCH.clear()
news2 = pd.concat([news.assign(id=news.id + 1000), pd.DataFrame({'id': [2000], 'title': ['Китай повысил пошлины на топливо'], 'text': ['пошлины']})])
events2 = pd.concat([events.assign(news_ids=[[i + 1000 for i in ids] for ids in events.news_ids]),
                     pd.DataFrame({'event': [200], 'news_ids': [[2000]], 'title': ['Китай повысил пошлины на топливо'],
                                   'first_date': ['2026-10-02'], 'n_sources': [1]})])
assert cand.hybrid_search(news2, events2).fingerprint != spec['input']['pool'], 'a new pool changes the fingerprint'
res2, _ = cand.candidates('U_TEST', facts, 'cfo', news=news2, events=events2, budget=4, quota=5, frozen_factory=True)
assert sum(doc_calls) - first_docs == 2, 'only the new text is encoded (once per model): vectors are keyed by text'
assert 200 in set(res2.event), 'new news are searched with the stored generators'
print(f'ok: {len(gens)} generators -> {len(res)} events; factory spec and text-hash vectors reused after an id change')
