"""Dedup gate: groups come from one call; unknown ids, repeats and singletons are dropped; < 2 candidates — no call."""
import asyncio

from common.llm.testing import ScriptedChatModel, calls, tool_call
from stage5_insight_detection.insight_dedup.dedup import DedupGate, render

NEWS = {'id': 'a', 'kind': 'news', 'title': 't', 'news_titles': ['ДОМ.РФ разместил облигации'], 'company_link': 'лизинг',
        'mechanism': 'фондирование'}
ITEMS = [NEWS, {**NEWS, 'id': 'b'}, {**NEWS, 'id': 'c'},
         {'id': 'd', 'kind': 'analytics', 'title': 'Ставка 14%', 'thesis': 'долг не подешевеет'}]


def test_groups_are_cleaned():
    model = ScriptedChatModel(responses=[calls(tool_call('Groups', groups=[
        {'ids': ['a', 'b', 'zz'], 'common': 'одно событие'}, {'ids': ['b', 'c'], 'common': 'b уже в группе'},
        {'ids': ['d'], 'common': 'одиночка'}]))])
    assert asyncio.run(DedupGate(model).agroups(ITEMS, 'cfo')) == [['a', 'b']]


def test_single_candidate_makes_no_call():
    assert asyncio.run(DedupGate(ScriptedChatModel(responses=[])).agroups(ITEMS[:1], 'cfo')) == []


def test_render_shows_what_the_gate_judges_by():
    assert 'новости: ДОМ.РФ разместил облигации' in render(NEWS) and 'механизм: фондирование' in render(NEWS)
    assert render(ITEMS[3]).startswith('[d] аналитика Банка: Ставка 14%')


class FakeGate:
    def __init__(self, groups):
        self.result, self.seen = groups, None

    def groups(self, items, role):  # the merge node calls the gate synchronously
        self.seen = items
        return self.result


def test_analytics_duplicates_of_news_and_of_each_other_are_dropped():
    from types import SimpleNamespace as NS
    from pipeline import analytics_dedup
    claim = NS(text='тезис')
    rate, fx, fx2 = (NS(title=t, why_it_matters=[claim], role_fit=f, impact=None, confidence='medium')
                     for t, f in (('Ставка 14%', 3), ('Курс 87', 2), ('Курс 93', 3)))
    news = [{'status': 'ready', 'insight': {}, 'title': 'ЦБ сохранил ставку', 'decision': {'company_link': 'кредит', 'mechanism': 'ставка'},
             'news_references': {'news:1': {'title': 'ЦБ сохранил ставку'}}}]
    gate = FakeGate([['n0', 'a0'], ['a1', 'a2']])
    assert analytics_dedup([rate, fx, fx2], news, gate, 'cfo') == [fx2]  # rate repeats the news; fx2 ranks above fx
    assert [i['kind'] for i in gate.seen] == ['news', 'analytics', 'analytics', 'analytics']
