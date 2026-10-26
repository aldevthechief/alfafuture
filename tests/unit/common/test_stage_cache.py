import asyncio

from common.llm.cache import JsonlCache
from common.stage_cache import StageCache, digest
from pipeline import InsightPipeline
from stage4_news_relevance.relevance_scoring.relevance import Relevance
from stage5_insight_detection.insight_judge.judge import InsightDecision

NEWS = dict(id=1, title='Ключевая ставка 14%', text='ЦБ сохранил ставку', url='http://x/1', date='2026-10-20', source='cbr')


class Counting:
    def __init__(self, value):
        self.value, self.calls = value, 0

    async def ascore(self, *_):
        self.calls += 1
        return self.value

    async def adecide(self, *_):
        self.calls += 1
        return self.value


def pipeline(tmp_path):
    p = object.__new__(InsightPipeline)  # no models, no corpus: only the cache logic
    p._stage_cache_on, p._bank_signature, p.bank, p.related = True, None, None, None
    p.stages = {n: StageCache(n, JsonlCache(f'stage_{n}', tmp_path)) for n in ('relevance', 'judge', 'writer')}
    p._version = {n: 'v1' for n in p.stages}
    p.relevance = Counting(Relevance(score=2, link='ставка', channel='ставка', direction='-', role_fit=2))
    decision = InsightDecision(is_insight=True, score=2, company_link='кредиты', mechanism='ставка выше')
    decision.calculation_facts, decision.bank_evidence = [{'id': 'calc:1'}], [{'id': 'bank:r:p1'}]
    p.judge = Counting(decision)
    p._params = dict(company_id='U_1', role='cfo', facts=[{'cat': 'a', 'text': 'долг 1 млрд'}], key_facts=['[a] долг 1 млрд'])
    return p


def test_same_inputs_are_not_recomputed_and_tool_fields_survive(tmp_path):
    p = pipeline(tmp_path)
    for _ in range(2):
        rel = asyncio.run(p._relevance(NEWS, 'долг', 'cfo'))
        decision = asyncio.run(p._judge(NEWS, {'matched_fragments': ['f']}))
    assert (p.relevance.calls, p.judge.calls) == (1, 1) and rel.channel == 'ставка'
    assert decision.calculation_facts == [{'id': 'calc:1'}] and decision.bank_evidence == [{'id': 'bank:r:p1'}]


def test_changed_inputs_recompute_only_what_they_touch(tmp_path):
    p = pipeline(tmp_path)
    asyncio.run(p._relevance(NEWS, 'долг', 'cfo')); asyncio.run(p._judge(NEWS, {}))
    p._params['role'] = 'ceo'                                  # another reader: both stages again
    asyncio.run(p._relevance(NEWS, 'долг', 'ceo')); asyncio.run(p._judge(NEWS, {}))
    p._params['role'] = 'cfo'
    p._params['facts'] = [{'cat': 'a', 'text': 'долг 2 млрд'}]  # company facts changed: the judge only, relevance reads key facts text
    asyncio.run(p._relevance(NEWS, 'долг', 'cfo')); asyncio.run(p._judge(NEWS, {}))
    assert (p.relevance.calls, p.judge.calls) == (2, 3)


def test_digest_ignores_dict_order():
    assert digest({'a': 1, 'b': 2}) == digest({'b': 2, 'a': 1}) != digest({'a': 1, 'b': 3})


def test_changed_consequences_invalidate_judge_without_recomputing_relevance(tmp_path):
    p=pipeline(tmp_path)
    for outcome in ('demand grows','demand shrinks','demand shrinks'):
        asyncio.run(p._relevance(NEWS,'debt','cfo'))
        asyncio.run(p._judge(NEWS,{'consequences':outcome}))
    assert (p.relevance.calls,p.judge.calls)==(1,2)


def test_model_settings_and_bank_page_ids_change_cache_identity(tmp_path):
    from types import SimpleNamespace
    from common.stage_cache import model_id
    a=SimpleNamespace(model_name='model',temperature=0.1)
    b=SimpleNamespace(model_name='model',temperature=0.7)
    assert model_id(a)!=model_id(b)
    p=pipeline(tmp_path)
    for page in ('bank:r:p1','bank:r:p2','bank:r:p2'):
        p.bank=SimpleNamespace(chunks=[{'id':page,'text':'same text'}])
        p._bank_signature=None
        asyncio.run(p._judge(NEWS,{}))
    assert p.judge.calls==2
