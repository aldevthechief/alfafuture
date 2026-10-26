from backend.top5 import select

N = lambda i: dict(group='news', event_id=None, news=[str(i)])
B = lambda i: dict(group='bank', event_id=None, news=[])
NO_SIM = lambda n: [[0] * n for _ in range(n)]


def test_quota_is_soft_and_order_is_the_llm_one():
    cards = [B(0), N(1), N(2), B(3), N(4), N(5), N(6)]
    chosen, _ = select(cards, [0, 1, 2, 3, 4, 5, 6], NO_SIM(7))
    assert chosen == [0, 1, 2, 3, 4]  # bank cards stay where the LLM put them: not grouped, not pushed to the end


def test_missing_group_is_filled_by_the_other():
    cards = [N(i) for i in range(7)]
    assert select(cards, list(range(7)), NO_SIM(7))[0] == [0, 1, 2, 3, 4]


def test_unranked_bank_card_does_not_take_a_quota_slot():
    cards = [N(i) for i in range(6)] + [B(6)]  # the bank card is outside the LLM top-10
    assert select(cards, list(range(6)), NO_SIM(7))[0] == [0, 1, 2, 3, 4]


def test_duplicate_is_skipped_and_next_card_comes_in():
    cards = [N(0), N(1), N(2), N(3), N(4), N(5)]
    sim = NO_SIM(6)
    sim[1][0] = sim[0][1] = 0.5
    chosen, skipped = select(cards, list(range(6)), sim)
    assert chosen == [0, 2, 3, 4, 5] and skipped[0][:2] == (1, 0)


def test_expired_cards_do_not_take_top_seats_and_expiring_ones_go_last(monkeypatch):
    monkeypatch.setenv('FEED_NOW', '2026-10-26')
    fresh = lambda i: dict(N(i), published='2026-10-20')
    expiring = dict(N(10), published='2026-10-01')  # 25 days: the last week
    stale = dict(N(11), published='2026-09-01')
    cards = [stale, expiring, *[fresh(i) for i in range(5)]]
    assert select(cards, list(range(7)), NO_SIM(7))[0] == [2, 3, 4, 5, 6]  # live cards fill the top-5, the expired one stays out
    few = [stale, expiring, fresh(2), fresh(3), fresh(4)]
    assert select(few, list(range(5)), NO_SIM(5))[0] == [2, 3, 4, 1, 0]  # nothing else left: expiring, then expired, last


def test_six_to_nine_cards_request_an_achievable_schema(monkeypatch):
    from types import SimpleNamespace
    from backend.top5 import llm_order
    import common.llm.models as models
    def structured(model,schema):
        count=schema.model_json_schema()['properties']['top']['minItems']
        return SimpleNamespace(invoke=lambda prompt:schema(top=[{'id':f'C{i:02}'} for i in range(count)]))
    monkeypatch.setattr(models,'structured',structured)
    factory=lambda:SimpleNamespace(_make=lambda *a:SimpleNamespace(model_copy=lambda **kw:None))
    for n in range(6,10):
        assert llm_order('U_1','cfo',[{'L3':'text'}]*n,factory)==list(range(n))


def test_read_only_rank_and_corrupt_cache_keep_all_cards_without_models(tmp_path,monkeypatch):
    from backend import top5
    cards=[dict(N(i),id=str(i),L2=f'text {i}',L3='text',rel=2,impact='high') for i in range(7)]
    monkeypatch.setattr(top5,'similarity',lambda cards:NO_SIM(len(cards)))
    calls=[]
    def ranked(*args):calls.append(args);return list(range(7))
    monkeypatch.setattr(top5,'llm_order',ranked)
    assert set(top5.rank('U_1','cfo',cards,tmp_path,cache_only=True)['order'])==set(map(str,range(7)))
    assert not calls
    top5.rank('U_1','cfo',cards,tmp_path)
    assert len(calls)==1
    cache=next((tmp_path/'top5').glob('*.json'));cache.write_text('{broken')
    assert len(top5.rank('U_1','cfo',cards,tmp_path,cache_only=True)['order'])==7
    assert len(calls)==1
    cards[0]['event_id']='new event'
    top5.rank('U_1','cfo',cards,tmp_path)
    assert len(calls)==2
