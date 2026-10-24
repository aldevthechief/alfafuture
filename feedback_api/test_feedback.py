"""Offline test of the insight discussion: import, reactions, questions, ownership, idempotency and the personalization
the next pipeline run applies. Scripted model, temporary SQLite, no case data, no network.
python -X utf8 feedback_api/test_feedback.py"""
import sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common.llm.models as llm_models
llm_models.LANGFUSE = []
from fastapi.testclient import TestClient

from common.llm.testing import ScriptedChatModel, calls, tool_call as c
from feedback_api.analyzer import FeedbackAnalyzer
from feedback_api.app import create_app
from feedback_api.store import Store

NEWS = {101: {'id': 101, 'date': '2026-10-20', 'source': 'interfax', 'title': 'ЦБ сохранил ключевую ставку 16%',
              'text': 'Банк России сохранил ключевую ставку. Кредиты для бизнеса останутся дорогими.', 'url': 'https://x/101'},
        202: {'id': 202, 'date': '2026-10-19', 'source': 'kommersant', 'title': 'Биржевые цены на бензин выросли',
              'text': 'Цены на бензин и дизельное топливо на бирже выросли.', 'url': 'https://x/202'}}
FACTS = [{'cat': 'bank', 'text': 'Продукты Альфа-Банка: кредитные продукты', 'depth': 1},
         {'cat': 'scale', 'text': 'Годовая выручка 16.0 млрд ₽', 'depth': 1}]


def result(news_id, title, channel):
    claim = {'text': title, 'sources': [f'news:{news_id}'], 'kind': 'fact'}
    return {'news_id': news_id, 'status': 'ready', 'judge_bank': [
        {'id': 'bank:rate.pdf:p1.1', 'report': 'Ключевая ставка ЦБ', 'page': 1, 'date': '2026-09-11', 'stale': False,
         'text': 'Банк ожидает паузу по ставке до конца года.'}],
        'insight': {'title': title, 'role': 'cfo', 'channel': channel, 'what_happened': [claim],
                    'why_it_matters': [{'text': 'У компании кредиты', 'sources': ['fact:1'], 'kind': 'fact'}],
                    'trigger': {'news_ids': [news_id]}}}


RESULTS = [result(101, 'Ставка ЦБ сохранена: долг остаётся дорогим', 'rate'),
           result(202, 'Бензин дорожает: логистика подорожает', 'fuel'),
           {'news_id': 303, 'status': 'review_rejected', 'draft': {'title': 'черновик', 'role': 'cfo'}}]

answer = c('Analysis', intent='question', response='Пауза по ставке продлится до конца года (позиция банка).',
           requested_action='answer', source_ids=['news:101', 'bank:rate.pdf:p1.1', 'fact:1'])
bad = c('Analysis', intent='question', response='…', requested_action='answer', source_ids=['news:999'])
model = ScriptedChatModel(responses=[calls(bad), calls(answer),       # 1st question: unknown source -> corrected
                                     calls(bad), calls(bad), calls(bad), calls(bad)])  # 2nd: never valid -> 502
tmp = Path(tempfile.mkdtemp())
store = Store(tmp / 'feedback.sqlite3')
app = create_app(store=store, analyzer=FeedbackAnalyzer(model=model), api_token='',
                 news_lookup=NEWS.get, facts=FACTS)
H = {'X-User-ID': 'demo'}

with TestClient(app) as cl:
    assert cl.get('/health').json()['status'] == 'ok'
    assert cl.get('/v1/companies/U_1/insights', params={'role': 'cfo'}).status_code == 422, 'no X-User-ID'
    assert cl.get('/v1/companies/U_1/insights', params={'role': 'cfo'}, headers={'X-User-ID': 'bad id!'}).status_code == 422

    recs = cl.post('/v1/insights/import', json={'company_id': 'U_1', 'results': RESULTS}, headers=H).json()
    assert len(recs) == 2, 'drafts are not registered'
    rate, fuel = recs
    ev = rate['evidence']
    assert [n['id'] for n in ev['news']] == [101] and [f['id'] for f in ev['facts']] == ['fact:1', 'fact:2']
    assert ev['bank'][0]['id'] == 'bank:rate.pdf:p1.1'

    # reactions: answered by rules (no model), scope future by default
    like = cl.post(f'/v1/insights/{rate["id"]}/feedback', json={'kind': 'like', 'request_id': 'r1'}, headers=H).json()
    assert like['scope'] == 'future' and like['analysis']['requested_action'] == 'rerank'
    again = cl.post(f'/v1/insights/{rate["id"]}/feedback', json={'kind': 'like', 'request_id': 'r1'}, headers=H).json()
    assert again['id'] == like['id'], 'retry with the same request_id is idempotent'
    assert cl.post(f'/v1/insights/{rate["id"]}/feedback', json={'kind': 'dislike', 'request_id': 'r1'}, headers=H).status_code == 409
    emoji = cl.post(f'/v1/insights/{fuel["id"]}/feedback', json={'kind': 'emoji', 'emoji': '👎', 'request_id': 'r2'}, headers=H).json()
    assert emoji['kind'] == 'dislike', 'an unambiguous emoji counts as dislike'
    cl.post(f'/v1/insights/{fuel["id"]}/feedback', json={'kind': 'already_known', 'request_id': 'r3'}, headers=H)
    assert cl.post(f'/v1/insights/{fuel["id"]}/feedback', json={'kind': 'comment', 'request_id': 'r4'}, headers=H).status_code == 422
    assert cl.get(f'/v1/insights/{rate["id"]}/reactions', headers=H).json()['counts'] == {'like': 1}

    # a question: the first answer cites an unknown source and is sent back; the corrected one is stored
    q = cl.post(f'/v1/insights/{rate["id"]}/ask', json={'question': 'Долго ли ставка будет высокой?', 'request_id': 'q1'}, headers=H)
    assert q.status_code == 201, q.text
    qa = q.json()['analysis']
    assert qa['source_ids'] == ['news:101', 'bank:rate.pdf:p1.1', 'fact:1'] and q.json()['scope'] == 'insight'
    q2 = cl.post(f'/v1/insights/{rate["id"]}/ask', json={'question': 'А что с курсом?', 'request_id': 'q2'}, headers=H)
    assert q2.status_code == 502, 'an answer that never cites valid sources is not stored as an answer'
    assert len(cl.get(f'/v1/insights/{rate["id"]}/feedback', headers=H).json()) == 3, 'the question itself is saved'

    # ownership: another user sees nothing
    assert cl.get(f'/v1/insights/{rate["id"]}', headers={'X-User-ID': 'other'}).status_code == 404
    assert cl.post(f'/v1/insights/{rate["id"]}/feedback', json={'kind': 'like', 'request_id': 'x'},
                   headers={'X-User-ID': 'other'}).status_code == 404

    # what the next pipeline run applies
    p = cl.get('/v1/companies/U_1/personalization', params={'role': 'cfo'}, headers=H).json()
    # fuel: 👎 then «уже знаю» — the latest verdict wins: the news is excluded and the dislike is withdrawn
    assert p['excluded_news_ids'] == [202] and p['topic_weights']['rate'] > 1 and 'fuel' not in p['topic_weights'], p
    assert cl.get('/v1/companies/U_1/personalization', params={'role': 'cfo'},
                  headers={'X-User-ID': 'other'}).json()['topic_weights'] == {}

pers = store.personalization('demo', 'U_1', 'cfo')
rate_news = {'id': 501, 'title': 'ЦБ может повысить ключевую ставку', 'text': 'ставка кредиты заемщики'}
fuel_news = {'id': 502, 'title': 'Цены на дизельное топливо растут', 'text': 'бензин АЗС биржа'}
other = {'id': 503, 'title': 'Новый закон о маркировке товаров', 'text': ''}
f_rate, f_fuel, f_other = pers.factor(rate_news), pers.factor(fuel_news), pers.factor(other)
assert f_rate > 1 >= f_fuel and f_other == 1.0 and pers.factor({'id': 202, 'title': ''}) == 0.0, (f_rate, f_fuel, f_other)
assert 'не подтверждённые факты' in pers.render()

secured = create_app(store=store, analyzer=FeedbackAnalyzer(model=model), api_token='s3cret', news_lookup=NEWS.get, facts=FACTS)
with TestClient(secured) as cl:
    assert cl.get('/v1/companies/U_1/insights', params={'role': 'cfo'}, headers=H).status_code == 401
    assert cl.get('/v1/companies/U_1/insights', params={'role': 'cfo'},
                  headers=H | {'Authorization': 'Bearer s3cret'}).status_code == 200
print(f'ok: import 2/3, reactions + emoji, idempotency, grounded answer, invalid answer rejected, ownership, token; '
      f'factors rate {f_rate} / fuel {f_fuel} / other {f_other}')
