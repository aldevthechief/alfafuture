"""«Если …» in the cards -> questions about the company -> answers -> facts of the next run. Scripted, no case data."""
import json

from fastapi.testclient import TestClient

from common.personalization import Personalization, ProfileAnswer
from feedback_api.app import create_app
from feedback_api.store import Store
from stage2_company_facts.questions.questions import questions
from feedback_api.profile import latest_results


def card(title, why=(), hypotheses=()):
    return {'insight': {'title': title, 'role': 'cfo', 'why_it_matters': [{'text': t, 'sources': []} for t in why],
                        'hypotheses': [{'text': t, 'sources': []} for t in hypotheses]}}


RESULTS = [
    card('Ставка 14% до конца года', ['Если часть долга компании плавающая, процентные расходы вырастут.'],
         ['Больше половины кредитного портфеля компании — под плавающую ставку. Подтвердят кредитные договоры.']),
    card('Облигационное окно сужается', ['Если часть долга компании плавающая, рефинансирование подорожает.']),
    card('Утильсбор растёт', ['Если автотехника — предмет лизинга, активы подорожают.']),  # cut clause: dropped
    {'status': 'review_rejected', 'draft': {'title': 'черновик'}},
]


def test_assumptions_become_ranked_questions():
    qs = questions(RESULTS)
    assert qs[0]['question'] == 'Верно ли, что часть долга компании плавающая?' and len(qs[0]['insights']) == 2
    assert all(q['question'].startswith('Верно ли, что ') and q['question'].endswith('?') for q in qs)
    assert not any('Подтвердят' in q['question'] for q in qs), 'the hypothesis keeps its claim, not the check'
    assert questions(RESULTS)[0]['id'] == qs[0]['id'], 'stable id: the answer stays attached across analyses'


def test_answers_flow_into_personalization_and_facts(tmp_path):
    store = Store(tmp_path / 'fb.sqlite3')
    app = create_app(store=store, api_token='', results_loader=lambda company, role, user: RESULTS)
    h = {'X-User-ID': 'demo'}
    with TestClient(app) as cl:
        qs = cl.get('/v1/companies/U_1/questions', params={'role': 'cfo'}, headers=h).json()
        q = qs[0]
        assert q['answer'] is None
        r = cl.put(f'/v1/companies/U_1/answers/{q["id"]}', headers=h,
                   json={'statement': q['text'], 'verdict': 'yes'})
        assert r.status_code == 200, r.text
        assert cl.put('/v1/companies/U_1/answers/bad', headers=h, json={'statement': q['text'], 'verdict': 'yes'}).status_code == 422
        assert cl.put(f'/v1/companies/U_1/answers/{q["id"]}', headers=h,
                      json={'statement': q['text'], 'verdict': 'other'}).status_code == 422, 'other needs a text'
        again = cl.get('/v1/companies/U_1/questions', params={'role': 'ceo'}, headers=h).json()
        assert again[0]['answer']['verdict'] == 'yes', 'an answer about the company holds for every role'
        assert cl.get('/v1/companies/U_1/questions', params={'role': 'cfo'},
                      headers={'X-User-ID': 'other'}).json()[0]['answer'] is None, 'answers are per user'
        p = Personalization.model_validate(cl.get('/v1/companies/U_1/personalization', params={'role': 'cfo'}, headers=h).json())
        assert not p.empty() and p.answers[0].verdict == 'yes'
        fact = p.answer_facts()[0]
        assert fact['cat'] == 'user_profile' and fact['kind'] == 'user' and fact['depth'] == 1
        assert fact['text'] == 'Со слов пользователя (не проверено банком): верно, что часть долга компании плавающая'
        assert 'верно, что часть долга' in p.render()
        assert cl.delete(f'/v1/companies/U_1/answers/{q["id"]}', headers=h).status_code == 204
        assert cl.get('/v1/companies/U_1/questions', params={'role': 'cfo'}, headers=h).json()[0]['answer'] is None


def test_other_answer_text():
    a = ProfileAnswer(question_id='q_0123456789ab', statement='топливо составляет более 20% затрат', verdict='other',
                      text='около 35%, своя АЗС')
    assert a.fact().endswith('на вопрос «топливо составляет более 20% затрат» — около 35%, своя АЗС')


def test_questions_use_cached_cabinet_rows_and_do_not_read_another_users_jobs(tmp_path):
    saved = tmp_path / 'analyses'
    saved.mkdir()
    for identifier, user, role in [('mine', 'alice', 'cfo'), ('shared', None, 'cfo'), ('private', 'bob', 'cfo'), ('role', 'alice', 'ceo')]:
        row = dict(status='ready', **card(identifier, hypotheses=[f'Компания использует плавающую ставку {identifier}']))
        row['insight']['id'] = identifier
        job = dict(id=identifier, status='succeeded', created_at='2026-10-25', finished_at='2026-10-25',
                   request=dict(company_id='U_1', role=role, user_id=user), result=dict(results=[row]))
        (saved / f'{identifier}.json').write_text(json.dumps(job), encoding='utf-8')
    rows = latest_results('U_1', 'cfo', 'alice', tmp_path)
    assert {r['insight']['title'] for r in rows} == {'mine', 'shared'}
    assert latest_results('U_2', 'cfo', 'alice', tmp_path) == []
    # Prepared cabinets have questions without a new API analysis or any model calls.
    assert questions(latest_results('U_14366', 'ceo', 'alice', tmp_path))


def test_saved_answers_survive_new_analyses_and_restart_and_are_isolated_by_company(tmp_path, monkeypatch):
    from unittest.mock import Mock
    monkeypatch.setenv('CABINET_MODE', 'full')
    rows = list(RESULTS)
    store = Store(tmp_path / 'answers.sqlite')
    model = Mock(side_effect=AssertionError('Profile answers must not invoke a model'))
    app = create_app(store=store, analyzer=model, api_token='', results_loader=lambda company, role, user: rows)
    headers = {'X-User-ID':'alice'}
    with TestClient(app) as client:
        q = client.get('/v1/companies/U_1/questions?role=cfo', headers=headers).json()[0]
        path = f'/v1/companies/U_1/answers/{q["id"]}'
        assert client.put(path, headers=headers, json={'statement':q['text'], 'verdict':'no'}).status_code == 200
        rows.clear()
        again = client.get('/v1/companies/U_1/questions?role=ceo', headers=headers).json()
        assert len(again) == 1 and again[0]['answer']['verdict'] == 'no'
        assert client.get('/v1/companies/U_2/questions?role=ceo', headers=headers).json() == []
        assert client.get('/v1/companies/U_1/questions?role=ceo', headers={'X-User-ID':'bob'}).json() == []
        assert client.get('/v1/companies/U_1/questions?role=unknown', headers=headers).status_code == 422
        preflight = client.options(path, headers={'Origin':'http://localhost:5173', 'Access-Control-Request-Method':'PUT',
                                                  'Access-Control-Request-Headers':'Content-Type,X-User-ID'})
        assert preflight.status_code == 200
        monkeypatch.setenv('CABINET_MODE', 'server')
        assert client.put(path, headers=headers, json={'statement':q['text'], 'verdict':'yes'}).status_code == 403
        assert client.delete(path, headers=headers).status_code == 403
    with TestClient(create_app(store=Store(store.path), api_token='', results_loader=lambda *args: [])) as client:
        assert client.get('/v1/companies/U_1/questions?role=cfo', headers=headers).json()[0]['answer']['verdict'] == 'no'
    model.assert_not_called()
