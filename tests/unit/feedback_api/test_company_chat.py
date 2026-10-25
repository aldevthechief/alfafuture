"""The shared answer engine with durable company/role/user isolation and per-answer source snapshots."""
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from common.evidence import EvidenceSnapshot, FactEvidence
from common.llm.testing import ScriptedChatModel, calls, tool_call
from common.personalization import ProfileAnswer
from feedback_api.analyzer import FeedbackAnalyzer
from feedback_api.app import create_app
from feedback_api.company_chat import load_company_context
from feedback_api.research import ResearchTools
from feedback_api.schemas import CompanyChatContext, FeedbackInput, RegisterInsight
from feedback_api.store import Store


class Model(ScriptedChatModel):
    runs: int = 0
    seen: list = []

    def _stream(self, messages, **kwargs):
        self.runs += 1
        self.seen.append(messages)
        args = json.dumps(self.responses[0].tool_calls[0]['args'], ensure_ascii=False)
        for index, offset in enumerate(range(0, len(args), 7)):
            yield ChatGenerationChunk(message=AIMessageChunk(content='', tool_call_chunks=[dict(
                name='Analysis' if index == 0 else None, id='answer' if index == 0 else None,
                index=0, args=args[offset:offset+7])]))


class EmptyResearch:
    def __init__(self, conversation):
        self.company_id, self.sources, self.trace = conversation.company_id, {}, []

    def as_tools(self):
        return []


def events(response):
    return [(frame.splitlines()[0][7:], json.loads(frame.splitlines()[1][6:]))
            for frame in response.text.split('\n\n') if frame.startswith('event: ')]


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setenv('CABINET_MODE', 'full')
    monkeypatch.setenv('ALFA_CACHE_REDIS_URL', '')
    store = Store(tmp_path/'chat.sqlite')
    store.initialize()
    model = Model(responses=[calls(tool_call('Analysis', intent='question', response='Опираемся на факт (fact:1).',
                                           requested_action='answer', source_ids=['fact:1']))])
    version = ['original']

    def context(db, user_id, company_id, role):
        return CompanyChatContext(card={'sector':f'{company_id} profile'}, evidence=EvidenceSnapshot(
            company_id=company_id, role=role, facts=[FactEvidence(id='fact:1', text=f'{company_id}/{user_id}/{version[0]}')]))

    app = create_app(store=store, analyzer=FeedbackAnalyzer(model=model, max_steps=1), api_token='',
                     company_context_loader=context, company_research_factory=EmptyResearch)
    with TestClient(app) as client:
        yield client, model, store, version


def opened(client, company='U_1', role='cfo', user='alice'):
    response = client.post('/v1/company-chats', json={'company_id':company,'role':role}, headers={'X-User-ID':user})
    assert response.status_code == 200, response.text
    return '/v1/company-chats/' + response.json()['id']


def test_same_engine_streams_but_company_history_never_contains_insight_other_role_company_or_user(fixture):
    client, model, store, version = fixture
    registered = store.register('alice', RegisterInsight(company_id='U_1', insight={'title':'INSIGHT_SECRET', 'role':'cfo'}))
    store.add('alice', registered.id, FeedbackInput(kind='question', comment='PRIVATE_INSIGHT_QUESTION', request_id='insight-question'))
    headers = {'X-User-ID':'alice'}
    base = opened(client)
    assert opened(client) == base
    assert client.get(base+'/messages', headers=headers).json()['items'] == []
    assert model.runs == 0
    payload = {'question':'COMPANY_FIRST_QUESTION', 'request_id':'first'}
    first = client.post(base+'/ask/stream', headers=headers, json=payload)
    assert first.status_code == 200
    frames = events(first)
    assert len([event for event in frames if event[0] == 'text']) > 2
    saved = frames[-1][1]
    assert frames[-1][0] == 'complete'
    assert saved['evidence']['facts'][0]['text'] == 'U_1/alice/original'
    assert 'context' not in saved
    assert 'INSIGHT_SECRET' not in model.seen[-1][-1].content
    assert 'PRIVATE_INSIGHT_QUESTION' not in model.seen[-1][-1].content
    assert events(client.post(base+'/ask/stream', headers=headers, json=payload)) == [('complete',saved)]
    assert model.runs == 1
    assert client.post(base+'/ask/stream', headers=headers,json={**payload,'question':'changed'}).status_code == 409
    version[0] = 'updated'
    followup = client.post(base+'/ask/stream', headers=headers,
                          json={'question':'SECOND_QUESTION','request_id':'second','selected_text':'Опираемся на факт'})
    assert events(followup)[-1][0] == 'complete'
    assert 'COMPANY_FIRST_QUESTION' in model.seen[-1][-1].content
    assert 'selected_answer_excerpt' in model.seen[-1][-1].content
    history = client.get(base+'/messages',headers=headers).json()['items']
    assert [item['evidence']['facts'][0]['text'] for item in history] == ['U_1/alice/original','U_1/alice/updated']
    for company,role,user in [('U_2','cfo','alice'),('U_1','ceo','alice'),('U_1','cfo','bob')]:
        other = opened(client,company,role,user)
        assert other != base
        assert client.get(other+'/messages',headers={'X-User-ID':user}).json()['items'] == []
        client.post(other+'/ask/stream',headers={'X-User-ID':user},json={'question':'OTHER','request_id':'new'})
        assert 'COMPANY_FIRST_QUESTION' not in model.seen[-1][-1].content
    assert client.get(base+'/messages',headers={'X-User-ID':'bob'}).status_code == 404
    assert client.post(base+'/ask/stream',headers={'X-User-ID':'bob'},json=payload).status_code == 404
    assert len(store.history('alice',registered.id)) == 1
    assert store.personalization('alice','U_1','cfo').empty()


def test_latest_first_pagination_restart_and_concurrent_retries_keep_one_answer(fixture):
    client, model, store, _ = fixture
    base = opened(client)
    headers = {'X-User-ID':'alice'}
    payload = {'question':'One question','request_id':'same'}
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: client.post(base+'/ask/stream',headers=headers,json=payload), range(2)))
    assert model.runs == 1
    assert events(results[0])[-1][1] == events(results[1])[-1][1]
    client.post(base+'/ask/stream',headers=headers,json={'question':'Next question','request_id':'next'})
    latest = client.get(base+'/messages?limit=1',headers=headers).json()
    assert latest['has_more'] and latest['items'][0]['comment'] == 'Next question'
    earlier = client.get(base+'/messages?limit=1&before='+latest['before'],headers=headers).json()
    assert not earlier['has_more'] and earlier['items'][0]['comment'] == 'One question'
    with TestClient(create_app(store=Store(store.path),api_token='')) as restarted:
        assert len(restarted.get(base+'/messages',headers=headers).json()['items']) == 2


def test_invalid_citations_stay_unsaved_and_server_cannot_invoke_a_company_chat(fixture, monkeypatch):
    client, model, store, _ = fixture
    base = opened(client)
    headers = {'X-User-ID':'alice'}
    model.responses = [calls(tool_call('Analysis',intent='question',response='Unknown fact',requested_action='answer',source_ids=['fact:999']))]
    result = client.post(base+'/ask/stream',headers=headers,json={'question':'Why?','request_id':'bad-source'})
    assert events(result)[-1][0] == 'error'
    assert client.get(base+'/messages',headers=headers).json()['items'][0]['analysis'] is None
    assert client.post('/v1/company-chats',headers=headers,json={'company_id':'U_1','role':'invalid'}).status_code == 422
    monkeypatch.setenv('CABINET_MODE','server')
    before = model.runs
    assert client.post('/v1/company-chats',headers=headers,json={'company_id':'U_1','role':'cfo'}).status_code == 403
    assert client.post(base+'/ask/stream',headers=headers,json={'question':'Why?','request_id':'server'}).status_code == 403
    assert model.runs == before


def test_default_context_uses_only_company_profile_and_explicit_saved_answers(tmp_path):
    store = Store(tmp_path/'context.sqlite'); store.initialize()
    store.set_answer('alice','U_1',ProfileAnswer(question_id='q_0123456789ab',statement='Ставка фиксированная',verdict='yes'))
    company = SimpleNamespace(card={'sector':'leasing'},facts=[dict(cat='bank',text='Есть кредит')])
    with patch('stage2_company_facts.profile.integration.load_company',return_value=company):
        context = load_company_context(store,'alice','U_1','ceo')
        assert len(context.evidence.facts) == 2
        assert 'Со слов пользователя' in context.evidence.facts[1].text
        assert len(load_company_context(store,'bob','U_1','ceo').evidence.facts) == 1
    tools = ResearchTools(company_id='U_1')
    assert tools.news_id is None
    assert not {'same_event_reports','follow_up_news'} & {tool.name for tool in tools.as_tools()}


def test_mismatched_context_is_rejected_before_creating_a_conversation(tmp_path, monkeypatch):
    monkeypatch.setenv('CABINET_MODE','full')
    context = CompanyChatContext(card={},evidence=EvidenceSnapshot(company_id='U_2',role='cfo'))
    store = Store(tmp_path/'mismatch.sqlite')
    with TestClient(create_app(store=store,api_token='',company_context_loader=lambda *args:context)) as client:
        result = client.post('/v1/company-chats',headers={'X-User-ID':'alice'},json={'company_id':'U_1','role':'cfo'})
        assert result.status_code == 404
        with store.connection() as db:
            assert db.execute('SELECT count(*) FROM company_chats').fetchone()[0] == 0
