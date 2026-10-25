"""Genuine model chunks, frozen evidence, durable retries and ownership on the streaming transport."""
import json
import pytest
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from common.llm.testing import ScriptedChatModel, calls, tool_call
from feedback_api.analyzer import FeedbackAnalyzer
from feedback_api.app import create_app
from feedback_api.store import Store


class StreamingModel(ScriptedChatModel):
    runs: int = 0
    seen: list[list] = []

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        self.runs += 1
        self.seen.append(messages)
        args = json.dumps(self.responses[0].tool_calls[0]['args'], ensure_ascii=True)
        for index, offset in enumerate(range(0, len(args), 9)):
            yield ChatGenerationChunk(message=AIMessageChunk(content='', tool_call_chunks=[dict(
                name='Analysis' if index == 0 else None, id='answer' if index == 0 else None,
                index=0, args=args[offset:offset + 9])]))


def events(response):
    result = []
    for frame in response.text.split('\n\n'):
        lines = frame.splitlines()
        if len(lines) >= 2 and lines[0].startswith('event: '):
            result.append((lines[0][7:], json.loads(lines[1][6:])))
    return result


def setup_client(tmp_path, source='fact:1'):
    model = StreamingModel(responses=[calls(tool_call('Analysis', intent='question',
        response='Первый вывод. Затем уточнение: «ставка».', requested_action='answer', source_ids=[source]))])
    store = Store(tmp_path / 'discussion.sqlite')
    app = create_app(store=store, analyzer=FeedbackAnalyzer(model=model))
    return TestClient(app), model


def register(client):
    response = client.post('/v1/insights', headers={'X-User-ID': 'alice'}, json={
        'company_id': 'U_1', 'insight': {'title': 'Ставка', 'role': 'cfo'},
        'evidence': {'company_id': 'U_1', 'role': 'cfo', 'captured_at': '2026-10-25',
                     'facts': [{'id': 'fact:1', 'text': 'Есть кредит'}]},
    })
    assert response.status_code == 201, response.text
    return f"/v1/insights/{response.json()['id']}"


def test_streams_partial_text_then_persists_validated_answer_and_replays_without_another_call(tmp_path):
    client, model = setup_client(tmp_path)
    with patch.dict('os.environ', {'CABINET_MODE': 'full', 'ALFA_CACHE_REDIS_URL': ''}), client:
        base = register(client)
        payload = {'question': 'Как повлияет ставка?', 'request_id': 'same-request'}
        response = client.post(base + '/ask/stream', headers={'X-User-ID': 'alice'}, json=payload)
        assert response.status_code == 200
        assert response.headers['x-accel-buffering'] == 'no'
        streamed = events(response)
        texts = [data['text'] for name, data in streamed if name == 'text']
        assert len(texts) > 2 and len(texts[0]) < len(texts[-1])
        assert streamed[-1][0] == 'complete'
        final = streamed[-1][1]
        assert texts[-1] == final['analysis']['response']
        assert final['analysis']['source_ids'] == ['fact:1']
        history = client.get(base + '/feedback', headers={'X-User-ID': 'alice'}).json()
        assert len(history) == 1 and history[0] == final
        replay = client.post(base + '/ask/stream', headers={'X-User-ID': 'alice'}, json=payload)
        assert events(replay) == [('complete', final)]
        assert model.runs == 1
        assert client.post(base + '/ask/stream', headers={'X-User-ID': 'bob'}, json=payload).status_code == 404
        assert client.post(base + '/ask/stream', headers={'X-User-ID': 'alice'},
                           json={**payload, 'question': 'Другой вопрос'}).status_code == 409


def test_unknown_sources_never_become_a_saved_answer_and_server_mode_cannot_start_a_stream(tmp_path):
    client, model = setup_client(tmp_path, source='fact:unknown')
    with patch.dict('os.environ', {'CABINET_MODE': 'full', 'ALFA_CACHE_REDIS_URL': ''}), client:
        base = register(client)
        payload = {'question': 'Почему?', 'request_id': 'invalid-sources'}
        response = client.post(base + '/ask/stream', headers={'X-User-ID': 'alice'}, json=payload)
        streamed = events(response)
        assert streamed[-1][0] == 'error'
        assert not any(name == 'complete' for name, _ in streamed)
        history = client.get(base + '/feedback', headers={'X-User-ID': 'alice'}).json()
        assert len(history) == 1 and history[0]['analysis'] is None
        before = model.runs
        with patch.dict('os.environ', CABINET_MODE='server'):
            assert client.post(base + '/ask/stream', headers={'X-User-ID': 'alice'}, json=payload).status_code == 403
        assert model.runs == before


@pytest.mark.parametrize('selected_source', ['insight', 'answer'])
def test_selected_excerpt_reaches_the_model_and_survives_history_retries_and_follow_up(tmp_path, selected_source):
    client, model = setup_client(tmp_path)
    with patch.dict('os.environ', {'CABINET_MODE': 'full', 'ALFA_CACHE_REDIS_URL': ''}), client:
        base = register(client)
        headers = {'X-User-ID': 'alice'}
        payload = {'question': 'Что это значит для нас?', 'selected_text': 'Есть кредит', 'selected_source':selected_source, 'request_id': 'quoted'}
        response = client.post(base + '/ask/stream', headers=headers, json=payload)
        final = events(response)[-1][1]
        assert final['selected_text'] == payload['selected_text']
        assert final['selected_source'] == selected_source
        assert final['comment'] == payload['question']
        prompt = model.seen[-1][-1].content
        assert '"selected_text": "Есть кредит"' in prompt
        assert ('Прежний ответ модели не является первоисточником' if selected_source == 'answer' else 'возможно сокращённой версии') in prompt
        assert client.get(base + '/feedback', headers=headers).json() == [final]
        assert events(client.post(base + '/ask/stream', headers=headers, json=payload)) == [('complete', final)]
        assert model.runs == 1
        assert client.post(base + '/ask/stream', headers=headers,
                           json={**payload, 'selected_text': 'Другая цитата'}).status_code == 409
        assert client.post(base + '/ask/stream', headers=headers,
                           json={**payload, 'selected_source':'answer' if selected_source == 'insight' else 'insight'}).status_code == 409
        assert client.post(base + '/ask/stream', headers=headers,
                           json={**payload, 'request_id': 'too-long', 'selected_text': 'x' * 2001}).status_code == 422
        client.post(base + '/ask/stream', headers=headers,
                    json={'question': 'А если условия изменятся?', 'request_id': 'follow-up'})
        label = 'ответа агента' if selected_source == 'answer' else 'инсайта'
        assert f'Выделенный фрагмент {label}: Есть кредит' in model.seen[-1][-1].content
        # The JSON transport supports the same excerpt contract as SSE.
        saved = client.post(base + '/ask', headers=headers, json={**payload, 'request_id': 'json-quoted'})
        assert saved.status_code == 201
        assert saved.json()['selected_text'] == 'Есть кредит'
        assert saved.json()['selected_source'] == selected_source
        assert client.post(base + '/ask', headers=headers,
                           json={**payload, 'request_id':'invalid-source', 'selected_source':'bank_fact'}).status_code == 422


def test_existing_discussion_database_keeps_history_when_excerpt_column_is_added(tmp_path):
    import sqlite3
    path = tmp_path / 'old.sqlite'
    with sqlite3.connect(path) as db:
        db.execute('''CREATE TABLE feedback (id TEXT PRIMARY KEY, insight_id TEXT NOT NULL,
            kind TEXT NOT NULL, comment TEXT NOT NULL, emoji TEXT NOT NULL, scope TEXT NOT NULL,
            request_id TEXT NOT NULL, analysis TEXT, created_at TEXT NOT NULL, UNIQUE(insight_id, request_id))''')
        db.execute("INSERT INTO feedback VALUES ('old', 'insight', 'question', 'Старый вопрос', '', 'insight', 'old-request', NULL, '2026-10-25')")
    store = Store(path)
    store.initialize()
    store.initialize()
    with store.connection() as db:
        record = store.feedback_record(db.execute('SELECT * FROM feedback').fetchone())
    assert record.comment == 'Старый вопрос'
    assert record.request_id == 'old-request'
    assert record.selected_text == ''
    assert record.selected_source == 'insight'
