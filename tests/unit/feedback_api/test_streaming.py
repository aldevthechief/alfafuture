"""Genuine model chunks, frozen evidence, durable retries and ownership on the streaming transport."""
import json
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

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        self.runs += 1
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
