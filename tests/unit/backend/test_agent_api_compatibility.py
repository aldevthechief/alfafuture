"""Maxon agents retain our HTTP contracts and server-mode restrictions."""
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.service import Service


def test_async_question_preserves_stream_and_user_identity(tmp_path):
    service = Service(tmp_path, models_factory=Mock(side_effect=AssertionError('No live models')))
    received = []

    async def ask(company, role, question, title, user):
        received.append((company, role, question, title, user))
        yield 'first '
        yield 'second'

    service.ask = ask
    with patch.dict('os.environ', CABINET_MODE='full'), TestClient(create_app(service)) as client:
        response = client.post('/v1/insights/ask', headers={'X-User-ID':'reader'},
            json={'company_id':'U_14366','role':'cfo','question':'Why?','insight_title':'Saved'})
        assert response.status_code == 200
        assert response.text == 'first second'
        assert received == [('U_14366','cfo','Why?','Saved','reader')]


def test_compact_enforces_owner_and_server_mode_before_running_models(tmp_path):
    service = Service(tmp_path, models_factory=Mock(side_effect=AssertionError('No live models')))
    service.agents.items['saved'] = dict(id='saved',status='succeeded',request={'user_id':'owner'},result={})
    service.compact = Mock(return_value={'card':{'title':'Short'}})
    with patch.dict('os.environ', CABINET_MODE='full'), TestClient(create_app(service)) as client:
        assert client.get('/v1/jobs/saved/compact',headers={'X-User-ID':'other'}).status_code == 404
        service.compact.assert_not_called()
        assert client.get('/v1/jobs/saved/compact',headers={'X-User-ID':'owner'}).json() == {'card':{'title':'Short'}}
    with patch.dict('os.environ', CABINET_MODE='server'), TestClient(create_app(service)) as client:
        assert client.get('/v1/jobs/saved/compact',headers={'X-User-ID':'owner'}).status_code == 403
    service.compact.assert_called_once_with('saved')
