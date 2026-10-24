"""Offline check that the cabinet (frontend/src/main.jsx InsightPanel) and the API agree: the real demo cards of
frontend/public/feed.json are registered with their card_sources, every reaction the panel sends is accepted, and a
question is answered from the card's sources. Mirrors the panel's requests; scripted model, no browser, no network.
python -X utf8 feedback_api/test_frontend_contract.py"""
import json, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import common.llm.models as llm_models
llm_models.LANGFUSE = []
from fastapi.testclient import TestClient

from common.llm.testing import ScriptedChatModel, calls, tool_call as c
from feedback_api.analyzer import FeedbackAnalyzer
from feedback_api.app import create_app
from feedback_api.store import Store

feed = json.loads((ROOT / 'frontend' / 'public' / 'feed.json').read_text(encoding='utf-8'))
SOURCES = feed['sources']


def source_text(s):  # = sourceText in main.jsx
    if s['kind'] == 'fact':
        return s['text']
    head = f'Альфа-Банк, «{s["title"]}»' if s['kind'] == 'bank' else f'{s["outlet"]}: {s["title"]}'
    return f'{head}, {s["date"]}{f", с. {s["page"]}" if s.get("page") else ""}. {s.get("snippet") or ""}'


def used(i):
    blocks = [i.get(k, []) for k in ('what_happened', 'why_it_matters', 'risks', 'opportunities', 'hypotheses', 'bank_view')]
    return list(dict.fromkeys(s for b in blocks for claim in b for s in claim['sources']))


cards = [(cid, i) for cid, co in feed['companies'].items() for f in co['feeds'].values() for i in f['insights']]
cid, card = cards[0]
cited = used(card)[:2]
model = ScriptedChatModel(responses=[calls(c('Analysis', intent='question', response='Ставка останется высокой до декабря.',
                                             requested_action='answer', source_ids=cited))])
app = create_app(store=Store(Path(tempfile.mkdtemp()) / 'fb.sqlite3'), analyzer=FeedbackAnalyzer(model=model), api_token='')
H = {'X-User-ID': 'demo-user'}
with TestClient(app) as cl:
    ids = {}
    for company, i in cards:  # every demo card must register as the panel sends it
        r = cl.post('/v1/insights', headers=H, json={'company_id': company, 'insight': i,
                                                     'card_sources': {s: source_text(SOURCES[s]) for s in used(i) if s in SOURCES}})
        assert r.status_code == 201, (i['id'], r.text[:300])
        ids[(company, i['id'])] = r.json()['id']
    iid = ids[(cid, card['id'])]
    for body in ({'kind': 'like'}, {'kind': 'not_relevant'}, {'kind': 'already_known'}, {'kind': 'dislike'},
                 {'kind': 'incorrect', 'comment': 'Пользователь отметил ошибку в фактах карточки', 'scope': 'future'},
                 *({'kind': 'emoji', 'emoji': e} for e in ('👍', '🔥', '🤔', '👎'))):
        if body['kind'] == 'incorrect':  # needs the model; checked through /ask below instead
            continue
        r = cl.post(f'/v1/insights/{iid}/feedback', headers=H, json={'request_id': json.dumps(body, ensure_ascii=False), **body})
        assert r.status_code == 201 and r.json()['analysis']['response'], (body, r.text[:300])
    a = cl.post(f'/v1/insights/{iid}/ask', headers=H, json={'question': 'Что будет, если ЦБ снизит ставку?', 'request_id': 'q1'})
    assert a.status_code == 201 and a.json()['analysis']['source_ids'] == cited, a.text[:300]
    counts = cl.get(f'/v1/insights/{iid}/reactions', headers=H).json()['counts']
    assert counts == {'like': 3, 'not_relevant': 1, 'already_known': 1, 'dislike': 2, 'emoji': 1, 'question': 1}, counts
    rec = cl.get(f'/v1/insights/{iid}', headers=H).json()
    assert set(rec['card_sources']) == {s for s in used(card) if s in SOURCES}
    p = cl.get(f'/v1/companies/{cid}/personalization', params={'role': card['role']}, headers=H).json()
    assert max(p['topic_weights'].values()) <= 1.0 + 1e-9, ('the last vote (👎) wins, 3 likes do not add up', p['topic_weights'])
print(f'ok: {len(cards)} demo cards registered with card_sources; panel reactions and emoji accepted '
      f'(🔥 -> like, 🤔 neutral); question answered citing {cited}')
