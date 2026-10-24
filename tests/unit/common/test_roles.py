"""Offline check of roles.json: every role is complete, its channels exist, an unknown role is an error.
python -X utf8 common/test_roles.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.roles import role, role_channels, role_context, roles
from stage3_news_search.facet_retrieval.candidates import CHANNELS

assert len(roles()) >= 10
for key, r in roles().items():
    assert {'title', 'source', 'focus', 'decisions', 'channels'} <= set(r) and r['decisions'], key
    assert set(r['channels']) <= set(CHANNELS), (key, set(r['channels']) - set(CHANNELS))
assert role_channels(None) == [] and role_channels('cfo') == ['rate', 'deposit', 'fx', 'tax']
assert role_context('cfo')['role_title'] == 'Финансовый директор'
try:
    role('cto')
    raise AssertionError('unknown role must raise')
except ValueError as e:
    assert 'cfo' in str(e)
print(f'ok: {len(roles())} roles, channels valid, unknown role raises')
