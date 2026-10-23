"""Offline test of the external factor rules on a synthetic card (no case data, no LLM).
python -X utf8 stage2_company_facts/external_factors/test_factors.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stage2_company_facts.external_factors.factors import factors, sector_entry

CARD = {'id': 'U_SYNTH', 'okved': '46.71.2', 'okved_name': 'Торговля оптовая моторным топливом', 'region': 'Омская область',
        'revenue': 16e9, 'group_revenue': None, 'products': ['loan', 'rko'], 'fts_countries': ['Китай', 'Германия'],
        'fts_flag': True, 'pay_out': {'46': 0.7, '49': 0.3}, 'pay_in': {'35': 0.1, '47': 0.9},
        'pay_out_total': 5e9, 'pay_in_total': 11e9}
FACTS = [{'id': 'fact:activity_1', 'cat': 'activity'}, {'id': 'fact:bank_1', 'cat': 'bank'},
         {'id': 'fact:trade_1', 'cat': 'trade'}, {'id': 'fact:geo_1', 'cat': 'geo'},
         {'id': 'fact:payments_out_1', 'cat': 'payments_out'}, {'id': 'fact:payments_in_1', 'cat': 'payments_in'},
         {'id': 'fact:ownership_1', 'cat': 'ownership', 'text': 'Иностранное участие в собственности'}]

fs = factors(CARD, FACTS)
by = {f['id']: f for f in fs}
rules = {f['rule'] for f in fs}
assert sector_entry('46.71.2')[0] == '46.71' and sector_entry('46.34')[0] == '46.34' and sector_entry('99.99')[0] is None
assert by['factor:ставка:ключевая_ставка_цб']['weight'] == 1.0, 'loan -> rate at full weight'
assert 'fact:bank_1' in by['factor:ставка:ключевая_ставка_цб']['basis']
assert any(f['rule'] == 'fts:unfriendly' for f in fs) and any('юань' in ' '.join(f['keywords']) for f in fs)
assert 'ownership:foreign' in rules
assert 'chain:in:47' in rules and 'chain:out:49' in rules, 'counterparty sectors with share >= 15%'
assert not any(r.startswith('chain:') and r.endswith(':46') for r in rules), 'same division is not a chain'
assert not any(r == 'chain:in:35' for r in rules), 'shares below 15% are ignored'
assert all(f['weight'] <= 0.6 for f in fs if f['rule'].startswith('chain:'))
assert all(f['basis'] and f['keywords'] for f in fs)
assert [f['weight'] for f in fs] == sorted((f['weight'] for f in fs), reverse=True)
print(f'ok: {len(fs)} factors, rules {sorted(rules)}')
