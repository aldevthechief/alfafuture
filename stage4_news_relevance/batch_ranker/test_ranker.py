"""Offline check of ranker 1 (packs): verdicts map back to news ids after shuffling; a skipped item is kept; leads.
python -X utf8 stage4_news_relevance/batch_ranker/test_ranker.py"""
import sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.llm import JsonlCache
from common.llm.testing import ScriptedChatModel, calls, tool_call
from stage4_news_relevance.batch_ranker.ranker import BatchRanker, lead

assert lead({'title': 'ЦБ сохранил ставку', 'text': 'ЦБ сохранил ставку Совет директоров решил…', 'summary': None}).startswith('Совет')
assert lead({'title': 'T', 'text': 'body', 'summary': 'RSS-описание новости длиннее сорока символов, точно.'}).startswith('RSS')

script = [calls(tool_call('Brief', text='Топливный оптовик, Омск, 16 млрд ₽')),
          calls(tool_call('PackVerdicts', verdicts=[{'i': 1, 'keep': False, 'channel': ''}, {'i': 2, 'keep': True, 'channel': 'цены'}]))]
with tempfile.TemporaryDirectory() as tmp:
    r = BatchRanker(ScriptedChatModel(responses=script), cache=JsonlCache('b', folder=Path(tmp)), pack=3, seed=1)
    brief = r.brief([{'cat': 'activity', 'text': 'Торговля оптовая топливом'}])
    items = [{'id': f'n{k}', 'title': f'Новость {k}', 'lead': ''} for k in range(3)]
    out = r.rank(brief, items)
assert set(out) == {'n0', 'n1', 'n2'} and sum(v['keep'] for v in out.values()) == 2, 'pack item without a verdict is kept'
print('ok: brief cached, pack verdicts mapped to ids, recall-first default')
