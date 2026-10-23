"""Offline checks: a chained component is split, a tight event is kept; the event summary is lazy and cached.
python -X utf8 stage4_news_relevance/plausibility/test_events.py"""
import sys, tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.llm import JsonlCache
from common.llm.testing import ScriptedChatModel, calls, tool_call
from stage4_news_relevance.plausibility.event_summary import EventSummarizer
from stage4_news_relevance.plausibility.events import cluster

# a chain of airport closures: neighbours overlap, ends do not -> one component at 0.45, several events after the split
cities = ['Пенза', 'Саратов', 'Ярославль', 'Калуга', 'Тамбов', 'Сочи', 'Ижевск', 'Пермь', 'Казань', 'Самара', 'Уфа', 'Омск']
chain = [f'В аэропорту {a} и аэропорту {b} введены ограничения на полеты' for a, b in zip(cities, cities[1:])]
tight = ['ЦБ сохранил ключевую ставку 14% на заседании совета директоров'] * 3
df = pd.DataFrame({'title': chain + tight, 'text': '', 'date': '2026-10-13'})
lab = cluster(df, max_size=4)
assert len(set(lab[:len(chain)])) > 1, 'chain must be split'
assert len(set(lab[len(chain):])) == 1, 'identical reports stay one event'

reply = calls(tool_call('EventSummary', title='13 октября закрытия аэропортов: Пенза, Саратов', summary='Ограничения на полеты.'))
with tempfile.TemporaryDirectory() as tmp:
    s = EventSummarizer(ScriptedChatModel(responses=[reply]), cache=JsonlCache('ev', folder=Path(tmp)), min_reports=3)
    reports = [{'id': i, 'date': '2026-10-13', 'source': 'interfax', 'title': t} for i, t in enumerate(chain[:4])]
    assert s.summarize(reports[:2]) is None  # small event: no LLM call
    first = s.summarize(reports)
    assert first['title'].startswith('13 октября') and s.summarize(reports[::-1]) == first  # cached, order-independent
print('ok: chained component split; event summary lazy and cached')
