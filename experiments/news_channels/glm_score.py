"""GLM-5.3-Flash scores n random events against the whole channel checklist in one call per event
(prompts/news_channels.md). Input = title + first TEXT_CHARS of the event's first report, the same text the rankers get.
Out: out/glm_scores.jsonl (key = event). Langfuse trace name: news_channels/glm.
python -X utf8 experiments/news_channels/glm_score.py [--n 400] [--limit 50] [--workers 12] [--dry]"""
import argparse, threading
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, create_model

from channels import ITEMS, OUT, TEXT_CHARS, HERE, random_events
from common.llm import ChatModels, JsonlCache, PromptLibrary, structured, trace_config
from judge import event_news  # experiments/judge_v2/judge.py

ItemId = Literal[tuple(ITEMS)]


class Hit(BaseModel):
    item: ItemId
    change: str = Field(description='что именно меняется и насколько')
    winners: list[str] = Field(description='кто выигрывает: отрасли, товары, регионы, типы компаний')
    losers: list[str] = Field(description='кто проигрывает')
    assumption: str = Field(description='допущение, на котором держится связь; пусто, если связь прямая')


Scores = create_model('Scores', **{k: (int, Field(ge=0, le=3, description=v)) for k, v in ITEMS.items()})


class ChannelScores(BaseModel):  # field order = reasoning order
    event: str = Field(description='что произошло и что именно изменилось, 1–2 предложения')
    significance: int = Field(ge=0, le=3)
    hits: list[Hit] = Field(default=[], description='каналы с оценкой 2–3')
    scores: Scores
    industries: list[str] = []
    goods: list[str] = []
    regions: list[str] = []
    countries: list[str] = []
    organizations: list[str] = []


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--limit', type=int, help='score only the first LIMIT events of the sample (pilot)')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    events = random_events(a.n)[:a.limit or a.n]
    texts = event_news(events)
    prompts = PromptLibrary(HERE / 'prompts')
    items = '\n'.join(f'- **{k}** {v}' for k, v in ITEMS.items())
    print(f'{len(events)} events; prompt {len(prompts.text("news_channels")) + len(items)} chars + news ≤ {TEXT_CHARS}', flush=True)
    if a.dry:
        print(prompts.chat('news_channels').format(items=items, title='T', source='S', date='D', text='X')[:600])
        raise SystemExit
    LABELS = JsonlCache('glm_scores', folder=OUT)
    lock = threading.Lock()  # JsonlCache appends to one file
    chain = prompts.chat('news_channels') | structured(ChatModels().agent(temperature=0.0), ChannelScores)

    def score(e):
        if LABELS.get(str(e)) is not None:
            return
        t = texts[e]
        for _ in range(2):  # the model sometimes answers without the tool call (None)
            try:
                v = chain.invoke({'items': items, 'title': t['title'], 'source': t['source'], 'date': t['date'],
                                  'text': t['text'][:TEXT_CHARS]}, trace_config('news_channels/glm'))
            except Exception as ex:
                print('  error', e, str(ex)[:150], flush=True); v = None
            if v is not None:
                with lock:
                    LABELS.put(str(e), {'news_id': t['id'], 'title': t['title'], **v.model_dump()})
                return
        print('  unscored', e, flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(score, events))
    rows = [LABELS.get(str(e)) for e in events]
    df = pd.DataFrame([r['scores'] for r in rows if r])
    print(f'{len(df)}/{len(events)} scored; share of events by score per item:')
    print(df.apply(lambda c: c.value_counts(normalize=True)).reindex([0, 1, 2, 3]).fillna(0).round(2).T.to_string())
    print('events with any item ≥ 2:', (df.max(axis=1) >= 2).mean().round(2))
