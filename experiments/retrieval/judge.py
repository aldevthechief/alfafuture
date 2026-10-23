"""Pooled relevance labels (TREC-style): the top-k events of every company generator under every listed method form the
pool; an allowed LLM grades each pooled event for the company (0 none, 1 possible / indirect, 2 clear impact that could
become an insight). Labels are cached in labels_<model>.jsonl (cheap = Gemma: 94% agreement with GLM on grade 2 vs not, 20.10); evaluate.py --labels uses grade >= 2 as positives.
python -X utf8 experiments/retrieval/judge.py --methods bm25-lemma/query frida/text ... [--k 5] [--model agent]"""
import argparse, json, random, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.data import DATA
from common.llm import ChatModels, JsonlCache, PromptLibrary, structured
from evaluate import LEXICAL, company_queries, corpus, event_ranks, kaggle_methods
from stage2_company_facts.deterministic_facts.facts import build
from stage4_news_relevance.batch_ranker.ranker import BatchRanker, lead

PROMPTS = PromptLibrary(HERE / 'prompts')
LABELS = None  # labels_<model>.jsonl, set in main


class Grade(BaseModel):
    i: int
    grade: int = Field(ge=0, le=2, description='0 — нет связи; 1 — возможная или косвенная; 2 — явно влияет на бизнес компании')


class Grades(BaseModel):
    grades: list[Grade]


def pool(methods: dict, k: int) -> dict[str, set]:
    _, _, event = corpus()
    cq = company_queries()
    out = {u: set() for u in cq}
    for name, f in methods.items():
        evid, E = event_ranks(f(), event)
        top = evid[np.argsort(-E, axis=0)[:k]]  # k x queries
        for u, qs in cq.items():
            out[u].update(int(e) for e in top[:, [q for q, _ in qs]].ravel())
        print(f'pooled {name}: total {sum(map(len, out.values()))}', flush=True)
    return out


def event_texts() -> dict[int, dict]:
    ids, _, event = corpus()
    news = pd.read_parquet(DATA / 'news' / 'news.parquet').set_index(pd.read_parquet(DATA / 'news' / 'news.parquet')['id'].astype(str))
    out = {}
    for i, e in zip(ids, event):
        if int(e) not in out and i in news.index:  # first (business-source-first) report stands for the event
            r = news.loc[i]
            out[int(e)] = {'title': r.title, 'lead': lead(r.to_dict())}
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', nargs='+', required=True)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--pack', type=int, default=20)
    ap.add_argument('--model', default='cheap', choices=['agent', 'cheap'])
    ap.add_argument('--workers', type=int, default=8, help='parallel LLM requests')
    a = ap.parse_args()
    LABELS = JsonlCache(f'labels_{a.model}', folder=HERE)
    allm = dict(LEXICAL) | kaggle_methods(HERE / 'out')
    P = pool({m: allm[m] for m in a.methods}, a.k)
    texts = event_texts()
    models = ChatModels()
    llm = models.agent(temperature=0.0, max_tokens=3000) if a.model == 'agent' else models.cheap(max_tokens=3000)
    chain = PROMPTS.chat('judge') | structured(llm, Grades)
    briefs = BatchRanker(models.cheap(max_tokens=1500))
    facts = build(list(P))
    ap_workers = a.workers
    lock = threading.Lock()

    def judge_pack(u, brief, pack):
        """The model sometimes answers without the tool call (None): retry once, then split the pack in halves."""
        lines = '\n'.join(f"{n}. {texts[e]['title']} — {texts[e]['lead']}" for n, e in enumerate(pack, 1))
        for _ in range(2):
            try:
                r = chain.invoke({'brief': brief, 'news': lines})
            except Exception as ex:  # malformed tool arguments, proxy errors
                print('  judge error:', str(ex)[:120]); r = None
            if r is not None:
                got = {g.i: g.grade for g in r.grades}
                with lock:  # JsonlCache appends to one file
                    for n, e in enumerate(pack, 1):
                        if n in got:
                            LABELS.put(f'{u}|{e}', got[n])
                return
        if len(pack) > 1:
            judge_pack(u, brief, pack[:len(pack) // 2]); judge_pack(u, brief, pack[len(pack) // 2:])
        else:
            print('  unlabelled', u, pack)

    jobs = []
    for u, evs in P.items():
        brief = briefs.brief(facts[u])
        todo = [e for e in sorted(evs) if e in texts and LABELS.get(f'{u}|{e}') is None]
        random.Random(0).shuffle(todo)
        jobs += [(u, brief, todo[s:s + a.pack]) for s in range(0, len(todo), a.pack)]
    print(f'{len(jobs)} packs to judge with {ap_workers} parallel requests', flush=True)
    with ThreadPoolExecutor(ap_workers) as ex:
        for k, _ in enumerate(as_completed([ex.submit(judge_pack, *j) for j in jobs]), 1):
            if k % 25 == 0:
                print(f'  {k}/{len(jobs)} packs', flush=True)
    for u, evs in P.items():
        g = [LABELS.get(f'{u}|{e}') for e in evs if LABELS.get(f'{u}|{e}') is not None]
        print(f'{u}: pooled {len(evs)}, labelled {len(g)}, grade2 {sum(x == 2 for x in g)}, grade1 {sum(x == 1 for x in g)}', flush=True)
