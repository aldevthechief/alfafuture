"""Pack judge vs single judge (judge v2, one company): the same system prompt (prompts/judge_v2.md, short refusals) grades
5–6 news per call (title + Gemma summary) instead of one. Pool: every event that is positive by the 3-input majority of
the single-judge run + `--neg` random negatives, shuffled into packs. Compared with the single judge's gemma_summary
verdicts (labels_v2_<company>.jsonl). Langfuse trace name: judge_v2_pack/gemma_summary.
python -X utf8 experiments/judge_v2/pack.py [--company U_106783] [--neg 23] [--pack 5] [--max-tokens 128000]"""
import argparse, sys, threading
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, create_model

sys.path.insert(0, str(Path(__file__).parent))
from judge import OUT, PROMPTS, REFUSAL, Verdict, deterministic, event_news, load_facts, parallel, positive  # also sets sys.path
from common.llm import ChatModels, JsonlCache, structured
from common.llm.models import trace_config

PACK_RULE = """

Сейчас тебе дают не одну, а несколько новостей под номерами. Оцени каждую независимо от остальных, как если бы она была
единственной: соседние новости не усиливают и не ослабляют оценку. Верни ответ по каждой новости, указав её номер `i`,
поля каждого ответа заполняй в том же порядке, что и для одной новости."""
PackVerdict = create_model('PackVerdict', i=(int, Field(description='номер новости')),
                           **{n: (f.annotation, f) for n, f in Verdict.model_fields.items()})
Verdicts = create_model('Verdicts', items=(list[PackVerdict], ...))


def prompt() -> ChatPromptTemplate:
    system = PROMPTS.chat('judge_v2').messages[0].prompt.template
    return ChatPromptTemplate.from_messages([('system', system + PACK_RULE),
                                             ('human', 'Профиль компании:\n{brief}\n\nНовости:\n{news}')])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--company', default='U_106783')
    ap.add_argument('--neg', type=int, default=23)
    ap.add_argument('--pack', type=int, default=5)
    ap.add_argument('--max-tokens', type=int, default=128000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    u = a.company
    single = pd.DataFrame(JsonlCache(f'labels_v2_{u}', folder=OUT)._load().values())
    single = single[single.refusal == 'short'].assign(pos=lambda d: d.apply(positive, axis=1))
    votes = single.groupby('event').pos.agg(['sum', 'size'])
    maj = votes['sum'] * 2 > votes['size']
    g1 = single[single.input == 'gemma_summary'].set_index('event')  # single-judge reference, same input as the packs
    rng = np.random.default_rng(a.seed)
    pos = [int(e) for e in maj[maj].index if e in g1.index]
    neg = [int(e) for e in rng.choice(sorted(e for e in maj[~maj].index if e in g1.index), a.neg, replace=False)]
    pool = rng.permutation(pos + neg).tolist()
    packs = [[int(e) for e in p] for p in np.array_split(pool, max(1, len(pool) // a.pack))]
    print(f'{u}: {len(pos)} positives + {len(neg)} negatives -> {len(packs)} packs of {[len(p) for p in packs]}', flush=True)

    texts = event_news(pool)
    sums = JsonlCache('summaries_gemma', folder=OUT)
    assert all(sums.get(texts[e]['id']) for e in pool), 'missing Gemma summary'
    brief = deterministic(load_facts()[u])
    llm = ChatModels().agent(temperature=0.0, max_tokens=a.max_tokens)
    llm.request_timeout = 3600  # up to 128k reasoning tokens
    chains = [prompt() | structured(m, Verdicts) for m in (llm, llm.model_copy(update={'cache': False}))]  # retry skips the cache
    LABELS, lock = JsonlCache(f'labels_pack_{u}', folder=OUT), threading.Lock()

    def grade(k):
        pack = packs[k]
        if all(LABELS.get(f'{a.seed}|{e}') for e in pack):
            return
        news = '\n\n'.join(f"{n}. {texts[e]['title']}\nВыжимка: {sums.get(texts[e]['id'])}" for n, e in enumerate(pack, 1))
        for chain in chains:
            try:
                r = chain.invoke({'brief': brief, 'news': news, **REFUSAL['short']},
                                 trace_config('judge_v2_pack/gemma_summary') | {'metadata': {'company': u, 'pack': k}})
            except Exception as ex:
                print('  pack error:', k, str(ex)[:200], flush=True); r = None
            if r is not None:
                got = {v.i: v for v in r.items}
                with lock:
                    for n, e in enumerate(pack, 1):
                        if n in got:
                            LABELS.put(f'{a.seed}|{e}', {'event': e, 'pack': k, 'pos_in_pack': n, 'title': texts[e]['title'],
                                                         **got[n].model_dump(exclude={'i'})})
                print(f'  pack {k}: {len(got)}/{len(pack)} verdicts', flush=True)
                return
        print('  pack failed', k, flush=True)

    parallel(grade, range(len(packs)), len(packs))

    P = pd.DataFrame([LABELS.get(f'{a.seed}|{e}') for e in pool if LABELS.get(f'{a.seed}|{e}')]).set_index('event')
    P['pos'] = P.apply(positive, axis=1)
    S, M = g1.loc[P.index], maj.loc[P.index]
    fields = ['significance', 'mechanism_strength', 'insight_potential', 'channel', 'direction', 'event_type', 'horizon']
    print(f'\njudged {len(P)}/{len(pool)}')
    print('pack vs single (gemma_summary), exact agreement:', {f: round((P[f] == S[f]).mean(), 2) for f in fields})
    print('mean |Δ|:', {f: round((P[f] - S[f]).abs().mean(), 2) for f in fields[:3]},
          '| mean pack − single:', {f: round((P[f] - S[f]).mean(), 2) for f in fields[:3]})
    for name, X in (('pack', P.pos), ('single', S.pos)):
        tp = int((X & M).sum())
        print(f'{name:6} vs majority label: TP {tp}, FP {int((X & ~M).sum())}, FN {int((~X & M).sum())}, '
              f'precision {tp / max(1, X.sum()):.2f}, recall {tp / max(1, M.sum()):.2f}')
    print('explanation length, chars: pack', int(P.impact_explanation.str.len().mean()), 'single', int(S.impact_explanation.str.len().mean()))
    d = P.assign(single=S.insight_potential, maj=M)
    print(d[d.insight_potential != d.single][['pack', 'pos_in_pack', 'insight_potential', 'single', 'maj', 'title']].to_string())
