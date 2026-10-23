"""Judge v2 check run on one company: n events from the candidate generators vs n random events, each graded by
GLM-5.3-Flash (reasons by default) on the multi-dimensional «insight potential» schema with 3 positive + 3 negative
+ 2 opportunity few-shots (prompts/judge_v2.md). News go in three ways (INPUTS): title + Gemma summary, title + GLM
summary (prompts/news_summary.md), title + raw text. Refusals: short «Механизма нет.» by default, `explain` variant asks
why (REFUSAL) on the random half of the first --explain events. Langfuse trace names: news_summary/<model>,
judge_v2/<input>[+explain_refusal]. The company is the
deterministic brief (d) from stage-2 code facts (experiments/briefs). Candidates = generators' weighted quotas over the
best retrieval of experiments/retrieval (RRF of e5-large-instruct/reason + qwen3-emb-0.6b/reason + bm25-lemma).
python -X utf8 experiments/judge_v2/judge.py [--company U_106783] [--n 100] [--limit 5] [--explain 5] [--max-tokens 32000] [--events ids.json --inputs gemma_summary] [--dry] [--workers 12]"""
import argparse, json, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / 'experiments/retrieval', ROOT / 'experiments/briefs'):
    sys.path.insert(0, str(p))
from common.data import DATA
from common.llm import ChatModels, JsonlCache, PromptLibrary, structured
from common.llm.log import RUN_ID
from common.llm.models import LANGFUSE
from evaluate import LEXICAL, UQ, company_queries, corpus, event_ranks, kaggle_methods, rrf
from run import deterministic, facts_for, load_facts  # experiments/briefs/run.py

PROMPTS = PromptLibrary(HERE / 'prompts')
OUT = HERE / 'out'
METHODS = ['e5-large-instruct/reason', 'qwen3-emb-0.6b/reason', 'bm25-lemma/query']
EVENT_TYPES = Literal['решение регулятора', 'проект регулирования или бюджета', 'макрорешение', 'макростатистика',
                      'сдвиг рынка или условий', 'корпоративное событие', 'сбой или авария', 'заявление или мнение',
                      'рутинная котировка', 'другое']
CHANNELS = Literal['объём спроса', 'цена продажи', 'доступ к рынку (лицензии, реестры, локализация, госзаказ)', 'цены закупок и сырья', 'логистика', 'персонал и ФОТ',
                   'налоги и взносы', 'оборотный капитал (отсрочки, авансы)', 'стоимость и доступность финансирования',
                   'валютный риск', 'стоимость активов и залогов', 'запасы', 'контрагентский риск', 'сбой поставок',
                   'регуляторный и правовой риск', 'конкуренция и доля рынка', 'новая возможность', 'нет']
INPUTS = ['gemma_summary', 'glm_summary', 'raw_text']
REFUSAL = {  # rule line + the three negative few-shot explanations (examples 4–6)
    'short': {'refusal': 'Если механизма нет — так и напиши: «Механизма нет.»',
              'why_4': 'Механизма нет.', 'why_5': 'Механизма нет.', 'why_6': 'Механизма нет.'},
    'explain': {'refusal': 'Если механизма нет — так и напиши и объясни почему.',
                'why_4': 'Механизма нет: колебание курса на 0,05% за день — рутинная котировка, условия бизнеса не меняются. '
                         'У компании есть страны ВЭД, но направление и объём операций неизвестны, а такой сдвиг несущественен при любом направлении.',
                'why_5': 'Это заявление без решения: ни Китай, ни ЕС пока ничего не изменили, условия торговли с Китаем для российских '
                         'компаний прежние. У компании в ФТС есть Китай, но направление и объём неизвестны; видимые поставщики — российские '
                         'оптовики и производители профилей, а обсуждаемые санкции касаются поставщиков электроники. Механизма нет.',
                'why_6': 'Решение заметное для банков и рынка жилья, но у оптового торговца топливом нет канала: среди видимых покупателей '
                         'и поставщиков нет застройщиков, кредитные продукты компании корпоративные, ипотечные лимиты их не касаются. '
                         'Влияние через спрос стройки на топливо — слишком длинная и слабая цепочка.'}}


class Verdict(BaseModel):  # field order = reasoning order: explanation first
    impact_explanation: str = Field(description='как событие влияет именно на эту компанию: событие → канал → факт профиля → эффект; или «механизма нет»')
    event_type: EVENT_TYPES
    significance: int = Field(ge=0, le=3, description='значимость события без учёта компании: масштаб и регулярность')
    channel: CHANNELS
    exposure_fact: str = Field(description='дословный фрагмент профиля, через который идёт влияние; пусто, если механизма нет')
    direction: Literal['риск', 'возможность', 'неоднозначно', 'нет']
    mechanism_strength: int = Field(ge=0, le=3)
    horizon: Literal['сейчас', 'квартал', 'год+', 'нет']
    insight_potential: int = Field(ge=0, le=3)


def positive(v: dict) -> bool:  # shortcut: preliminary threshold, recalibrate on the human-labelled set
    return v['insight_potential'] >= 2 and v['significance'] >= 2 and v['mechanism_strength'] >= 2


def candidate_events(u: str, n: int) -> dict[int, dict]:
    """Event -> (best rank, generator) for the generators' quota union at the smallest budget giving >= n news events."""
    _, _, event = corpus()
    every = dict(LEXICAL) | kaggle_methods(ROOT / 'experiments/retrieval/out')
    evid, E = event_ranks(rrf(*[every[m]() for m in METHODS]), event)
    real = set(pd.read_parquet(DATA / 'news' / 'events.parquet').event)  # drops planted needles
    qi = [q for q, _ in company_queries()[u]]
    w = np.array([w for _, w in company_queries()[u]])
    order = np.argsort(-E[:, qi], axis=0)
    for b in range(n, 20 * n, 10):
        chosen = {}
        for j, h in enumerate(np.maximum(1, np.round(b * w / w.sum())).astype(int)):
            for r, row in enumerate(order[:h, j]):
                e = int(evid[row])
                if e in real and (e not in chosen or r < chosen[e]['rank']):
                    chosen[e] = {'rank': r, 'reason': UQ[qi[j]]['reason'], 'query': UQ[qi[j]]['query'][:80]}
        if len(chosen) >= n:
            return chosen
    return chosen


def event_news(events) -> dict[int, dict]:
    """First listed report stands for the event. shortcut: not the best-written report, pick by source tier if it matters."""
    ev = pd.read_parquet(DATA / 'news' / 'events.parquet').set_index('event')
    news = pd.read_parquet(DATA / 'news' / 'news.parquet')
    news = news.set_index(news['id'].astype(str))
    out = {}
    for e in events:
        i = next(str(x) for x in ev.loc[e, 'news_ids'] if str(x) in news.index)
        r = news.loc[i]
        out[e] = {'id': i, 'title': r.title, 'source': r.source, 'date': str(r.date)[:10], 'text': (r.text or '')[:5000]}
    return out


def parallel(fn, items, workers, rate: float = 8.0):
    """Up to `workers` concurrent calls (team rule: up to 64), new requests started at most `rate` per second."""
    with ThreadPoolExecutor(workers) as ex:
        futures = []
        for x in items:
            futures.append(ex.submit(fn, x))
            time.sleep(1 / rate)
        for f in futures:
            f.result()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--company', default='U_106783')
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--limit', type=int, help='smoke: judge only this many pairs per source')
    ap.add_argument('--dry', action='store_true', help='only build pairs and prompts, no LLM calls')
    ap.add_argument('--explain', type=int, default=5, help='random events also judged with the explain-refusal prompt')
    ap.add_argument('--workers', type=int, default=64)
    ap.add_argument('--max-tokens', type=int, default=16000, help='judge cap; reasoning counts toward it')
    ap.add_argument('--events', help='JSON list of event ids to judge instead of candidates + random (source "listed")')
    ap.add_argument('--inputs', nargs='+', default=INPUTS, choices=INPUTS)
    a = ap.parse_args()
    u = a.company
    if a.events:
        cand, rand = {}, []
        pairs = [(int(e), 'listed', None) for e in json.loads(Path(a.events).read_text())]
    else:
        cand = candidate_events(u, a.n)
        rest = sorted(set(pd.read_parquet(DATA / 'news' / 'events.parquet').event) - set(cand))
        rand = [int(e) for e in np.random.default_rng(0).choice(rest, a.n, replace=False)]
        pairs = [(e, 'candidate', cand[e]) for e in sorted(cand, key=lambda e: cand[e]['rank'])] + [(e, 'random', None) for e in rand]
    if a.limit:
        pairs = [p for p in pairs if p[1] == 'candidate'][:a.limit] + [p for p in pairs if p[1] == 'random'][:a.limit]
    texts = event_news([e for e, _, _ in pairs])
    brief = deterministic(facts_for(u))
    print(f'{u}: {len(cand)} candidate + {len(rand)} random events; judging {len(pairs)}; brief {len(brief)} chars, '
          f'judge prompt {len(PROMPTS.text("judge_v2"))} chars', flush=True)
    print('candidates by reason:', pd.Series([c['reason'] for c in cand.values()], dtype=str).value_counts().to_dict(), flush=True)
    if a.dry:
        sys.exit()

    OUT.mkdir(exist_ok=True)
    SUMS = {m: JsonlCache(f'summaries_{m}', folder=OUT) for m in ('gemma', 'glm')}
    LABELS = JsonlCache(f'labels_v2_{u}', folder=OUT)
    lock = threading.Lock()  # JsonlCache appends to one file
    models = ChatModels()
    summarizers = {'gemma': models.cheap(), 'glm': models.agent(temperature=0.0)}
    summarize = {m: PROMPTS.chat('news_summary') | llm | StrOutputParser() for m, llm in summarizers.items()}
    verdict = PROMPTS.chat('judge_v2') | structured(models.agent(temperature=0.0, max_tokens=a.max_tokens), Verdict)

    def trace(name, **meta):  # handler at the chain root: Langfuse sees run_name and nests the model call under it
        return {'run_name': name, 'callbacks': LANGFUSE,
                'metadata': {'langfuse_session_id': RUN_ID, 'langfuse_tags': ['judge_v2', name], **meta}}

    def summary(job):
        m, e = job
        t = texts[e]
        if SUMS[m].get(t['id']) is None:
            s = summarize[m].invoke({k: t[k] for k in ('title', 'source', 'date', 'text')},
                                    trace(f'news_summary/{m}', news_id=t['id'], event=e)).strip()
            with lock:
                SUMS[m].put(t['id'], s)

    def news_block(t, inp):
        if inp == 'raw_text':
            return f"Текст: {t['text']}"
        return f"Выжимка: {SUMS[inp.split('_')[0]].get(t['id'])}"

    def grade(job):
        (e, source, gen), inp, refusal = job
        key = f'{e}|{inp}|{refusal}'
        if LABELS.get(key) is not None:
            return
        t = texts[e]
        name = f'judge_v2/{inp}' + ('+explain_refusal' if refusal == 'explain' else '')
        cfg = trace(name, company=u, event=e, pool=source, news_id=t['id'])
        for _ in range(2):  # the model sometimes answers without the tool call (None)
            try:
                v = verdict.invoke({'brief': brief, 'title': t['title'], 'news': news_block(t, inp), **REFUSAL[refusal]}, cfg)
            except Exception as ex:
                print('  judge error:', key, str(ex)[:150], flush=True); v = None
            if v is not None:
                with lock:
                    LABELS.put(key, {'event': e, 'input': inp, 'refusal': refusal, 'source': source, 'generator': gen,
                                     'news_id': t['id'], 'title': t['title'], **v.model_dump()})
                return
        print('  unlabelled', key, flush=True)

    parallel(summary, [(m, e) for m in SUMS if f'{m}_summary' in a.inputs for e in texts], a.workers)
    print('summaries done', flush=True)
    jobs = [(p, inp, 'short') for p in pairs for inp in a.inputs]
    jobs += [(p, 'gemma_summary', 'explain') for p in [p for p in pairs if p[1] == 'random'][:a.explain]]
    print(f'{len(jobs)} judge calls', flush=True)
    parallel(grade, jobs, a.workers)

    rows = [LABELS.get(f'{p[0]}|{inp}|{r}') for p, inp, r in jobs]
    df = pd.DataFrame([r for r in rows if r]).assign(positive=lambda d: d.apply(positive, axis=1))
    df['variant'] = df.input + '/' + df.refusal
    cols = ['significance', 'mechanism_strength', 'insight_potential', 'positive']
    print(df.groupby(['variant', 'source'])[cols].mean().round(2).to_string())
    print(df.pivot_table(index=['source', 'title'], columns='variant', values='insight_potential').to_string())
