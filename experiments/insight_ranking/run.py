"""Top-5 selection experiments over the pool of an insight cabinet. LLM calls run in a thread pool (parallel); every call carries
a marker [exp:<config>] so its billed cost is read back from data/logs/calls.jsonl per configuration.
python experiments/insight_ranking/run.py <phases>   phase A: listwise, self-consistency, two-stage; phase B: pointwise, pairwise"""
import concurrent.futures as cf
import itertools
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
from pydantic import BaseModel, Field  # noqa: E402

from common.llm.log import RUN_ID  # noqa: E402
from common.llm.models import ChatModels, structured  # noqa: E402
from common.llm.quota import key_info  # noqa: E402
from gold import TODAY, evaluate  # noqa: E402,F401

DATA = ROOT / 'data' / 'experiments' / 'ranking'
LOG = ROOT / 'data' / 'logs' / 'calls.jsonl'
KEYS = ['U_14366_cfo', 'U_42222_owner']
WORKERS = 16
BUDGET_STOP = 48.0  # ₽ billed to this script: hard stop before the 55 ₽ the user allowed
GLM, DEEPSEEK, QWEN = 'z-ai/glm-5.3-flash', 'deepseek/deepseek-v4.1-flash', 'qwen/qwen3.6-35b-a3b'
MODELS = ChatModels()


def model(name=GLM, effort='low', temperature=0.0, max_tokens=64000):
    return MODELS._make(name, temperature, max_tokens).model_copy(update={'reasoning_effort': effort})


class Pick(BaseModel):
    id: str = Field(description='id карточки из списка, например C07')
    reason: str = Field(max_length=300, description='одна фраза: чем ценна и чем отличается от остальных в топе')


class Ranking(BaseModel):
    top5: list[Pick] = Field(min_length=5, max_length=5, description='ровно 5 карточек, от самой ценной к менее ценной')


class Score(BaseModel):
    value: int = Field(ge=1, le=10, description='общая ценность для этого читателя')
    urgency: int = Field(ge=1, le=10, description='срочность: есть ли срок или событие в ближайшие недели')
    relevance: int = Field(ge=1, le=10, description='насколько связано с бизнесом именно этой компании')
    theme: str = Field(max_length=40, description='тема в 1–3 словах, одинаковая у карточек про одно и то же')


class Winner(BaseModel):
    winner: str = Field(description='A или B')


def profile(key):
    from common.data import load, money
    from common.roles import roles
    company, role = key.rsplit('_', 1)
    r = load()[0].loc[company]
    prod = [n for c, n in (('CL_PRODUCT_LOA_MARK', 'кредиты'), ('CL_PRODUCT_LEA_MARK', 'лизинг'), ('CL_PRODUCT_DEP_MARK', 'депозиты'),
                           ('CL_PRODUCT_GAR_MARK', 'гарантии'), ('CL_PRODUCT_ZPP_MARK', 'зарплатный проект')) if r[c] == 1]
    return (f"Компания: вид деятельности «{r.MAINOKVED_NAME}», выручка {money(r.REVENUE_COMPANY)} за 2025, штат {r.AMOUNT_WORKERS_FNS_GRADATION}, "
            f"возраст {r.COMPANY_AGE_YEARS_GRADATION} лет, продукты банка: {', '.join(prod) or 'нет'}, "
            f"внешняя торговля: {r.FTS_COUNTRY_NAME if r.FTS_FLAG == 1 else 'нет'}.\nЧитатель: {roles()[role]['title']}.")


RUBRIC = ("Ценность карточки для этого читателя = (1) связь с бизнесом именно этой компании, (2) существенность для денег и решений, "
          "(3) срочность: срок или событие в ближайшие недели, свежесть на сегодня " + TODAY + ", (4) конкретность действия для роли, "
          "(5) надёжность (уверенность, опора на данные). Карточки про одно и то же событие или механизм дублируют друг друга.")
DIVERSITY = ("Топ-5 должен быть разнообразным: пять разных механизмов или тем, не два варианта одной новости; в нём должны быть и риски, "
             "и возможности, а карточка с позицией Банка идёт в топ, если она существенна. Из дублей возьми одну лучшую.")


def listwise_prompt(key, level, cards, diversity=True):
    body = '\n\n'.join(f"[C{n:02d}] {c[level]}" for n, c in enumerate(cards))
    return (f"{profile(key)}\n\n{RUBRIC}\n{DIVERSITY if diversity else ''}\n\nНиже {len(cards)} готовых карточек-инсайтов этой компании. "
            f"Выбери топ-5 и расположи по убыванию ценности.\n\n{body}")


def parse_top(r, n):
    out = []
    for p in r.top5:
        m = re.search(r'\d+', p.id)
        if m and int(m.group()) < n and int(m.group()) not in out:
            out.append(int(m.group()))
    return out


def call(tag, chain, prompt):
    t = time.time()
    try:
        return chain.invoke(f'[exp:{tag}]\n{prompt}'), round(time.time() - t, 1), None
    except Exception as e:  # one lost call must not kill the experiment
        return None, round(time.time() - t, 1), repr(e)[:160]


# ---------------- heuristic prescore (no LLM) ----------------
def prescore(c):
    days = (time.mktime(time.strptime(TODAY, '%Y-%m-%d')) - time.mktime(time.strptime(c['published'] or '2026-10-01', '%Y-%m-%d'))) / 86400
    fresh = math.exp(-max(days, 0) / 14)
    return (.30 * ((c['rel'] or 2) / 3) + .10 * ((c['role_fit'] or 2) / 3) + .15 * {'high': 1, 'mid': .6, 'low': .3}.get(c['impact'], .5) +
            .10 * {'high': 1, 'medium': .75, 'low': .45}.get(c['confidence'], .6) + .15 * fresh + .10 * (1 if c['has_bank'] else 0) +
            .10 * min(c['n_gen'], 5) / 5)


def words(t):
    return set(re.findall(r'[а-яё]{5,}', t.lower()))


def overlap(a, b):
    a, b = words(a), words(b)
    return len(a & b) / max(1, min(len(a), len(b)))


def diversify(cards, scores, k=5, penalty=.25):
    """Greedy: the next card = best score minus a penalty for the same channel, the same event and shared title words."""
    left, chosen = list(range(len(cards))), []
    while left and len(chosen) < k:
        def adj(i):
            pen = sum(.12 * (cards[i]['channel'] == cards[j]['channel']) + 1.0 * (cards[i]['event_id'] is not None and cards[i]['event_id'] == cards[j]['event_id'])
                      + .4 * overlap(cards[i]['title'], cards[j]['title']) for j in chosen)
            return scores[i] - penalty * pen
        best = max(left, key=adj)
        chosen.append(best)
        left.remove(best)
    return chosen


def borda(lists):
    pts = Counter()
    for top in lists:
        for r, c in enumerate(top):
            pts[c] += 5 - r
    return [c for c, _ in sorted(pts.items(), key=lambda kv: (-kv[1], kv[0]))][:5]


def log_costs(start):
    """Billed ₽ per [exp:tag] marker and in total, from the calls this process appended to the log."""
    per, total = defaultdict(float), 0.0
    if not LOG.exists():
        return per, total
    with open(LOG, 'rb') as f:
        f.seek(start)
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get('run') != RUN_ID:
                continue
            m = re.search(r'\[exp:([^\]]+)\]', str(d.get('prompt')))
            c = float(d.get('cost') or 0)
            total += c
            if m:
                per[m.group(1)] += c
    return per, total


def build_jobs(phase, pools):
    jobs = []  # (config, key, kind, callable)
    rank = lambda name, eff, temp=0.0: structured(model(name, eff, temp), Ranking)
    for key in KEYS:
        cards = pools[key]
        if phase == 'A':
            # M1: one listwise call over the whole pool: detail x reasoning, without the diversity line, other models
            for level in ('L1', 'L2', 'L3'):
                for eff in ('low', 'high'):
                    jobs.append((f'M1_{level}_{eff}', key, 'list', lambda key=key, cards=cards, level=level, eff=eff: call(
                        f'M1_{level}_{eff}_{key}', rank(GLM, eff), listwise_prompt(key, level, cards))))
            jobs.append(('M1_L3_low_nodiv', key, 'list', lambda key=key, cards=cards: call(
                f'M1_L3_low_nodiv_{key}', rank(GLM, 'low'), listwise_prompt(key, 'L3', cards, diversity=False))))
            for name, tag in ((DEEPSEEK, 'deepseek'), (QWEN, 'qwen')):
                for eff in ('low', 'high'):
                    jobs.append((f'M1_L2_{eff}_{tag}', key, 'list', lambda key=key, cards=cards, name=name, eff=eff, tag=tag: call(
                        f'M1_L2_{eff}_{tag}_{key}', rank(name, eff), listwise_prompt(key, 'L2', cards))))
            # M2: self-consistency: 5 runs at temperature 0.7 (a run number keeps the cache from replaying one answer), Borda
            for run in range(5):
                jobs.append(('M2_L2_low_x5', key, 'list', lambda key=key, cards=cards, run=run: call(
                    f'M2_L2_low_x5_{key}_{run}', rank(GLM, 'low', .7), listwise_prompt(key, 'L2', cards) + f'\n(прогон {run + 1})')))
            # M4: two stages: the heuristic prescore keeps 10, the LLM ranks them at the full detail
            order = sorted(range(len(cards)), key=lambda i: -prescore(cards[i]))[:10]
            sub = [cards[i] for i in order]
            for eff in ('low', 'high'):
                jobs.append((f'M4_pre10_L3_{eff}', key, ('map', order), lambda key=key, sub=sub, eff=eff: call(
                    f'M4_pre10_L3_{eff}_{key}', rank(GLM, eff), listwise_prompt(key, 'L3', sub))))
        elif phase in 'BC':
            # M3: pointwise scoring of every card at L3, then code picks the top with diversity
            for eff in (('low',) if phase == 'B' else ('high',)):
                for n, c in enumerate(cards):
                    prompt = f"{profile(key)}\n\n{RUBRIC}\n\nОцени одну карточку-инсайт по шкале 1–10.\n\n{c['L3']}"
                    jobs.append((f'M3_L3_{eff}', key, ('point', n), lambda key=key, prompt=prompt, eff=eff, n=n: call(
                        f'M3_L3_{eff}_{key}_{n}', structured(model(GLM, eff, max_tokens=4000), Score), prompt)))
            # M5: pairwise duels of the prescore top-6 in both orders (position bias), wins counted, then diversify
            order = sorted(range(len(cards)), key=lambda i: -prescore(cards[i]))[:6]
            for a, b in (itertools.combinations(order, 2) if phase == 'B' else ()):
                for x, y in ((a, b), (b, a)):
                    prompt = (f"{profile(key)}\n\n{RUBRIC}\n\nКакая карточка ценнее для этого читателя? Ответь A или B.\n\n"
                              f"[A]\n{cards[x]['L2']}\n\n[B]\n{cards[y]['L2']}")
                    jobs.append(('M5_pair_L2_low', key, ('pair', x, y), lambda key=key, prompt=prompt, x=x, y=y: call(
                        f'M5_pair_L2_low_{key}_{x}_{y}', structured(model(GLM, 'low', max_tokens=3000), Winner), prompt)))
    return jobs


def run_phase(phase, pools, state):
    jobs = build_jobs(phase, pools)
    t0 = time.time()
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        futs = [(cfg, key, kind, ex.submit(fn)) for cfg, key, kind, fn in jobs]
        raw = [(cfg, key, kind, f.result()) for cfg, key, kind, f in futs]
    print(f'phase {phase}: {len(jobs)} calls in {time.time() - t0:.0f} s', flush=True)
    state.setdefault('raw', []).extend(dict(cfg=c, key=k, kind=str(kd), sec=r[1], err=r[2], out=(r[0].model_dump() if r[0] else None))
                                       for c, k, kd, r in raw)
    return raw


def aggregate(raw, pools):
    res, lists, points, wins = defaultdict(dict), defaultdict(list), defaultdict(dict), defaultdict(lambda: defaultdict(float))
    for cfg, key, kind, (out, sec, err) in raw:
        if out is None:
            continue
        if kind == 'list':
            lists[(cfg, key)].append(parse_top(out, len(pools[key])))
        elif kind[0] == 'map':
            lists[(cfg, key)].append([kind[1][i] for i in parse_top(out, len(kind[1]))])
        elif kind[0] == 'point':
            points[(cfg, key)][kind[1]] = out
        elif kind[0] == 'pair':
            wins[(cfg, key)][kind[1] if out.winner.strip().upper().startswith('A') else kind[2]] += 1
    for (cfg, key), ls in lists.items():
        ls = [x for x in ls if x]
        if ls:
            res[cfg][key] = borda(ls) if len(ls) > 1 else ls[0]
    for (cfg, key), sc in points.items():
        cards = pools[key]
        score = {n: (.5 * s.value + .25 * s.urgency + .25 * s.relevance) / 10 for n, s in sc.items()}
        res[cfg + '_raw'][key] = sorted(score, key=lambda i: -score[i])[:5]
        themes, chosen, left = {n: s.theme.strip().lower() for n, s in sc.items()}, [], sorted(score, key=lambda i: -score[i])
        while left and len(chosen) < 5:
            best = max(left, key=lambda i: score[i] - .12 * sum(themes[i] == themes[j] for j in chosen)
                       - .3 * sum(overlap(cards[i]['title'], cards[j]['title']) for j in chosen))
            chosen.append(best)
            left.remove(best)
        res[cfg + '_div'][key] = chosen
    for (cfg, key), w in wins.items():
        cards = pools[key]
        res[cfg + '_raw'][key] = sorted(w, key=lambda i: -w[i])[:5]
        res[cfg + '_div'][key] = diversify(cards, [w.get(i, 0) / 5 + .01 * prescore(cards[i]) for i in range(len(cards))])
    return res


if __name__ == '__main__':
    phases = sys.argv[1] if len(sys.argv) > 1 else 'A'
    pools = {k: json.loads((DATA / f'pool_{k}.json').read_text(encoding='utf-8')) for k in KEYS}
    spent0 = key_info()['spend']
    start = LOG.stat().st_size if LOG.exists() else 0
    state_file = DATA / 'state.json'
    state = json.loads(state_file.read_text(encoding='utf-8')) if state_file.exists() else {}
    state.setdefault('spent_by_phase', {})
    for ph in phases:
        print('key spend before phase', round(key_info()['spend'] - spent0, 2), flush=True)
        raw = run_phase(ph, pools, state)
        per, total = log_costs(start)
        state['spent_by_phase'][ph] = round(total, 2)
        state.setdefault('per_config_cost', {}).update({k: round(v, 3) for k, v in per.items()})
        res = aggregate(raw, pools)
        state.setdefault('results', {}).update({cfg: {k: evaluate(k, top) for k, top in by.items()} for cfg, by in res.items()})
        state_file.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding='utf-8')
        print('billed so far this process: %.2f ₽' % total, flush=True)
        if total > BUDGET_STOP:
            print('budget guard hit, stopping')
            break
    print('done; key spend delta', round(key_info()['spend'] - spent0, 2))
