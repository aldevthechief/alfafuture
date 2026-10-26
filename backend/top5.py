"""Default top-5 of a reader's feed: 3 news + 2 bank-analytics cards as a *soft* quota, ranked by one listwise LLM call.
The LLM ranks the whole pool (top-10 by value for this company and role); code walks that order, fills each group's quota only
from LLM-ranked cards, skips duplicates (same event, shared news ids, TF-IDF cosine of the short text) and, when a group runs short
or its cards are not in the LLM top-10, the best remaining card of the other group takes the place. The result keeps the LLM order:
no group sits before another by rule. Experiments and numbers: experiments/insight_ranking/REPORT.md."""
import hashlib
import json
import os
import re
import threading
from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field, create_model

from common.clock import today
from common.text import stems

QUOTA = {'news': 3, 'bank': 2}
COS, JAC = 0.30, 0.5
RANK_TOP = 10
HORIZON_DAYS, EXPIRING_DAYS = 30, 7  # prototype: every insight lives a month; the last week of it ranks below the fresh ones
_LOCKS: dict[str, threading.Lock] = {}

RUBRIC = ("Ценность карточки для этого читателя = (1) связь с бизнесом именно этой компании, (2) существенность для денег и решений, "
          "(3) срочность: срок или событие в ближайшие недели, свежесть на сегодня {today}, (4) конкретность действия для роли, "
          "(5) надёжность (уверенность, опора на данные). Карточки про одно и то же событие или механизм дублируют друг друга.")


class Pick(BaseModel):
    id: str = Field(description='id карточки из списка, например C07')


class Ranking(BaseModel):
    top: list[Pick] = Field(min_length=RANK_TOP, max_length=RANK_TOP, description=f'ровно {RANK_TOP} карточек, от самой ценной к менее ценной')


def age_days(card) -> int:
    try:
        return (today() - date.fromisoformat((card.get('published') or '')[:10])).days
    except ValueError:
        return 0  # no date: not aged out


def _texts(items, n=None):
    return [x['text'] if isinstance(x, dict) else str(x) for x in (items or [])][:n]


def card(c, row):
    """Pool card: preview fields + the signals of the row + a short (L2) and a full (L3) text."""
    i = row['insight']
    s4 = (row.get('stage4') or {}).get('relevance') or {}
    ret, d = row.get('retrieval') or {}, row.get('decision') or {}
    acts = [a['text'] if isinstance(a, dict) else str(a) for a in (i.get('actions') or [])]
    why, impact = _texts(i.get('why_it_matters'), 4), (i.get('impact') or {}).get('level')
    published = (c.get('published') or '')[:10]
    group = 'bank' if (row.get('source') == 'bank_analytics' or i.get('origin') == 'bank') else 'news'
    news = sorted({str(n) for n in [row.get('news_id'), *(d.get('related_news_ids') or [])] if n})
    return dict(id=c['id'], title=c['title'], channel=c['channel'], group=group, event_id=ret.get('event_id'), news=news,
                L2=f"{c['title']}\nПочему важно: {why[0] if why else ''}\nДействия: {'; '.join(acts[:2])}",
                L3='\n'.join([c['title'], f"Канал: {c['channel']}; уверенность: {i.get('confidence')}; влияние: {impact}; дата: {published}",
                              'Что произошло: ' + ' '.join(_texts(i.get('what_happened'), 3)), 'Почему важно: ' + ' '.join(why),
                              'Риски: ' + ' | '.join(_texts(i.get('risks'), 3)), 'Возможности: ' + ' | '.join(_texts(i.get('opportunities'), 3)),
                              'Действия: ' + ' | '.join(acts[:3]), 'Позиция Банка: ' + ' '.join(_texts(i.get('bank_view'), 2))]),
                rel=s4.get('score') or 2, impact=impact, published=published)


def lifecycle(card) -> str:
    age = age_days(card)
    return 'stale' if age > HORIZON_DAYS else 'expiring' if age > HORIZON_DAYS - EXPIRING_DAYS else 'active'


def similarity(cards):
    from sklearn.feature_extraction.text import TfidfVectorizer
    m = TfidfVectorizer(analyzer=stems).fit_transform([c['L2'] for c in cards])
    return (m @ m.T).toarray()


def duplicate_of(i, chosen, cards, sim):
    for j in chosen:
        a, b = cards[i], cards[j]
        if a['event_id'] and a['event_id'] == b['event_id']:
            return j, 'event'
        if a['news'] and b['news'] and len(set(a['news']) & set(b['news'])) / len(set(a['news']) | set(b['news'])) >= JAC:
            return j, 'news'
        if sim[i][j] >= COS:
            return j, f'tfidf {sim[i][j]:.2f}'
    return None


def select(cards, order, sim, quota=QUOTA, k=5):
    """-> (chosen indices in LLM order, [(skipped index, duplicate of, why)]). Quota phase uses LLM-ranked cards only."""
    chosen, skipped = [], []
    rest = [i for i in range(len(cards)) if i not in order]
    state = [lifecycle(c) for c in cards]
    # expired cards never take a seat while a live one is left; the last week of a card's life ranks below the fresh ones
    pools = [('quota', [i for i in order if state[i] == 'active']), ('fill', [i for i in order + rest if state[i] == 'active']),
             ('fill', [i for i in order + rest if state[i] == 'expiring']), ('fill', [i for i in order + rest if state[i] == 'stale'])]
    for phase, pool in pools:
        for i in pool:
            if len(chosen) >= k or i in chosen or any(i == s[0] for s in skipped):
                continue
            if phase == 'quota' and sum(cards[j]['group'] == cards[i]['group'] for j in chosen) >= quota[cards[i]['group']]:
                continue
            dup = duplicate_of(i, chosen, cards, sim)
            if dup:
                skipped.append((i, *dup))
                continue
            chosen.append(i)
    chosen.sort(key=lambda i: ({'active': 0, 'expiring': 1, 'stale': 2}[state[i]], order.index(i) if i in order else len(order) + i))
    return chosen, skipped


def prescore_order(cards):
    return sorted(range(len(cards)), key=lambda i: -(cards[i]['rel'] / 3 + {'high': 1, 'mid': .6, 'low': .3}.get(cards[i]['impact'], .5)))


def profile(company, role):
    try:
        from common.data import load, money
        from common.roles import roles
        r = load()[0].loc[company]
        return f"Компания: вид деятельности «{r.MAINOKVED_NAME}», выручка {money(r.REVENUE_COMPANY)} за 2025.\nЧитатель: {roles()[role]['title']}."
    except Exception:  # company tables are not installed in every deployment
        return f'Компания {company}. Читатель: {role}.'


def llm_order(company, role, cards, models_factory=None):
    from common.llm.models import ChatModels, structured
    model = (models_factory or ChatModels)()._make('z-ai/glm-5.3-flash', 0.0, 8000).model_copy(update={'reasoning_effort': 'low'})
    count=min(RANK_TOP,len(cards))
    schema=create_model(f'Ranking{count}',top=(list[Pick],Field(min_length=count,max_length=count)))
    body = '\n\n'.join(f"[C{n:02d}] {c['L3']}" for n, c in enumerate(cards))
    prompt = (f"[top5]\n{profile(company, role)}\n\n{RUBRIC.format(today=today().isoformat())}\n\nНиже {len(cards)} готовых карточек-инсайтов "
              f"этой компании. Выбери {count} самых ценных без повторений и расположи по убыванию ценности.\n\n{body}")
    order = []
    for p in structured(model, schema).invoke(prompt).top:
        m = re.search(r'\d+', p.id)
        if m and int(m.group()) < len(cards) and int(m.group()) not in order:
            order.append(int(m.group()))
    if len(order)!=count: raise ValueError('Ranking contains missing, repeated or unknown card IDs')
    return order


def rank(company, role, cards, root: Path, *, cache_only=False, models_factory=None):
    """-> dict(ids: top-5 card ids, order: all ids best first, method, skipped). Cached by the set of card ids and their texts."""
    if len(cards) <= 5:  # nothing to choose: no model call, the lifecycle order only
        chosen, skipped = select(cards, prescore_order(cards), similarity(cards) if len(cards) > 1 else [[1.0]] * len(cards))
        return dict(ids=[cards[i]['id'] for i in chosen], order=[cards[i]['id'] for i in chosen], method='all', lifecycle={},
                    skipped=[dict(id=cards[i]['id'], duplicate_of=cards[j]['id'], why=w) for i, j, w in skipped])
    key = hashlib.sha1(json.dumps(['integration-v1',company, role, today().isoformat(), cards],ensure_ascii=False,sort_keys=True).encode()).hexdigest()[:20]
    path = Path(root) / 'top5' / f'{company}_{role}_{key}.json'
    with _LOCKS.setdefault(key, threading.Lock()):
        if path.exists():
            try:
                saved=json.loads(path.read_text(encoding='utf-8'))
                ids={c['id'] for c in cards}
                if (isinstance(saved,dict) and isinstance(saved.get('order'),list)
                        and all(isinstance(i,str) for i in saved['order'])
                        and len(saved['order'])==len(ids) and set(saved['order'])==ids
                        and isinstance(saved.get('ids'),list) and len(saved['ids'])<=5
                        and all(isinstance(i,str) and i in ids for i in saved['ids'])):
                    return saved
            except (OSError,ValueError):
                pass  # stale/corrupt optional ranking: fall back without losing the feed
        sim, method = similarity(cards), 'llm'
        try:
            if cache_only: raise LookupError('Cached ranking unavailable')
            order = llm_order(company, role, cards,models_factory)
        except Exception:  # a lost call must not empty the cabinet: heuristic order, not cached
            order, method = prescore_order(cards), 'prescore'
        chosen, skipped = select(cards, order, sim)
        tail = [i for i in order if i not in chosen] + [i for i in range(len(cards)) if i not in chosen and i not in order]
        out = dict(ids=[cards[i]['id'] for i in chosen], order=[cards[i]['id'] for i in chosen + tail], method=method,
                   lifecycle={c['id']: lifecycle(c) for c in cards if lifecycle(c) != 'active'},
                   skipped=[dict(id=cards[i]['id'], duplicate_of=cards[j]['id'], why=w) for i, j, w in skipped])
        if method == 'llm':
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')
        return out
