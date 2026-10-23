"""Snowball expansion of a company profile: from one table row to as many related terms and news as the corpus supports.

Round 0 seeds = what stage 2 already knows (activities, goods, inputs, factor keywords, region, FTS countries,
counterparty sectors). Each round searches the news pool for the new terms (common.search, TF-IDF over stems) and
proposes the next terms by pseudo-relevance feedback: words and bigrams that are frequent in the found news and rare
in the corpus. A term is accepted only with support in >= `min_support` found news; its weight decays per round.
Stops on saturation (fewer than `min_new` new terms) or after `rounds`. Optional `--llm`: one cheap-model call
proposes extra seed terms from the card (traced to Langfuse by common.llm).
Every term and news keeps its origin, so a lead can be traced back to the profile field it grew from.
The company itself is never searched for: the data is anonymised, expansion goes through sector, context and links.
Usage: python -X utf8 stage2_company_facts/expansion/expand.py U_42222 [--rounds 2] [--llm]
"""
import argparse, json, re, sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.data import DATA
from common.search import news_document, news_index
from common.text import STOP

GENERIC = set('года году году течение января февраля марта апреля мая июня июля августа сентября октября ноября декабря '
              'также ранее сообщает сообщил сообщила заявил заявила отметил отметила словам данным фото источник время '
              'является будут может могут более менее около около свыше этого этом этой которые который которая '
              'тысячу тысяч тысячи процентов процента рублей млрд млн тыс компании компания компаний россии российских россия рф '
              'новости новость интерфакс тасс риа ведомости коммерсант bfm'.split()) | STOP
DECAY = 0.6


@lru_cache(None)
def term_space():
    """Word uni/bigram TF-IDF over the news pool: candidate terms for feedback (readable words, not stems)."""
    df, index = news_index()
    vec = TfidfVectorizer(token_pattern=r'(?u)\b[а-яё][а-яё-]{3,}\b', ngram_range=(1, 2), min_df=3, max_df=0.05,
                          sublinear_tf=True, lowercase=True)
    X = vec.fit_transform(news_document(df).tolist())
    return df, index, vec, X, np.array(vec.get_feature_names_out())


def seeds(profile: dict) -> dict[str, tuple[float, str]]:
    """term -> (weight, origin, facet). Factor keywords inherit the factor weight; the facet comes from the factor rule."""
    out = {}

    def put(term, w, origin, facet):
        t = term.strip().lower()
        if len(t) >= 3 and not re.fullmatch(r'[\d.]+', t) and (t not in out or out[t][0] < w):
            out[t] = (w, origin, facet)

    st = profile['search_terms']
    for k in ('goods', 'inputs'):
        for t in st.get(k, []):
            put(t, 1.0, f'seed:search_terms.{k}', 'sector')
    for t in st.get('activities', []):
        put(t, 0.8, 'seed:search_terms.activities', 'sector')
    for t in st.get('regions', []):
        put(t, 0.5, 'seed:search_terms.regions', 'geo')
    for t in st.get('trade_countries', []):
        put(t, 0.5, 'seed:search_terms.trade_countries', 'trade')
    for f in profile.get('external_factors', []):
        for t in f['keywords']:
            put(t, f['weight'], f'seed:{f["id"]}', FACET.get(f['rule'].split(':')[0], 'macro'))
    return out


FACET = {'sector': 'sector', 'chain': 'chain', 'fts': 'trade', 'ownership': 'trade', 'geo': 'geo', 'product': 'macro', 'base': 'macro'}


def llm_seeds(profile: dict, n: int = 15) -> list[str]:
    """One cheap call: extra plain-word search terms for the sector and its chain (no company names)."""
    from pydantic import BaseModel
    from common.llm import ChatModels, structured

    class Terms(BaseModel):
        terms: list[str]

    c = profile['card']
    prompt = (f'Компания (обезличена): ОКВЭД {c["okved"]} «{c["okved_name"]}», регион {c["region"]}, '
              f'страны ВЭД: {", ".join(c["fts_countries"]) or "нет"}, отрасли поставщиков по платежам: {", ".join(c["pay_out"]) or "нет"}.\n'
              f'Дай {n} коротких поисковых запросов (1–3 слова) для поиска российских новостей о событиях, '
              'которые влияют на бизнес такой компании: товары, сырьё, цены, регуляторы, законы, рынки сбыта, логистика. '
              'Без названий конкретных компаний.')
    res = structured(ChatModels().cheap(), Terms).invoke(prompt)
    return [t for t in res.terms if isinstance(t, str)][:n]


def search(terms: dict[str, float], k: int, min_score: float) -> dict[int, dict]:
    """news row -> {score, terms}: sum of term weight x cosine over the terms that found it."""
    _, index, *_ = term_space()
    hits = {}
    for t, w in terms.items():
        for row, s in index.search(t, k):
            if s < min_score:
                continue
            h = hits.setdefault(row, {'score': 0.0, 'terms': []})
            h['score'] += w * s
            h['terms'].append(t)
    return hits


@lru_cache(None)
def templated() -> np.ndarray:
    """News whose title opens like >= 20 others (exchange press releases, 'Курс юаня на ...'): kept as hits, never feedback."""
    df, *_ = term_space()
    key = df.title.fillna('').str.lower().str.split().str[:3].str.join(' ')
    return (key.map(key.value_counts()) >= 20).to_numpy()


@lru_cache(None)
def doc_freq() -> np.ndarray:
    _, _, _, X, _ = term_space()
    return np.asarray((X > 0).mean(axis=0)).ravel()


def feedback(rows: list[int], known: set[str], top: int, min_support: int, min_lift: float
             ) -> list[tuple[str, float, int, list[int]]]:
    """Terms frequent in the found news and rare in the corpus: (term, score, support, news rows).
    lift = share of found news with the term / share of all news with it; low lift = generic word, rejected."""
    if not rows:
        return []
    _, _, _, X, names = term_space()
    sub = X[rows]
    support = np.asarray((sub > 0).sum(axis=0)).ravel()
    lift = support / len(rows) / np.maximum(doc_freq(), 1e-9)
    score = np.asarray(sub.mean(axis=0)).ravel() * np.log1p(support) * np.log1p(lift)
    out = []
    for j in np.argsort(-score):
        if score[j] <= 0 or len(out) >= top:
            break
        t = names[j]
        if support[j] < min_support or lift[j] < min_lift or t in known or any(w in GENERIC for w in t.split()) \
                or any(t in x or x in t for x in known):
            continue
        out.append((t, float(score[j]), int(support[j]), [rows[i] for i in sub[:, j].nonzero()[0]]))
    return out


def snowball(seed: dict[str, tuple[float, str, str]], rounds: int, k: int, min_score: float, per_round: int,
             min_support: int, min_new: int, min_lift: float, anchor_share: float) -> tuple[dict, dict, list]:
    """One expansion from `seed` (term -> (weight, origin, facet)): (terms, news by row, round stats).
    anchor: similarity of each news to all seed terms at once; news found from round 1 on must reach
    `anchor_share` x the median anchor of round-0 news, so later rounds cannot drift off the seed's context."""
    df, index, *_ = term_space()
    facet = next(iter(seed.values()))[2] if seed else ''
    terms = {t: {'term': t, 'weight': round(w, 3), 'round': 0, 'origin': o, 'facet': fc, 'support': None, 'from_news': []}
             for t, (w, o, fc) in seed.items()}
    news, stats, frontier = {}, [], {t: v['weight'] for t, v in terms.items()}
    anchor = index.scores(' '.join(terms))
    floor = 0.0
    for r in range(rounds + 1):
        queried = len(frontier)
        hits = search(frontier, k, min_score)
        if r == 0:
            floor = anchor_share * float(np.median([anchor[i] for i in hits])) if hits else 0.0
        else:
            hits = {i: h for i, h in hits.items() if anchor[i] >= floor}
        new_news = 0
        for row, h in hits.items():
            n = news.setdefault(row, {'row': row, 'score': 0.0, 'terms': [], 'first_round': r})
            new_news += n['score'] == 0
            n['score'] += h['score']
            n['terms'] = list(dict.fromkeys(n['terms'] + h['terms']))
        # feedback only from the on-topic core: >= 2 distinct terms matched, not a template
        core = [i for i in hits if len(news[i]['terms']) >= 2 and not templated()[i]]
        found = sorted(core, key=lambda x: -news[x]['score'])[:100]
        proposed = feedback(found, set(terms), per_round, min_support, min_lift) if r < rounds else []
        top = max((s for _, s, _, _ in proposed), default=1.0)
        frontier = {}
        for t, s, sup, rows in proposed:
            w = round(DECAY ** (r + 1) * s / top, 3)
            terms[t] = {'term': t, 'weight': w, 'round': r + 1, 'origin': f'news_feedback:round{r}', 'facet': facet,
                        'support': sup, 'from_news': [int(df.id.iat[i]) for i in rows[:5]]}
            frontier[t] = w
        stats.append({'facet': facet, 'round': r, 'queried_terms': queried, 'news_found': len(hits), 'new_news': int(new_news),
                      'new_terms': len(frontier), 'total_terms': len(terms), 'total_news': len(news)})
        if len(frontier) < min_new:
            break
    return terms, news, stats


def expand(profile: dict, rounds: int = 2, faceted: bool = True, k: int = 15, min_score: float = 0.08,
           per_round: int = 20, min_support: int = 2, min_new: int = 3, min_lift: float = 5.0, anchor_share: float = 1.0,
           use_llm: bool = False) -> dict:
    """faceted: seeds are split by facet (own sector, counterparty chain, trade, geo, macro, llm) and each facet grows on
    its own anchor with per_round / 2 new terms, so a loud facet (currency, mortgage) cannot pull the others off topic."""
    df, *_ = term_space()
    seed = seeds(profile)
    if use_llm:
        for t in llm_seeds(profile):
            seed.setdefault(t.strip().lower(), (0.7, 'seed:llm', 'llm'))
    groups = {}
    for t, v in seed.items():
        groups.setdefault(v[2] if faceted else 'all', {})[t] = v
    terms, news, stats = {}, {}, []
    for fc, g in groups.items():
        t, n, s = snowball(g, rounds, k, min_score, per_round // 2 if faceted else per_round, min_support, min_new,
                           min_lift, anchor_share)
        for x, v in t.items():
            terms.setdefault(x, v)
        for row, v in n.items():
            m = news.setdefault(row, {'row': row, 'score': 0.0, 'terms': [], 'first_round': v['first_round'], 'facets': []})
            m['score'] += v['score']
            m['terms'] = list(dict.fromkeys(m['terms'] + v['terms']))
            m['first_round'] = min(m['first_round'], v['first_round'])
            m['facets'].append(fc)
        stats += s
    ranked = sorted(news.values(), key=lambda x: -x['score'])
    return {
        'id': profile['id'],
        'terms': sorted(terms.values(), key=lambda x: (x['round'], -x['weight'])),
        'news': [{'news_id': int(df.id.iat[n['row']]), 'title': df.title.iat[n['row']], 'date': str(df.date.iat[n['row']]),
                  'source': df.source.iat[n['row']], 'score': round(n['score'], 3), 'first_round': n['first_round'],
                  'facets': n['facets'], 'terms': n['terms'][:8]} for n in ranked],
        'rounds': stats,
        'saturated': all(s['new_terms'] < min_new for s in stats if s['round'] == max(x['round'] for x in stats if x['facet'] == s['facet'])),
    }


if __name__ == '__main__':
    from stage2_company_facts.profile.profile import profiles
    ap = argparse.ArgumentParser()
    ap.add_argument('ids', nargs='+')
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--llm', action='store_true', help='one cheap LLM call for extra seed terms (Langfuse-traced)')
    ap.add_argument('--out', default=str(DATA / 'companies' / 'expansion'))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for u, p in profiles(a.ids).items():
        res = expand(p, rounds=a.rounds, use_llm=a.llm)
        (out / f'{u}.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
        print(u, 'rounds:', [(s['round'], s['new_terms'], s['new_news']) for s in res['rounds']],
              f'-> {len(res["terms"])} terms, {len(res["news"])} news, saturated={res["saturated"]}')
