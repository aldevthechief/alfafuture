"""Candidate generators: each one answers «why could a news become an insight for this company» from one stage-2 fact
(«it hits the industry of a key supplier», «the company has loans in the bank», …), searches the pool on its own
(hybrid: TF-IDF over stems + dense encoder, plus optional HyDE headlines) and returns between n/k and l/k events.
Generators are not equal: `weight` sets each one's share of the [n, l] budget; weights are to be tuned post factum
by where the useful insights come from (backlog.md).
python -X utf8 stage3_news_search/candidate_generators/generators.py U_31505 [--role cfo] [--n 200 --l 800] [--hyde]
"""
import argparse, re, sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.data import DATA
from common.roles import role_channels
from common.search import news_index
from common.fusion import reciprocal_rank_fusion
from stage3_news_search.facet_retrieval.candidates import CHANNELS, DIV_CHANNELS, clean

# default share of the budget per reason; shortcut: hand-set, tune from insight yield per generator (backlog.md item 5)
WEIGHTS = {'own_industry': 1.0, 'supplier_industry': 0.7, 'customer_industry': 0.7, 'hypothesis': 0.7,
           'bank_product': 0.5, 'macro': 0.4, 'trade_country': 0.4, 'second_hop': 0.3, 'region': 0.3, 'group_industry': 0.3}
WHY = {'own_industry': 'регулирование, цены, спрос и конкуренция в отрасли компании',
       'supplier_industry': 'события, сильно влияющие на сферу ключевых поставщиков компании',
       'customer_industry': 'события, сильно влияющие на сферу ключевых покупателей компании',
       'second_hop': 'события у поставщиков и покупателей контрагентов компании',
       'group_industry': 'события в отраслях других компаний группы',
       'bank_product': 'условия по продукту банка, которым пользуется компания',
       'macro': 'макрофактор, к которому чувствителен бизнес компании',
       'trade_country': 'торговля и отношения со страной, с которой связана компания или её контрагенты',
       'region': 'экономика региона, где работает компания или её контрагенты',
       'hypothesis': 'гипотеза о бизнесе компании'}
PRODUCT_CHANNELS = {'кредитные продукты': 'rate', 'лизинг': 'leasing', 'депозиты': 'deposit', 'банковские гарантии': 'guarantee',
                    'зарплатный проект': 'labor'}
SHARE = re.compile(r'\((<?\d+)% суммы\)')
OKVED = re.compile(r'^\d{2}(\.\d+)*$')


@dataclass
class Generator:
    reason: str
    query: str
    weight: float
    facts: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:  # natural-language query for the encoder and HyDE
        return f'{WHY[self.reason]}: {self.query}'


def _share(text: str) -> int:
    m = SHARE.search(text)
    return 0 if not m or m.group(1).startswith('<') else int(m.group(1))


def generators(facts: list[dict], hypotheses: list[str] = (), role: str | None = None,
               min_share: int = 5, min_region_share: int = 10) -> list[Generator]:
    """facts: stage-2 records {cat, text, depth, entities}. Same query from several facts -> one generator (max weight)."""
    out: dict[str, Generator] = {}

    def add(reason, query, fact, weight=None):
        query = clean(query) if reason.endswith('industry') or reason == 'second_hop' else query
        if not query:
            return
        w = WEIGHTS[reason] if weight is None else weight
        g = out.get(query)
        if g is None or w > g.weight:
            out[query] = Generator(reason, query, w, (g.facts if g else []) + [fact])
        else:
            g.facts.append(fact)

    division, countries = None, False
    for f in facts:
        cat, text, ent = f['cat'], f['text'], f.get('entities', [])
        if cat == 'activity' and text.startswith('Основной вид деятельности') and len(ent) >= 2:
            division = ent[0][:2]
            add('own_industry', ent[1], text)
        elif cat in ('payments_out', 'payments_in') and len(ent) == 2 and OKVED.match(ent[0]) and _share(text) >= min_share:
            add('supplier_industry' if cat == 'payments_out' else 'customer_industry', ent[1], text)
        elif cat in ('payments_out', 'payments_in') and len(ent) == 1 and 'региона' in text and _share(text) >= min_region_share:
            add('region', f'{ent[0]} экономика предприятия бизнес', text)
        elif cat in ('payments_out', 'payments_in') and 'участники ВЭД' in text:
            for c in ent:
                add('trade_country', f'{c} торговля поставки пошлины санкции логистика', text)
        elif cat == 'payments_hop2':
            for name in ent[:2]:
                add('second_hop', name, text)
        elif cat in ('trade', 'group_trade') and ent and ent != ['ВЭД']:
            countries = True
            for c in ent:
                add('trade_country', f'{c} торговля поставки пошлины санкции логистика', text)
        elif cat == 'geo' and len(ent) >= 2:
            add('region', f'{ent[1]} экономика предприятия бизнес', text)
        elif cat == 'group_activity' and len(ent) >= 2:
            add('group_industry', ent[1], text)
        elif cat == 'bank' and text.startswith('Продукты'):
            for p in ent:
                if p in PRODUCT_CHANNELS:
                    add('bank_product', CHANNELS[PRODUCT_CHANNELS[p]], text)
    macro = set(DIV_CHANNELS.get(division, [])) | ({'fx'} if countries else set()) | set(role_channels(role))
    for c in sorted(macro):
        add('macro', CHANNELS[c], f'макроканал {c}: раздел ОКВЭД {division}, ВЭД, роль {role}')
    for h in hypotheses:
        add('hypothesis', h, h)
    gens = list(out.values())
    n_of = pd.Series([g.reason for g in gens]).value_counts()
    for g in gens:  # a reason's weight is its budget share: ten trade countries must not outweigh the own industry
        g.weight /= n_of[g.reason]
    return gens


def select(scores: np.ndarray, sims: np.ndarray, lo: int, hi: int, threshold: float) -> np.ndarray:
    """Indices of events for one generator: the top `hi` by fused score, of which those beyond the first `lo`
    must also pass the dense-similarity threshold. Keeps >= lo for a thin news flow, <= hi to spare the rankers."""
    order = np.argsort(-scores)[:hi]
    return order[(np.arange(len(order)) < lo) | (sims[order] >= threshold)]


def events_frame(news: pd.DataFrame | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """events.parquet (stage 4 clustering) and, for each news_pool row, the index of its event."""
    from common.search import news_pool
    from stage4_news_relevance.plausibility.events import ensure_events
    news = news_pool() if news is None else news
    ev = ensure_events(news).reset_index(drop=True)
    pos = {str(i): k for k, ids in enumerate(ev.news_ids) for i in ids}
    return ev, np.array([pos.get(str(i), -1) for i in news['id']])


def candidates(facts: list[dict], hypotheses: list[str] = (), role: str | None = None, n: int = 200, l: int = 800,
               threshold: float = 0.55, hyde=None, rrf_k: int = 60) -> tuple[pd.DataFrame, list[Generator]]:
    """Union of all generators' events. hyde: optional HydeQueryWriter — its headlines are extra queries per generator.
    Returns events with the generators that found them (reason, query, rank) and the generators list."""
    from common.embed import encode, news_embeddings
    gens = generators(facts, hypotheses, role)
    df, index = news_index()
    ev, event_of = events_frame(df)
    if not gens or ev.empty:
        return ev.head(0).assign(generators=[], best_rank=[]), gens
    if (event_of < 0).any():
        raise ValueError('Event membership does not match the news corpus')
    emb = news_embeddings(news=df)
    from common.chunk_search import chunk_index
    chunks = chunk_index(df)
    chunk_scores = chunks.article_scores([g.query for g in gens])
    from common.news_fact_index import FactIndex, FactStore
    fact_index = FactIndex(df, FactStore().records(df))
    total = sum(g.weight for g in gens)
    picked: dict[int, list] = {}
    for gi, g in enumerate(gens):
        queries = [g.text] + (hyde.queries(g.text) if hyde else [])
        lex = (index.matrix @ index.transform([g.query] + queries[1:]).T).toarray().max(1)
        lex = np.maximum(lex, chunk_scores[:, gi])
        dense = (emb @ encode(queries).T).max(1)
        fact_scores = fact_index.scores(g.query)
        fused = reciprocal_rank_fusion(lex, dense, np.where(fact_scores > 0, fact_scores, np.nan), k=rrf_k)
        ev_fused = pd.Series(fused).groupby(event_of).max().reindex(range(len(ev)), fill_value=0).to_numpy()
        ev_dense = pd.Series(dense).groupby(event_of).max().reindex(range(len(ev)), fill_value=0).to_numpy()
        lo, hi = max(1, round(n * g.weight / total)), max(1, round(l * g.weight / total))
        for rank, e in enumerate(select(ev_fused, ev_dense, lo, hi, threshold)):
            picked.setdefault(int(e), []).append({'reason': g.reason, 'query': g.query[:80], 'rank': rank, 'sim': round(float(ev_dense[e]), 3)})
    res = ev.loc[list(picked)].assign(generators=[picked[e] for e in picked],
                                      best_rank=[min(x['rank'] for x in picked[e]) for e in picked])
    from common.search import corpus_fingerprint
    res = res.sort_values('best_rank')
    res.attrs['corpus_fingerprint'] = corpus_fingerprint(df)
    return res, gens


if __name__ == '__main__':
    from stage2_company_facts.deterministic_facts.facts import build
    ap = argparse.ArgumentParser()
    ap.add_argument('company')
    ap.add_argument('--role')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--l', type=int, default=800)
    ap.add_argument('--threshold', type=float, default=0.55)
    ap.add_argument('--hyde', action='store_true', help='add HyDE headlines (needs LLM_API_KEY)')
    a = ap.parse_args()
    hyde = None
    if a.hyde:
        from stage3_news_search.hyde_queries.hyde import HydeQueryWriter
        hyde = HydeQueryWriter()
    res, gens = candidates(build([a.company])[a.company], role=a.role, n=a.n, l=a.l, threshold=a.threshold, hyde=hyde)
    print(f'{len(gens)} generators -> {len(res)} events')
    by = pd.DataFrame([x for xs in res.generators for x in xs])
    for g in gens:
        k = by[by['query'] == g.query[:80]]
        print(f'  {g.reason:17} w={g.weight:.2f} kept={len(k):4d} | {g.query[:70]}')
    for r in res.head(15).itertuples():
        print(f'{r.best_rank:3d} {str(r.first_date)[:10]} n={r.n_reports} {r.generators[0]["reason"]:17} | {r.title[:100]}')
