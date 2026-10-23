"""Offline check of the 10 proposed candidate generators (experiments/judge_v2/analysis.md §7) against the current ones,
U_106783. Same hybrid as the judge-v2 candidates: e5-large-instruct + qwen3-emb-0.6b (`Instruct: <reason>\nQuery: <q>`)
+ BM25 on pymorphy3 lemmas, RRF (k=60) over every (method, facet query) ranking of a generator, event = max of its news,
weighted quota union at the smallest budget giving >= N events. Queries are encoded locally (documents: Kaggle `out/`).
Targets: 13 judge-v2 positives + 28 useful negatives (experiments/judge_v2/useful_negatives.json).
Caveat: the new facet queries were written after reading the targets — an optimistic upper bound, not a fair test.
python -X utf8 experiments/retrieval/new_generators.py"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
import evaluate as ev
from push import REASON_INSTRUCTIONS

U = 'U_106783'
OKVED = 'электрической распределительной и регулирующей аппаратуры'
NEW = {  # reason: (instruction, facet queries); filled by hand from U_106783's facts (OKVED 27.12, foreign owner, FTS, leasing, payroll)
    'own_industry': (REASON_INSTRUCTIONS['own_industry'], [OKVED]),
    'ownership_foreign': ('Найди новости о регулировании и рисках для компаний с иностранным собственником',
                          ['компании с иностранным капиталом выход иностранных инвесторов правительственная комиссия',
                           'дивиденды нерезидентам счета типа С ограничения вывода',
                           'временное управление активами иностранных компаний']),
    'buyer_capex': ('Найди новости об инвестициях и закупках в отраслях, которые покупают продукцию компании',
                    ['инвестиционная программа сетевой компании строительство и реконструкция подстанций',
                     'строительство электростанций и объектов генерации энергосистема',
                     'новые промышленные объекты и центры обработки данных энергоснабжение подключение к сетям']),
    'product_regulation': ('Найди новости о регулировании продукции этой отрасли: локализация, реестры, стандарты, сборы',
                           ['локализация промышленной продукции баллы реестр российской продукции электротехническое оборудование',
                            'технический регламент сертификация электротехнической продукции',
                            'технологический сбор электроника электротехника']),
    'competition_import': ('Найди новости о конкурентах, импорте и ценах на продукцию этой отрасли',
                           ['производители электротехнического оборудования конкуренция рынок',
                            'импортозамещение электрооборудования уход иностранных производителей',
                            'цены на электротехническую продукцию']),
    'input_costs': ('Найди новости об издержках производителя этой отрасли: энергия, сырьё, комплектующие',
                    ['тарифы на электроэнергию для промышленности оптовый рынок цены',
                     'цены на медь и металлы для промышленности',
                     'дефицит электронных компонентов поставки комплектующих']),
    'sales_regions': ('Найди новости о крупных проектах в регионах России, где нужна продукция отрасли компании',
                      ['региональная программа модернизации электросетей',
                       'инфраструктурные проекты регионов строительство объектов',
                       'развитие энергосистемы региона']),
    'trade_regime': ('Найди новости о режиме внешней торговли России: санкции, пошлины, платежи',
                     ['санкции против России вторичные санкции пошлины',
                      'параллельный импорт ввоз оборудования ограничения',
                      'платежи с иностранными контрагентами расчёты в национальных валютах']),
    'fx_shift': ('Найди новости о сдвигах валютного курса и их причинах, а не ежедневные котировки',
                 ['Минфин покупки валюты бюджетное правило', 'ослабление рубля прогноз курса на год',
                  'Банк России решения валютный рынок']),
    'financing_terms': ('Найди новости об условиях финансирования для компаний-заёмщиков и лизингополучателей',
                        ['ставки по кредитам для бизнеса льготное кредитование субсидирование ставки',
                         'условия лизинга для лизингополучателей ставки авансы',
                         'Банк России требования к банкам риск-веса кредиты бизнесу']),
    'state_budget': ('Найди новости о федеральном и региональных бюджетах и госрасходах',
                     ['проект федерального бюджета расходы налоги', 'сокращение бюджетных расходов госзакупки нацпроекты',
                      'региональный бюджет инвестиции в инфраструктуру']),
    'labour': (REASON_INSTRUCTIONS['bank_product'], ['дефицит кадров зарплаты рынок труда мигранты']),  # kept from current
}
MODELS = {'e5-large-instruct': ('intfloat/multilingual-e5-large-instruct', 'Query: ', None),
          'qwen3-emb-0.6b': ('Qwen/Qwen3-Embedding-0.6B', 'Query:', 'left')}


def encode(slug, texts):
    """As kaggle_embed.run_dense: fp16, max_len 512, normalized; checked against the Kaggle query vectors in main."""
    import torch
    from sentence_transformers import SentenceTransformer
    name, _, pad = MODELS[slug]
    m = SentenceTransformer(name, device='cuda', model_kwargs={'torch_dtype': torch.float16},
                            tokenizer_kwargs={'padding_side': pad} if pad else None)
    m.max_seq_length = 512
    v = m.encode(texts, batch_size=16, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    del m; torch.cuda.empty_cache()
    return v


def bm25_scorer():
    import bm25s
    _, texts, _ = ev.corpus()
    tok = ev._lemmatizer()
    toks = [tok(t) for t in texts]
    vocab = {w: i for i, w in enumerate(sorted({w for ts in toks for w in ts}))}
    r = bm25s.BM25()
    r.index(bm25s.tokenization.Tokenized(ids=[[vocab[w] for w in ts] for ts in toks], vocab=vocab), show_progress=False)
    return lambda q: r.get_scores([vocab[w] for w in tok(q) if w in vocab] or [0])


def rank(s):
    return np.argsort(np.argsort(-s))


def candidate_set(gens, scores, n):
    """gens: [(name, weight)], scores: name -> fused doc scores. Smallest budget whose quota union has >= n events."""
    _, _, event = ev.corpus()
    real = set(pd.read_parquet(ev.DATA / 'news' / 'events.parquet').event)
    ev_score = {g: pd.Series(scores[g]).groupby(event).max() for g, _ in gens}
    order = {g: [e for e in s.sort_values(ascending=False).index if e in real] for g, s in ev_score.items()}
    w = np.array([wt for _, wt in gens])
    for b in range(n, 20 * n, 5):
        chosen = set()
        for (g, _), h in zip(gens, np.maximum(1, np.round(b * w / w.sum())).astype(int)):
            chosen.update(order[g][:h])
        if len(chosen) >= n:
            return chosen
    return chosen


if __name__ == '__main__':
    useful = json.loads((HERE.parents[0] / 'judge_v2/useful_negatives.json').read_text(encoding='utf-8'))['items']
    L = pd.read_pickle(HERE.parents[0] / 'judge_v2/out/single.pkl')
    L['p'] = (L.insight_potential >= 2) & (L.significance >= 2) & (L.mechanism_strength >= 2)
    v = L.groupby('event').p.agg(lambda s: s.sum() * 2 > len(s))
    targets = {'positives': set(v[v].index)} | {
        'useful strong+medium': {u['event'] for u in useful if u['tier'] != 'weak'}, 'useful all': {u['event'] for u in useful}}
    targets['positives + useful'] = targets['positives'] | targets['useful all']

    cur = [(i, q) for i, q in ev.company_queries()[U]]  # (UQ index, weight)
    gens = {f'cur:{ev.UQ[i]["reason"]}:{k}': (REASON_INSTRUCTIONS[ev.UQ[i]['reason']], [ev.UQ[i]['query']], w) for k, (i, w) in enumerate(cur)}
    gens |= {f'new:{r}': (ins, qs, 1.0) for r, (ins, qs) in NEW.items()}
    jobs = [(g, j, ins, q) for g, (ins, qs, _) in gens.items() for j, q in enumerate(qs)]

    D = {s: np.load(HERE / 'out' / f'{s}.npz')['doc'].astype(np.float32) for s in MODELS}
    S = {}  # (generator, facet, method) -> doc scores
    for slug, (_, sep, _) in MODELS.items():
        Q = encode(slug, [f'Instruct: {ins}\n{sep}{q}' for _, _, ins, q in jobs])
        for (g, j, _, _), qv in zip(jobs, Q):
            S[(g, j, slug)] = D[slug] @ qv
        kq = np.load(HERE / 'out' / f'{slug}.npz')[f'q_reason'].astype(np.float32)  # compatibility with the Kaggle vectors
        cos = [float(Q[[k for k, jb in enumerate(jobs) if jb[0] == f'cur:{ev.UQ[i]["reason"]}:{n}'][0]] @ kq[i]) for n, (i, _) in enumerate(cur)]
        print(f'{slug}: local vs Kaggle query cosine min {min(cos):.4f}', flush=True)
    bm = bm25_scorer()
    for g, j, _, q in jobs:
        S[(g, j, 'bm25')] = bm(q)

    def fused(g, facets=None):
        keys = [k for k in S if k[0] == g and (facets is None or k[1] < facets)]
        return sum(1 / (60 + rank(S[k])) for k in keys)

    cur_g = [g for g in gens if g.startswith('cur:')]
    new_g = [g for g in gens if g.startswith('new:')]
    sets = {
        'current (orig. weights)': ([(g, gens[g][2]) for g in cur_g], None),
        'current (equal weights)': ([(g, 1.0) for g in cur_g], None),
        'new, 1st facet only': ([(g, 1.0) for g in new_g], 1),
        'new, 3 facets RRF': ([(g, 1.0) for g in new_g], None),
        'current + new, 3 facets': ([(g, gens[g][2]) for g in cur_g] + [(g, 1.0) for g in new_g], None),
    }
    rows, found = [], {}
    for name, (gs, facets) in sets.items():
        sc = {g: fused(g, facets) for g, _ in gs}
        for n in (100, 200):
            cs = candidate_set(gs, sc, n)
            found[(name, n)] = cs
            rows.append({'set': name, 'budget': n, 'events': len(cs),
                         **{t: f'{len(cs & ts)}/{len(ts)}' for t, ts in targets.items()}})
    print(pd.DataFrame(rows).to_string(index=False))
    title = {u['event']: (u['title'], u['tier']) for u in useful}
    allt = targets['positives + useful']
    print('\nTargets found at budget 200 (C = current orig., N = new 3 facets):')
    for e in sorted(allt):
        c, nn = e in found[('current (orig. weights)', 200)], e in found[('new, 3 facets RRF', 200)]
        t = title.get(e, (L[L.event == e].title.iloc[0], 'positive'))
        print(f"  {'C' if c else '-'}{'N' if nn else '-'} {e:5} {t[1]:8} {t[0][:90]}")

    # which targets each set could not have «by construction», labelled coverage, per-generator yield
    E = pd.read_pickle(HERE.parents[0] / 'judge_v2/out/consensus.pkl')
    judged, cand0 = set(E.index), set(E[E.source == 'candidate'].index)
    outside = allt - cand0
    print(f'\ntargets outside the judged candidate set: {len(outside)} (from the random sample)')
    for name in sets:
        for n in (100, 200):
            cs = found[(name, n)]
            lab = cs & judged
            print(f'  {name:26} {n}: outside-targets {len(cs & outside)}/{len(outside)}, labelled {len(lab)}/{len(cs)}, '
                  f'targets among labelled {len(cs & allt)}/{len(lab)} = {len(cs & allt) / max(1, len(lab)):.2f}')
    gs = sets['new, 3 facets RRF'][0]
    sc = {g: fused(g) for g, _ in gs}
    _, _, event = ev.corpus()
    real = set(pd.read_parquet(ev.DATA / 'news' / 'events.parquet').event)
    cs = found[('new, 3 facets RRF', 200)]
    print('\nnew generators at budget 200: own events in the set, labelled, targets')
    w = 200 // len(gs)
    for g, _ in gs:
        top = [e for e in pd.Series(sc[g]).groupby(event).max().sort_values(ascending=False).index if e in real][:w + 2]
        top = [e for e in top if e in cs]
        print(f'  {g:24} {len(top):3}  labelled {len(set(top) & judged):3}  targets {len(set(top) & allt):2}  '
              f'top-3: {" | ".join(E.title.get(e, "?")[:40] if e in judged else "(не размечено)" for e in top[:3])}')

    # events of the new sets at budget 200 without judge-v2 labels -> input for judge.py --events
    todo = sorted((found[('new, 1st facet only', 200)] | found[('new, 3 facets RRF', 200)]) - judged)
    (HERE.parents[0] / 'judge_v2/out/new_gen_unlabelled.json').write_text(json.dumps([int(e) for e in todo]))
    print(f'\n{len(todo)} unlabelled events of the new sets at 200 -> judge_v2/out/new_gen_unlabelled.json')
    (HERE.parents[0] / 'judge_v2/out/generator_sets.json').write_text(json.dumps(
        {f'{k[0]}|{k[1]}': sorted(int(e) for e in v) for k, v in found.items()}, ensure_ascii=False))

    # every generator's own ranked events (top 300, news events only) for attribution and overlap analysis
    _, _, event = ev.corpus()
    real = set(pd.read_parquet(ev.DATA / 'news' / 'events.parquet').event)
    tops = {}
    for g in gens:
        for f, tag in ((None, '3'), (1, '1')):
            if tag == '1' and g.startswith('cur:'):
                continue
            s = pd.Series(fused(g, f)).groupby(event).max().sort_values(ascending=False)
            tops[f'{g}|{tag}'] = [int(e) for e in s.index if e in real][:300]
    (HERE.parents[0] / 'judge_v2/out/generator_tops.json').write_text(json.dumps(
        {'tops': tops, 'weights': {g: gens[g][2] for g in gens},
         'queries': {g: {'instruction': gens[g][0], 'facets': gens[g][1]} for g in gens}}, ensure_ascii=False))
