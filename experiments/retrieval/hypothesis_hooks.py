"""Hooks from the research agent's hypotheses (U_106783, data/companies/agent/U_106783.json, 3rd live run) as candidate
generators, no LLM: the same hybrid as new_generators.py (e5-large-instruct + qwen3-emb-0.6b with an instruction,
BM25 on lemmas, RRF, event = max of its news). Two ways to use a hypothesis:
  raw  — the hypothesis text itself is the query (what generators.py's `hypothesis` reason does today);
  hook — a news angle derived from the hypothesis by hand (what would make an event matter if the hypothesis holds).
Each pick is marked against the current set (C) and the 10 proposed generators (N) at budget 200, and the judge-v2
targets (13 positives + 28 useful negatives). Labels exist only for ~200 events, so most picks are read by hand.
python -X utf8 experiments/retrieval/hypothesis_hooks.py [--top 10]"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
import evaluate as ev
from new_generators import MODELS, bm25_scorer, encode, rank

U = 'U_106783'
HOOKS = {  # hook: (hypotheses it comes from, instruction, query)
    'sales_channel': ('H1 H2 H8', 'Найди новости о рынке сбыта через дистрибьюторов и торговые дома в этой отрасли',
                      'дистрибьюторы и торговые дома электротехнического оборудования, рынок сбыта, банкротство дилеров'),
    'import_components': ('H4 H8', 'Найди новости о поставках импортных комплектующих для производителя',
                          'поставки импортных комплектующих и электронных компонентов из Китая, задержки, рост цен'),
    'foreign_payments': ('H4 H8', 'Найди новости о трудностях с международными платежами компаний',
                         'проблемы с платежами в Китай и ОАЭ, банки отклоняют платежи, платёжные агенты, комиссии'),
    'baltic_transit': ('H8 (FTS Литва)', 'Найди новости о транзите и логистике грузов через страну',
                       'транзит грузов через Литву, Калининград, ограничения на перевозки, граница'),
    'factoring': ('H5', 'Найди новости о факторинге и финансировании оборотного капитала поставщиков',
                  'рынок факторинга, финансирование дебиторской задолженности, ставки факторинга'),
    'liquidity': ('H6', 'Найди новости об условиях размещения свободных денег компаний',
                  'ставки по корпоративным депозитам, размещение свободной ликвидности компаний, казначейство'),
    'payment_control': ('H3 H7', 'Найди новости о контроле банков и налоговой за платежами компаний',
                        'блокировка подозрительных операций банками, 115-ФЗ, запросы налоговой о платежах, проверки контрагентов'),
    'transfer_pricing': ('H1 H6 + иностранный собственник', 'Найди новости о налоговом контроле сделок со связанными лицами',
                         'контроль трансфертных цен, сделки с иностранными связанными компаниями, налоговые проверки'),
}


def main(top: int):
    agent = json.loads((HERE.parents[1] / 'data/companies/agent' / f'{U}.json').read_text(encoding='utf-8'))
    raw = {f"raw_{h['id']}": (h['id'], 'Найди новости, связанные с гипотезой о бизнесе компании', h['text'])
           for h in agent['hypotheses']}
    gens = HOOKS | raw
    names, texts, queries = list(gens), [], []
    for g in names:
        _, ins, q = gens[g]
        queries.append(q)
    S = {g: 0 for g in names}
    for slug, (_, sep, _) in MODELS.items():
        D = np.load(HERE / 'out' / f'{slug}.npz')['doc'].astype(np.float32)
        Q = encode(slug, [f'Instruct: {gens[g][1]}\n{sep}{gens[g][2]}' for g in names])
        for g, qv in zip(names, Q):
            S[g] = S[g] + 1 / (60 + rank(D @ qv))
    bm = bm25_scorer()
    for g in names:
        S[g] = S[g] + 1 / (60 + rank(bm(gens[g][2])))

    _, _, event = ev.corpus()
    evs = pd.read_parquet(ev.DATA / 'news' / 'events.parquet').set_index('event')
    sets = json.loads((HERE.parents[0] / 'judge_v2/out/generator_sets.json').read_text(encoding='utf-8'))
    cur, new = set(sets['current (orig. weights)|200']), set(sets['new, 1st facet only|200'])
    E = pd.read_pickle(HERE.parents[0] / 'judge_v2/out/consensus.pkl')
    L = pd.read_pickle(HERE.parents[0] / 'judge_v2/out/single.pkl')
    L['p'] = (L.insight_potential >= 2) & (L.significance >= 2) & (L.mechanism_strength >= 2)
    v = L.groupby('event').p.agg(lambda s: s.sum() * 2 > len(s))
    pos = set(v[v].index)
    useful = {u['event']: u['tier'] for u in json.loads((HERE.parents[0] / 'judge_v2/useful_negatives.json')
                                                       .read_text(encoding='utf-8'))['items']}
    rows = []
    for g in names:
        order = [e for e in pd.Series(S[g]).groupby(event).max().sort_values(ascending=False).index if e in evs.index][:top]
        for r, e in enumerate(order):
            label = 'POS' if e in pos else useful.get(e, 'neg' if e in E.index else '')
            rows.append({'gen': g, 'from': gens[g][0], 'rank': r, 'event': int(e), 'C': e in cur, 'N': e in new,
                         'label': label, 'title': str(evs.title.get(e, ''))[:110]})
    df = pd.DataFrame(rows)
    out = HERE / 'out' / f'hypothesis_hooks_{U}.csv'
    df.to_csv(out, index=False, encoding='utf-8')
    for g, d in df.groupby('gen', sort=False):
        print(f"\n== {g} ({gens[g][0]}): in current {d.C.sum()}/{len(d)}, in proposed {d.N.sum()}/{len(d)}, "
              f"new to both {(~d.C & ~d.N).sum()}")
        for r in d.itertuples():
            print(f"  {r.rank:2} {'C' if r.C else '-'}{'N' if r.N else '-'} {r.label:7} {r.event:5} {r.title}")
    print(f'\n-> {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--top', type=int, default=10)
    main(ap.parse_args().top)
