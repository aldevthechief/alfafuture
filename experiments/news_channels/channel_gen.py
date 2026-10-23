"""News-first candidate generator «channel × company exposure», evaluated against judge v2 on the labelled companies.
Company side, deterministic from stage-2 facts (no LLM): facets = own OKVED, up to 3 counterparty OKVEDs per side
(weight = link strength × visible flow share of revenue; off below 5% of revenue), region; each facet is a
mini-generator ranking events by max over its items of pct_item(event) × aff_facet(event)^alpha, where pct is the
channel percentile (channels_events / one model) and aff the event's percentile of e5-large-instruct cosine to the facet
query (retrieval doc embeddings, max over reports; the facet instructions of the current generators). Attribute
channels (credit/leasing → D1 D2, foreign owner → C5, FTS → D3 E1, staff > 250 → B4, buyer concentration → D4,
all → C1) rank by pct alone. Pool = quota union at a budget, quotas ∝ weights (as judge.candidate_events).
alpha = ∞ is the «affinity only» ablation (≈ current facet generators), alpha = 0 «channels only».
python -X utf8 experiments/news_channels/channel_gen.py [--companies U_106783 U_150736 U_212593 U_9250]"""
import argparse, json, re, sys

import numpy as np
import pandas as pd

from channels import DATA, ITEMS, OUT, ROOT
from ensemble import model_scores
from judge import positive
sys.path.insert(0, str(ROOT / 'experiments/retrieval')); sys.path.insert(0, str(ROOT / 'experiments/briefs'))
import evaluate as rev
from new_generators import encode
from push import REASON_INSTRUCTIONS
from run import facts_for

IDS = list(ITEMS)
COMPANIES = ['U_106783', 'U_150736', 'U_212593', 'U_9250']
OWN = ['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'C2', 'C3', 'C4', 'F1', 'G1', 'H1']
SIDE = {'payments_out': ('supplier_industry', ['B1', 'B3', 'C2', 'C3', 'G1', 'F1']),
        'payments_in': ('customer_industry', ['A1', 'A2', 'A3', 'A4', 'C3', 'C4', 'F1', 'D4'])}
REGION = ['A1', 'A3', 'B2', 'B4', 'C4', 'G1']
GF, JV2 = ROOT / 'experiments/generator_factory/out', ROOT / 'experiments/judge_v2/out'


def facets(facts: list[dict]) -> list[dict]:
    """[{name, items, query (None = attribute channel), weight}] from fact texts (formats of experiments/briefs/run.py)."""
    by = {}
    for f in facts:
        by.setdefault(f['cat'], []).append(f['text'])
    t = lambda cat, pre: next((x for x in by.get(cat, []) if x.startswith(pre)), '')
    okved = t('activity', 'Основной').split(': ', 1)[1]
    out = [{'name': 'own', 'items': OWN, 'query': (REASON_INSTRUCTIONS['own_industry'], okved), 'weight': 1.0}]
    for cat, (reason, items) in SIDE.items():
        flow = re.search(r'это (<?)(\d+)% годовой', t(cat, 'Входящие платежи:' if cat == 'payments_in' else 'Исходящие платежи:'))
        share = 0 if not flow or flow[1] else min(1.0, int(flow[2]) / 30)
        links = [(float(m[3]), m[2]) for x in by.get(cat, []) if (m := re.search(r'с ОКВЭД ([\d.]+) «(.+?)»: .+\((\d+)% суммы\)', x))]
        for p, name in sorted(links, reverse=True)[:3]:
            w = (1.0 if p >= 30 else 0.5 if p >= 5 else 0) * share
            if w >= 0.5 * 5 / 30:  # visible flow ≥ 5% of revenue for a medium link
                out.append({'name': f'{reason}:{name[:30]}', 'items': items, 'query': (REASON_INSTRUCTIONS[reason], name), 'weight': w})
        conc = re.search(r'крупнейший контрагент (\d+)%', t(cat, 'Концентрация'))
        if cat == 'payments_in' and conc and int(conc[1]) >= 30 and share >= 1:
            out.append({'name': 'buyer_concentration', 'items': ['D4'], 'query': None, 'weight': 0.5})
    geo = t('geo', 'Зарегистрирована').split(': ', 1)[-1]
    out.append({'name': 'region', 'items': REGION, 'query': (REASON_INSTRUCTIONS['region'], geo), 'weight': 0.5})
    prods = t('bank', 'Продукты')
    if re.search('кредит|лизинг', prods):
        out.append({'name': 'credit', 'items': ['D1', 'D2'], 'query': None, 'weight': 0.5})
    if any('иностран' in x for x in by.get('ownership', [])):
        out.append({'name': 'foreign_owner', 'items': ['C5'], 'query': None, 'weight': 0.5})
    if any('из страны' in x for x in by.get('trade', [])):
        out.append({'name': 'fts', 'items': ['D3', 'E1'], 'query': None, 'weight': 0.3})
    if 'более 251' in t('profile', 'Численность'):
        out.append({'name': 'staff', 'items': ['B4'], 'query': None, 'weight': 0.3})
    out.append({'name': 'taxes', 'items': ['C1'], 'query': None, 'weight': 0.2})
    return out


def event_pct(models=None) -> pd.DataFrame:
    """event × item: mean over models of per-item percentile over news, max over the event's reports."""
    M = model_scores()
    M = {k: v for k, v in M.items() if models is None or k in models}
    E = sum(S.rank(pct=True) for S in M.values()) / len(M)
    ev = pd.read_parquet(DATA / 'news' / 'events.parquet')[['event', 'news_ids']].explode('news_ids')
    ev['news_ids'] = ev['news_ids'].astype(str)
    return ev.join(E, on='news_ids').dropna().groupby('event')[IDS].max()


class Affinity:
    """event × facet query percentile of e5-large-instruct cosine (retrieval doc embeddings, max over reports)."""

    def __init__(self, events: pd.Index):
        z = np.load(ROOT / 'experiments/retrieval/out/e5-large-instruct.npz')
        ids, _, event = rev.corpus()
        assert list(z['doc_ids']) == ids
        self.D, self.event, self.events = z['doc'].astype(np.float32), event, events

    def __call__(self, queries: list[tuple[str, str]]) -> pd.DataFrame:
        Q = encode('e5-large-instruct', [f'Instruct: {i}\nQuery: {q}' for i, q in queries])
        S = pd.DataFrame(self.D @ Q.T)
        S['e'] = self.event
        return S.groupby('e').max().reindex(self.events).rank(pct=True)


def generator_scores(fs, P, aff, alpha) -> pd.DataFrame:
    """event × facet score."""
    out = {}
    for k, f in enumerate(fs):
        ch = P[f['items']].max(axis=1)
        if f['query'] is None:
            out[k] = ch
        elif alpha == np.inf:
            out[k] = aff[k]
        else:
            out[k] = P[f['items']].max(axis=1) * aff[k] ** alpha if alpha else ch
    return pd.DataFrame(out)


def quota_union(S: pd.DataFrame, w: np.ndarray, budget: int) -> list[int]:
    order = {k: S[k].sort_values(ascending=False).index for k in S}
    for b in range(budget, 20 * budget, 5):
        chosen = dict.fromkeys(e for k, h in zip(S, np.maximum(1, np.round(b * w / w.sum())).astype(int)) for e in order[k][:h])
        if len(chosen) >= budget:
            return list(chosen)
    return list(chosen)


def labels(u):
    L = pd.DataFrame(json.loads(l)['value'] for l in open(JV2 / f'labels_v2_{u}.jsonl', encoding='utf-8'))
    L = L[(L.input == 'gemma_summary') & (L.refusal == 'short')].drop_duplicates('event', keep='last').set_index('event')
    L['pos'] = L.apply(positive, axis=1)
    L['rel'] = L.mechanism_strength >= 2
    return L


def reference_sets(u) -> dict[str, set]:
    tops = lambda f, pre: {e for k, v in json.loads((GF / f).read_text()).items() if k.startswith(pre) for e in v[:13]}
    if u == 'U_106783':
        G = json.loads((JV2 / 'generator_sets.json').read_text())
        return {'current@200': set(G['current (orig. weights)|200']), 'hand-written@200': set(G['new, 3 facets RRF|200']),
                'factory v1': tops('factory_U_106783_tops13.json', 'fin:'), 'factory v2-B': tops('factory_U_106783_B_tops.json', 'sel:')}
    return {'factory v2-B': tops(f'factory_{u}_B_tops.json', 'sel:')}


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--companies', nargs='+', default=COMPANIES)
    a = ap.parse_args()
    PCT = {'ens': event_pct(), 'e5': event_pct(['e5-large-instruct'])}
    aff_of = Affinity(PCT['ens'].index)
    rows, union_rows = [], []
    FS = {u: facets(facts_for(u)) for u in a.companies}
    QK = [(u, k) for u in a.companies for k, f in enumerate(FS[u]) if f['query']]
    AFF = aff_of([FS[u][k]['query'] for u, k in QK])  # one encode call: the model is loaded once (4 GB GPU)
    AFF.columns = pd.MultiIndex.from_tuples(QK)
    for u in a.companies:
        fs = FS[u]
        print(f'\n===== {u}: ' + '; '.join(f"{f['name']} ({f['weight']:.2f}: {','.join(f['items'])})" for f in fs))
        qi = [k for k, f in enumerate(fs) if f['query']]
        A = AFF[u]
        L = labels(u)
        npos = int(L.pos.sum())
        w = np.array([f['weight'] for f in fs])
        refs = reference_sets(u)
        for name, s in refs.items():
            lab = [e for e in s if e in L.index]
            rows.append({'company': u, 'set': name, 'events': len(s), 'labelled': len(lab), 'pos': int(L.loc[lab].pos.sum()),
                         'pos_share': L.loc[lab].pos.sum() / npos, 'pos_per_labelled': L.loc[lab].pos.mean()})
        for model, P in PCT.items():
            for alpha in (0, 1, 2, np.inf):
                S = generator_scores(fs, P, A, alpha)
                if alpha == np.inf:
                    S = S[qi]
                score = S.rank(pct=True).max(axis=1)  # facet-agnostic score for AUC vs the judge
                lab = L.index.intersection(score.index)
                for budget in (100, 200):
                    pool = quota_union(S, w[S.columns] if alpha == np.inf else w, budget)
                    labp = [e for e in pool if e in L.index]
                    name = f'channel/{model}/a={alpha}@{budget}'
                    rows.append({'company': u, 'set': name, 'events': len(pool), 'labelled': len(labp), 'pos': int(L.loc[labp].pos.sum()),
                                 'pos_share': L.loc[labp].pos.sum() / npos, 'pos_per_labelled': L.loc[labp].pos.mean(),
                                 'judge_pos_auc': auc(L.loc[lab].pos, score[lab]), 'judge_rel_auc': auc(L.loc[lab].rel, score[lab])})
                    for ref, s in refs.items():
                        if model == 'e5' or alpha not in (1, np.inf):
                            continue
                        new = set(pool) - s
                        union_rows.append({'company': u, 'channel set': name, 'reference': ref, 'overlap': len(set(pool) & s) / len(pool),
                                           'ref_pos': int(L.loc[[e for e in s if e in L.index]].pos.sum()),
                                           'new_pos': int(L.loc[[e for e in new if e in L.index]].pos.sum()),
                                           'new_labelled': sum(e in L.index for e in new), 'new_events': len(new)})
        print(f'{len(L)} labelled, {npos} positives')
    R = pd.DataFrame(rows).round(3)
    U = pd.DataFrame(union_rows).round(3)
    R.to_csv(OUT / 'channel_gen_eval.csv', index=False); U.to_csv(OUT / 'channel_gen_union.csv', index=False)
    pd.set_option('display.width', 250)
    print(R.to_string(index=False)); print(U.to_string(index=False))
