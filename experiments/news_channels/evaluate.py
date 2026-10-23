"""Do the rankers' (news, item) scores agree with GLM and separate useful news? Reads out/rank/<slug>.npz (kernel
output) and out/glm_scores.jsonl. Per model × query variant:
- random set vs GLM: per-item ROC-AUC against GLM score ≥ 2 (macro over items with ≥ 3 positives), «any channel»
  AUC (max over items vs GLM max ≥ 2), top-1 item agreement on news GLM gave a hit;
  `centred` = score minus the news' mean over items, so only the item choice within a news counts.
- labelled U_106783 set vs judge v2: AUC of the max item score for judge positives / relevant (mechanism ≥ 2) /
  positives + manually useful negatives, and top-3 agreement with the judge's channel mapped onto the checklist.
python -X utf8 experiments/news_channels/evaluate.py"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from channels import ITEMS, OUT, ROOT, labelled_events
from judge import positive

JUDGE_CHANNEL = {  # judge v2 channel -> checklist items it falls under
    'объём спроса': ['A1', 'A2', 'A3', 'A4'], 'цена продажи': ['B1'], 'цены закупок и сырья': ['B1', 'B2'],
    'доступ к рынку (лицензии, реестры, локализация, госзаказ)': ['C3', 'A3'], 'логистика': ['B3'], 'персонал и ФОТ': ['B4'],
    'налоги и взносы': ['C1', 'C2'], 'оборотный капитал (отсрочки, авансы)': ['D4'],
    'стоимость и доступность финансирования': ['D1', 'D2'], 'валютный риск': ['D3'], 'контрагентский риск': ['D4', 'F1'],
    'сбой поставок': ['G1'], 'регуляторный и правовой риск': ['C3', 'C5'], 'конкуренция и доля рынка': ['F1'],
    'новая возможность': ['H1', 'C4'], 'стоимость активов и залогов': ['D2'], 'запасы': ['B1']}
IDS = list(ITEMS)


def auc(y, s):
    return roc_auc_score(y, s) if 0 < sum(y) < len(y) else np.nan


def load_models():
    for f in sorted((OUT / 'rank').glob('*.npz')):
        z = np.load(f)
        assert list(z['items']) == IDS, f
        for v, name in enumerate(z['variants']):
            yield f'{f.stem}/{name}', pd.DataFrame(z['score'][:, :, v], index=z['keys'], columns=IDS)


def vs_glm(S, G):
    S = S.loc[G.index]
    C = S.sub(S.mean(axis=1), axis=0)
    per_item = [auc(G[i] >= 2, S[i]) for i in IDS if (G[i] >= 2).sum() >= 3]
    hit = G.max(axis=1) >= 2
    return {'glm_item_auc': np.nanmean(per_item), 'glm_item_auc_centred': np.nanmean([auc(G[i] >= 2, C[i]) for i in IDS if (G[i] >= 2).sum() >= 3]),
            'glm_any_auc': auc(hit, S.max(axis=1)), 'glm_pair_auc': auc((G >= 2).values.ravel(), S.values.ravel()),
            'glm_top1': (C[hit].idxmax(axis=1) == G[hit].idxmax(axis=1)).mean(), 'items_scored': len(per_item),
            'glm_recall@10%': ((S >= S.quantile(0.9)) & (G >= 2)).values.sum() / (G >= 2).values.sum()}


def vs_judge(S, J, useful):
    S = S.loc[[e for e in J if e in S.index]]
    m = S.max(axis=1)
    pos = pd.Series({e: positive(J[e]) for e in S.index})
    rel = pd.Series({e: J[e]['mechanism_strength'] >= 2 for e in S.index})
    C = S.sub(S.mean(axis=1), axis=0)
    ranked = [(J[e]['channel'], list(C.loc[e].sort_values(ascending=False).index[:3])) for e in S.index if J[e]['mechanism_strength'] >= 1]
    top3 = np.mean([bool(set(JUDGE_CHANNEL[ch]) & set(top)) for ch, top in ranked if ch in JUDGE_CHANNEL])
    return {'judge_pos_auc': auc(pos, m), 'judge_rel_auc': auc(rel, m), 'judge_useful_auc': auc(pos | S.index.isin(useful), m),
            'judge_channel_top3': top3, 'judge_A2_pos_auc': auc(pos, S['A2'])}


if __name__ == '__main__':
    G = pd.DataFrame({int(d['key']): d['value']['scores'] for d in map(json.loads, open(OUT / 'glm_scores.jsonl', encoding='utf-8'))}).T
    J = labelled_events()
    useful = [x['event'] for x in json.loads((ROOT / 'experiments/judge_v2/useful_negatives.json').read_text(encoding='utf-8'))['items']]
    rows, ranks = [], {}
    models = dict(load_models())
    rng = np.random.default_rng(0)
    S0 = next(iter(models.values()))
    models['random baseline'] = pd.DataFrame(rng.random(S0.shape), index=S0.index, columns=IDS)
    best = ['e5-large-instruct/item', 'giga-emb-480m/item', 'ru-reranker-edge-150m/item', 'giga-emb-10b/item', 'bge-reranker-v2-m3/item']
    models['ensemble (mean pct rank, 5 models)'] = sum(models[m].rank(pct=True) for m in best) / len(best)
    for name, S in models.items():
        rows.append({'model': name, **vs_glm(S, G.loc[G.index.isin(S.index)]), **vs_judge(S, J, useful)})
    df = pd.DataFrame(rows).set_index('model').round(3)
    df.to_csv(OUT / 'eval.csv')
    print(f'GLM: {len(G)} random events, {(G.max(axis=1) >= 2).sum()} with a hit; judge: {len(J)} events,',
          sum(map(positive, J.values())), 'positives')
    print(df.to_string())
