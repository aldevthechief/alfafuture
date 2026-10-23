"""Stage A of candidate-news selection: cheap retrieval without an LLM reading the news.

A company is turned into weighted *facets* (own activity, group activities, supplier / customer industries from payments,
regions, FTS countries, macro channels implied by bank products and OKVED division). Every facet is a short query;
news and facets are embedded with word TF-IDF over 6-letter stems (a crude Russian stemmer: «застройщик/застройщиков» -> «застро»).
score(news) = max over facets of weight * cosine; the matched facet is the explanation.
python -X utf8 stage3_news_search/facet_retrieval/candidates.py U_76122 [--top 30] [--role cfo]
"""
import argparse, re, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.data import countries, load
from common.roles import role_channels
from common.search import news_document, news_index

GENERIC = re.compile(r'торговля оптовая|торговля розничная|деятельность|производство|прочих|прочими|прочие|не включенн\w+ в другие группировки|'
                     r'за вознаграждение или на договорной основе|в специализированных магазинах|кроме деятельности агентов|услуги по|'
                     r'собственным или арендованным|включая', re.I)
# macro channels: which bank-product / industry features open which news topics (mirrors analytics-usage.md)
CHANNELS = {
    'rate': 'ключевая ставка ЦБ решение по ставке стоимость кредитов процентные ставки заемщики',
    'deposit': 'ставки по депозитам доходность размещение ликвидности',
    'leasing': 'лизинг техники лизинговые компании изъятия льготный лизинг',
    'guarantee': 'госзакупки тендеры банковские гарантии 44-ФЗ 223-ФЗ контракты',
    'fx': 'курс рубля юань доллар валютный контроль санкции пошлины импорт параллельный импорт',
    'housing': 'ипотека семейная ипотека новостройки застройщики эскроу проектное финансирование ввод жилья',
    'construction': 'стройматериалы цемент арматура металлопрокат подрядчики строительная отрасль',
    'fuel': 'цены на бензин дизельное топливо нефтепродукты АЗС биржевые цены топлива',
    'consumer': 'потребительский спрос розничная торговля средний чек покупательская активность',
    'it': 'импортозамещение ПО реестр отечественного ПО ИТ-компании аккредитация облака кибербезопасность ИИ',
    'horeca': 'рестораны кафе общепит доставка еды средний чек продукты питания',
    'tax': 'налоги НДС УСН налоговая нагрузка страховые взносы',
    'labor': 'дефицит кадров зарплаты рынок труда мигранты',
}
DIV_CHANNELS = {'41': ['housing', 'construction'], '42': ['construction'], '43': ['construction'], '71': ['construction'],
                '68': ['housing'], '46': ['consumer', 'fx'], '47': ['consumer'], '49': ['fuel'], '52': ['fuel'],
                '56': ['horeca', 'consumer'], '55': ['horeca'], '62': ['it'], '63': ['it'], '10': ['consumer'], '35': ['rate']}
# news about crime, courts and accidents rarely carry a business consequence: damp them (macro facets excepted)
NOISE = re.compile(r'приговор|осужд|мошеннич|прокуратур|уголовн|задержан|суд |суда |судом|авари|пожар|погиб|ДТП|убийств', re.I)


def clean(name):
    return re.sub(r'\s+', ' ', GENERIC.sub(' ', name or '')).strip(' ,')


def facets(u, role=None, top=6):
    """[(kind, text, weight)] for company u; only from table facts."""
    U, P, _, _ = load()
    r = U.loc[u]
    out = [('own_activity', clean(r.MAINOKVED_NAME), 1.0)]
    g = U[(U.GROUP_ID == r.GROUP_ID) & (U.index != u)]
    for nm, _ in g.groupby('MAINOKVED_NAME').REVENUE_COMPANY.sum().nlargest(3).items():
        out.append(('group_activity', clean(nm), 0.4))
    for side, me, cp, w in [('supplier', 'PAYER', 'BENEFICIARY', 0.7), ('customer', 'BENEFICIARY', 'PAYER', 0.7)]:
        t = P[P[me] == u]
        k = t.assign(o=t[cp].map(U.MAINOKVED_NAME)).dropna(subset=['o'])
        for nm, v in k.groupby('o').PAY_AMT.sum().nlargest(top).items():
            if v / max(t.PAY_AMT.sum(), 1) >= 0.02:  # ignore counterparties below 2% of the flow
                out.append((f'{side}_industry', clean(nm), w))
    for c in (countries(r.FTS_COUNTRY_NAME) if isinstance(r.FTS_COUNTRY_NAME, str) else []):
        out.append(('fts_country', f'{c} торговля поставки пошлины санкции логистика', 0.6))
    out.append(('region', f'{r.REGION} экономика бизнес предприятия', 0.4))
    ch = set(DIV_CHANNELS.get(r.okdiv, []))
    ch |= {k for f, k in [('CL_PRODUCT_LOA_MARK', 'rate'), ('CL_PRODUCT_LEA_MARK', 'leasing'), ('CL_PRODUCT_DEP_MARK', 'deposit'),
                          ('CL_PRODUCT_GAR_MARK', 'guarantee')] if r[f] == 1}
    if isinstance(r.FTS_COUNTRY_NAME, str):
        ch.add('fx')
    ch |= set(role_channels(role))
    out += [(f'macro:{c}', CHANNELS[c], 0.5) for c in sorted(ch)]
    return [f for f in out if f[1]]


def candidates(u, role=None, top=30, extra=None, per_facet=None):
    """extra: optional DataFrame of additional news (same columns as news.parquet) — used to plant synthetic needles.
    per_facet: quota of news kept per facet before merging (diversity: one noisy facet cannot fill the list)."""
    df, index = news_index()
    X = index.matrix
    if extra is not None:
        df = pd.concat([df, extra], ignore_index=True)
        X = index.transform(news_document(df).tolist())
    fs = facets(u, role)
    if df.empty or not fs:
        return df.head(0).assign(matched_fragments=[]), fs
    Q = index.transform([t for _, t, _ in fs])
    from common.chunk_search import chunk_index
    chunks = chunk_index(df)
    S = np.maximum((X @ Q.T).toarray(), chunks.article_scores([t for _, t, _ in fs])) * np.array([w for *_, w in fs])
    noise = (df.title.fillna('') + ' ' + df.text.fillna('').str[:300]).str.contains(NOISE).to_numpy()
    macro = np.array([k.startswith('macro') for k, *_ in fs])
    S[np.ix_(noise, ~macro)] *= 0.3
    tier = df.get('tier', pd.Series(1, index=df.index)).fillna(1).to_numpy()
    S *= np.where(tier == 2, 0.7, 1.0)[:, None]  # general media rubrics: weaker prior
    q = per_facet or max(3, top // len(fs) + 2)
    keep = set()
    for j in range(len(fs)):
        keep.update(np.argsort(-S[:, j])[:q])
    keep = np.array(sorted(keep))
    best = S[keep].argmax(1)
    res = df.iloc[keep].assign(score=S[keep].max(1), facet=[fs[i][0] for i in best], facet_text=[fs[i][1][:60] for i in best])
    res = res.sort_values('score', ascending=False).head(top).copy()
    res['matched_fragments'] = [chunks.fragments(i, [t for _, t, _ in fs]) for i in res['id']]
    from common.search import corpus_fingerprint
    res.attrs['corpus_fingerprint'] = corpus_fingerprint(df)
    return res, fs


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('company')
    ap.add_argument('--top', type=int, default=30)
    ap.add_argument('--role')
    a = ap.parse_args()
    res, fs = candidates(a.company, a.role, a.top)
    print('facets:', *[f'  {k} ({w}): {t[:80]}' for k, t, w in fs], sep='\n')
    for r in res.itertuples():
        print(f'{r.score:.3f} {r.facet:18} {str(r.date)[:10]} {r.source[:10]:10} {r.title[:110]}')
