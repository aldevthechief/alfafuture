"""Baseline news side: lever strengths of any news by a stem lexicon (no model, no corpus statistics), and
news x company matching. Stand-in for the model ensemble of experiments/news_channels until its scores are shipped:
`match()` takes any [news x 21] matrix in LEVERS order.

importance(news, company) = sum_l lever(news, l) x sensitivity(company, l) + sector(news, company)
- lever: share of the lever's lexicon stems found in the news, saturating (3+ hits -> 1.0);
- sector: hits of the company's goods / inputs / activity stems (what the news is about), weighted 0.5,
  so a lever-only news (rate, weather) still needs the company to be sensitive, and a sector news needs no lever.
python -X utf8 stage2_company_facts/sensitivity/news_levers.py   -> recall on the stage-3 golden set
"""
import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.text import stems
from stage2_company_facts.sensitivity.levers import LEVERS, vector

LEXICON = {
    'A1': 'потребительский спрос, розничные продажи, доходы населения, потребкредиты, ипотека, покупатели, траты населения',
    'A2': 'инвестпрограмма, капвложения, инвестиции компаний, стройка, закупки оборудования, ввод мощностей',
    'A3': 'госзаказ, госзакупки, бюджет, нацпроект, гособоронзаказ, бюджетные расходы, тендер',
    'A4': 'экспорт, внешние рынки, поставки за рубеж, экспортёры, открытие рынка, запрет ввоза',
    'B1': 'цены, подорожание, подешевление, биржевые цены, стоимость сырья, котировки, нефть, металл, зерно, урожай цены',
    'B2': 'тарифы, электроэнергия, цены на газ, теплоснабжение, ЖКХ, индексация тарифов, энергосистема',
    'B3': 'логистика, перевозки, грузоперевозки, железная дорога, РЖД, порт, фрахт, морские перевозки, транзит',
    'B4': 'зарплаты, дефицит кадров, МРОТ, мигранты, рынок труда, безработица, найм',
    'C1': 'налог на прибыль, НДС, страховые взносы, налоговая нагрузка, налоговый кодекс, спецрежим, УСН',
    'C2': 'пошлина, акциз, утильсбор, экспортная пошлина, ввозная пошлина, сбор, демпфер',
    'C3': 'регулирование, лицензия, маркировка, реестр, запрет, квота, требования, стандарт, проверки, закон',
    'C4': 'господдержка, субсидии, льготные кредиты, гранты, компенсация, льготная ипотека, поддержка отрасли',
    'C5': 'изъятие активов, временное управление, национализация, иностранные владельцы, выход иностранных компаний',
    'D1': 'ключевая ставка, Банк России, ставка ЦБ, процентные ставки, стоимость кредитов, ставки по кредитам, ставки по депозитам',
    'D2': 'кредитование бизнеса, корпоративные кредиты, облигации, размещение облигаций, надбавки ЦБ, доступ к финансированию',
    'D3': 'курс рубля, курс доллара, курс юаня, валютная выручка, валютный контроль, девальвация, укрепление рубля',
    'D4': 'неплатежи, отсрочка платежа, просроченная задолженность, дебиторская задолженность, оборотный капитал',
    'E1': 'санкции, санкционный, ограничения против России, вторичные санкции, параллельный импорт, третьи страны',
    'F1': 'конкуренция, слияние, поглощение, банкротство, уход с рынка, импортозамещение, демпинг',
    'G1': 'авария, пожар, наводнение, засуха, заморозки, погода, атака БПЛА, отключение, стихийное бедствие, ураган',
    'H1': 'технологии, искусственный интеллект, цифровизация, новый продукт, автоматизация, роботизация',
}
LEX = {k: {s for phrase in v.split(', ') for s in stems(phrase)} for k, v in LEXICON.items()}
SECTOR_WEIGHT = 0.5


def lever_matrix(texts: list[str]) -> np.ndarray:
    """[news x 21] lever strengths in LEVERS order: min(1, matched lexicon stems / 3)."""
    out = np.zeros((len(texts), len(LEVERS)))
    keys = list(LEVERS)
    for i, t in enumerate(texts):
        st = set(stems(t))
        for j, k in enumerate(keys):
            out[i, j] = min(1.0, len(st & LEX[k]) / 3)
    return out


def sector_stems(profile: dict) -> set[str]:
    st = profile['search_terms']
    return {s for t in st.get('goods', []) + st.get('inputs', []) + st.get('activities', [])[1::2] for s in stems(t)}


def match(profile: dict, levers: np.ndarray, news_stems: list[set[str]]) -> np.ndarray:
    """importance of every news for the company."""
    lever_part = levers @ np.array(vector(profile['sensitivity']))
    sec = sector_stems(profile)
    sector_part = np.array([min(1.0, len(s & sec) / 2) for s in news_stems]) * SECTOR_WEIGHT
    return lever_part + sector_part


if __name__ == '__main__':
    from common.search import news_document, news_pool
    from stage2_company_facts.profile.profile import profiles
    df = news_pool()
    docs = news_document(df).tolist()
    L, S = lever_matrix(docs), [set(stems(d)) for d in docs]
    print('news with any lever >= 2/3:', f'{(L.max(axis=1) >= 2 / 3).mean():.0%}',
          '| per lever:', {k: int((L[:, j] >= 2 / 3).sum()) for j, k in enumerate(LEVERS)})
    gold_file = Path(__file__).resolve().parents[2] / 'stage3_news_search' / 'golden_set' / 'real_pairs.json'
    pairs = {u: set(v) for u, v in json.loads(gold_file.read_text(encoding='utf-8'))['pairs'].items()}
    P = profiles(list(pairs))
    ids = df.id.to_numpy()
    tot = {100: 0, 300: 0, 500: 0}
    for u, gold in pairs.items():
        order = ids[np.argsort(-match(P[u], L, S))]
        for k in tot:
            tot[k] += len(gold & set(order[:k].tolist()))
    n = sum(map(len, pairs.values()))
    print('levers x sensitivity + sector:', ', '.join(f'top-{k} {v}/{n} ({v / n:.0%})' for k, v in tot.items()))
