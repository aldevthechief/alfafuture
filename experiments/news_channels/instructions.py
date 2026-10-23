"""Ablation of the ranker query side on the 398 GLM-scored random events (no LLM): instruction wording for the
instruct bi-encoder e5-large-instruct (local GPU), and few-shot exemplars (3 typical headlines per item, written by
hand, not taken from the corpus) for e5 and the cross-encoders bge-reranker-v2-m3 / ru-reranker-edge-150m (Kaggle:
--push builds kernel alfa-rank-ablation with queries = item, item + examples, each example). Metrics as evaluate.vs_glm.
python -X utf8 experiments/news_channels/instructions.py [--push]   (cross part read from out/rank_ablation/ when there)"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from channels import ITEMS, OUT, news_text, random_events
from push import INSTRUCTION  # before judge: judge puts experiments/retrieval (its own push.py) first on the path
from judge import event_news

IDS = list(ITEMS)
EXAMPLES = {  # typical headlines per item; generic on purpose (no corpus titles)
    'A1': ['Розничные продажи в России замедлились третий месяц подряд', 'Россияне сократили траты на одежду и электронику', 'Выдачи потребкредитов упали на 20% после ужесточения требований ЦБ'],
    'A2': ['«Россети» увеличат инвестпрограмму на 2027 год на 15%', 'Компании откладывают закупки оборудования из-за высокой ставки', 'Стройка новых заводов в промышленных парках выросла на треть'],
    'A3': ['Расходы бюджета на нацпроект «Инфраструктура» сократят на 10%', 'Регион увеличит закупки школьных автобусов по госконтрактам', 'Гособоронзаказ на 2027 год вырастет'],
    'A4': ['Китай снял ограничения на ввоз российской свинины', 'Экспорт российской пшеницы в Египет вырос вдвое', 'Индия сократила закупки российского угля из-за падения спроса'],
    'B1': ['Цены на алюминий на LME выросли до максимума за год', 'Биржевые цены на дизельное топливо подскочили на 8% за неделю', 'Сахар подешевел на 12% из-за рекордного урожая свёклы'],
    'B2': ['Тарифы на электроэнергию для бизнеса с июля вырастут на 11%', 'ФАС утвердила индексацию тарифов РЖД на грузоперевозки', 'Тарифы ЖКХ проиндексируют сверх инфляции'],
    'B3': ['Ставки на контейнерные перевозки из Китая выросли на 30%', 'РЖД ограничили погрузку угля на восток из-за загрузки БАМа', 'Порт Новороссийск запустил новый зерновой терминал'],
    'B4': ['Дефицит рабочих на заводах достиг рекорда, зарплаты растут на 20%', 'Правительство ужесточит правила найма мигрантов', 'МРОТ с 1 января увеличат на 18%'],
    'C1': ['Ставку налога на прибыль повысят до 25%', 'Порог НДС для упрощённой системы налогообложения снизят', 'Пониженные страховые взносы для МСП отменят'],
    'C2': ['Утильсбор на грузовики с 2027 года вырастет вдвое', 'Правительство ввело вывозную пошлину на подсолнечное масло', 'Акцизы на сладкие напитки повысят'],
    'C3': ['Маркировку «Честный знак» распространят на строительные материалы', 'Требования к локализации электротехнического оборудования ужесточат', 'Лицензирование производства готовой еды станет обязательным'],
    'C4': ['Льготную программу кредитования аграриев продлят и увеличат лимит', 'Фонд развития промышленности выдаст займы под 3% на станкостроение', 'Правительство компенсирует экспортёрам часть логистических затрат'],
    'C5': ['Суд обратил в доход государства акции крупного завода', 'Активы иностранного владельца переданы во временное управление', 'Для выхода иностранных компаний из российских активов повысили обязательный взнос'],
    'D1': ['Банк России повысил ключевую ставку до 18%', 'Ставки по корпоративным кредитам снизились вслед за ключевой', 'Банки повысили ставки по депозитам для юрлиц'],
    'D2': ['ЦБ ввёл надбавки к риск-весам по кредитам крупным компаниям', 'Объём размещений корпоративных облигаций упал вдвое', 'Банки ужесточили требования к заёмщикам-застройщикам'],
    'D3': ['Рубль ослаб до 100 за доллар', 'Требование об обязательной продаже валютной выручки экспортёров отменят', 'ЦБ ужесточил валютный контроль по внешнеторговым контрактам'],
    'D4': ['Торговые сети требуют от поставщиков увеличить отсрочку платежа до 90 дней', 'Неплатежи заказчиков строителям выросли на треть', 'Госзаказчики сократят авансы по контрактам до 30%'],
    'E1': ['ЕС принял новый пакет санкций против российских металлургов', 'США ввели вторичные санкции против банков, обслуживающих расчёты с Россией', 'Поставки комплектующих через третьи страны затруднены после санкций'],
    'F1': ['Крупнейший производитель отрасли объявил о банкротстве', 'Китайские производители заняли 40% российского рынка грузовиков', 'Два лидера рынка объявили о слиянии'],
    'G1': ['Авария на крупном НПЗ остановила выпуск бензина на месяц', 'Наводнение парализовало работу предприятий в регионе', 'Кибератака остановила работу логистической компании'],
    'H1': ['Производители перешли на отечественные промышленные роботы', 'Запуск 5G-сетей откроет рынок промышленного интернета вещей', 'Новая технология переработки снизит себестоимость удобрений на 20%'],
}
TASK = ('Дан канал влияния на бизнес. Найди новости о событиях, которые через этот канал заметно меняют выручку, '
        'затраты, доступ к рынку или к деньгам понятной группы компаний: принятые решения, сдвиги цен и объёмов, '
        'аварии; не ежедневные котировки, хроника и мнения без решений')
WEB = 'Given a web search query, retrieve relevant passages that answer the query'


def metrics(S: pd.DataFrame, G: pd.DataFrame) -> dict:
    auc = lambda y, s: roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan
    C = S.sub(S.mean(axis=1), axis=0)
    items = [i for i in IDS if (G[i] >= 2).sum() >= 3]
    hit = G.max(axis=1) >= 2
    P = S.rank(pct=True)
    return {'pair_auc': auc((G >= 2).values.ravel(), P.values.ravel()),
            'item_auc_centred': np.nanmean([auc(G[i] >= 2, C[i]) for i in items]),
            'any_auc': auc(hit, P.max(axis=1)), 'top1': (C[hit].idxmax(axis=1) == G[hit].idxmax(axis=1)).mean(),
            'recall@10%': ((P >= 0.9) & (G >= 2)).values.sum() / (G >= 2).values.sum()}


def e5_variants(news_ids):
    """CPU: documents = the precomputed retrieval embeddings (title twice + text, all variants share them), only the
    ~150 short queries are encoded here (the local GPU is shared with other sessions)."""
    from sentence_transformers import SentenceTransformer
    z = np.load(OUT.parents[1] / 'retrieval/out/e5-large-instruct.npz')
    row = {i: k for k, i in enumerate(z['doc_ids'])}
    D = z['doc'][[row[i] for i in news_ids]].astype(np.float32)
    m = SentenceTransformer('intfloat/multilingual-e5-large-instruct', device='cpu')
    enc = lambda xs: m.encode(xs, batch_size=16, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    q = lambda instr, texts: enc([f'Instruct: {instr}\nQuery: {t}' for t in texts])
    out = {f'e5 / {k}': D @ q(instr, list(ITEMS.values())).T for k, instr in (('current', INSTRUCTION), ('web', WEB), ('task', TASK))}
    out['e5 / no instruction'] = D @ enc(list(ITEMS.values())).T
    ex = np.stack([D @ q(INSTRUCTION, [ITEMS[i]] + EXAMPLES[i]).T for i in IDS], axis=1)  # news × item × (item + 3)
    out['e5 / current + examples (max)'] = ex.max(axis=2)
    out['e5 / current + examples (mean)'] = ex.mean(axis=2)
    out['e5 / current, item + examples in query'] = D @ q(INSTRUCTION, [f"{ITEMS[i]}. Например: {'; '.join(EXAMPLES[i])}" for i in IDS]).T
    return out


def cross_queries() -> dict[str, str]:
    q = {f'{i}|item': ITEMS[i] for i in IDS} | {f"{i}|item+ex": f"{ITEMS[i]}. Например: {'; '.join(EXAMPLES[i])}" for i in IDS}
    return q | {f'{i}|ex{k}': x for i in IDS for k, x in enumerate(EXAMPLES[i])}


def cross_from_kaggle(events) -> dict:
    out = {}
    for f in sorted((OUT / 'rank_ablation').glob('*.npz')):
        z = np.load(f)
        S = pd.DataFrame(z['raw'][:, :, 0], index=z['keys'], columns=z['items']).loc[events]
        out[f'{f.stem} / item'] = S[[f'{i}|item' for i in IDS]].values
        out[f'{f.stem} / item + examples in query'] = S[[f'{i}|item+ex' for i in IDS]].values
        out[f'{f.stem} / examples only (max)'] = np.stack([S[[f'{i}|ex{k}' for k in range(3)]].max(axis=1) for i in IDS], axis=1)
    return out


def cross_variants(docs, name='sshalimov04/ru-reranker-edge-150m'):  # shortcut: local GTX 1650 is ~20x slower than Kaggle; kept for small runs
    from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained(name)
    m = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=torch.float16).cuda().eval()

    def score(queries):
        pairs = [(q, d) for d in docs for q in queries]
        out = []
        for i in range(0, len(pairs), 32):
            b = pairs[i:i + 32]
            x = tok([p[0] for p in b], [p[1] for p in b], padding=True, truncation='only_second', max_length=512, return_tensors='pt').to('cuda')
            with torch.no_grad():
                out.append(m(**x).logits[:, 0].float().cpu().numpy())
        return np.concatenate(out).reshape(len(docs), len(queries))

    res = {'ru-reranker / item': score(list(ITEMS.values())),
           'ru-reranker / item + examples in query': score([f"{ITEMS[i]}. Например: {'; '.join(EXAMPLES[i])}" for i in IDS])}
    ex = np.stack([score(EXAMPLES[i]) for i in IDS], axis=1)
    res['ru-reranker / examples only (max)'] = ex.max(axis=2)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--push', action='store_true', help='push the cross-encoder kernel and exit')
    a = ap.parse_args()
    if a.push:
        import os, subprocess
        from push import CREDS, KERNELS, build
        account, groups = KERNELS['small'][0], [('0', ['bge-reranker-v2-m3']), ('1', ['ru-reranker-edge-150m'])]
        evs = random_events(); t = event_news(evs)
        d = build('ablation', account, groups, [{'key': e, 'id': t[e]['id']} for e in evs], ['item'], cross_queries())
        subprocess.run(['kaggle', 'kernels', 'push', '-p', str(d)], check=False,
                       env={**os.environ, 'KAGGLE_CONFIG_DIR': str(CREDS / f'kaggle_{account}' / '.kaggle')})
        raise SystemExit
    G = pd.DataFrame({int(d['key']): d['value']['scores'] for d in map(json.loads, open(OUT / 'glm_scores.jsonl', encoding='utf-8'))}).T
    events = [e for e in random_events() if e in G.index]
    texts = event_news(events)
    docs = [news_text(texts[e]) for e in events]
    G = G.loc[events]
    rows = {}
    for k, v in (cross_from_kaggle(events) | e5_variants([texts[e]['id'] for e in events])).items():
        rows[k] = metrics(pd.DataFrame(v, index=events, columns=IDS), G)
        print(k, {m: round(x, 3) for m, x in rows[k].items()}, flush=True)
    df = pd.DataFrame(rows).T.round(3)
    df.to_csv(OUT / 'instructions_eval.csv')
    print(df.to_string())
