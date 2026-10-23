"""News-first channel checklist (checklist_draft.md, v1) and the news sets of the pilot.
Each event is scored once against every item: can this news, through this channel, change the money of some concrete
company's business. Items are direction-neutral; who is affected and the sign come from separate fields."""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
OUT = HERE / 'out'
for p in (ROOT, ROOT / 'experiments/judge_v2'):
    sys.path.insert(0, str(p))
from common.data import DATA
from common.news_corpus import assert_experiment_corpus

assert_experiment_corpus(ROOT / 'experiments/retrieval/corpus.sources.json')

ITEMS = {
    'A1': 'Потребительский спрос (доходы и расходы населения, розничные продажи, потребкредиты, ипотека)',
    'A2': 'Инвестиционный спрос компаний (инвестпрограммы и капвложения, в том числе госкомпаний и монополий; стройки, закупки оборудования)',
    'A3': 'Госрасходы и госзаказ (бюджеты всех уровней, нацпроекты, гособоронзаказ, госзакупки)',
    'A4': 'Внешние рынки сбыта (спрос и цены за рубежом, открытие или закрытие рынков других стран для российских товаров)',
    'B1': 'Цены на товары, сырьё и материалы (биржевые, мировые и внутренние цены; затрагивают и продавцов, и покупателей)',
    'B2': 'Энергия и регулируемые тарифы (электроэнергия, газ, тепло, ЖКХ, тарифы естественных монополий)',
    'B3': 'Логистика (стоимость и доступность перевозок, маршруты, порты, транспортная инфраструктура)',
    'B4': 'Труд (зарплаты, дефицит кадров, правила найма и миграции)',
    'C1': 'Общие налоги и взносы (налог на прибыль, НДС, страховые взносы, спецрежимы)',
    'C2': 'Отраслевые сборы, пошлины, акцизы и налоговые льготы (утильсбор, технологический сбор, ввозные и вывозные пошлины)',
    'C3': 'Отраслевое регулирование продукта и деятельности (лицензии, реестры, локализация, маркировка, стандарты, запреты, квоты, проверки)',
    'C4': 'Господдержка (субсидии, льготные кредиты и лизинг, гранты, компенсации затрат)',
    'C5': 'Собственность и корпоративное право (изъятие активов, внешнее управление, правила для иностранных и государственных владельцев)',
    'D1': 'Ставки и стоимость кредита (ключевая ставка, рыночные ставки по кредитам и депозитам)',
    'D2': 'Доступ к финансированию (кредитование бизнеса, требования ЦБ к банкам, лимиты и надбавки, долговой рынок)',
    'D3': 'Валютный курс и валютное регулирование (курс рубля, продажа валютной выручки, валютный контроль)',
    'D4': 'Платёжная дисциплина и оборотный капитал (отсрочки платежей, неплатежи, авансы, сроки расчётов)',
    'E1': 'Санкции и внешние ограничения против российских компаний, их ужесточение или снятие (в том числе платежи и поставки через третьи страны)',
    'F1': 'Конкуренция и структура рынка (вход и уход игроков, слияния, банкротства, импортозамещение, демпинг импорта)',
    'G1': 'Сбои и форс-мажор (аварии, атаки, стихийные бедствия, погода, отключения, киберинциденты)',
    'H1': 'Технологии и новые рынки (новые продукты, технологии и бизнес-модели, меняющие спрос или издержки)',
}
TEXT_CHARS = 2000  # rankers see ~512 tokens; GLM gets the same text so the two are comparable


def labelled_events(company='U_106783') -> dict[int, dict]:
    """judge-v2 verdicts on title + Gemma summary, one per event."""
    rows = (json.loads(l)['value'] for l in open(ROOT / f'experiments/judge_v2/out/labels_v2_{company}.jsonl', encoding='utf-8'))
    return {r['event']: r for r in rows if r['input'] == 'gemma_summary' and r['refusal'] == 'short'}


def random_events(n=400, seed=0) -> list[int]:
    """Random real events outside the judge-labelled ones. Frozen in out/random_events.json once GLM has scored them:
    the labelled set keeps growing (another session), which reshuffles a fresh sample."""
    frozen = OUT / 'random_events.json'
    if frozen.exists():
        return json.loads(frozen.read_text())
    every = set(pd.read_parquet(DATA / 'news' / 'events.parquet').event) - set(labelled_events())
    return [int(e) for e in np.random.default_rng(seed).choice(sorted(every), n, replace=False)]


def news_text(t: dict) -> str:
    return f"{t['title']}. {t['text'][:TEXT_CHARS]}"
