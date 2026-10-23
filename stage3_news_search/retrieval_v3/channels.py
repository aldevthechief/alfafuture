"""News-first candidates of retrieval v3: «keys of influence» × company exposure (experiments/news_channels, checklist v2).
- News side: every news gets a percentile per checklist leaf (30 leaves under 20 parents + sink X0) from e5-large-instruct
  — the encoder retrieval v2 already loads and caches — with the query `Instruct: INSTRUCTION / leaf. Например: 3 headlines`
  (22.10 ablation: the leaf with examples in one query is e5's best wording). Parent = max of its leaves, event = max of
  its news. One encode of 30 short queries per corpus: no extra model in prod.
- Company side, deterministic from stage-2 fact texts (no LLM): facets = own OKVED, up to 3 counterparty OKVEDs per side
  (weight = link strength × visible flow share of revenue, off below 5% of revenue), region; each facet ranks events by
  max over its parents of channel percentile × facet affinity (percentile of the v2 hybrid score of the facet query).
  Attribute facets rank by the channel alone: credit / leasing → D, foreign owner → C5, FTS → D3 E1, staff > 250 → B4,
  buyer concentration → D4, every company → TX C6. The sink X0 belongs to no facet.
NEUTRAL (22.10, after a hardcode audit): leaf texts name mechanisms, not sectors; the 3 example headlines of a leaf come
from different sectors and never from the sectors of the companies the retriever is evaluated on (packaging and paper,
electrical equipment and grids, construction, restaurants, marketplaces) — test_retrieval_v3.py guards it.
In experiments (judge-v2 labels, checklist v1, 5-ranker ensemble) this added 4 positives to standard + factory on U_106783
(45 → 49 of 74): experiments/candidate_sets/README.md."""
import re

import pandas as pd

ENCODER = 'e5-large-instruct'
LEAVES = {  # leaf: (parent, text); experiments/news_channels/checklist_v2.md has the boundaries between neighbours
    'A1': ('A1', 'Потребительский спрос (доходы и расходы населения, розничные продажи, потребкредиты, ипотека)'),
    'A2a': ('A2', 'Капвложения госкомпаний и монополий (инвестпрограммы естественных монополий и госкорпораций, их закупки оборудования и подряды)'),
    'A2b': ('A2', 'Частные инвестиции и стройка (новые заводы, коммерческая и жилая стройка, закупки оборудования)'),
    'AS': ('AS', 'Государственные деньги (бюджеты всех уровней, нацпроекты, гособоронзаказ, госзакупки, субсидии, льготные кредиты и лизинг, гранты)'),
    'A4': ('A4', 'Внешние рынки сбыта (спрос и цены за рубежом, пошлины и запреты других стран на российские товары)'),
    'B1a': ('B1', 'Рыночные цены на энергоносители и топливо (нефть, нефтепродукты, газ, уголь, оптовая электроэнергия)'),
    'B1b': ('B1', 'Рыночные цены на металлы и материалы (сталь, цветные металлы, химия, стройматериалы, комплектующие)'),
    'B1c': ('B1', 'Рыночные цены на агро- и пищевое сырьё (зерно, масличные, сахар, мясо, молоко)'),
    'B2a': ('B2', 'Регулируемые тарифы (тарифы естественных монополий, ЖКХ, транспортные тарифы)'),
    'B2b': ('B2', 'Государственное регулирование цен на продукцию (заморозка цен, предельные наценки, демпфер)'),
    'B3': ('B3', 'Логистика (стоимость и доступность перевозок, маршруты, порты, новая транспортная инфраструктура)'),
    'B4': ('B4', 'Труд (зарплаты, дефицит кадров, МРОТ, правила найма и миграции)'),
    'TX': ('TX', 'Налоги, сборы и пошлины (налог на прибыль, НДС, взносы, утильсбор, экологические сборы, акцизы, ввозные и вывозные пошлины, налоговые льготы)'),
    'C3a': ('C3', 'Допуск к рынку (лицензии, реестры, локализация, запреты и квоты, параллельный импорт)'),
    'C3b': ('C3', 'Правила для продукта (маркировка, стандарты, сертификация, санитарные и ветеринарные требования)'),
    'C3c': ('C3', 'Контроль и санкции регулятора (проверки, ФАС, Роспотребнадзор, штрафы, в том числе за экологический ущерб)'),
    'C5': ('C5', 'Собственность и корпоративное право (изъятие активов, внешнее управление, правила для иностранных и госвладельцев, ответственность руководителей, банкротство)'),
    'C6': ('C6', 'Общие обязательные требования к компаниям (бухучёт и отчётность, персональные данные, отечественное ПО и кибербезопасность, электронный документооборот)'),
    'Da': ('D', 'Стоимость и доступность денег (ключевая ставка, ставки по кредитам и депозитам, требования ЦБ к банкам, облигации)'),
    'Db': ('D', 'Платёжные сервисы и расчёты (эквайринг, СБП, банковские комиссии, цифровой рубль, наличные)'),
    'D3': ('D3', 'Валютный курс и валютное регулирование (курс рубля, продажа валютной выручки, валютный контроль)'),
    'D4': ('D4', 'Платёжная дисциплина и оборотный капитал (отсрочки платежей, неплатежи, авансы, сроки расчётов)'),
    'E1': ('E1', 'Санкции и внешние ограничения против российских компаний (вторичные санкции, платежи и поставки через третьи страны)'),
    'F1a': ('F1', 'Конкуренция и структура рынка (вход и уход игроков, слияния, банкротства, импортозамещение, демпинг)'),
    'F1b': ('F1', 'Посредники и платформы (маркетплейсы, агрегаторы, их комиссии и правила)'),
    'G1a': ('G1', 'Сбой на производстве, в инфраструктуре или поставках (аварии, атаки на объекты, отключения, киберинциденты)'),
    'G1b': ('G1', 'Погода, климат и стихийные бедствия (в том числе влияние на урожай)'),
    'H1': ('H1', 'Технологии и новые рынки (новые продукты, технологии и бизнес-модели, меняющие спрос или издержки)'),
    'R1': ('R1', 'Недвижимость и аренда (ставки аренды, рынок коммерческой недвижимости, стоимость залогов)'),
    'X0': ('X0', 'Происшествия и хроника без последствий для бизнеса (криминал, ДТП, бытовые пожары, суды частных лиц)'),
}
PARENT = {k: p for k, (p, _) in LEAVES.items()}
INSTRUCTION = ('Дан канал влияния на бизнес. Найди новости о событиях, которые через этот канал заметно меняют выручку, '
               'затраты, доступ к рынку или к деньгам понятной группы компаний: принятые решения, сдвиги цен и объёмов, '
               'аварии; не ежедневные котировки, хроника и мнения без решений')
EXAMPLES = {  # see NEUTRAL in the module docstring: 3 sectors per leaf, none of the evaluation companies' sectors
    'A1': ['Продажи новых легковых автомобилей упали на 15%', 'Россияне стали реже покупать бытовую технику', 'Спрос на зарубежные туры снизился'],
    'A2a': ['Газовая госкомпания сократит инвестпрограмму на следующий год', 'Железная дорога увеличит закупки локомотивов', 'Нефтяная госкомпания отложила освоение нового месторождения'],
    'A2b': ['Металлурги откладывают модернизацию прокатных станов', 'Агрохолдинг построит новый животноводческий комплекс', 'Фармкомпании увеличили закупки производственного оборудования'],
    'AS': ['Расходы федерального бюджета на здравоохранение сократят на 5%', 'Аграриям продлят льготные кредиты и увеличат лимит', 'Регион увеличит закупки школьных автобусов'],
    'A4': ['Страна Азии сняла запрет на ввоз российской свинины', 'Экспорт пшеницы в Египет вырос вдвое', 'Индия сократила закупки российского угля'],
    'B1a': ['Биржевые цены на дизельное топливо подскочили на 8% за неделю', 'Нефть подешевела до минимума за полгода', 'Энергетический уголь для промышленных потребителей подорожал на 12%'],
    'B1b': ['Цены на алюминий достигли максимума за год', 'Никель подешевел на фоне избытка предложения', 'Производители удобрений повысили цены на аммиак'],
    'B1c': ['Пшеница подешевела на 10% на фоне рекордного урожая', 'Цены на сахар-песок выросли из-за неурожая свёклы', 'Закупочные цены на молоко снизились'],
    'B2a': ['Тарифы на грузовые перевозки по железной дороге проиндексируют на 10%', 'Тарифы на транспортировку газа по трубопроводам повысят', 'Тарифы ЖКХ повысят сверх инфляции'],
    'B2b': ['Правительство ограничит наценку на социально значимые продукты', 'Демпфер для нефтяников сократят', 'Цены на бензин заморозят до конца года'],
    'B3': ['Ставки на контейнерные перевозки из Китая выросли на 30%', 'Железная дорога ограничила погрузку на восток', 'В порту открыт новый терминал для зерна'],
    'B4': ['Дефицит рабочих на заводах вырос, зарплаты поднялись на 20%', 'Правила найма мигрантов ужесточат', 'МРОТ с января увеличат на 18%'],
    'TX': ['Ставку налога на прибыль повысят до 25%', 'Утильсбор на грузовики с 2027 года удвоят', 'Введена вывозная пошлина на подсолнечное масло'],
    'C3a': ['Требования к локализации производства лекарств ужесточат', 'Деятельность по утилизации отходов станет лицензируемой', 'Ряд товаров исключат из параллельного импорта'],
    'C3b': ['Обязательную маркировку распространят на корма для животных', 'Новые ветеринарные требования к перевозке скота вступят в силу', 'Утверждён новый стандарт на трубы для нефтепроводов'],
    'C3c': ['ФАС оштрафовала производителя за картельный сговор', 'Роспотребнадзор приостановил продажу партии детского питания', 'Предприятие оштрафовали на 2 млрд за загрязнение реки'],
    'C5': ['Суд обратил в доход государства акции завода', 'Активы иностранного владельца переданы во временное управление', 'Расширят субсидиарную ответственность директоров'],
    'C6': ['Компании обяжут хранить персональные данные только в России', 'Новые стандарты бухучёта вступят в силу с 2027 года', 'Госкомпании обязаны перейти на отечественное ПО'],
    'Da': ['Банк России повысил ключевую ставку до 18%', 'Ставки по корпоративным кредитам снизились', 'Объём размещений корпоративных облигаций упал вдвое'],
    'Db': ['Комиссии за эквайринг для бизнеса снизят', 'Оплата через СБП станет обязательной для крупных сетей', 'Расчёты цифровым рублём запустят для бюджетных выплат'],
    'D3': ['Рубль ослаб до 100 за доллар', 'Обязательную продажу валютной выручки отменят', 'ЦБ ужесточил валютный контроль по внешнеторговым контрактам'],
    'D4': ['Торговые сети требуют от поставщиков отсрочку платежа 90 дней', 'Неплатежи агрохолдингов поставщикам техники выросли на треть', 'Авансы по госконтрактам сократят до 30%'],
    'E1': ['ЕС принял новый пакет санкций против металлургов', 'Банки третьих стран отказываются проводить платежи российских компаний', 'США ввели вторичные санкции против поставщиков комплектующих'],
    'F1a': ['Крупный производитель отрасли объявил о банкротстве', 'Китайские производители заняли 40% рынка грузовиков', 'Два лидера рынка объявили о слиянии'],
    'F1b': ['Агрегатор такси повысил комиссию для перевозчиков', 'Платформа бронирования отелей изменила правила выплат', 'Закон о платформенной экономике вступил в силу'],
    'G1a': ['Авария на крупном НПЗ остановила выпуск бензина на месяц', 'Атака беспилотников остановила работу химического завода', 'Кибератака остановила работу логистической компании'],
    'G1b': ['Заморозки погубили часть урожая в южных регионах', 'Наводнение парализовало работу предприятий в регионе', 'Засуха снизила урожай зерновых в Поволжье'],
    'H1': ['Заводы переходят на отечественных промышленных роботов', 'Новая технология снизит себестоимость удобрений на 20%', 'Запуск сетей 5G откроет рынок промышленного интернета вещей'],
    'R1': ['Ставки аренды торговых центров выросли на 15%', 'Вакантность офисов в Москве упала до минимума', 'Банки снизили оценку коммерческой недвижимости в залоге'],
    'X0': ['В Подмосковье задержали угонщиков автомобилей', 'Пожар в жилом доме унёс жизни двух человек', 'Суд вынес приговор по делу о мошенничестве с квартирами'],
}
AFFINITY = {  # facet -> encoder instruction, as retrieval_v2.candidates.REASON_INSTRUCTIONS for the same reasons
    'own_industry': 'Найди новости о регулировании, ценах, спросе и конкуренции в этой отрасли',
    'supplier_industry': 'Найди новости о событиях, сильно влияющих на эту отрасль поставщиков компании: цены, дефицит, регулирование, сбои',
    'customer_industry': 'Найди новости о событиях, сильно влияющих на эту отрасль покупателей компании: спрос, финансы, регулирование',
    'region': 'Найди новости об экономике и бизнесе этого региона',
}
OWN = ['A1', 'A2', 'AS', 'A4', 'B1', 'B2', 'B3', 'TX', 'C3', 'F1', 'G1', 'H1', 'R1']
SIDE = {'payments_out': ('supplier_industry', ['B1', 'B3', 'TX', 'C3', 'G1', 'F1']),
        'payments_in': ('customer_industry', ['A1', 'A2', 'AS', 'A4', 'C3', 'F1', 'D4'])}
REGION = ['A1', 'AS', 'B2', 'B4', 'G1']


def facets(facts: list[dict]) -> list[dict]:
    """[{name, parents, spec (instruction, [query]) or None for an attribute channel, weight, fact}] from stage-2 fact texts."""
    by = {}
    for f in facts:
        by.setdefault(f['cat'], []).append(f['text'])
    first = lambda cat, pre: next((x for x in by.get(cat, []) if x.startswith(pre)), '')
    out = []
    if act := first('activity', 'Основной'):
        out.append({'name': 'own_industry', 'parents': OWN, 'spec': (AFFINITY['own_industry'], [act.split(': ', 1)[1]]),
                    'weight': 1.0, 'fact': act})
    for cat, (reason, parents) in SIDE.items():
        flow = re.search(r'это (<?)(\d+)% годовой', first(cat, 'Входящие платежи:' if cat == 'payments_in' else 'Исходящие платежи:'))
        visible = 0 if not flow or flow[1] else min(1.0, int(flow[2]) / 30)  # a flow < 30% of revenue says little of the business
        links = [(int(m[2]), m[1], x) for x in by.get(cat, []) if (m := re.search(r'с ОКВЭД [\d.]+ «(.+?)»: .+\((\d+)% суммы\)', x))]
        for share, name, fact in sorted(links, reverse=True)[:3]:
            w = (1.0 if share >= 30 else 0.5 if share >= 5 else 0) * visible
            if w >= 0.5 * 5 / 30:  # at least a medium link with a visible flow >= 5% of revenue
                out.append({'name': reason, 'parents': parents, 'spec': (AFFINITY[reason], [name]), 'weight': w, 'fact': fact})
        conc = first(cat, 'Концентрация')
        if cat == 'payments_in' and (m := re.search(r'крупнейший контрагент (\d+)%', conc)) and int(m[1]) >= 30 and visible >= 1:
            out.append({'name': 'buyer_concentration', 'parents': ['D4'], 'spec': None, 'weight': 0.5, 'fact': conc})
    if geo := first('geo', 'Зарегистрирована'):
        out.append({'name': 'region', 'parents': REGION, 'spec': (AFFINITY['region'], [geo.split(': ', 1)[1]]), 'weight': 0.5, 'fact': geo})
    if re.search('кредит|лизинг', prods := first('bank', 'Продукты')):
        out.append({'name': 'credit', 'parents': ['D'], 'spec': None, 'weight': 0.5, 'fact': prods})
    if owner := next((x for x in by.get('ownership', []) if 'иностран' in x), None):
        out.append({'name': 'foreign_owner', 'parents': ['C5'], 'spec': None, 'weight': 0.5, 'fact': owner})
    if fts := next((x for x in by.get('trade', []) if 'из страны' in x), None):
        out.append({'name': 'fts', 'parents': ['D3', 'E1'], 'spec': None, 'weight': 0.3, 'fact': fts})
    if 'более 251' in (staff := first('profile', 'Численность')):
        out.append({'name': 'staff', 'parents': ['B4'], 'spec': None, 'weight': 0.3, 'fact': staff})
    out.append({'name': 'taxes_and_rules', 'parents': ['TX', 'C6'], 'spec': None, 'weight': 0.2, 'fact': 'все компании'})
    return out


def leaf_queries() -> list[str]:
    return [f"Instruct: {INSTRUCTION}\nQuery: {text}. Например: {'; '.join(EXAMPLES[k])}" for k, (_, text) in LEAVES.items()]


def leaf_cosines(search):
    """news × leaf cosine: from a pre-indexed search (precompute.Frozen.leaf) or e5 over the cached document vectors."""
    if hasattr(search, 'leaf'):
        return search.leaf
    return search.docs[ENCODER] @ search.encoder(ENCODER, leaf_queries()).T


def parent_scores(search) -> pd.DataFrame:
    """event × parent channel percentile; search = retrieval_v2.hybrid.HybridSearch (or a pre-indexed Frozen one)."""
    leaves = pd.DataFrame(leaf_cosines(search), columns=list(LEAVES)).rank(pct=True)
    return leaves.T.groupby(PARENT).max().T.groupby(search.event).max()


def generators(facts: list[dict], search, k: int = 300) -> list[dict]:
    """Channel generators of one company: facet name, parents, the fact behind it, weight and its top-k events."""
    fs = facets(facts)
    P = parent_scores(search)
    with_spec = [f for f in fs if f['spec']]
    A = pd.DataFrame(search.scores([f['spec'] for f in with_spec])).groupby(search.event).max().rank(pct=True) if with_spec else None
    out = []
    for f in fs:
        s = P[f['parents']].max(axis=1)
        if f['spec']:
            s = s * A[with_spec.index(f)].reindex(s.index).fillna(0)
        top = [int(e) for e in s.sort_values(ascending=False, kind='stable').index[:k]]
        out.append({'source': 'channel', 'reason': f['name'], 'channel': ','.join(f['parents']), 'fact': f['fact'],
                    'query': f"{f['name']}: {f['fact']}"[:80], 'weight': f['weight'], 'top': top})
    return out


assert set(EXAMPLES) == set(LEAVES) and all(len(v) == 3 for v in EXAMPLES.values())
assert set(OWN + REGION + [p for _, ps in SIDE.values() for p in ps]) <= set(PARENT.values())
