# News-channel checklist v1 (used in the pilot; run approved without edits on 21.10.2026)

Each news (event) is scored once against every item: «может ли эта новость через этот канал изменить деньги
бизнеса какой-то конкретной компании». Items are direction-neutral; direction, significance and «who» are
separate per-news fields. Company side: each item maps to company facts (last column).

| id | item (Russian text the scorer sees, examples in parentheses) | company fact for matching |
|---|---|---|
| A1 | Потребительский спрос (доходы и расходы населения, розница, потребкредит, ипотека) | OKVED B2C (industry profile) |
| A2 | Инвестиционный спрос компаний (инвестпрограммы, капвложения госкомпаний и монополий, стройки, закупки оборудования) | OKVED capital goods / construction |
| A3 | Госрасходы и госзаказ (бюджеты всех уровней, нацпроекты, гособоронзаказ, госзакупки) | OKVED budget-dependent, region |
| A4 | Внешние рынки сбыта (спрос и цены за рубежом, открытие или закрытие рынков других стран) | OKVED exporter-type; FTS country (weak) |
| B1 | Цены на товары, сырьё и материалы (биржевые, мировые, внутренние; бьёт и по продавцам, и по покупателям) | OKVED producer/trader of X; supplier industries |
| B2 | Энергия и регулируемые тарифы (электроэнергия, газ, тепло, ЖКХ, тарифы монополий) | OKVED energy-intensity; utilities |
| B3 | Логистика (стоимость и доступность перевозок, маршруты, порты, инфраструктура) | OKVED transport / trade; supplier industries |
| B4 | Труд (зарплаты, дефицит кадров, миграционные правила) | headcount, region |
| C1 | Общие налоги и взносы (прибыль, НДС, страховые взносы, спецрежимы) | everyone (low specificity), scale |
| C2 | Отраслевые сборы, пошлины, акцизы, льготы (утильсбор, техсбор, ввозные и вывозные пошлины, налоговые льготы) | OKVED; supplier industries |
| C3 | Отраслевое регулирование продукта и деятельности (лицензии, реестры, локализация, маркировка, стандарты, запреты, квоты, проверки) | OKVED |
| C4 | Господдержка (субсидии, льготные кредиты и лизинг, гранты, компенсации) | OKVED, bank products (credit, leasing) |
| C5 | Собственность и корпоративное право (изъятие, внешнее управление, правила для иностранных и госвладельцев) | ownership type, foreign owner |
| D1 | Ставка и стоимость кредита (ключевая ставка, рыночные ставки, требования ЦБ к банкам) | bank products (credit, leasing), deposits |
| D2 | Доступ к финансированию (долговой рынок, банковские лимиты и надбавки, выдачи кредитов) | bank products, scale |
| D3 | Валютный курс и валютное регулирование (курс, продажа выручки, валютный контроль) | FTS presence (weak, no direction) |
| D4 | Платёжная дисциплина и оборотный капитал (отсрочки, неплатежи, авансы, сроки расчётов в отрасли) | customer concentration, buyer industries |
| E1 | Санкции и ограничения против российских компаний и их снятие (в т.ч. вторичные, платежи, маршруты) | FTS countries (weak), foreign owner |
| F1 | Конкуренция и структура рынка (вход и уход игроков, слияния, банкротства, импортозамещение, демпинг импорта) | OKVED, market rank |
| G1 | Сбои и форс-мажор (аварии, атаки, погода, отключения, киберинциденты) | region, OKVED, supplier industries |
| H1 | Технологии и новые рынки (новые продукты, технологии, бизнес-модели, меняющие спрос или издержки отрасли) | OKVED |

Per-news fields beside item scores: significance 0–3 (company-independent, judge-v2 scale), direction per item
(+/−/±), targets: industries (free text → embedded against OKVED names), goods/commodities, regions, countries,
named organisations.

Dropped / folded from the starting draft: «state spending & programmes» → A3 + C4; «regional economy» → region
target slot; «counterparties' health» → F1/D4 by industry (case data is anonymised, named counterparties can't be
linked); «labour & migration» → B4; «foreign trade» → A4 + E1 + C2.
