# alfafuture-hack — новостная аналитика для бизнес-решений (кейс Альфа-Банка)

Пайплайн: новости + аналитика Банка + данные о компании → персонализированные инсайты для ЛПР.
Каждый этап — своя папка, каждый подход — подпапка этапа, golden set этапа — в `golden_set/` этого этапа.
Промпты лежат отдельными `.md` файлами в `prompts/` рядом с кодом, который их использует.

**Для команды:** [reference текущей MVP-интеграции и обзор PR](docs/MVP_REFERENCE.md) — два режима пайплайна,
контракты данных, статусы выдачи, подготовка OCR, диагностика и ограничения. Начинать интеграцию новых модулей с него.

| Этап команды | Папка | Подходы | Golden set |
|---|---|---|---|
| 1. Набираем большой пул новостей | [`stage1_news_pool/`](stage1_news_pool/README.md) | `web_scraping` | — |
| 2. Набираем много фактов о компании | [`stage2_company_facts/`](stage2_company_facts/README.md) | `deterministic_facts`, `research_agent` (LangGraph) | `company_facts_golden.json` (9 компаний) |
| 3. Ищем потенциально полезную новость | [`stage3_news_search/`](stage3_news_search/README.md) | `retrieval_v2` (по умолчанию), `candidate_generators`, `facet_retrieval`, `hyde_queries` | `synthetic_needles.json` |
| 4. Может ли новость быть интересна компании + правдоподобие | [`stage4_news_relevance/`](stage4_news_relevance/README.md) | `consequence_extraction`, `relevance_scoring`, `plausibility` | — |
| 5. Есть ли в новости инсайт: дополнения из пула, аналитика Банка | [`stage5_insight_detection/`](stage5_insight_detection/README.md) | `news_enrichment`, `bank_analytics`, `insight_judge` (LangGraph), `analyst_loop` | синтетика `analyst_loop/synthetic/` |
| 6. Формируем инсайт и вывод под роль | [`stage6_insight_generation/`](stage6_insight_generation/README.md) | `insight_writer` | план бенчмарка в README |

Очередь LLM-запусков, их входы и стоимость — [`experiments/llm_runs/`](experiments/llm_runs/README.md).
Сквозной запуск этапов 2–6 для пары (компания, роль): [`pipeline.py`](pipeline.py), офлайн-тест — [`test_pipeline.py`](test_pipeline.py).

**Фронтенд демо** — [`frontend/`](frontend/), вид от лица клиента: копия веб-кабинета Альфа-Бизнеса (`index.html`,
ванильный JS) + наш сервис на React и [core-components](https://core-ds.github.io/core-components/master/) (`src/main.jsx`):
однострочный блок «Инсайты» под строкой статусов на главной (топ-1 + «Больше инсайтов»), пункт «Инсайты» в «Мои сервисы»,
страница ленты (фильтры), профиль компании с HITL-вопросами «Верно ли, что …?» (да / нет /
другое / пропустить, по гипотезам агента этапа 2, которые данные не подтвердили), панель инсайта (факты / вывод AI /
гипотезы / позиция Банка раздельно, источники, действия, отзыв с причиной, вопрос по событию, «как мы пришли к выводу»),
переключатель компании группы и роли в шапке. Данные — `frontend/public/feed.json` (синтетика по схеме `Insight` этапа 6
+ гипотезы профиля; заменить выходом пайплайна). Отзывы и ответы пока пишутся в консоль браузера. Только десктоп.
`cd frontend && npm ci && npm run dev` → http://localhost:5173; `npm run build` → статический `dist/`.

## Общий код — `common/`
| Модуль | Что |
|---|---|
| `config.py` | `Settings` из переменных окружения: ключ, модели |
| `data.py` | Загрузка таблиц кейса, форматирование денег и долей |
| `text.py`, `search.py` | 6-буквенные основы; `StemIndex` (TF-IDF + косинус) и кешированный индекс пула новостей |
| `llm/models.py` | `ChatModels`: `agent()` (GLM-5.3-Flash), `cheap()` (массовые вызовы), `vision()` (OCR) — `ChatOpenAI` через LLM-прокси организаторов (LiteLLM), трейсы в Langfuse; `structured()` — ответ в pydantic-схему |
| `llm/prompts.py` | `PromptLibrary`: `.md` → `ChatPromptTemplate` (секции `## system` / `## human`) |
| `llm/cache.py` | `JsonlCache`: ответ LLM на новость или грань оплачивается один раз |
| `llm/tool_loop.py` | `ToolLoop`: подцикл ReAct на LangGraph, завершается вызовом финального инструмента с pydantic-схемой |
| `llm/log.py` | Локальный лог каждого вызова LLM и инструмента агента → `data/logs/calls.jsonl` (промпт, ответ, токены, ₽, кеш); `python -X utf8 common/llm/log.py` — сводка по запускам |
| `llm/testing.py` | `ScriptedChatModel` для офлайн-тестов |
| `safety.py` | Защита от инъекций: чистка спарсенного текста и обёртка `<untrusted_item>` для промптов |
| `dedup.py` | Дедупликация новостей из разных источников с сохранением всех источников |
| `trace.py` | `AgentTrace`: журнал шагов агента в JSONL (`data/traces/<run>.jsonl`) |

**LangChain** — для простых вызовов (промпт → модель → pydantic). **LangGraph** — для циклов с подциклами и памятью:
агент исследования компании (этап 2) и судья инсайта (этап 5).

## Запуск

Локальный запуск на машине с GPU 4 ГБ (GTX 1650): только CPU-режим — оба энкодера (e5 + qwen3) плюс
параллельные агенты не помещаются в видеопамять (CUDA OOM). Перед командой задайте `CUDA_VISIBLE_DEVICES=-1`
(PowerShell: `$env:CUDA_VISIBLE_DEVICES="-1"`; пустая строка на Windows CUDA не отключает); вектор статей
предвычислен (`data/retrieval_v2/*.npz`), на CPU кодируется только запрос — это секунды.

Команды ниже выполняются из корня репозитория. Базовое окружение — Python 3.12 (см. `requirements.txt`);
офлайн-проверки текущей интеграции также прошли на Python 3.13 в Windows.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Локально положить три таблицы кейса в `data/`: `df_Uid_data_hack.csv`, `df_Uid_payment_hack.csv`,
`df_Fid_Uid_link_hack.csv`. Новостной корпус `data/news/news.parquet` уже отслеживается в Git.
PDF аналитики — в `data/analytics/`, их подготовка описана в [reference](docs/MVP_REFERENCE.md#аналитика-банка-и-ocr).

Создать `common/llm/.env` с ключом команды:

```dotenv
LLM_API_KEY=your_team_key
```

Это ключ LLM-прокси организаторов. Ключи Langfuse необязательны и не заменяют `LLM_API_KEY`.

Для первого запуска через аналитический цикл, без скачивания локального энкодера:

```powershell
.\.venv\Scripts\python.exe -X utf8 pipeline.py U_68622 --role ceo --candidates 8 --judge 1 --engine loop --retrieval lexical
```

`--judge 1` задаёт максимум новостей для глубокого разбора, а не обещает один готовый инсайт.
JSON результата: `data/insights/U_68622_ceo.json`; HTML-карточки режима `loop`: `data/insights/cards/`.
Повторный запуск для той же компании и роли заменяет JSON; для сравнения запусков сохраните его копию.
Если окружение уже активировано, вместо `.\.venv\Scripts\python.exe` можно использовать `python`:

```bash
pip install -r requirements.txt
python -X utf8 test_pipeline.py                                   # офлайн, без ключа
python -X utf8 stage2_company_facts/research_agent/test_agent.py  # офлайн, без ключа
python -X utf8 pipeline.py U_68622 --role cfo                      # нужен LLM_API_KEY
python -X utf8 stage5_insight_detection/analyst_loop/test_analyst_loop.py  # офлайн, без ключа и данных
python -X utf8 pipeline.py U_68622 --role cfo --engine loop        # этапы 4–6 через агентный цикл аналитики Банка
python -X utf8 pipeline.py U_68622 --role cfo --retrieval lexical  # прежний поиск по граням
python -X utf8 -m unittest test_merge_regressions test_news_corpus test_pipeline_failures # регрессии, синтетика, без сети
python -X utf8 -m unittest test_bundle_integration # CLI, OCR-кеш/бюджет, совместимость старого и нового анализа
python -X utf8 -m unittest test_deep_integration   # смешанные PDF, таблицы + графики, обновление кешей и границы публикации
python -X utf8 -m unittest test_bank_preparation   # флаги подготовки, общий запуск и кеши всех этапов PDF
```
В обоих режимах (`judge` и `loop`) по умолчанию используется ретривер v2 ([`stage3_news_search/retrieval_v2/`](stage3_news_search/retrieval_v2/README.md)):
стандартные генераторы + фабрика генераторов (GLM, ~4 ₽ один раз на компанию, затем кеш) на поиске e5 + qwen + BM25.
Векторы новостей кешируются по хешу текста; без кеша первый запуск кодирует весь пул (см. README ретривера).
`--retrieval hybrid` — прежний гибридный поиск событий: FRIDA + TF-IDF.
При первом поиске энкодер скачивает веса; эмбеддинги затем кешируются. `--candidates` ограничивает число событий.
FRIDA используется локально только для эмбеддингов; её допустимость по регламенту требует отдельного подтверждения
организаторов. `--retrieval lexical` работает без загрузки энкодера. Генерация текста идёт через LLM API.
На этапы 4–6 передаётся исходная новость-представитель с ID участников события и причинами подбора.
`events.parquet` создаётся автоматически при обращении и пересобирается, если изменились ID, тексты, даты или источники
корпуса. Эмбеддинги также обновляются при изменении текста с прежним ID.
Результаты с `review.ok=False` сохраняются в `draft`, исключаются из числа готовых инсайтов и не показываются как инсайт.
В обоих режимах финальный аудит блокирует выдачу при `unknown_ref` или `no_source` (сверка чисел с источниками убрана: ложные срабатывания):
результат получает `status="audit_rejected"`, текст остаётся в `draft`, поля `insight` нет.
Остальные предупреждения аудита сохраняются без блокировки. Правдоподобие `0` не заменяется значением по умолчанию.
Проверяются также числа в заголовке и итоговом выводе. Измерение графика допускается лишь с «≈» / «около»
непосредственно перед соответствующим числом. Цитаты ЦБ и других источников остаются данными, собственная позиция
Банка выделяется отдельно. Оба движка используют общую библиотеку PDF, таблиц и графиков; при запуске она обновляется
по изменениям входных файлов без автоматического обращения к vision API.
Некорректный структурированный ответ LLM повторяется один раз с уточнением запроса; после второй неудачи новость
получает `status="model_error"` и диагностику `error` (этап, схема, число попыток). Обработка остальных новостей продолжается
в обоих движках. Судья также повторяет попытку, если не получил корректный финальный `decide`.

При пересборке сборщик сначала сохраняет старые и новые снимки в `data/news/versions/<fingerprint>/`, готовит события,
затем атомарно заменяет `news.parquet`. ID — 63-битный хеш нормализованного URL (`url-sha256-63-v1`).
Векторы принадлежат конкретному снимку, проверяются по fingerprint и порядку ID и пересчитываются лениво.
`data/news/rebuild_report.json` содержит итоги миграции ID и проверки сохранённых результатов.

JSON пайплайна содержит `corpus_fingerprint` и `news_references` (URL, хеш содержимого, исходный ID).
Старые ID можно сопоставить с новыми только по тому же URL и неизменённому содержимому; удалённые/изменённые статьи
и старые результаты без доказанной версии источников считаются устаревшими/непроверенными. Исторические выводы не
переписываются под новый текст новости. Для чтения используйте `common.news_corpus.load_results` (актуальные ссылки)
или `load_history` (исходные материалы снимка).

```bash
python -X utf8 common/news_corpus.py --check data/insights/U_68622_cfo.json
python -X utf8 common/news_corpus.py --history data/insights/U_68622_cfo.json
```

Реальные пары golden set связаны с исходными URL в `real_pairs.sources.json`; `load_pairs` возвращает новые ID или
явную ошибку при изменении источника. Экспериментальные векторы/метки событий закреплены за исходным корпусом
в `experiments/retrieval/corpus.sources.json`; после его изменения старые результаты нельзя использовать для оценки
нового корпуса — их нужно пересоздать вместе с привязкой.

Переменные окружения (секреты — в `common/llm/.env`, он в .gitignore; переменные окружения важнее файла):
`LLM_API_KEY` — ключ команды к LLM-прокси организаторов (LiteLLM, `LLM_BASE_URL`, по умолчанию
`https://changellenge.up.railway.app/v1`); модели `AGENT_MODEL` (по умолчанию `z-ai/glm-5.3-flash`), `CHEAP_MODEL`,
`VISION_MODEL` — id ровно как в `/v1/models` прокси; `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` —
трейсинг каждого LLM-вызова в Langfuse (один запуск скрипта = одна сессия `<скрипт>-<время>`); `ALFA_DATA` — путь к данным.
```
python -X utf8 common/llm/smoke.py   # живая проверка: вызов, structured output, ToolLoop (~0.02 ₽)
python -X utf8 common/llm/quota.py   # потрачено сегодня из 500 ₽ (сброс 00:00 UTC = 03:00 МСК) + цены моделей
```
Все команды — из корня репозитория, кроме этапа 1 (см. его README).

## PDF, таблицы и графики после интеграции

Восстановлены CLI `--retrieval`, снимок корпуса при сохранении и различение уровня/изменения показателя.
Новые модули Вовы сохранены: полный OCR, очистка страниц, таблицы, оцифровка графиков, выжимки отчётов и банковский reviewer.
Распознанные моделью, измеренные кодом и подтверждённые числа учитываются раздельно; приближённые значения требуют «≈» / «около».

Для локального распознавания надписей и чисел графиков/таблиц один раз подготовить языковые данные:

```powershell
python -m common.ocr_local --download
```

Скачиваются только `rus` и `eng` из [официального tessdata_fast](https://github.com/tesseract-ocr/tessdata_fast)
в игнорируемую `.cache/tessdata/`. Само распознавание выполняется локально: используется установленный Tesseract
через `pytesseract` либо встроенный Tesseract PyMuPDF. Системный `tesseract.exe` для второго варианта не нужен.
При распознавании сеть не используется. Можно указать готовые языковые данные через `TESSDATA_PREFIX`.

```powershell
# Явно подготовить OCR и пересобрать библиотеку перед общим запуском.
python -X utf8 pipeline.py U_68622 --role cfo --engine loop --retrieval lexical --ocr

# Весь процесс вместе: полный OCR → графики → таблицы → выжимки → агентский цикл.
python -X utf8 pipeline.py U_68622 --role ceo --candidates 8 --judge 1 --engine loop --retrieval lexical --bank-prepare

# Выбрать только сверку графиков и таблиц; необходимый OCR включится автоматически.
python -X utf8 pipeline.py U_68622 --role ceo --engine loop --retrieval lexical --bank-charts --bank-tables

# Дополнительно подключить LLM-reviewer банковских фрагментов с бюджетом на этот запуск.
python -X utf8 pipeline.py U_68622 --role cfo --engine loop --retrieval lexical --bank-review --bank-budget 1.5
```

Подготовка выключена по умолчанию. `--bank-charts`, `--bank-tables`, `--bank-digest` выбирают отдельные этапы;
`--bank-prepare` включает их все и полный OCR. Эти флаги требуют `--engine loop`. Результаты сразу передаются циклу,
кеши сохраняются между запусками. Подготовка общая для всех новостей запуска; `--candidates` не ограничивает число PDF.
Бюджеты на запуск: `--ocr-budget` (15 ₽), `--charts-budget` (8 ₽), `--tables-budget` (7 ₽), `--digest-budget` (3 ₽).
Они относятся к подготовке, генерация инсайтов оплачивается отдельно. При исчерпании бюджета используется успевшая
подготовиться аналитика, в `data/analytics/preparation.json`, журнале и поле результата `bank_preparation` виден статус `partial`.
`--ocr` сохраняет прежнее поведение простого OCR; reviewer включается отдельно через `--bank-review`.
Подробности — в [reference](docs/MVP_REFERENCE.md#аналитика-банка-и-ocr).

**Данные**:
- `data/news/news.parquet` — в репозитории;
- таблицы кейса `data/df_*.csv` и `data/analytics/*.pdf` конфиденциальны: их кладут в `data/` локально (они в .gitignore).


## Фрагменты статей и подтверждение фактов

В обоих режимах retrieval (`hybrid` и `lexical`) дополнительный TF-IDF индекс оценивает все фрагменты
статьи (1000 символов, перекрытие 150), а не только начало. Оценка статьи — максимум по её фрагментам;
в гибридном режиме она дополняет исходную лексическую оценку перед объединением с dense-поиском.
Для события выбирается исходная статья с лучшим совпадением. Её ID, URL и полный текст сохраняются.
`matched_fragments` содержит до трёх совпадений: `news_id`, `chunk_index`, `text`, `start`, `end`, `score`.
Позиции — индексы символов Python, `end` не включается. Фрагменты передаются в оценку релевантности,
batch-классификатор, судью и генератор текста, а также сохраняются в JSON результата.

Извлечение последствий читает статью блоками по 3000 символов с перекрытием 250. Это может увеличить
число LLM-вызовов для длинных статей; результаты кешируются по содержимому и схеме.
`stage4.consequences.facts` содержит факты с дословными цитатами и `evidence` (source, quote, start, end, url).
Цитаты проверяются кодом по исходному тексту; выдуманные помещаются в `rejected_facts`.
`affected` — выводы о последствиях, отдельные от извлечённых фактов.

Каждый итоговый `what_happened` также содержит `evidence`. Код пересчитывает позиции цитаты и URL,
отклоняет отсутствие/подмену цитаты даже при положительном ответе LLM-проверяющего. После одной попытки
исправления результат остаётся `draft` со статусом `review_rejected`, если проблема не устранена.
HTML-карточка показывает цитаты рядом с фактами. Наличие цитаты подтверждает её происхождение;
смысловую связь утверждения и цитаты проверяет LLM-review.

Пополнение новостного корпуса запускается отдельно: [инструкция](stage1_news_pool/UPDATER.md).

```powershell
.\.venv\Scripts\python.exe -X utf8 pipeline.py U_68622 --role cfo --engine judge --retrieval lexical
.\.venv\Scripts\python.exe -X utf8 pipeline.py U_68622 --role cfo --engine loop --retrieval hybrid
.\.venv\Scripts\python.exe -X utf8 -m unittest test_news_updates test_merge_regressions test_news_corpus test_pipeline_failures
```

Гибридный поиск, cross-encoder и инкрементальное извлечение фактов: [инструкция](docs/HYBRID_NEWS_SEARCH.md).

## API и параллельная обработка

REST API со Swagger, очередями независимых анализов и фоновой загрузкой PDF: [docs/API.md](docs/API.md).
