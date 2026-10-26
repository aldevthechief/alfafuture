# THIRD_PARTY_NOTICES

Перечень сторонних компонентов, моделей, данных, сервисов и смоделированных
интеграций проекта `alfafuture-hack`. Состояние: 26 октября 2026 года.
Права на сторонние материалы остаются у их авторов и правообладателей.
Этот перечень не меняет их лицензии и не предоставляет права на данные кейса,
публикации СМИ, изображения или товарные знаки.

## Фиксация версий и границы перечня

- CPython **3.12.10**: `.python-version`, корневой `Dockerfile`.
- Node.js **20.19.0**, npm **10.8.2**: `frontend/.node-version`,
  `frontend/.nvmrc`, `frontend/package.json`, `frontend/Dockerfile`.
- `requirements.txt` фиксирует через `==` прямые и ключевые Python-зависимости,
  включая SDK моделей, Transformers, токенизаторы и русский словарь.
  `requirements-dev.txt` добавляет фиксированную версию pytest.
- `frontend/package-lock.json` (lockfile v3) фиксирует также транзитивные
  npm-зависимости, адреса архивов и integrity; установка — `npm ci`.
- Python-перечень ниже относится к явно закреплённым пакетам. Он не является
  полным SBOM транзитивных пакетов или системных библиотек контейнера.
  Лицензии и copyright notices из устанавливаемых дистрибутивов сохраняются.

## Python-компоненты

Версии соответствуют `requirements.txt`; ссылки ведут к исходным проектам.
Обозначения лицензий сверены с метаданными установленных пакетов там, где они
доступны. Комплект конкретного wheel может содержать дополнительные лицензии.

| Компонент / источник | Версия | Назначение | Лицензия / условия |
|---|---|---|---|
| [langchain-core](https://github.com/langchain-ai/langchain) | 1.6.12 | Сообщения, инструменты, runnable | MIT |
| [langchain-openai](https://github.com/langchain-ai/langchain) | 1.7.2 | Клиент OpenAI-совместимого прокси | MIT |
| [openai Python SDK](https://github.com/openai/openai-python) | 3.37.0 | HTTP-клиент и ошибки API | Apache-2.0 |
| [langchain-text-splitters](https://github.com/langchain-ai/langchain) | 1.1.3 | Разбиение текстов | MIT |
| [langgraph](https://github.com/langchain-ai/langgraph) | 1.2.16 | Граф агентского пайплайна | MIT |
| [langgraph-checkpoint-sqlite](https://github.com/langchain-ai/langgraph) | 3.1.2 | SQLite-чекпоинты графа | MIT |
| [langfuse Python SDK](https://github.com/langfuse/langfuse-python) | 5.0.3 | Трассировка LLM-вызовов | MIT; условия сервиса отдельно |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | 1.2.3 | Загрузка конфигурации окружения | BSD-3-Clause |
| [pydantic](https://github.com/pydantic/pydantic) | 2.14.1 | Схемы и валидация | MIT |
| [pandas](https://github.com/pandas-dev/pandas) | 3.0.5 | Табличные данные | BSD-3-Clause |
| [NumPy](https://github.com/numpy/numpy) | 2.5.2 | Массивы и вычисления | BSD-3-Clause; метаданные дистрибутива также указывают 0BSD, MIT, Zlib, CC0-1.0 |
| [PyArrow](https://github.com/apache/arrow) | 25.0.1 | Parquet | Apache-2.0 |
| [scikit-learn](https://github.com/scikit-learn/scikit-learn) | 1.9.0 | TF-IDF и метрики | BSD-3-Clause |
| [SciPy](https://github.com/scipy/scipy) | 1.18.1 | Численные методы | BSD-3-Clause; дополнительные notices в wheel |
| [PyMuPDF / MuPDF](https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright) | 1.28.2 | PDF, рендеринг, встроенный OCR | GNU AGPL v3 либо коммерческая лицензия Artifex |
| [Pillow](https://github.com/python-pillow/Pillow) | 11.0.0 | Предобработка изображений | MIT-CMU / HPND, см. LICENSE дистрибутива |
| [HTTPX](https://github.com/encode/httpx) | 0.28.1 | HTTP-запросы | BSD-3-Clause |
| [Requests](https://github.com/psf/requests) | 2.34.2 | HTTP-запросы | Apache-2.0 |
| [lxml](https://github.com/lxml/lxml) | 6.1.3 | XML/HTML | BSD-3-Clause; libxml2/libxslt имеют собственные notices |
| [lxml_html_clean](https://github.com/fedora-python/lxml_html_clean) | 0.4.5 | Очистка HTML | BSD-3-Clause |
| [Trafilatura](https://github.com/adbar/trafilatura) | 2.3.1 | Извлечение текста новостей | Apache-2.0 |
| [PyTorch](https://github.com/pytorch/pytorch) | 2.6.0 | Локальный инференс | BSD-3-Clause; дополнительные notices в дистрибутиве |
| [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) | 5.1.1 | Энкодеры и cross-encoder | Apache-2.0 |
| [Transformers](https://github.com/huggingface/transformers) | 4.57.6 | Загрузка и инференс моделей | Apache-2.0 |
| [huggingface-hub](https://github.com/huggingface/huggingface_hub) | 0.36.2 | Получение весов и конфигураций | Apache-2.0 |
| [Tokenizers](https://github.com/huggingface/tokenizers) | 0.22.2 | Токенизация | Apache-2.0 |
| [bm25s](https://github.com/xhluca/bm25s) | 0.2.14 | BM25-поиск | MIT |
| [PyStemmer / Snowball](https://github.com/snowballstem/pystemmer) | 3.0.0 | Стемминг | MIT и BSD, см. лицензии обёртки и Snowball |
| [pymorphy3](https://github.com/no-plagiarism/pymorphy3) | 2.0.4 | Лемматизация | MIT |
| [pymorphy3-dicts-ru](https://github.com/no-plagiarism/pymorphy3-dicts) | 2.4.417150.4580142 | Русские морфологические словари | Метаданные пакета: MIT; исходные словари OpenCorpora — [CC BY-SA 3.0](https://opencorpora.org/dict.php) |
| [FastAPI](https://github.com/fastapi/fastapi) | 0.161.1 | REST API и OpenAPI | MIT |
| [Uvicorn](https://github.com/Kludex/uvicorn) | 0.54.0 | ASGI-сервер | BSD-3-Clause |
| [python-multipart](https://github.com/Kludex/python-multipart) | 0.0.32 | Загрузка файлов через API | Apache-2.0 |
| [XlsxWriter](https://github.com/jmcnamara/XlsxWriter) | 3.3.0 | XLSX-экспорт | BSD-2-Clause |
| [redis-py](https://github.com/redis/redis-py) | 6.4.0 | Клиент кеша Redis | MIT; лицензия сервера отдельно |
| [pytest](https://github.com/pytest-dev/pytest) | 9.1.1 | Офлайн-тесты, `requirements-dev.txt` | MIT |

В Docker отдельно устанавливается CPU-wheel `torch==2.6.0+cpu` из
[официального индекса PyTorch](https://download.pytorch.org/whl/cpu).
Вариант CUDA 12.4 описан комментарием в `requirements.txt`; это отдельный вариант
окружения с дополнительными условиями поставщиков GPU-компонентов.

## Фронтенд, окружение и инфраструктура

| Компонент / источник | Версия или источник фиксации | Назначение | Лицензия / условия |
|---|---|---|---|
| [Alfa core-components](https://github.com/core-ds/core-components) | 50.37.0 | UI-компоненты | MIT |
| [React](https://github.com/facebook/react) | 18.3.1 | UI | MIT |
| [React DOM](https://github.com/facebook/react) | 18.3.1 | DOM-рендеринг | MIT |
| [Vite](https://github.com/vitejs/vite) | 6.3.5 | Сборка фронтенда | MIT |
| [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react) | 4.5.2 | React-плагин сборки | MIT |
| [CPython](https://www.python.org/downloads/release/python-31210/) | 3.12.10 | Python runtime | PSF License и включённые notices |
| [Node.js](https://nodejs.org/en/blog/release/v20.19.0) / [npm](https://github.com/npm/cli) | 20.19.0 / 10.8.2 | JS runtime и менеджер пакетов | Node.js: MIT и включённые notices; npm CLI: Artistic-2.0 |
| [nginx](https://nginx.org/en/LICENSE) | Образ `nginx:1.27-alpine` | Статика и reverse proxy | BSD-2-Clause |
| [Redis server](https://redis.io/legal/licenses/) | Образ `redis:7-alpine` | Кеш ответов, AOF/RDB | Для 7.2 и ниже BSD-3-Clause; для 7.4 — RSALv2 либо SSPLv1; зависит от разрешённого тега |
| [SQLite](https://www.sqlite.org/copyright.html) | В составе Python/образа | История, календарь, индексы, чекпоинты | Public domain |
| [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) | PyMuPDF либо системный пакет | Локальное распознавание | Apache-2.0 |
| [tessdata_fast](https://github.com/tesseract-ocr/tessdata_fast) | `eng`, `rus`; локальная загрузка из `main` либо apt-пакеты Docker | Языковые данные OCR | Apache-2.0 |
| [Debian](https://www.debian.org/legal/licenses/) / [Alpine](https://www.alpinelinux.org/) | Базовые образы в Dockerfile | ОС, g++, Git, системные библиотеки | Лицензии отдельных пакетов, включая GPL для GCC/Git |

Точные транзитивные версии npm и заявленные ими лицензии перечислены в
`frontend/package-lock.json`. Теги Redis/nginx, пакеты apt/apk и OCR-данные не
закреплены digest/хешами; таблица не утверждает побитовую воспроизводимость образа.

Дополнительные экспериментальные компоненты не входят в стандартную установку:
[EasyOCR](https://github.com/JaidedAI/EasyOCR) (Apache-2.0,
`stage5_insight_detection/bank_analytics/ocr_easy.py`),
[pytesseract](https://github.com/madmaze/pytesseract) (Apache-2.0, необязательная
обёртка в `common/ocr_local.py`), [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
(1.3.5, MIT), [Accelerate](https://github.com/huggingface/accelerate) (Apache-2.0)
и [Kaggle CLI](https://github.com/Kaggle/kaggle-api) (Apache-2.0).
Версии, отсутствующие в манифестах экспериментов, не фиксируются этим документом.

## Модели и веса

| Модель / поставщик | Роль и статус | Источник и условия |
|---|---|---|
| `intfloat/multilingual-e5-large-instruct` | Основной dense retrieval, новости и аналитика | [Карточка, MIT](https://huggingface.co/intfloat/multilingual-e5-large-instruct) |
| `Qwen/Qwen3-Embedding-0.6B` | Второй dense-канал retrieval v2 | [Карточка, Apache-2.0](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) |
| `BAAI/bge-reranker-v2-m3` | Необязательный cross-encoder, выключен по умолчанию в пайплайне | [Карточка, Apache-2.0](https://huggingface.co/BAAI/bge-reranker-v2-m3) |
| `z-ai/glm-5.3-flash` | `AGENT_MODEL`: агент, судья, writer | ID API организаторов в `common/config.py`; условия прокси и поставщика Z.ai |
| `google/gemma-4-26b-a4b-it` | `CHEAP_MODEL`: массовые текстовые задачи | ID API организаторов в `common/config.py`; условия прокси и поставщика Google |
| `qwen/qwen3-vl-30b-a3b-instruct` | `VISION_MODEL`: vision/OCR через API | ID API организаторов в `common/config.py`; условия прокси и поставщика Qwen |

Для API-моделей указаны именно идентификаторы конфигурации, а не утверждение о
доступности публичных весов или открытой лицензии. Они переопределяются через
`AGENT_MODEL`, `CHEAP_MODEL`, `VISION_MODEL`; `ENCODER_MODEL` и `RERANKER_MODEL`
также допускают другие локальные модели. При переопределении нужно обновить перечень.
Код не закрепляет HF commit/revision весов; API-прокси не предоставляет в репозитории
неизменяемую версию backend-модели. Пины Python-пакетов этого не обеспечивают.

Модели из экспериментов `experiments/retrieval/push.py`, не все используются
в рабочем пайплайне:

- [ai-forever/FRIDA](https://huggingface.co/ai-forever/FRIDA);
- [deepvk/USER-bge-m3](https://huggingface.co/deepvk/USER-bge-m3);
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3);
- [opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1);
- [Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B);
- [ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826](https://huggingface.co/ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826).

Их условия определяются LICENSE и карточкой каждой модели, включая условия базовых
моделей. FRIDA сохранена в экспериментальных конфигурациях, основной поиск заменён на E5.

## Датасеты, производные материалы и визуальные источники

| Набор / расположение | Происхождение и использование | Условия / статус |
|---|---|---|
| `data/df_Uid_data_hack.csv`, `data/df_Uid_payment_hack.csv`, `data/df_Fid_Uid_link_hack.csv` | Организаторы кейса Альфа-Банка; компании, платежи, связи физлиц и юрлиц | Конфиденциальные данные кейса; локальная установка, исключены из Git. Открытая лицензия в репозитории не предоставлена |
| `data/analytics/*.pdf` | Аналитические материалы Банка, включая загружаемые пользователем PDF | Локальные материалы; права у авторов/поставщиков. OCR, таблицы, графики и выжимки — производные материалы с теми же ограничениями исходников |
| `data/news/news.parquet` | Собранные публикации СМИ и официальных источников с URL и датами | Отслеживается в Git; единой открытой лицензии корпуса не заявлено. Права и условия принадлежат каждому источнику |
| `maxamsecret/alfa-hack-ru-business-news` | [Kaggle-набор](https://www.kaggle.com/datasets/maxamsecret/alfa-hack-ru-business-news), указанный в `experiments/retrieval/push.py` | Канал доставки корпуса для вычисления эмбеддингов, не независимый источник прав на статьи |
| `stage2_company_facts/golden_set/` | Подборки компаний и эталонные факты команды по данным кейса | Производные от данных кейса, не независимый открытый датасет |
| `stage3_news_search/golden_set/real_pairs.json`, `real_pairs.sources.json` | Размеченные командой пары компания/новость | Ссылки и хеши привязывают разметку к исходным публикациям |
| `stage3_news_search/golden_set/synthetic_needles.json` | Синтетические новости, цепочки и отрицательные примеры команды | Только оценка поиска; не реальные публикации |
| `stage5_insight_detection/analyst_loop/synthetic/`, `stage6_insight_generation/golden_set/synthetic_insights.json` | Синтетические компании, новости, страницы аналитики и инсайты | Тестовые фикстуры; не сведения о реальных событиях и не официальная позиция Банка |
| `evals/retrieval_golden/golden/`, `experiments/retrieval/` | Разметка, генераторные запросы, оценки и эмбеддинги | Производные от корпуса/кейса и результатов моделей; происхождение см. MANIFEST и `corpus.sources.json` |
| `data/llm_cache/llm.jsonl.gz`, `data/news/llm_cache/`, `data/api/redis-warmup.rdb` | Сохранённые результаты LLM и кеши для повторного запуска | Производные от входных материалов; не новые независимые источники фактов |
| `backend/demo_cache/`, `stage5_insight_detection/analytics_insights/cache/`, `frontend/public/feed.json`, `data/calendar/events.db` | Сохранённые результаты и демонстрационные материалы кабинета | Предрассчитанные данные; происхождение фактов определяется исходными статьями, кейсом и аналитикой |
| `frontend/reference/*.png` | Скриншоты команды дизайна Альфа-Банка из [публикации 2024 года](https://habr.com/ru/companies/alfa/articles/809991/) и [публикации 2023 года](https://habr.com/ru/companies/alfa/articles/770064/) | Визуальный референс; подробности в `frontend/reference/README.md`. Открытая лицензия на скриншоты и бренд в проекте не заявлена |

Конфигурируемые источники новостей перечислены в
[`stage1_news_pool/web_scraping/sources.py`](stage1_news_pool/web_scraping/sources.py):
РБК (включая Недвижимость и Тренды), Интерфакс, Коммерсантъ, Ведомости, Forbes,
Прайм, Frank Media, Банки.ру, Эксперт, Бизнес FM, The Bell, Профиль, Секрет фирмы,
VC.ru, Деловой Петербург, Финам, Московская биржа, Финмаркет, Банк России,
Минфин, ФАС, ФТС, Минпромторг, Минэнерго, КонсультантПлюс, Клерк, БУХ.1С,
Главбух, Bigpowernews, Fishnews, Retail.ru, New Retail, Строительная газета,
Гудок, Logirus, Корабел.ру, SeaNews, Автостат, CNews, TAdviser, N+1,
Naked Science, Индикатор, Научная Россия, РИА Новости, Лента.ру, Известия,
Российская газета, Московский комсомолец, NEWS.ru, Парламентская газета, REGNUM.
Исторический корпус также может содержать источники, уже исключённые из конфига
(например, Lesprom); фактическое происхождение записи определяется её `source` и URL.
Локальная инвентаризация условий источников:
[`sources_validity.md`](stage1_news_pool/web_scraping/sources_validity.md).
Наличие URL в конфиге не означает наличие публикаций этого источника в корпусе
или разрешение на их распространение.

## Сторонние сервисы и реальные подключения

| Сервис | Как используется | Статус |
|---|---|---|
| LiteLLM-прокси организаторов, `https://changellenge.up.railway.app/v1` | OpenAI-совместимые LLM-вызовы по `LLM_API_KEY`; адрес меняется через `LLM_BASE_URL` | Реальный внешний API; условия, квоты и доступ определяются организаторами. Наличие OpenAI SDK не означает прямые вызовы сервиса OpenAI |
| [Langfuse](https://langfuse.com/) | Трассировка при наличии `LANGFUSE_*`, адрес через `LANGFUSE_BASE_URL` | Необязательное реальное подключение; лицензия SDK отделена от условий выбранного сервиса |
| [Hugging Face Hub](https://huggingface.co/) | Скачивание локальных весов и конфигураций | Реальное обращение при отсутствии локального кеша; условия Hub и лицензии моделей |
| [Kaggle](https://www.kaggle.com/) | Экспериментальные GPU kernels и подготовка векторов | Необязательная внешняя инфраструктура, не обязательный онлайн-сервис инференса |
| Сайты СМИ и официальные RSS/sitemap | Сбор и обновление корпуса | Реальные HTTP-запросы отдельного сборщика; условия каждого сайта |
| GitHub, PyPI, npm registry, индекс PyTorch, реестры контейнеров и apt | Установка кода, образов и OCR-данных | Источники поставки зависимостей; для стандартного инференса OCR-данные готовятся заранее |
| Redis и SQLite | Локальный кеш и хранение состояния | Реальные локальные компоненты; внешняя банковская инфраструктура не требуется |

## Смоделированные интеграции и демонстрационные ограничения

- **Кабинет Альфа-Бизнеса:** собственная демонстрационная оболочка по визуальным
  референсам. Логотип, наименования и товарные знаки принадлежат их правообладателям.
  Подключение к промышленному кабинету или core banking в проекте отсутствует.
- **Вход и идентичность:** выбор компании/роли и `X-User-ID` (по умолчанию
  `demo-user`) задают контекст прототипа. Банковские SSO, IAM и проверка полномочий
  сотрудника не интегрированы; см. `docs/CABINET_ENTRY.md` и `docs/API.md`.
- **Банковские остатки и продукты в оболочке:** `backend/cabinet.py` содержит
  демонстрационные значения; `_generated_account()` рассчитывает баланс, кредит
  и депозит как доли выручки. Они не получены из банковских счетов.
- **Платежи и контрагенты:** `backend/payments.py` действительно считает агрегаты
  по предоставленным CSV кейса. Это файловая интеграция со снимком данных,
  а не доступ к текущим транзакциям или исполнение платежей.
- **Готовые кабинеты и статическая лента:** читают сохранённые анализы и `feed.json`.
  Показ предрассчитанного ответа не означает новый онлайн-вызов модели.
  Новый анализ в полной версии использует настоящий пайплайн и внешний LLM API.
- **Диалог по инсайтам, реакции, календарь и документы:** реализованы локальными
  API проекта, SQLite и обработкой загруженных PDF. Синхронизация с промышленными
  банковскими чатами, CRM, ЭДО и внешними календарями не реализована.
- **Рекомендации действий:** текстовые выводы и гипотезы модели; автоматическое
  открытие банковских продуктов, отправка поручений и сделок не интегрированы.
- **Офлайн-тесты:** `common/llm/testing.py` (`ScriptedChatModel`) и тестовые
  заглушки воспроизводят заранее заданные ответы, инструменты и энкодеры без
  обращения к поставщикам. Синтетические фикстуры не выдаются за живые интеграции.

При добавлении зависимости, смене модели, нового датасета или реального сервиса
обновляются соответствующий манифест/lock-файл и этот перечень.
