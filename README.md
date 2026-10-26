# AlfaFuture Insights

**[Инструкция по запуску](https://mishanya000.github.io/alfa-future-hack-page/)** · **[Демо с сохранёнными результатами](https://alfafuture-insights.duckdns.org/)**

Персональная аналитика для руководителей компаний: система сопоставляет новости, профиль бизнеса и аналитические отчёты Банка, затем формирует инсайты с источниками и объяснением возможного влияния на компанию.

Демо показывает результаты предыдущих прогонов моделей и служит дополнительным сценарием проверки. Для нового анализа, обсуждения инсайтов и динамической подгрузки новостей запустите полную версию через Docker. Если размещённое демо недоступно, сохранённые кабинеты можно посмотреть локально в режиме `server`.

Команды ниже сверены с веткой `integration-demo`, коммит `4dd237a` от 26 октября 2026 года. В текущей `main` Docker-сборки ещё нет.

## Требования для запуска

- Git либо архив нужной ветки проекта.
- Docker Desktop на Windows/macOS либо Docker Engine с плагином Compose на Linux.
- Интернет для первой сборки, загрузки отсутствующих весов и обращения к LLM-прокси.
- Для полного анализа: таблицы кейса, доступный ключ LLM и необходимые источники.

Python и Node.js на хосте для запуска через Docker не нужны. Python 3.12 и зависимости устанавливаются в образе; GPU не требуется. Первая сборка загружает в том числе CPU-версию PyTorch, поэтому её время зависит от сети и компьютера.

Проверьте Docker:

```shell
docker --version
docker compose version
```

## Полный запуск проекта

Добавьте в `common/llm/.env` настройки, заменив `your_team_key` своим действующим ключом:

```dotenv
# Copy to common/llm/.env for live LLM calls. The frontend works without a key.
LLM_API_KEY=sk-viBWE0gDSjn7YrFV1h1hEQ
LLM_BASE_URL=https://changellenge.up.railway.app/v1
AGENT_MODEL=z-ai/glm-5.3-flash
CHEAP_MODEL=google/gemma-4-26b-a4b-it
VISION_MODEL=qwen/qwen3-vl-30b-a3b-instruct
LANGFUSE_SECRET_KEY=sk-lf-aeda40eb-03c8-4715-869a-bcb205e5506b
LANGFUSE_PUBLIC_KEY=pk-lf-3db0055d-48d1-4ab8-b00f-71478ec7569e
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
```

Ключ остаётся в локальном `.env`, исключённом из Git. Переменные окружения процесса имеют приоритет над файлом. Идентификаторы моделей должны поддерживаться выбранным прокси. Для замены ключа в запущенной системе отредактируйте файл и выполните `docker compose restart backend feedback-api`.

Разложите данные:

```text
data/
├── df_Uid_data_hack.csv
├── df_Uid_payment_hack.csv
├── df_Fid_Uid_link_hack.csv
├── news/news.parquet
└── analytics/
    └── *.pdf
```

Новостной корпус входит в ветку проекта. CSV и PDF предоставляются отдельно. Компания для нового анализа должна присутствовать в таблицах кейса.

Переключитесь в полный режим в том же терминале:

```powershell
# Windows PowerShell
$env:CABINET_MODE='full'
docker compose up -d --build
```

```bash
# Linux/macOS
export CABINET_MODE=full
docker compose up -d --build
```

| Переменная | Назначение |
| --- | --- |
| `LLM_API_KEY` | Ключ LLM-прокси; нужен для новых вызовов моделей |
| `LLM_BASE_URL` | Адрес OpenAI-совместимого прокси |
| `AGENT_MODEL` | Модель агентского анализа |
| `CHEAP_MODEL` | Модель массовых вызовов |
| `VISION_MODEL` | Модель обработки изображений и PDF |
| `CABINET_MODE` | `server`: сохранённые кабинеты; `full`: новый анализ и чат; задаётся в терминале для Compose |
| `BACKEND_PORT`, `FEEDBACK_PORT` | Внешние порты API, по умолчанию `18000` и `18001`; задаются для Compose |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` | Необязательный трейсинг; не заменяет ключ LLM |
| `ALFA_CACHE_REDIS_URL` | Адрес Redis; в Compose уже задан как `redis://redis:6379/0` |

Без явного `CABINET_MODE` Compose использует `full`. PDF при старте Docker автоматически не обрабатываются: подготовка запускается через API или CLI. Загрузка новых PDF через `POST /v1/reports/upload` выполняется в фоне; подробности — в [контракте API](docs/API.md).

Swagger: [основной API](http://localhost:18000/docs), [API обсуждений](http://localhost:18001/docs). Дополнительные флаги и схемы запросов — [docs/API.md](docs/API.md).

## Динамическое пополнение новостей

Обновлятор запускается отдельно от агентского цикла. Обычная команда `docker compose up` не включает регулярный сбор. Для одного прохода:

```shell
docker compose run --rm pipeline python -X utf8 -m stage1_news_pool.updater --once --only cbr --limit 20
```

Для регулярного сбора оставьте в отдельном терминале:

```shell
docker compose run --rm pipeline python -X utf8 -m stage1_news_pool.updater --interval 900 --days 7
```

Сам сборщик не требует LLM-ключа. Он сохраняет прежние статьи, обрабатывает новые и изменённые публикации и атомарно публикует корпус. Текущий анализ остаётся на своём снимке; следующие запросы получают обновление. Статус — `data/news/updater_status.json`. Выход — Ctrl+C. Подробности — [stage1_news_pool/UPDATER.md](stage1_news_pool/UPDATER.md).

## Кеши и управление

Дополнительный импорт сохранённых ответов моделей в Redis:

```shell
docker compose run --rm pipeline python -X utf8 scripts_seed_redis.py
```

## Материалы проекта

- [Вход в кабинеты и различия режимов](docs/CABINET_ENTRY.md).
- [API и фоновые задачи](docs/API.md).
- [Контракты пайплайна и интеграция](docs/MVP_REFERENCE.md).
- [Гибридный поиск и факты из новостей](docs/HYBRID_NEWS_SEARCH.md).