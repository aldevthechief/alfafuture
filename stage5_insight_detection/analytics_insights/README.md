# Инсайты из аналитики Банка

Агент ищет инсайты в отчётах Банка сам, без новости: находит позицию Банка, привязывает её к фактам компании и роли, проверяет
цитаты и числа кодом. Работает параллельно с новостными движками и пишет в тот же список результатов.

## Как запускать
```bash
python -X utf8 pipeline.py U_14366 --role cfo                           # новости + аналитика (по умолчанию --source both)
python -X utf8 pipeline.py U_14366 --role cfo --source analytics --analytics-cache-only   # только кэш, без вызова моделей
```
| Флаг | Что делает |
|---|---|
| `--source news\|analytics\|both` | `both` по умолчанию: новостные движки как раньше + аналитика параллельно |
| `--analytics-cache-only` | аналитика только из кэша (в том числе устаревшего, с пометкой `stale_cache`), модель не вызывается |
| `--analytics-refresh` | игнорировать кэш, запустить агента (≈3–5 ₽ на пару компания × роль, GLM, температура 0) |
| `--analytics-share 0.4` | доля инсайтов из аналитики среди готовых: не меньше одного и не больше этой доли |

Результат — `data/insights/<компания>_<роль>.json`: строки новостей (как раньше) и строки аналитики (`source: bank_analytics`,
`news_id: null`) и строка итога прогона (`status: analysis_run`: покрытые каналы, отброшенные темы, вопросы клиенту).
Ошибка аналитики не мешает новостям: она печатается строкой «analytics: …», строк аналитики нет.

## Кэш
Пайплайн и `run.py` получают факты через профиль компании Кирилла (`profiles.jsonl`, с построением из таблиц при отсутствии записи).
Пояснения чувствительности и внешних факторов передаются отдельным блоком вне лимита первых 25 фактов. Они не получают
номера `fact:N` и не могут использоваться как доказательства; исходные факты доступны через `company_facts`.
Изменение пояснений также меняет ключ кэша.

Альтернативный сценарий Вовчика включается явно: `--engine loop --source news --bank-insights`.
С `--source analytics` или `both` этот флаг несовместим: так два генератора инсайтов из отчётов не запускаются одновременно.
При `--bank-prepare` подготовка отчётов заканчивается до открытия поискового индекса и запуска параллельных агентов.

`cache/<компания>_<роль>.json` лежит в репозитории: демо-компании отдаются без вызова модели. Ключ кэша — хеш роли, фактов компании
и PDF в папке аналитики: новые факты или новый отчёт делают кэш устаревшим, и агент запускается заново (пока не указан
`--analytics-cache-only`). Полный прогон без ошибок кэшируется сам. Положить в кэш готовый прогон `run.py`:
```bash
python -X utf8 stage5_insight_detection/analytics_insights/service.py --import-runs data/analytics_agent/runs.json
```
В кэше сейчас три пары (U_14366 cfo, U_57706 treasury, U_177581 ceo) из прогона GLM-5.3-Flash, температура 0.

## Контракт инсайта v2
[insight_v2.py](../../stage6_insight_generation/insight_writer/insight_v2.py): надмножество прежнего `Insight`. Старый JSON читается как v2,
`to_v1` возвращает старый вид; в сохранённых результатах каждый `insight` новостей тоже в v2 (`migrate_result`).
Единица — утверждение с цитатой (`Claim` + `Evidence`: источник, страница, смещения внутри фрагмента, итог проверки кодом), триггер —
новость или отчёт (`origin`: `news` или `bank`). Гипотеза хранит `check` (что её подтвердит) и `supports` (на чём держится); у утверждений,
действий и оценки обратная ссылка `depends_on`. `AnalysisRun` — контейнер прогона на пару компания × роль.

## Что агент отдаёт и что проверяет код
Агент: `bank_claims` с цитатами, `company_facts` (номера), `link_inference`, `mechanism`, `effect`, `actions`, `assumptions`, `impact`,
`conclusion`. Код без LLM превращает это в `Insight` v2 ([to_insight.py](to_insight.py)) и проверяет: цитаты на странице отчёта
([quotes.py](../bank_analytics/quotes.py): точно, с правкой распознавания, страница исправляется), числа
([number_check.py](../bank_analytics/number_check.py)), факты о компании и придуманные суммы, привязку гипотез, оценку влияния, дату события.
Замечания приходят агенту один раз с номером W1, W2…; ничего не блокируется, принять замечание можно с причиной (`accepted_warnings`).

## Тесты и справочные файлы
```bash
python -X utf8 stage5_insight_detection/analytics_insights/test_agent.py      # агент, проверки, цикл
python -X utf8 stage5_insight_detection/analytics_insights/test_to_insight.py # контракт v2, отображение кандидата
python -X utf8 stage5_insight_detection/analytics_insights/test_service.py    # кэш, квота, строки результата, карточка
python -X utf8 stage5_insight_detection/bank_analytics/test_quotes.py         # проверка цитат
python -X utf8 stage5_insight_detection/bank_analytics/test_number_check.py   # проверка чисел
```
Дизайн и предложения: [analytics_agent_design.md](../bank_analytics/analytics_agent_design.md), разбор трейсов: [trace_review_v4.md](trace_review_v4.md).
