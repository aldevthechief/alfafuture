# Этап 2: какие файлы нужны от других этапов (21.10.2026)

Файлы не в git (данные, `.gitignore`), их нужно передать отдельно — диск или чат.

| От кого | Файл | Зачем этапу 2 | Где используется |
|---|---|---|---|
| Максон | `experiments/news_channels/out/channels_news.parquet` (news_id × A1..H1, ансамбль 5 моделей) | модельная «сторона новости» вместо словаря: сила новости по 21 рычагу | `stage2_company_facts/sensitivity/news_levers.py`, `match()` принимает матрицу [новости × 21] |
| Максон | те же оценки для корпуса Миши на 31,9 тыс. новостей (если считались) | замер на большом корпусе | то же |
| Миша | `data/news/news.parquet` на 31,9 тыс. новостей (та же схема, что 9,1 тыс.) | замеры расширения и сопоставления на большом корпусе | `stage2_company_facts/expansion/`, `sensitivity/news_levers.py` |
| Миша | `data/news/export/business_contexts.jsonl`, `fact_candidates.jsonl` | контекст события для объяснения «почему важно» | задача «обогащение профиля из новостной базы» |

## Что этап 2 отдаёт другим

`data/companies/profiles.jsonl` — `python -X utf8 stage2_company_facts/profile/profile.py` (115 крупных компаний, ~6 с),
описание полей в [`profile/README.md`](profile/README.md). Для этапа 3 полезны `search_terms.goods / inputs`,
`external_factors[].keywords`, `sensitivity`; этап 5 строит `CompanyProfile(**p['card'])` без чтения CSV.
