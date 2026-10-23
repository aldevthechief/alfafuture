# Terms of use of the news sources (checked 2026-10-21)

Rule from the organizers (TG-8053, `org-facts/facts/news-sources.md` N-04): legal and free methods only; respect the sites' own rules,
if automated collection is forbidden, pick another source. Old (`news/`) and new (`web_scraping/`) collectors use the **same 53 sources**
(`sources.py` differs only by `recent_months()`), so the finding applies to both. Neither reads robots.txt; the User-Agent is spoofed as Chrome.

Method: terms pages found from each homepage footer and grepped for automation/copy clauses (`tos_raw.json` not kept), Vedomosti / RBC / Kommersant /
CBR read in full. 8 hosts did not open from here; terms of ~20 sources were not found (listed below) — "not found" is not "allowed".

## Explicit ban on automated collection (11.8% of the repo corpus, 10.8% of the rebuilt one)
| source | share (repo corpus) | clause |
|---|---|---|
| vedomosti | 655 | vedomosti.ru/info/reprint_rules, п. 1.4: «Запрещается использование материалов, полученных с помощью любых сервисов автоматизированного извлечения информации» без разрешения АО «Бизнес Ньюс Медиа»; п. 4.2: создание интерактивных баз данных; п. 4.3: коммерческое использование |
| lenta | 216 | lenta.ru/legal: «Запрещается использовать автоматизированные скрипты (программы, боты, краулеры) для сбора информации на Ресурсе … без специального на то разрешения» |
| lesprom | 151 | lesprom.com/legal/agreement: no copying/scraping/crawling by automated means except the official API or a public search engine obeying robots.txt | **Исключён из `sources.py` 22.10.2026** (в пуле были только генерируемые таблицы пошлин, не новости).
| buh | 46 | buh.ru/pravila-ispolzovaniya-materialov-sayta/: «Запрещается автоматизированное извлечение информации сайта любыми сервисами без официального разрешения ООО «1С-Паблишинг»» |
| klerk | 12 | klerk.ru/about/331994/: не использовать без письменного разрешения автоматизированные программы для сбора информации |

## Ban on copying / downloading / extraction, no word about robots (5.2% more)
- rbc, rbc-trends, rbc-realty (497): rbc.ru/legal п. 8.2: материалы не могут быть скопированы, скачаны или иным образом извлечены из Продукта РБК, переработаны, распространены без разрешения правообладателя; использование только в рамках функционала сайта.

## Copyright terms only (62% of the corpus): no scraping clause, but redistribution is limited
- kommersant (2647): /copyright — use under ст. 1274 ГК без разрешения запрещено; цитирование ≤30%, без переработки, со ссылкой; photos not at all. (robots.txt disallows /doc-rss, /rules, `?` URLs; articles `/doc/…` allowed.)
- 1prime (992): terms_of_use — информация «для личного (некоммерческого) использования», воспроизведение и распространение запрещены без письменного согласия; коммерческое использование запрещено.
- interfax (830): /license — свободно только в рамках ст. 1273/1274 ГК (личные цели, цитирование со ссылкой).
- ria (400): copyright.html — любое использование контента без предварительного письменного разрешения запрещено.
- rg (263): «РГ» — скачивание/копирование без разрешения только гражданами для личного использования.
- iz (206): свободно для некоммерческого использования, коммерческое — только с согласия.
- mk (181): письменное разрешение правообладателя + прямая ссылка. forbes (174): условия перепечатки (автор, ссылка, дата).
- Permissive: bfm (435) — воспроизведение разрешено, кроме контента агентств-партнёров; cbr (8) — свободное воспроизведение со ссылкой (научные доклады — только с согласия авторов).

## Terms not found / not checked
dp, gudok, news.ru, moex (индексы: коммерческое использование только по договору), finam, glavbukh, cnews, consultant, expert, thebell, pnp,
tadviser, sostav, autostat, stroygaz, pharmvestnik, seanews, korabel, vademec, retail (общее соглашение 1С — на странице правил нет отдельного пункта), bigpowernews, frankmedia, secretmag, new-retail, ria-hosts that did not open.

## Not violated
- robots.txt Disallow: 0 of 9147 corpus URLs (custom matcher with wildcards; `urllib.robotparser` gives false hits).
- Crawl-delay ignored: stroygaz 30 s, pharmvestnik 10 s, moex 10 s, comnews 5 s, logirus 3 s (collector: 3 parallel requests per host).

## Options (decision pending)
1. Drop the 5 explicit-ban sources (+RBC) from `sources.py` — ~12–17% of the corpus; Vedomosti and RBC are big business sources, replace via RSS/aggregators with a permissive licence (cbr, bfm, tass-type feeds).
2. Keep them, document the risk in README (hackathon, non-commercial, internal analysis, only metadata + short quotes shown in UI) — the organizers' N-05 (headline/date/link/lead only) is still unanswered.
3. Whatever is chosen: add `allowed_automation` per source in `sources.py` (RC-19), read robots.txt + crawl-delay in `fetch.py`, store only what the UI shows.
