"""How much a news-based card can be trusted, in words a client reads: the outlets behind the event, their kind and the
plausibility score of stage 4 (corroboration, tier, official source, numbers that disagree).

The corpus is a whitelist of 44 outlets (stage1 sources.py): no tabloids; tier 1 = business / industry / official,
tier 2 = general media. Most events of the demo come from one outlet, so the label names the outlet and its kind
instead of saying «1 источник». No LLM.
"""
NAMES = {'kommersant': 'Коммерсантъ', '1prime': 'ПРАЙМ', 'interfax': 'Интерфакс', 'vedomosti': 'Ведомости', 'rbc': 'РБК',
         'rbc-realty': 'РБК Недвижимость', 'rbc-trends': 'РБК Тренды', 'bfm': 'BFM', 'ria': 'РИА Новости', 'dp': 'Деловой Петербург',
         'rg': 'Российская газета', 'gudok': 'Гудок', 'lenta': 'Lenta.ru', 'iz': 'Известия', 'mk': 'МК', 'forbes': 'Forbes',
         'cnews': 'CNews', 'news.ru': 'News.ru', 'moex': 'Московская биржа', 'expert': 'Эксперт', 'pnp': 'ПНП', 'thebell': 'The Bell',
         'glavbukh': 'Главбух', 'buh': 'Бухгалтерия.ру', 'vademec': 'Vademecum', 'finam': 'Финам', 'retail': 'Retail.ru',
         'bigpowernews': 'BigpowerNews', 'autostat': 'Автостат', 'new-retail': 'New Retail', 'seanews': 'SeaNews',
         'stroygaz': 'Стройгазета', 'klerk': 'Клерк', 'sostav': 'Sostav', 'cbr': 'Банк России', 'consultant': 'КонсультантПлюс',
         'garant': 'Гарант', 'banki': 'Банки.ру', 'fishnews': 'Fishnews', 'tadviser': 'TAdviser', 'pharmvestnik': 'Фармвестник',
         'regnum': 'Regnum', 'korabel': 'Корабел.ру', 'vc': 'vc.ru', 'logirus': 'Logirus', 'frankmedia': 'Frank Media'}
OFFICIAL = {'cbr', 'moex'}


def credibility(sources: list[str], tiers: dict[str, int] | None = None, plausibility: float | None = None) -> dict:
    """sources: outlets that reported the event (the card's news and its same-event reports)."""
    tiers = tiers or {}
    src = list(dict.fromkeys(s.strip() for s in sources if isinstance(s,str) and s.strip()))
    names = [NAMES.get(s, s) for s in src]
    official = [NAMES.get(s, s) for s in src if s in OFFICIAL]
    business = all(tiers.get(s) == 1 for s in src) if src else False
    if official:
        label = f'Официальный источник: {", ".join(official)}' + (f'; ещё {len(src) - len(official)} изд.' if len(src) > len(official) else '')
    elif len(src) >= 2:
        label = f'Публикации в {len(src)} изданиях: {", ".join(names[:4])}' + ('…' if len(names) > 4 else '')
    elif src:
        kind = ' — деловое издание' if business else ' — общественно-политическое издание' if tiers.get(src[0])==2 else ''
        label = f'{names[0]}{kind}; в сохранённых материалах одно издание'
    else:
        label = 'Источник не указан'
    level = 'high' if official or len(src) >= 3 else 'medium' if len(src) == 2 or business else 'low'
    return {'label': label, 'level': level, 'outlets': names, 'n_outlets': len(src), 'official': bool(official),
            'business_only': business, 'plausibility': plausibility}


def for_result(news: dict, same_event: list[dict], plausibility: float | None) -> dict:
    """news / same_event rows: {'source', 'tier'} — the card's main news and the other reports of its event."""
    rows = [news, *same_event]
    def tier(value):
        try: return int(value) if value is not None else None
        except (ValueError,TypeError): return None
    return credibility([r.get('source', '') for r in rows], {r.get('source', ''): tier(r.get('tier')) for r in rows},
                       plausibility)


def from_saved(result):
    """Only known same-event articles count; related evidence may cover different events."""
    references=result.get('news_references') or {}
    evidence=(result.get('evidence') or {}).get('news') or []
    articles={str(n.get('id')):n for n in evidence if isinstance(n,dict)}
    articles.update({str(key).removeprefix('news:'):value for key,value in references.items() if isinstance(value,dict)})
    main=articles.get(str(result.get('news_id')))
    if not main: return None
    ids={str(value) for value in (result.get('retrieval') or {}).get('event_news_ids') or []}
    same=[value for key,value in articles.items() if key in ids and key!=str(result.get('news_id'))]
    return for_result(main,same,(result.get('stage4') or {}).get('plausibility'))
