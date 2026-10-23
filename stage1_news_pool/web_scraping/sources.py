from datetime import datetime, timezone


def recent_months(count: int = 2) -> tuple[str, ...]:
    """Current UTC month and previous months; no third-party dependency."""
    now = datetime.now(timezone.utc)
    year, month = now.year, now.month
    result = []
    for _ in range(count):
        result.append(f"{year:04d}-{month:02d}")
        month -= 1
        if month == 0:
            year -= 1
            month = 12
    return tuple(result)


# source -> entry points + priority.
# tier 1: economy / business / industry / official / legal-accounting — fetched fully.
# tier 2: science + general media — only business-relevant rubrics (keep regex on url, or on page categories in build.py).
# keys: sitemap (index or urlset, recursed, pruned by date), rss, archive (daily pages), keep (url regex), topic (default topic).
LENTA_RUBRICS = ('economics', 'science', 'russia', 'world', 'forces')

SOURCES = {
    # --- tier 1: business media ---
    'rbc':        {'tier': 1, 'topic': 'economy', 'sitemap': ['https://www.rbc.ru/sitemap_index.xml'],
                   'rss': ['https://rssexport.rbc.ru/rbcnews/news/30/full.rss'],
                   'keep': r'rbc\.ru/(economics|business|finances|technology_and_media|politics|rbcfreenews|society|investment|own_business)/'},
    'rbc-realty': {'tier': 1, 'topic': 'real_estate', 'sitemap': ['https://realty.rbc.ru/sitemap_index.xml']},
    'rbc-trends': {'tier': 1, 'topic': 'economy', 'sitemap': ['https://trends.rbc.ru/trends/sitemap_index.xml']},
    'interfax':   {'tier': 1, 'topic': 'economy', 'sitemap': ['https://www.interfax.ru/SEO_SiteMapIndex.xml'],
                   'rss': ['https://www.interfax.ru/rss.asp'], 'keep': r'interfax\.ru/(business|russia|world|moscow)/'},
    'kommersant': {'tier': 1, 'topic': 'economy', 'sitemap': [f'https://www.kommersant.ru/sitemaps/archive/sitemap-{m}.xml' for m in recent_months()] + ['https://www.kommersant.ru/sitemaps/news.xml'],
                   'rss': ['https://www.kommersant.ru/RSS/news.xml']},
    'vedomosti':  {'tier': 1, 'topic': 'economy', 'sitemap': ['https://www.vedomosti.ru/sitemap.xml'],
                   'rss': ['https://www.vedomosti.ru/rss/news', 'https://www.vedomosti.ru/rss/rubric/economics', 'https://www.vedomosti.ru/rss/rubric/business', 'https://www.vedomosti.ru/rss/rubric/finance', 'https://www.vedomosti.ru/rss/rubric/technology', 'https://www.vedomosti.ru/rss/rubric/politics', 'https://www.vedomosti.ru/rss/rubric/realty'],
                   'keep': r'vedomosti\.ru/(business|economics|finance|technology|politics|realty|investments|auto|management|press_releases|strana)/'},
    'forbes':     {'tier': 1, 'topic': 'economy', 'sitemap': ['https://www.forbes.ru/sitemap.xml'],
                   'keep': r'forbes\.ru/(biznes|finansy|investicii|tekhnologii|milliardery|society|nedvizhimost|karera-i-svoy-biznes|svoi-biznes|mneniya)'},
    '1prime':     {'tier': 1, 'topic': 'economy', 'sitemap': ['https://1prime.ru/sitemap_article_index.xml'],
                   'rss': ['https://1prime.ru/export/rss2/index.xml']},
    'frankmedia': {'tier': 1, 'topic': 'finance', 'sitemap': ['https://frankmedia.ru/sitemap_index.xml']},
    'banki':      {'tier': 1, 'topic': 'finance', 'rss': ['https://www.banki.ru/xml/news.rss']},
    'expert':     {'tier': 1, 'topic': 'economy', 'sitemap': ['https://expert.ru/sitemap-index.xml']},
    'bfm':        {'tier': 1, 'topic': 'economy', 'sitemap': ['https://www.bfm.ru/sitemap.xml'], 'keep': r'bfm\.ru/news/'},
    'thebell':    {'tier': 1, 'topic': 'economy', 'sitemap': ['https://thebell.io/sitemap_index.xml']},
    'profile':    {'tier': 1, 'topic': 'economy', 'rss': ['https://profile.ru/feed/']},  # sitemap 429
    'secretmag':  {'tier': 1, 'topic': 'business', 'sitemap': ['https://secretmag.ru/sitemap.xml.gz']},
    'vc':         {'tier': 1, 'topic': 'business', 'rss': ['https://vc.ru/rss/all']},
    'dp':         {'tier': 1, 'topic': 'regions', 'sitemap': ['https://www.dp.ru/sitemap-index.xml']},
    # --- tier 1: markets / investing ---
    'finam':      {'tier': 1, 'topic': 'markets', 'rss': ['https://www.finam.ru/analysis/conews/rsspoint/', 'https://www.finam.ru/international/advanced/rsspoint/']},
    'moex':       {'tier': 1, 'topic': 'markets', 'rss': ['https://www.moex.com/export/news.aspx?cat=100']},
    'finmarket':  {'tier': 1, 'topic': 'markets', 'rss': ['https://www.finmarket.ru/rss/mainnews.asp']},  # terms not found
    # --- tier 1: regulators / government ---
    'cbr':        {'tier': 1, 'topic': 'monetary', 'rss': ['https://www.cbr.ru/rss/RssPress', 'https://www.cbr.ru/rss/eventrss', 'https://www.cbr.ru/rss/RssNews']},
    # official sites, RSS only (checked 2026-10-22). minfin / fas publish under CC BY 4.0 (attribution only);
    # minpromtorg / minenergo: no terms page found, official body information; minenergo article pages give no text -> RSS lead is used
    'minfin':     {'tier': 1, 'topic': 'finance', 'rss': ['https://minfin.gov.ru/rss_news?mod=news&lim=50', 'https://minfin.gov.ru/rss_news?mod=lib&lim=50']},
    'fas':        {'tier': 1, 'topic': 'economy', 'rss': ['https://fas.gov.ru/news.rss', 'https://fas.gov.ru/publications.rss']},
    'customs':    {'tier': 1, 'topic': 'economy', 'rss': ['https://customs.gov.ru/press/federal/rss']},  # FTS: terms page customs.gov.ru/info_open not read
    'minpromtorg': {'tier': 1, 'topic': 'economy', 'rss': ['https://minpromtorg.gov.ru/rss']},
    'minenergo':  {'tier': 1, 'topic': 'energy', 'rss': ['https://minenergo.gov.ru/rss']},
    # --- tier 1: legal / accounting (CFO, chief accountant) ---
    'consultant': {'tier': 1, 'topic': 'legal', 'rss': ['https://www.consultant.ru/rss/hotdocs.xml', 'https://www.consultant.ru/rss/zw.xml', 'https://www.consultant.ru/rss/nw.xml']},
    'klerk':      {'tier': 1, 'topic': 'accounting', 'rss': ['https://www.klerk.ru/export/news.rss', 'https://www.klerk.ru/export/articles.rss']},  # sitemap: lastmod-only
    'buh':        {'tier': 1, 'topic': 'accounting', 'sitemap': ['https://buh.ru/sitemap.xml'], 'keep': r'buh\.ru/news/'},
    'glavbukh':   {'tier': 1, 'topic': 'accounting', 'sitemap': ['https://www.glavbukh.ru/sitemap.xml'], 'keep': r'glavbukh\.ru/news/'},
    # --- tier 1: industry ---
    'bigpowernews': {'tier': 1, 'topic': 'energy', 'sitemap': ['https://www.bigpowernews.ru/sitemap.xml']},
    'fishnews':   {'tier': 1, 'topic': 'agro', 'rss': ['https://fishnews.ru/rss']},  # sitemap: lastmod-only
    'retail':     {'tier': 1, 'topic': 'retail', 'sitemap': ['https://www.retail.ru/sitemap.xml'], 'keep': r'retail\.ru/(news|articles)/'},
    'new-retail': {'tier': 1, 'topic': 'retail', 'sitemap': ['https://new-retail.ru/sitemap-new.xml'], 'rss': ['https://new-retail.ru/rss/']},
    'stroygaz':   {'tier': 1, 'topic': 'construction', 'rss': ['https://stroygaz.ru/rss/']},
    'gudok':      {'tier': 1, 'topic': 'transport', 'sitemap': ['https://gudok.ru/sitemap_index.xml']},
    'logirus':    {'tier': 1, 'topic': 'transport', 'rss': ['https://logirus.ru/news/rss/']},
    'korabel':    {'tier': 1, 'topic': 'transport', 'rss': ['https://www.korabel.ru/rss/news.xml']},
    'seanews':    {'tier': 1, 'topic': 'transport', 'sitemap': ['https://seanews.ru/sitemap.xml']},
    'autostat':   {'tier': 1, 'topic': 'auto', 'sitemap': ['https://www.autostat.ru/sitemap_index.xml'], 'keep': r'autostat\.ru/news/'},
    'vademec':    {'tier': 1, 'topic': 'pharma', 'sitemap': ['https://vademec.ru/sitemap-news.xml', 'https://vademec.ru/sitemap.xml'], 'keep': r'vademec\.ru/news/'},
    'pharmvestnik': {'tier': 1, 'topic': 'pharma', 'rss': ['https://pharmvestnik.ru/rss.xml']},
    'sostav':     {'tier': 1, 'topic': 'media_ads', 'rss': ['https://www.sostav.ru/rss']},
    'comnews':    {'tier': 1, 'topic': 'telecom', 'rss': ['https://www.comnews.ru/rss']},
    'cnews':      {'tier': 1, 'topic': 'tech', 'sitemap': ['https://www.cnews.ru/inc/sitemap.xml'], 'rss': ['https://www.cnews.ru/inc/rss/news.xml'], 'keep': r'cnews\.ru/(news|articles)/'},
    'tadviser':   {'tier': 1, 'topic': 'tech', 'rss': ['https://www.tadviser.ru/xml/tadviser.xml']},
    # --- tier 2: science ---
    'nplus1':     {'tier': 2, 'topic': 'science', 'sitemap': ['https://nplus1.ru/sitemap.xml']},
    'naked-science': {'tier': 2, 'topic': 'science', 'sitemap': ['https://naked-science.ru/sitemap-news.xml'], 'rss': ['https://naked-science.ru/feed']},
    'indicator':  {'tier': 2, 'topic': 'science', 'sitemap': ['https://indicator.ru/sitemap.xml.gz']},
    'scientificrussia': {'tier': 2, 'topic': 'science', 'sitemap': ['https://scientificrussia.ru/sitemap.xml']},
    # --- tier 2: general media, business-relevant rubrics only ---
    'ria':        {'tier': 2, 'topic': None, 'sitemap': ['https://ria.ru/sitemap_article_index.xml']},  # no rubric in url -> filtered by page categories
    'lenta':      {'tier': 2, 'topic': None, 'archive': [{'url': f'https://lenta.ru/rubrics/{r}/{{y}}/{{m}}/{{d}}/page/{{p}}/',
                                                         'link': r'href="(/news/{y}/{m}/{d}/[a-z0-9_-]+/)"', 'rubric': r} for r in LENTA_RUBRICS]},
    'iz':         {'tier': 2, 'topic': None, 'sitemap': ['https://iz.ru/sitemap.xml']},
    'rg':         {'tier': 2, 'topic': None, 'sitemap': ['https://rg.ru/sitemaps/index.xml']},
    'mk':         {'tier': 2, 'topic': None, 'sitemap': ['https://www.mk.ru/sitemap/index.xml'], 'keep': r'mk\.ru/(economics|politics|science|social)/'},
    'news.ru':    {'tier': 2, 'topic': None, 'sitemap': ['https://news.ru/sitemap.xml'], 'keep': r'news\.ru/(economics|vlast|world|europe|usa|asia|russia|tech|science)/'},
    'pnp':        {'tier': 2, 'topic': 'legal', 'sitemap': ['https://www.pnp.ru/site-map/sitemap_index.xml'], 'keep': r'pnp\.ru/(economics|politics|social|law|in-world)/'},
    'regnum':     {'tier': 2, 'topic': None, 'sitemap': [f'https://regnum.ru/sitemap/news/{m}.xml' for m in recent_months()]},
}
