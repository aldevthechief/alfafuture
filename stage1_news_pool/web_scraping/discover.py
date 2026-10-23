"""Stage 1: discover news URLs for the last N days from sitemaps + RSS -> data/index.jsonl"""
import argparse, gzip, json, re, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlsplit, urlunsplit, parse_qsl, urlencode

import requests
from lxml import etree

try:
    from .sources import SOURCES
except ImportError:
    from sources import SOURCES

H = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36'}
DATA = Path(__file__).resolve().parents[2] / 'data' / 'news'
MAX_SITEMAPS = 150  # per source, guards against huge archive indexes
DATE_IN_URL = re.compile(r'(20\d\d)[-/_]?(0[1-9]|1[0-2])[-/_]?(0[1-9]|[12]\d|3[01])')
# skip non-article pages: tags, authors, rubrics, galleries, videos
JUNK = re.compile(r'/(tags?|authors?|person|persons|rubrics?|themes?|story|photo|video|gallery|tv|specials?|search|comments)/|/person-|[?&]page=', re.I)
# rubrics with no business value; matched against url and page categories
OFFTOPIC = re.compile(r'sport|спорт|футбол|хоккей|showbiz|шоу-бизнес|звезд|culture|kultura|культур|style|стиль жизни|zozh|зож|'
                      r'wine|forbeslife|nopolitics|afisha|афиша|kupony|horoscop|гороскоп|recipe|рецепт|travel|туризм и отдых|lifestyle|beauty|moda|мода|kino', re.I)


# Drop only known tracking parameters. Keep article-identifying query parameters.
TRACKING_PARAMS = {"fbclid", "gclid", "yclid", "mc_cid", "mc_eid", "_openstat"}


def canonical_url(url: str) -> str:
    parts = urlsplit(url.strip())
    params = [
        (key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in TRACKING_PARAMS
    ]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, urlencode(params), ""))


def is_article(u):
    # article URLs carry a numeric id, a date, or a long slug; section/rubric pages don't
    path = urlparse(u).path.rstrip('/')
    last = path.rsplit('/', 1)[-1]
    if JUNK.search(u):
        return False
    has_id = re.search(r'\d{5,}', path)
    dated_slug = DATE_IN_URL.search(path) and re.search(r'[a-z]{3}', last, re.I)  # /2026/10/15/vodka, /2026-10-19_slug
    return bool(has_id or dated_slug or last.count('-') >= 3)


def parse_date(s):
    if not s:
        return None
    s = s.strip()
    try:
        d = datetime.fromisoformat(s.replace('Z', '+00:00'))
    except ValueError:
        try:
            d = parsedate_to_datetime(s)
        except Exception:
            return None
    return d if d.tzinfo else d.replace(tzinfo=timezone(timedelta(hours=3)))  # Russian sites default to MSK


def url_date(u):
    m = DATE_IN_URL.search(u)
    if m:
        try:
            d = datetime(*map(int, m.groups()), tzinfo=timezone.utc)
            return d if d <= datetime.now(timezone.utc) + timedelta(days=1) else None  # digits of numeric ids
        except ValueError:
            pass
    return None


def fetch_xml(s, url):
    r = s.get(url, headers=H, timeout=30)
    r.raise_for_status()
    b = r.content
    if b[:2] == b'\x1f\x8b':
        b = gzip.decompress(b)
    return etree.fromstring(b, etree.XMLParser(recover=True, huge_tree=True))


def txt(el, name):
    r = el.xpath(f'.//*[local-name()="{name}"]/text()')
    return r[0].strip() if r else None


def crawl_sitemap(s, url, cutoff, out, budget):
    if budget[0] <= 0:
        return
    budget[0] -= 1
    root = fetch_xml(s, url)
    if root is None:
        return
    tag = etree.QName(root).localname
    if tag == 'sitemapindex':
        kids = []
        for sm in root.xpath('./*[local-name()="sitemap"]'):
            loc, lm = txt(sm, 'loc'), parse_date(txt(sm, 'lastmod'))
            if not loc:
                continue
            d = lm or url_date(loc)
            if d and d < cutoff:
                continue
            if not d and DATE_IN_URL.search(loc) is None and re.search(r'(tag|author|person|rubric|theme|categor|section|static|page|video|photo)', loc, re.I):
                continue
            kids.append((d or datetime.max.replace(tzinfo=timezone.utc), loc))
        for _, loc in sorted(kids, reverse=True):  # freshest first
            try:
                crawl_sitemap(s, loc, cutoff, out, budget)
            except Exception as e:
                print(f'  ! {loc}: {e}', file=sys.stderr)
    else:
        rows, lastmod_only = [], Counter()
        for u in root.xpath('./*[local-name()="url"]'):
            loc = txt(u, 'loc')
            if not loc or not is_article(loc):
                continue
            pub = parse_date(txt(u, 'publication_date'))
            d = pub or url_date(loc)
            src = 'news' if pub else 'url'
            if not d:
                d, src = parse_date(txt(u, 'lastmod')), 'lastmod'
                if d:
                    lastmod_only[d.date()] += 1
            if d and cutoff <= d <= datetime.now(timezone.utc) + timedelta(days=1):
                rows.append({'url': loc, 'date': d.isoformat(), 'date_src': src, 'title': txt(u, 'title')})
        # sites that re-stamp lastmod on every page at once: such lastmod says nothing about publication
        if lastmod_only and max(lastmod_only.values()) > 1500:
            rows = [r for r in rows if r['date_src'] != 'lastmod']
        out += rows


def crawl_rss(s, url, cutoff, out):
    root = fetch_xml(s, url)
    for it in root.xpath('//*[local-name()="item" or local-name()="entry"]'):
        link = txt(it, 'link') or (it.xpath('./*[local-name()="link"]/@href') or [None])[0]
        d = parse_date(txt(it, 'pubDate') or txt(it, 'published') or txt(it, 'updated') or txt(it, 'date'))
        if link and d and d >= cutoff:
            out.append({'url': link.strip(), 'date': d.isoformat(), 'date_src': 'rss', 'title': txt(it, 'title'),
                        'summary': txt(it, 'description') or txt(it, 'summary'), 'rss_category': txt(it, 'category')})


def crawl_archive(s, cfg, cutoff, out):
    """Daily archive pages: cfg['url'] has {y} {m} {d} {p}; walk pages until no new links."""
    day = datetime.now(timezone.utc)
    while day >= cutoff - timedelta(days=1):
        seen = set()
        for p in range(1, 200):
            t = s.get(cfg['url'].format(y=day.year, m=f'{day.month:02}', d=f'{day.day:02}', p=p), headers=H, timeout=30).text
            new = set(re.findall(cfg['link'].format(y=day.year, m=f'{day.month:02}', d=f'{day.day:02}'), t)) - seen
            if not new:
                break
            seen |= new
        out += [{'url': urljoin(cfg['url'], u), 'date': day.date().isoformat() + 'T00:00:00+03:00', 'date_src': 'archive', 'title': None, 'rss_category': cfg.get('rubric')} for u in seen]
        day -= timedelta(days=1)


def discover(name, cfg, cutoff):
    s, out, budget = requests.Session(), [], [MAX_SITEMAPS]
    arch = cfg.get('archive', [])
    for ac in arch if isinstance(arch, list) else [arch]:
        try:
            crawl_archive(s, ac, cutoff, out)
        except Exception as e:
            print(f'  ! {name} archive {ac["url"]}: {e}', file=sys.stderr)
    for u in cfg.get('sitemap', []):
        try:
            crawl_sitemap(s, u, cutoff, out, budget)
        except Exception as e:
            print(f'  ! {name} {u}: {e}', file=sys.stderr)
    for u in cfg.get('rss', []):
        try:
            crawl_rss(s, u, cutoff, out)
        except Exception as e:
            print(f'  ! {name} {u}: {e}', file=sys.stderr)
    # rss/archive items are already fresh articles; sitemap items keep the url filters
    out = [r for r in out if not OFFTOPIC.search(r['url']) and (r['date_src'] == 'rss' or 'keep' not in cfg or re.search(cfg['keep'], r['url']))]
    seen, res = {}, []
    for r in out:  # merge sitemap+rss records per url, keep richest fields
        key = canonical_url(r['url'])
        r['url'] = key
        if key in seen:
            for k, v in r.items():
                if v and not seen[key].get(k):
                    seen[key][k] = v
            continue
        r.update(source=name, tier=cfg['tier'], topic=cfg.get('topic'))
        seen[key] = r
        res.append(r)
    print(f'{name:14} {len(res):6}  (sitemaps used {MAX_SITEMAPS - budget[0]})', flush=True)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=31)
    ap.add_argument('--only', nargs='*', help='refresh only these sources in an existing index')
    ap.add_argument('--out', default=str(DATA / 'index.jsonl'))
    a = ap.parse_args()
    cutoff = datetime.now(timezone.utc) - timedelta(days=a.days)
    srcs = {k: v for k, v in SOURCES.items() if not a.only or k in a.only}
    with ThreadPoolExecutor(16) as ex:
        results = list(ex.map(lambda kv: discover(*kv, cutoff), srcs.items()))
    rows = [r for rs in results for r in rs]
    if a.only and Path(a.out).exists():
        rows += [r for r in map(json.loads, open(a.out, encoding='utf-8')) if r['source'] not in srcs and r['source'] in SOURCES]
    with open(a.out, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'total {len(rows)} urls -> {a.out}')
