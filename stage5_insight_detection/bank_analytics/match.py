"""Task 2: pick the bank's analytics for an event and a company profile.

Score of a fragment = 0.45 lexical similarity to the event (stem TF-IDF, normalized by the best hit)
                    + 0.25 overlap of event and fragment topics
                    + 0.20 fit to the company: max exposure of the profile over the fragment topics
                    + 0.10 the fragment is the bank's own view, not a quote.
Freshness: aging -0.05; stale or undated fragments are multiplied by 0.3 and always ranked after fresh ones, so the
2021 steel report can be shown as context but never as the current position. A chart whose values were read off the
scale (`unverified`) is multiplied by 0.85: its title matches an event well, but it carries no usable numbers, so a text
with the bank's view goes first (verified charts and tables are not touched). Every match carries its score parts and a
short reason, they go to the trace. Pure code, no LLM.
"""
import re
from dataclasses import dataclass

from common.safety import render_untrusted
from common.text import stems
from stage5_insight_detection.bank_analytics.library import TOPICS, BankLibrary, Fragment

W_LEX, W_TOPIC, W_PROFILE, W_OWN = 0.45, 0.25, 0.20, 0.10
SHAKY_CHART = 0.85
# select(): two stages, absolute text similarity (cosine of stem TF-IDF), not divided by the best hit
LEX_FULL = 0.25  # a cosine at which the text counts as fully similar
MIN_EVENT = 0.5  # the event part (text + topics) a candidate needs: a shared topic and some text, or a clearly similar
# text alone; none left = «аналитики Банка нет». Company fit, own view and freshness only rerank, they never let one in
W_EVENT, W_COMPANY, W_OWN2, W_FRESH = 0.55, 0.25, 0.10, 0.20
FRESH = {'fresh': 1.0, 'aging': 0.25}  # on one topic a fresh view goes above an aging one
# a bank review of one issuer («(далее – Компания)», «Наше мнение: Компания…») fits news about that issuer only
ISSUER = re.compile(r'далее\s*[–-]\s*Компани|\bКомпани[яиюей]\b')
ISSUER_LEX = 0.2
BRIEF_CAP = 8  # bank fragments in one brief, the enrichment included
# another country's economy: its institutions, indicators, currencies or the country itself named and Russia not; a report
# most of whose topical fragments are such (a digest of the US gold market) makes the rest of it foreign too
FOREIGN = re.compile(r'(?<![а-яa-z])(сша|американск|европ|еврозон|китай|китая|кнр|пекин|инди[ияйю]|япони|британ|турци|бразили|'
                     r'мексик|германи|франци|фрс|fomc|ецб|wgc|dxy|cpi|pce|etf|nasdaq|s&p)|[€£¥]', re.I)
DOMESTIC = re.compile(r'(?<![а-яa-z])(росси|рф\b|рубл|минфин(?! сша)|банк\w* росси|цб рф)', re.I)
MACRO = {'ставка', 'инфляция', 'кредит', 'бюджет_и_налоги', 'рынок_труда', 'ставка_мир'}  # every company's, not its business
SECTORAL = {'золото', 'металлургия', 'строительство', 'энергия_и_сырьё'}  # topics of a sector, not of the economy at large
GENERIC = set(stems('прочие прочих виды видов деятельность деятельности специализированных магазинах предоставление услуг '
                    'области связанная использованием помощью общая оптовая розничная торговля производство '
                    'включая кроме исключением'))


@dataclass
class BankMatch:
    fragment: Fragment
    score: float
    parts: dict
    reason: str

    def dict(self) -> dict:
        f = self.fragment
        return {'id': f.id, 'report': f.report, 'date': f.date, 'page': f.page, 'freshness': f.freshness, 'own_view': f.own_view,
                'score': self.score, 'parts': self.parts, 'reason': self.reason, 'text': f.text[:400], 'kind': f.kind,
                'label': f.caption(), 'unverified': f.unverified, 'approx': sorted(f.approx()), 'measured': sorted(f.measured()),
                'figures': f.figures}

    def prompt_item(self, limit: int = 900, page: bool = True) -> str:
        """The fragment for a model prompt: a chart is "график Банка, рис. N, стр. M", its values read off the scale are
        not passed at all (Fragment.brief_text)."""
        f = self.fragment
        review = self.parts.get('review') or {}
        meta = {'дата': f.date, 'стр.': f.page if page else None, 'свежесть': f.freshness,
                'позиция Банка': ('да' if f.own_view else 'цитата/данные') if page else None,
                'канал': review.get('channel') or None, 'взгляд Банка и событие': review.get('agrees') or None,
                'агент сомневается': review.get('why') if review.get('doubt') else None}
        return render_untrusted(f.id, f.caption(), f.brief_text()[:limit], meta)


class BankMatcher:
    def __init__(self, library: BankLibrary, min_score: float = 0.15):
        self.library, self.min_score = library, min_score

    def match(self, query: str, topics: list[str], exposure: dict[str, float], k: int = 5,
              exclude: set[str] = frozenset()) -> list[BankMatch]:
        if not self.library.fragments:
            return []
        lex = list(self.library.scores(query))
        top = max(lex) if lex and max(lex) > 0 else 1.0
        out = []
        for f, s in zip(self.library.fragments, lex):
            s = s if s >= 0.05 else 0.0  # a stray shared word is not similarity, even if it is the best hit
            if f.id in exclude or not f.text.strip():
                continue
            common = sorted(set(topics) & set(f.topics))
            topic = len(common) / max(1, len(set(topics))) if topics else 0.0
            fit = max((exposure.get(t, 0.0) for t in f.topics), default=0.0)
            parts = {'lexical': round(float(s / top), 3), 'topics': round(topic, 3), 'profile': round(fit, 3), 'own_view': float(f.own_view)}
            score = W_LEX * parts['lexical'] + W_TOPIC * topic + W_PROFILE * fit + W_OWN * parts['own_view']
            if not common and s < 0.15:
                continue  # no shared topic and weak text overlap: profile fit alone is not a match
            if f.freshness == 'aging':
                score -= 0.05
            if f.stale:
                score *= 0.3
            if f.kind == 'chart' and f.unverified:
                score *= SHAKY_CHART
            score = round(float(score), 3)
            if score < self.min_score:
                continue
            reason = (f'темы {", ".join(common) or "—"}; близость текста {parts["lexical"]:.2f}; '
                      f'чувствительность компании {fit:.1f}{"; позиция Банка" if f.own_view else ""}; {f.freshness}'
                      f'{"; график с ненадёжными значениями ×0.85" if f.kind == "chart" and f.unverified else ""}')
            out.append(BankMatch(f, score, parts, reason))
        out.sort(key=lambda m: (not m.fragment.stale, m.score), reverse=True)
        return out[:k]


def abroad(text: str) -> bool:
    return bool(FOREIGN.search(text)) and not DOMESTIC.search(text)


def foreign_ids(library: BankLibrary) -> set[str]:
    """Fragments about another country's economy: named so themselves, or in a report mostly about it (and not naming
    Russia themselves)."""
    flagged = {f.id for f in library.fragments if abroad(f.text)}
    reports: dict[str, list] = {}
    for f in library.fragments:
        if f.topics:
            reports.setdefault(f.file, []).append(f.id in flagged)
    abroad_reports = {r for r, xs in reports.items() if sum(xs) >= 0.5 * len(xs)}
    return flagged | {f.id for f in library.fragments if f.file in abroad_reports and not DOMESTIC.search(f.text)}


def company_terms(profile) -> tuple[set[str], list[str]]:
    """What the company's business is called in text: stems of its OKVED name (generic OKVED words left out) and the word
    beginnings of the topics its sector reaches hardest (exposure >= 0.8 from the OKVED rules, not the base weights)."""
    from stage5_insight_detection.analyst_loop.profile import CLASS, SECTOR
    from stage5_insight_detection.bank_analytics.library import topics_of
    name = {w for w in stems(profile.okved_name or '') if w not in GENERIC and len(w) >= 5}
    sector = {**SECTOR.get(profile.okdiv, {}), **CLASS.get(profile.okved[:5], {}), **{t: 1.0 for t in topics_of(profile.okved_name or '')},
              **profile.exposure}  # exposure adds the channels of its foreign trade (курс) and counterparties
    keys = [k for t, w in sector.items() if w >= 0.8 and t not in MACRO for k in TOPICS.get(t, [])]
    return name, keys


def company_fit(f: Fragment, profile, terms) -> float:
    """0..1: the fragment speaks of the company's own business (its terms) and of channels the company is exposed to
    beyond the event's topic (a rate fragment about corporate loans for a borrower)."""
    name, keys = terms
    low = f.text.lower().replace('ё', 'е')
    hits = len(name & set(stems(low))) + sum(k in low for k in keys)
    return 0.5 * min(1.0, hits / 2) + 0.5 * max((profile.exposure.get(t, 0.0) for t in f.topics), default=0.0) * \
        (1.0 if hits else 0.6)


def _select(self, event, profile, k: int = 5, exclude: set[str] = frozenset(), query: str | None = None) -> list[BankMatch]:
    """Task 2 in two stages. 1) Candidates by the event alone: absolute text similarity and shared topics must reach
    MIN_EVENT (the text alone when the event has no topic); a review of one issuer needs a close text, a foreign
    fragment a foreign event or a shared sector. 2) Rerank for the company: its business terms and exposures, the
    bank's own view, freshness; stale fragments go last. Nothing left = «аналитики Банка нет»."""
    if not self.library.fragments:
        return []
    lex = list(self.library.scores(event.query()))
    extra = list(self.library.scores(query)) if query else [0.0] * len(lex)
    terms = company_terms(profile)
    ev_topics = set(event.topics)
    if getattr(self, '_foreign', None) is None:
        self._foreign = foreign_ids(self.library)
    event_abroad = abroad(f'{event.title} {event.text[:600]}')
    from stage5_insight_detection.bank_analytics.library import topics_of
    headline = set(topics_of(event.title)) & ev_topics
    out = []
    for f, s, query_score in zip(self.library.fragments, lex, extra):
        s = float(s)
        if f.id in exclude or not f.text.strip():
            continue
        common = sorted(ev_topics & set(f.topics))
        # the topic part is full for the headline's topic (the lead adds side topics: a mortgage news mentions the
        # budget); an event with no topic found is judged by its text alone, not marked down for topics it has not
        share = 1.0 if set(common) & headline else len(common) / len(ev_topics) if ev_topics else 0.0
        ev = 0.6 * min(1.0, s / LEX_FULL) + 0.4 * share if ev_topics else min(1.0, s / LEX_FULL)
        if ev < MIN_EVENT:
            continue
        if s < ISSUER_LEX and ISSUER.search(f.text):
            continue
        if f.id in self._foreign and not event_abroad and not set(common) & SECTORAL:
            continue  # a foreign fragment for a domestic event, unless they share a sector (gold for a gold miner)
        fit = company_fit(f, profile, terms)
        # the event's move against the bank's view on the same topic: a disagreement needs a warning in the insight
        moves = {t: 'совпадает' if f.stance[t] == event.moves[t] else 'расходится'
                 for t in common if t in getattr(event, 'moves', {}) and t in f.stance}
        parts = {'event': round(ev, 3), 'lexical': round(s, 3), 'topics': common, 'company': round(fit, 3),
                 'own_view': float(f.own_view), 'fresh': FRESH.get(f.freshness, 0.0), 'direction': moves}
        score = W_EVENT * ev + W_COMPANY * fit + W_OWN2 * parts['own_view'] + W_FRESH * parts['fresh']
        score += 0.05 * min(1.0, float(query_score) / LEX_FULL)  # rerank only after the event eligibility gates
        if f.kind == 'chart' and f.unverified:
            score *= SHAKY_CHART
        if f.stale:
            score *= 0.3  # kept as context after the fresh ones, never as the current view
        reason = (f'событие {ev:.2f} (текст {s:.2f}; темы {", ".join(common) or "—"}); компания {fit:.2f}'
                  f'{"; позиция Банка" if f.own_view else ""}; {f.freshness}' +
                  ''.join(f'; направление по теме «{t}»: {v}' + (' — Банк ждёт иного, нужна оговорка' if v == 'расходится' else '')
                          for t, v in moves.items()))
        out.append(BankMatch(f, round(score, 3), parts, reason))
    out.sort(key=lambda m: (not m.fragment.stale, m.score), reverse=True)
    return out[:k]


BankMatcher.select = _select


def same_fact(a: Fragment, b: Fragment) -> bool:
    """Two fragments of different reports that say the same: most of their stems or of their figures are shared."""
    if a.file == b.file:
        return False
    sa, sb = set(stems(a.text)), set(stems(b.text))
    if sa and sb and len(sa & sb) / min(len(sa), len(sb)) >= 0.7:
        return True
    fig = lambda f: {(x['indicator'], x['value'], x['unit']) for x in f.figures if x['indicator']}
    fa, fb = fig(a), fig(b)
    return len(fa & fb) >= 2 and len(fa & fb) >= 0.6 * min(len(fa), len(fb))


def brief_bank(matches: list[BankMatch], cap: int = BRIEF_CAP) -> list[BankMatch]:
    """The bank fragments of one brief: in order, without repeats of one fact from another report, at most `cap`."""
    out: list[BankMatch] = []
    for m in matches:
        if any(m.fragment.id == o.fragment.id or same_fact(m.fragment, o.fragment) for o in out):
            continue
        out.append(m)
        if len(out) == cap:
            break
    return out
