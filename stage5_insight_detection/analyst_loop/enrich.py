"""Task 3: go back to the news pool and the bank's analytics to complete what the first match found.

Three moves, all code:
1. confirmations: other reports of the same event and later follow-ups (independent outlets, official sources);
2. updates: for every matched bank fragment, news published after the report on the same indicator. If a news figure
   differs from the bank's figure for the same indicator, it is recorded as `newer_data` (the bank view may be dated)
   for the consistency check (grounds.py);
3. more analytics: the bank library is searched again with the terms of the confirmations and with the company's
   strongest channels that the event touches, which adds fragments the event text alone did not reach.
"""
from dataclasses import dataclass, field

from stage5_insight_detection.analyst_loop.event import EventCard
from stage5_insight_detection.analyst_loop.pool import NewsPool
from stage5_insight_detection.analyst_loop.profile import CompanyProfile
from stage5_insight_detection.bank_analytics.library import TOPICS, figures_of, topics_of
from stage5_insight_detection.bank_analytics.match import BankMatch, BankMatcher


@dataclass
class Enrichment:
    confirmations: list[dict] = field(default_factory=list)  # news rows of the same event / follow-ups
    updates: list[dict] = field(default_factory=list)  # {'news', 'fragment', 'indicator', 'news_value', 'bank_value'}
    extra_bank: list[BankMatch] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    same_event_ids: set[str] = field(default_factory=set)

    def independent_confirmations(self, event: EventCard) -> int:
        return len({str(r.get('source') or '').strip().lower() for r in self.confirmations
                    if str(r['id']) in self.same_event_ids and str(r.get('source') or '').strip()
                    and str(r['source']).strip().lower() != str(event.source or '').strip().lower()})

    def news_ids(self) -> list[str]:
        return list(dict.fromkeys(f'news:{r["id"]}' for r in self.confirmations + [u['news'] for u in self.updates]))


class Enricher:
    def __init__(self, pool: NewsPool, matcher: BankMatcher, max_confirmations: int = 5):
        self.pool, self.matcher, self.max_confirmations = pool, matcher, max_confirmations

    def enrich(self, event: EventCard, matches: list[BankMatch], profile: CompanyProfile) -> Enrichment:
        out = Enrichment()
        seen = {event.news_id}
        is_news = getattr(event, 'origin', 'news') == 'news'
        same = self.pool.same_event(event.news_id) if is_news else []
        out.same_event_ids = {str(r['id']) for r in same}
        for row in same + (self.pool.follow_ups(event.news_id) if is_news else []):
            if str(row['id']) not in seen and len(out.confirmations) < self.max_confirmations:
                seen.add(str(row['id']))
                out.confirmations.append(row)
        for m in matches:
            f = m.fragment
            for topic in [t for t in f.topics if t in event.topics][:2]:
                q = f'{" ".join(TOPICS[topic][:3])} {f.heading}'.strip()
                out.queries.append(q)
                for row in self.pool.search(q, k=5):
                    if f.date and str(row.get('date', ''))[:10] <= f.date or topic not in topics_of(row['title']):
                        continue
                    if any(u['news']['id'] == row['id'] and u['fragment'] == f.id for u in out.updates):
                        continue
                    for nf in figures_of(f'{row["title"]}. {str(row.get("text", ""))[:600]}'):
                        bank_values = [bf for bf in f.figures if bf['indicator'] == topic == nf.indicator and bf['unit'] == nf.unit
                                       and bf['kind'] == 'actual' and bf.get('verified', True)
                                       and bf.get('measure') != 'change'
                                       and (not nf.period or not bf.get('period') or nf.period == bf['period'])]
                        if nf.kind == 'actual' and nf.measure != 'change' and bank_values and abs(bank_values[0]['value'] - nf.value) > 1e-9:
                            out.updates.append({'news': row, 'fragment': f.id, 'indicator': topic, 'news_value': nf.value,
                                                'bank_value': bank_values[0]['value'], 'unit': nf.unit,
                                                'news_date': str(row.get('date'))[:10], 'bank_date': f.date})
                            break
        known = {m.fragment.id for m in matches}
        channels = [t for t, w in sorted(profile.exposure.items(), key=lambda kv: -kv[1]) if w >= 0.6 and t in event.topics]
        extra_queries = [r['title'] for r in out.confirmations[:3]] + [' '.join(TOPICS[t][:4]) for t in channels[:2]]
        for q in extra_queries:
            out.queries.append(q)
            for m in self.matcher.select(event, profile, k=3, exclude=known, query=q):
                known.add(m.fragment.id)
                out.extra_bank.append(m)
        return out
