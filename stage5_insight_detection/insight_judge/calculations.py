"""Per-event adapter to the loop's Python calculator. No agent, search or model is run here."""
from dataclasses import asdict
from datetime import date
import json
from math import isfinite
from types import SimpleNamespace

from stage2_company_facts.profile.integration import load_company
from stage5_insight_detection.analyst_loop.compute import compute
from stage5_insight_detection.analyst_loop.event import EventCard
from stage5_insight_detection.analyst_loop.facts import Fact
from stage5_insight_detection.bank_analytics.library import figures_of, freshness, stance_of


class JudgeCalculations:
    def __init__(self, news: dict, company_id: str | None, bank):
        self.news, self.company_id, self.bank = news, company_id, bank
        self.facts, self.bank_sources, self.selected = {}, {}, {}
        self.replies, self.round = {}, 0
        self.company = None

    def observe_bank(self, passages: list[dict]) -> None:
        """Reuse the core search hits the judge has read. Never start an extra retrieval for the calculator."""
        chunks = {c['id']: c for c in self.bank.chunks}
        rows = {f['id']: f for f in getattr(self.bank, 'rows', self.bank.chunks)}
        for p in passages:
            for cid in p['ids']:
                for fid in chunks.get(cid, {}).get('frag_ids', [cid]):
                    if fid in rows:
                        self.selected[fid] = rows[fid]

    def _profile(self) -> dict[str, Fact]:
        if self.company is None:
            if not self.company_id:
                return {}
            card = load_company(self.company_id).card
            self.company = {}
            for name, label in [('revenue', 'Выручка компании'), ('group_revenue', 'Выручка группы')]:
                value = card.get(name)
                if isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value) and value > 0:
                    if name == 'revenue' and value <= 1000:
                        continue
                    # Keep exact rubles for arithmetic; do not round to billions before calculating a share.
                    self.company[f'fact:{name}'] = Fact(f'fact:{name}', label, value, '₽',
                                                        source=f'stage2_company_facts.profile.card.{name}')
        return self.company

    def _bank(self) -> list:
        out = []
        for f in self.selected.values():
            try:
                when = date.fromisoformat(f.get('date') or '')
            except ValueError:
                when = None
            fresh, _ = freshness(when, date.today())
            figures = f.get('figures')
            if figures is None:
                figures = [asdict(x) for x in figures_of(f['text'])] if f['kind'] == 'text' else []
            out.append(SimpleNamespace(fragment=SimpleNamespace(
                id=f['id'], date=f.get('date'), stale=fresh in ('stale', 'unknown') or f.get('unverified', False),
                figures=figures)))
        return out

    def run(self, query: str) -> str:
        key = query.strip().lower()
        if key in self.replies:
            return self.replies[key]
        try:
            needs_company = any(word in key for word in ('доля', 'долю', 'выручк'))
            company = self._profile() if needs_company else {}
            context = SimpleNamespace(event=EventCard.from_news(self.news),
                                      profile=SimpleNamespace(facts=company), bank=self._bank())
            results, why = compute(query, context, self.round + 1)
            if any(isinstance(f.value, (int, float)) and not isfinite(f.value) for f in results):
                results, why = [], 'результат не является конечным числом'
            if not results:
                return json.dumps({'ok': False, 'reason': why}, ensure_ascii=False)

            refs = {part.strip() for f in results for part in f.source.split(';')}
            needed = [f for f in company.values() if any(f.id in r.formula or f.id in r.source for r in results)]
            for f in [*needed, *results]:
                self.facts[f.id] = f.dict()
            for ref in refs & self.selected.keys():
                f = self.selected[ref]
                when = date.fromisoformat(f['date']) if f.get('date') else None
                fresh, _ = freshness(when, date.today())
                self.bank_sources[ref] = dict(f, stale=fresh in ('stale', 'unknown'), freshness=fresh,
                                              own_view=f.get('own_view', stance_of(f['text'])[0]))
            self.round += 1
            reply = json.dumps({'ok': True, 'facts': [f.dict() for f in [*needed, *results]],
                                'note': 'kind=hypothesis — сценарий или непроверенное число, не установленный эффект компании'},
                               ensure_ascii=False, allow_nan=False)
            self.replies[key] = reply
            return reply
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            return json.dumps({'ok': False, 'reason': f'недостаточно корректных данных для расчёта: {error}'}, ensure_ascii=False)
