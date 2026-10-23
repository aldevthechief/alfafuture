"""Everything the loop has gathered about one (event, company, role): the single object every step reads and extends."""
from dataclasses import dataclass, field

from stage5_insight_detection.analyst_loop.enrich import Enrichment
from stage5_insight_detection.analyst_loop.event import EventCard
from stage5_insight_detection.analyst_loop.facts import Fact
from stage5_insight_detection.analyst_loop.impact import Consequence
from stage5_insight_detection.analyst_loop.profile import CompanyProfile
from stage5_insight_detection.analyst_loop.risks import RiskItem
from stage5_insight_detection.bank_analytics.match import BankMatch, brief_bank

MACRO = {'ставка', 'инфляция', 'курс', 'потребление', 'кредит', 'бюджет_и_налоги', 'облигации', 'рынок_труда', 'инвестиции',
         'промышленность', 'строительство', 'золото', 'металлургия', 'энергия_и_сырьё'}


@dataclass
class Dossier:
    event: EventCard
    profile: CompanyProfile
    role: str
    matches: list[BankMatch] = field(default_factory=list)
    enrichment: Enrichment = field(default_factory=Enrichment)
    consequences: list[Consequence] = field(default_factory=list)
    computed: list[Fact] = field(default_factory=list)
    risks: list[RiskItem] = field(default_factory=list)
    extra_facts: list[str] = field(default_factory=list)  # answers of company_facts requests (stage 2 texts)
    explanations: dict[str, dict] = field(default_factory=dict)  # consequence id -> {'text', 'refs', 'kind'} from the model
    profile_context: str = ''  # advisory only; excluded from facts(), refs() and computed verdicts
    requested: list[Fact] = field(default_factory=list)  # results survive consequence recomputation
    deltas: dict = field(default_factory=dict)  # checked bank-thesis changes

    @property
    def bank(self) -> list[BankMatch]:
        """Task 2 matches first, then the enrichment's: one fact once, at most BRIEF_CAP fragments."""
        return brief_bank(self.matches + self.enrichment.extra_bank)

    def facts(self) -> dict[str, Fact]:
        return {**self.profile.facts, **{f.id: f for f in self.computed + self.requested}}

    def news(self) -> dict[str, dict]:
        rows = {self.event.id: {'id': self.event.news_id, 'title': self.event.title, 'text': self.event.text,
                                'date': self.event.date, 'source': self.event.source, 'url': self.event.url,
                                'matched_fragments': self.event.matched_fragments}} if getattr(self.event, 'origin', 'news') == 'news' else {}
        for r in self.enrichment.confirmations + [u['news'] for u in self.enrichment.updates]:
            rows.setdefault(f'news:{r["id"]}', r)
        return rows

    def refs(self) -> set[str]:
        """Every id a claim may cite."""
        return set(self.facts()) | set(self.news()) | {m.fragment.id for m in self.bank} | \
            {f'fact:stage2_{i}' for i in range(1, len(self.extra_facts) + 1)}
