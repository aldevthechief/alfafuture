"""Task 6: risks and opportunities, with the bank's position kept apart from the AI's.

Each consequence becomes a risk (sign -1), an opportunity (+1) or a point to watch (0). The bank's stance on the topic
(own-view sentences of fresh fragments first) is compared with the direction of the event:
- the same direction -> `bank_supported`, the bank refs back the item and it is shown in the bank block;
- the bank expects the opposite or a pause -> `bank_disagrees`, the item stays an AI conclusion and the card says so;
- no fresh bank fragment on the topic -> basis 'ai' and the "не официальная позиция Банка" mark.
Roles are attached from a fixed topic -> role map (jobs of each reader). Pure code, no LLM.
"""
from dataclasses import asdict, dataclass

from stage5_insight_detection.analyst_loop.impact import Consequence
from stage5_insight_detection.bank_analytics.match import BankMatch

ROLES = {'ставка': ['cfo', 'treasury', 'owner'], 'облигации': ['cfo', 'treasury'], 'кредит': ['cfo', 'treasury'],
         'курс': ['cfo', 'treasury', 'ceo'], 'инфляция': ['ceo', 'cfo'], 'потребление': ['ceo', 'owner'],
         'бюджет_и_налоги': ['accountant', 'cfo'], 'рынок_труда': ['ceo'], 'строительство': ['ceo', 'owner'],
         'металлургия': ['ceo', 'cfo'], 'энергия_и_сырьё': ['ceo', 'cfo'], 'золото': ['treasury', 'cfo'],
         'внешняя_торговля': ['ceo', 'cfo', 'accountant'], 'инвестиции': ['owner', 'ceo'], 'промышленность': ['ceo', 'owner']}


@dataclass
class RiskItem:
    id: str
    type: str  # risk | opportunity | watch
    consequence: str
    topic: str
    text: str  # code-made line; the model rewrites it in the assessment, refs stay
    level: str
    basis: str  # bank | ai
    bank_supported: bool
    bank_disagrees: bool
    bank_refs: list[str]
    refs: list[str]
    roles: list[str]

    def dict(self) -> dict:
        return asdict(self)


def bank_stance(topic: str, matches: list[BankMatch]) -> tuple[str | None, list[str]]:
    fresh = [m.fragment for m in matches if not m.fragment.stale and m.fragment.own_view and topic in m.fragment.stance]
    fresh.sort(key=lambda f: not f.own_view)  # the bank's own view first, then match order
    return (fresh[0].stance[topic], [f.id for f in fresh[:2]]) if fresh else (None, [])


def split(consequences: list[Consequence], matches: list[BankMatch]) -> list[RiskItem]:
    out = []
    for i, c in enumerate(consequences, 1):
        stance, bank_refs = bank_stance(c.topic, matches)
        supported = stance is not None and stance == c.move
        disagrees = c.sign != 0 and stance is not None and c.move in ('up', 'down') and stance != c.move
        kind = 'risk' if c.sign < 0 else 'opportunity' if c.sign > 0 else 'watch'
        est = f'; оценка {c.estimate.text()}' if c.estimate else ''
        text = f'{c.topic}: {c.mechanism} ({"−" if c.sign < 0 else "+" if c.sign > 0 else "±"}){est}'
        out.append(RiskItem(f'R{i}', kind, c.id, c.topic, text, c.level, 'bank' if supported else 'ai', supported, disagrees,
                            bank_refs, list(dict.fromkeys(c.facts + c.news_refs + (bank_refs if supported else []))),
                            ROLES.get(c.topic, ['ceo'])))
    return out
