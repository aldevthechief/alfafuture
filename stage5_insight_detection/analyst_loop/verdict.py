"""Task 8: hand a grounded conclusion to the insight writer, or refuse the event with a reason.

Refuse when the grounds are insufficient (fatal issues of grounds.py, a blocking request still open) or when the event
does not matter enough: every consequence is low and the event importance is <= 2. Otherwise the verdict carries an
`InsightBrief`: the only material the writer may use. It lists the allowed refs, computed facts with formulas,
consequences with levels, risks and opportunities split by basis (bank / AI), hypotheses, the impact of the strongest
consequence and the flags the card needs (not the bank's position, warnings). Pure code, no LLM.
"""
from dataclasses import dataclass, field

from stage5_insight_detection.analyst_loop.dossier import Dossier
from stage5_insight_detection.analyst_loop.grounds import GroundsReport
from stage5_insight_detection.analyst_loop.impact import IMPACT_CRITERIA
from stage5_insight_detection.analyst_loop.data_requests import DataRequest


@dataclass
class InsightBrief:
    refs: list[str]
    facts: list[dict]
    consequences: list[dict]
    risks: list[dict]
    bank: list[dict]
    hypotheses: list[dict]
    impact: dict
    not_bank_position: bool
    warnings: list[str]
    consistent: bool = True
    origin: str = 'news'
    event_ref: str = ''

    def render(self) -> str:
        """Text for the writer prompt (`decision`)."""
        lines = ['Последствия (посчитаны кодом, уровни по правилу: ' + IMPACT_CRITERIA + '):']
        if self.origin == 'bank':
            lines.insert(0, f'Событие — тезис Банка из {self.event_ref}, не новость. what_happened: только слова '
                            'этого фрагмента bank:…; последствия для компании — вывод AI, допущения — гипотезы.')
        for c in self.consequences:
            est = f'; оценка {c["estimate"]["value"]} {c["estimate"]["unit"]} ({c["estimate"]["id"]})' if c['estimate'] else ''
            expl = f'\n  объяснение: {c["explanation"]}' if c.get('explanation') else ''
            lines.append(f'- {c["id"]} [{c["level"]}] {c["topic"]}: {c["mechanism"]} '
                         f'({"риск" if c["sign"] < 0 else "возможность" if c["sign"] > 0 else "зависит от данных"}){est}; '
                         f'ссылки: {", ".join(c["facts"] + c["bank_refs"] + c["news_refs"])}{expl}')
            if c.get('level_why'):
                lines.append(f'  основание уровня: {c["level_why"]}')
        lines.append('Риски и возможности (basis=bank — подтверждено аналитикой Банка, basis=ai — вывод AI):')
        lines += [f'- {r["id"]} {r["type"]} basis={r["basis"]}{" БАНК ОЖИДАЕТ ИНОГО" if r["bank_disagrees"] else ""}: {r["text"]}; '
                  f'ссылки: {", ".join(r["refs"])}' for r in self.risks]
        if self.hypotheses:
            lines.append('Гипотезы (допущения, не подтверждены данными; только в hypotheses с kind=hypothesis):')
            lines += [f'- {h["id"]} {h["label"]}: {h["value"]} {h["unit"]} ({h["formula"]})' for h in self.hypotheses]
        if self.not_bank_position:
            lines.append('Свежей аналитики Банка по теме нет: bank_view оставь пустым, вывод — AI, не официальная позиция Банка.')
        if self.warnings:
            lines.append('Предупреждения проверки оснований: ' + '; '.join(self.warnings))
        lines.append('Разрешённые ссылки: ' + ', '.join(self.refs))
        return '\n'.join(lines)

    def dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Verdict:
    status: str  # insight | reject
    reasons: list[str]
    brief: InsightBrief | None = None
    open_requests: list[DataRequest] = field(default_factory=list)

    def dict(self) -> dict:
        return {'status': self.status, 'reasons': self.reasons, 'brief': self.brief.dict() if self.brief else None,
                'open_requests': [r.model_dump() for r in self.open_requests]}


def decide(d: Dossier, grounds: GroundsReport, open_requests: list[DataRequest]) -> Verdict:
    reasons = [i.detail for i in grounds.issues if i.severity == 'fatal']
    blocking = [r for r in open_requests if r.blocking]
    reasons += [f'не хватает данных: {r.query} ({r.reason})' for r in blocking]
    if not d.consequences:
        reasons.append('событие не касается каналов, к которым чувствительна компания')
    elif all(c.level == 'low' for c in d.consequences) and d.event.importance <= 2:
        reasons.append('влияние низкое по всем каналам, важность события ≤ 2')
    if reasons:
        return Verdict('reject', reasons, None, open_requests)
    facts = d.facts()
    used = {f for c in d.consequences for f in c.facts} | {f for r in d.risks for f in r.refs if f.startswith('fact:')}
    used |= {f.id for f in d.requested}
    top = d.consequences[0]
    impact = {'level': top.level, 'criteria': IMPACT_CRITERIA + ('; ' + top.level_why if top.level_why else ''),
              'estimate': (f'{top.estimate.text()} = {top.estimate.formula}' if top.estimate else ''),
              'sources': top.facts + top.bank_refs + top.news_refs}
    consequences = []
    for c in d.consequences:
        row = c.dict()
        row['explanation'] = d.explanations.get(c.id, {}).get('text', '')
        consequences.append(row)
    fresh_bank = [m for m in d.bank if not m.fragment.stale and m.fragment.own_view]
    brief = InsightBrief(
        refs=sorted(d.refs()), facts=[facts[f].dict() for f in sorted(used) if f in facts and facts[f].kind == 'fact'] +
        [{'id': f'fact:stage2_{i}', 'label': t, 'value': '', 'unit': '', 'formula': '', 'source': 'stage2_company_facts',
          'kind': 'fact'} for i, t in enumerate(d.extra_facts, 1)],
        consequences=consequences, risks=[r.dict() for r in d.risks], bank=[m.dict() for m in d.bank],
        hypotheses=[facts[f].dict() for f in sorted(used) if f in facts and facts[f].kind == 'hypothesis'], impact=impact,
        not_bank_position=not any(r.bank_supported for r in d.risks) and not fresh_bank,
        warnings=[i.detail for i in grounds.issues if i.severity == 'warn'], consistent=grounds.consistent,
        origin=getattr(d.event, 'origin', 'news'), event_ref=d.event.id)
    return Verdict('insight', [f'основания достаточны; сильнейшее последствие {top.id} [{top.level}]'], brief, open_requests)
