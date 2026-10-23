"""Task 7: are the grounds for the insight sufficient and consistent? Deterministic checks over the dossier.

Fatal (no insight): no source or no date of the event; the event is implausible (< 0.3); no consequence links the event
to a concrete fact about the company; a blocking data request stayed open.
Warnings (the insight can go, the card shows them, confidence drops):
- no_bank_view: a macro channel without a fresh bank fragment -> "вывод AI, не официальная позиция Банка";
- stale_bank_only: only stale or undated analytics on the topic;
- bank_disagrees: the bank expects the opposite direction or a pause;
- newer_data: a news figure newer than the bank's figure for the same indicator differs from it (the bank view may be dated);
- numbers_conflict: confirmations of the same event give different actual values for one indicator;
- single_source_high: a high impact resting on one outlet and no bank support -> the level is capped at mid.
Gaps that a tool can close are returned as DataRequests, so the loop asks before it judges.
"""
from dataclasses import asdict, dataclass, field

from stage5_insight_detection.analyst_loop.dossier import MACRO, Dossier
from stage5_insight_detection.analyst_loop.data_requests import DataRequest
from stage5_insight_detection.bank_analytics.library import TOPICS, figures_of

MIN_PLAUSIBILITY = 0.3


@dataclass
class Issue:
    code: str
    severity: str  # fatal | warn
    detail: str
    refs: list[str] = field(default_factory=list)


@dataclass
class GroundsReport:
    issues: list[Issue]
    requests: list[DataRequest]

    @property
    def sufficient(self) -> bool:
        return not any(i.severity == 'fatal' for i in self.issues)

    @property
    def consistent(self) -> bool:
        return not any(i.code in ('bank_disagrees', 'numbers_conflict') for i in self.issues)

    def codes(self) -> set[str]:
        return {i.code for i in self.issues}

    def dict(self) -> dict:
        return {'sufficient': self.sufficient, 'consistent': self.consistent, 'issues': [asdict(i) for i in self.issues],
                'requests': [r.model_dump() for r in self.requests]}


def check(d: Dossier) -> GroundsReport:
    issues, requests = [], []
    e = d.event
    if not (e.url or e.source):
        issues.append(Issue('no_source', 'fatal', 'у события нет ни ссылки, ни источника', [e.id]))
    if e.date is None:
        issues.append(Issue('no_date', 'fatal', 'дата публикации не распознана: свежесть не проверить', [e.id]))
    if e.plausibility is not None and e.plausibility < MIN_PLAUSIBILITY:
        issues.append(Issue('implausible', 'fatal', f'правдоподобие события {e.plausibility:.2f} < {MIN_PLAUSIBILITY}', [e.id]))
    linked = [c for c in d.consequences if c.exposure >= 0.5 and c.facts]
    if not linked:
        issues.append(Issue('no_company_link', 'fatal', 'ни одно последствие не опирается на факт о компании с чувствительностью ≥ 0.5'))
        if e.topics:
            requests.append(DataRequest(kind='company_facts', query=' '.join(e.topics[:2]), blocking=True,
                                        reason='найти факт о компании, через который событие её касается'))
    bank = d.bank
    for topic in sorted({c.topic for c in d.consequences} & MACRO):
        on_topic = [m.fragment for m in bank if topic in m.fragment.topics and m.fragment.own_view]
        if not on_topic:
            issues.append(Issue('no_bank_view', 'warn', f'нет аналитики Банка по теме «{topic}»: вывод AI, не официальная позиция Банка'))
            requests.append(DataRequest(kind='bank_analytics', query=' '.join(TOPICS[topic][:4]),
                                        reason=f'позиция Банка по теме «{topic}»'))
        elif all(f.stale for f in on_topic):
            issues.append(Issue('stale_bank_only', 'warn', f'по теме «{topic}» есть только устаревшая аналитика',
                                [f.id for f in on_topic]))
    for r in d.risks:
        if r.bank_disagrees:
            issues.append(Issue('bank_disagrees', 'warn', f'Банк ожидает другого направления по теме «{r.topic}»', r.bank_refs))
    for u in d.enrichment.updates:
        issues.append(Issue('newer_data', 'warn', f'«{u["indicator"]}»: в новости от {u["news_date"]} {u["news_value"]}{u["unit"]}, '
                                                  f'в аналитике Банка от {u["bank_date"]} {u["bank_value"]}{u["unit"]} (news:{u["news"]["id"]})',
                            [f'news:{u["news"]["id"]}', u['fragment']]))
    values: dict[str, dict[float, str]] = {}
    for ref, row in d.news().items():
        if ref != e.id and str(row['id']) not in d.enrichment.same_event_ids:
            continue  # later updates and follow-ups need not have the same value
        first = {}
        for f in figures_of(f'{row["title"]}. {str(row.get("text", ""))[:600]}'):
            if f.kind == 'actual' and f.indicator and f.unit == '%':
                first.setdefault((f.indicator, 'change' if f.measure == 'change' else 'level', f.period), f.value)
        for key, v in first.items():
            values.setdefault(key, {}).setdefault(v, ref)
    for (indicator, measure, period), vals in values.items():
        if len(vals) > 1 and max(vals) - min(vals) > 0.05 * max(abs(v) for v in vals):
            issues.append(Issue('numbers_conflict', 'warn', f'источники расходятся по «{indicator}»: '
                                + ', '.join(f'{v}% ({r})' for v, r in vals.items()), list(vals.values())))
            requests.append(DataRequest(kind='news', query=f'{" ".join(TOPICS[indicator][:2])} официально',
                                        reason=f'официальное значение «{indicator}»'))
    for c in d.consequences:
        if getattr(e, 'origin', 'news') == 'news' and c.level == 'high' and len(set(e.sources)) < 2 and not d.enrichment.independent_confirmations(e) and not any(r.bank_supported for r in d.risks
                                                                                                    if r.consequence == c.id):
            c.level = 'mid'
            c.level_why += '; один новостной источник без поддержки Банка: уровень ограничен mid'
            issues.append(Issue('single_source_high', 'warn', f'{c.id}: высокое влияние на одном источнике, уровень снижен до mid', [e.id]))
    return GroundsReport(issues, requests)
