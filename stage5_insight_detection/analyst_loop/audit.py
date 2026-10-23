"""Post-answer checks of the written insight. The prompt asks, the code checks, because models sometimes ignore asks.
Checks do not rewrite the text; they set flags the UI shows and move claims between layers:
- every claim needs refs, and every ref must exist among the material that was really given (brief.refs);
  otherwise the claim is `unconfirmed` ("не подтверждено" in the UI);
- numbers are not matched against the sources: that check flagged page numbers («стр. 12»), signs («−10,3%» vs «10,3%»)
  and correct figures cited from a neighbouring fact, and as a blocking code it rejected whole insights the checker had
  passed (U_68622, 22.10: 2 of 5) — removed;
- why_it_matters / risks / opportunities without any number get a warning (a conclusion without a figure);
- bank_view accepts only bank:* refs to fresh fragments; anything else moves to why_it_matters as an AI conclusion,
  so the separation of the bank's position is guaranteed by code, not by the prompt;
- layer kinds are set by section; hypotheses, impact and the "not the bank's position" flag come from the brief;
- a bank product stays only if it answers a risk (or, for deposits, an opportunity) of the insight: no mis-selling;
- confidence is capped at medium without bank support or when the bank expects another direction.
Idea after FinRobot (engine/compute/operators/audit/); our own code.
"""
import re

from common.quantities import SCALE, UNIT, quantities
from stage5_insight_detection.analyst_loop.verdict import InsightBrief
from stage6_insight_generation.insight_writer.insight import Claim, Impact, Insight

MONTHS = r'(?:январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр)[а-я]*'
DATES = re.compile(r'\b(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b'                # 2026-09-02
                   r'|(?<![\d.,])(?:0?[1-9]|[12]\d|3[01])\.(?:0[1-9]|1[0-2])(?:\.(?:\d{4}|\d{2}))?(?!\d|[.,]\d)'  # 02.09.2026, 02.09.26, 02.09
                   r'|\b(?:19|20)\d{2}\s*[–—-]\s*(?:19|20)\d{2}\b'                                 # 2027–2028
                   r'|(?<![\w.,])\d{1,2}[КкQqПпМмHН]\d{2}\b'                                       # 4К26, 3Q26, 1П26, 7М26
                   r'|(?<![\d.,])(?:[12]?\d|3[01])(?:\s*[–—-]\s*(?:[12]?\d|3[01]))?\s+' + MONTHS)    # 25–31 августа
RISK, OPP = {'risk', 'watch'}, {'opportunity'}
PRODUCT_TOPICS = {'кредит': ({'ставка', 'кредит', 'облигации', 'инвестиции'}, RISK), 'лизинг': ({'ставка', 'инвестиции'}, RISK),
                  'депозит': ({'ставка', 'облигации'}, OPP), 'вэд': ({'курс', 'внешняя_торговля'}, RISK),
                  'валют': ({'курс', 'внешняя_торговля'}, RISK), 'хедж': ({'курс', 'энергия_и_сырьё', 'металлургия', 'золото'}, RISK),
                  'факторинг': ({'потребление', 'строительство'}, RISK), 'гарант': ({'строительство', 'бюджет_и_налоги'}, RISK)}


def without_dates(text: str) -> str:
    # «1.05%» and «1.05 млн ₽» are decimal quantities, although their digits look like a day/month.
    return DATES.sub(lambda m: m.group() if UNIT.match(text, m.end()) or SCALE.match(text, m.end()) else ' ', text)


def number_groups(text: str) -> list[set[float]]:
    """Compatibility helper returning only canonical values; validation also compares their units."""
    return [{q.value} for q in quantities(without_dates(text))]


def numbers(text: str) -> set[float]:
    return set().union(*number_groups(text))


def known(ref: str, refs: set[str]) -> bool:
    return ref in refs or (ref.startswith('bank:') and any(
        r.startswith(ref + '.') or (ref.endswith(':') and r.startswith(ref + 'p')) for r in refs if r.startswith('bank:')))


def audit(insight: Insight, brief: InsightBrief, sources: dict[str, str]) -> tuple[Insight, list[dict]]:
    """sources: ref -> full text of news and bank fragments that were given to the writer (kept for callers)."""
    refs, warnings = set(brief.refs), []
    insight.origin = brief.origin
    fresh_bank = {b['id'] for b in brief.bank if b['freshness'] in ('fresh', 'aging') and b.get('own_view', False)}

    def check(section: str, claim: Claim, kind: str, need_number: bool) -> Claim:
        claim.kind = kind
        bad = [r for r in claim.sources if not known(r, refs)]
        if not claim.sources:
            claim.unconfirmed = True
            warnings.append({'section': section, 'code': 'no_source', 'text': claim.text})
        if bad:
            claim.unconfirmed = True
            warnings.append({'section': section, 'code': 'unknown_ref', 'refs': bad, 'text': claim.text})
        if need_number and not numbers(claim.text):
            warnings.append({'section': section, 'code': 'no_number', 'text': claim.text})
        return claim

    bank_ok, moved = [], []
    for c in insight.bank_view:
        if c.sources and all(r.startswith('bank:') and any(known(r, {b}) for b in fresh_bank) for r in c.sources):
            bank_ok.append(c)
        else:
            moved.append(c)
            warnings.append({'section': 'bank_view', 'code': 'moved_to_ai', 'text': c.text, 'refs': c.sources})
    insight.bank_view = [check('bank_view', c, 'bank', False) for c in bank_ok]
    insight.what_happened = [check('what_happened', c, 'bank' if brief.origin == 'bank' else 'fact', False)
                            for c in insight.what_happened]
    insight.why_it_matters = [check('why_it_matters', c, 'ai', True) for c in insight.why_it_matters + moved]
    insight.risks = [check('risks', c, 'ai', True) for c in insight.risks]
    insight.opportunities = [check('opportunities', c, 'ai', True) for c in insight.opportunities]
    insight.hypotheses = [Claim(text=f'{h["label"]}: {h["value"]} {h["unit"]} ({h["formula"]})', sources=[h['id']], kind='hypothesis')
                          for h in brief.hypotheses] + [check('hypotheses', c, 'hypothesis', False) for c in insight.hypotheses
                                                        if c.sources and c.sources[0] not in {h['id'] for h in brief.hypotheses}]
    insight.impact = Impact(**brief.impact)
    insight.not_bank_position = brief.not_bank_position or not insight.bank_view
    if (insight.not_bank_position or not brief.consistent) and insight.confidence == 'high':
        insight.confidence = 'medium'  # no bank support or the bank expects otherwise: never "high"
    if insight.bank_product:
        fits = [k for k, (topics, types) in PRODUCT_TOPICS.items() if k in insight.bank_product.lower()
                and any(r['topic'] in topics and r['type'] in types for r in brief.risks)]
        if not fits:
            warnings.append({'section': 'bank_product', 'code': 'mis_selling_guard', 'text': insight.bank_product})
            insight.bank_product = ''
    return insight, warnings
