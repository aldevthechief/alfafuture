"""Business value of a checked card, computed by code after the writer (no LLM):
- confidence: the writer marked 32 of 35 demo cards `medium`, which ranks nothing; here it follows the evidence:
  who reported the event (common/credibility), a fresh bank view behind the card, code-computed numbers, and how much of
  the reasoning rests on assumptions («если …», hypotheses). `basis` says why, in the client's words;
- bank_product: the case allows native offers of the bank's products; the writer left 29 of 35 empty and the rest named
  products the company already has («кредитные продукты»). Here the offer follows the card's topic and the company's
  data (its Alfa products, payment concentration), with the reason, and never offers what the company already uses.
Both go to result['business_value']; insight.confidence is replaced, insight.bank_product filled only when empty or vague.
"""
import re
from functools import lru_cache
from datetime import date
from math import isfinite

from common.personalization import topics_of

PRODUCTS = {  # key -> (offer, when the company has `has`, it is not offered)
    'fixed_rate': ('Фиксация ставки по кредиту в Альфа-Банке (процентный своп или перевод на фиксированную ставку)', None),
    'credit_line': ('Кредитная линия Альфа-Банка с фиксированной ставкой под пиковую потребность в оборотных средствах', 'loan'),
    'deposit': ('Депозит или неснижаемый остаток в Альфа-Банке для временно свободных средств', 'deposit'),
    'fx': ('Валютный контроль и форвард на курс в Альфа-Банке для расчётов с иностранными контрагентами', None),
    'factoring': ('Факторинг Альфа-Банка: деньги за отгрузку сразу, без ожидания оплаты от крупных покупателей', 'factoring'),
    'guarantee': ('Банковская гарантия Альфа-Банка для участия в госзакупках и исполнения контрактов', 'guarantee'),
    'salary': ('Обсудить зарплатный проект Альфа-Банка для организации выплат сотрудникам', 'salary'),
    'leasing': ('Лизинг техники и транспорта через Альфа-Банк', 'leasing'),
}
LIQUIDITY = re.compile(r'ликвидн|свободн\w* (средств|остат)|остатк|кассов\w* разрыв|оборотн', re.I)
RECEIVABLES = re.compile(r'дебитор|отсрочк|концентрац|топ-5|крупнейш\w* (покупател|плательщик)', re.I)
PROCUREMENT = re.compile(r'госзакуп|госзаказ|тендер|44-ФЗ|223-ФЗ', re.I)
IF = re.compile(r'(^|[\s(«])если\b', re.I)


def _profile(company_id: str) -> dict:
    from stage2_company_facts.profile.load import PROFILES
    try:
        info=PROFILES.stat()
        version=(str(PROFILES),info.st_mtime_ns,info.st_ctime_ns,info.st_size)
    except OSError: version=None
    return _cached_profile(company_id,version)


@lru_cache(64)
def _cached_profile(company_id,version):
    from stage2_company_facts.profile.load import load_profile
    try:
        return load_profile(company_id)
    except Exception:  # an unknown company: no company-based offer, confidence still computed
        return {}


def _claims(ins: dict) -> list[str]:
    return [c.get('text', '') for b in ('what_happened', 'why_it_matters', 'risks', 'opportunities') for c in ins.get(b, [])]


def confidence(result: dict, ins: dict) -> tuple[str, list[str]]:
    score, basis = 0, []
    cred = result.get('credibility') or {}
    plaus = (result.get('stage4') or {}).get('plausibility')
    if result.get('source') == 'bank_analytics' or ins.get('origin') == 'bank':
        score += 2  # the bank's own view
        basis.append('основано на отчёте аналитиков Альфа-Банка')
        exact = (ins.get('review') or {}).get('verification', {}).get('exact', 0)
        if exact:
            score += 1
            basis.append(f'цитаты сверены с отчётом ({exact})')
    elif cred:
        score += {'high': 2, 'medium': 1}.get(cred.get('level'), 0)
        basis.append(cred.get('label', ''))
    elif isinstance(plaus,(int,float)) and isfinite(plaus) and plaus >= 0.6:
        score += 1
        basis.append('оценка подтверждения события по публикациям достаточно высокая')
    bank = [*result.get('judge_bank', []), *result.get('bank', []), *((result.get('evidence') or {}).get('bank') or [])]
    cited={source for claim in ins.get('bank_view') or [] for source in claim.get('sources') or []}
    def fresh(item):
        if item.get('stale') or item.get('superseded_by') or item.get('freshness') in ('stale','unknown'): return False
        try: return 0<=(date.today()-date.fromisoformat(item.get('date') or '')).days<=365
        except (TypeError,ValueError): return item.get('stale') is False
    relevant=[item for item in bank if (item.get('id') in cited or any(source in (item.get('frag_ids') or []) for source in cited))]
    if (result.get('source')=='bank_analytics' or ins.get('origin')=='bank') and relevant and not any(fresh(b) for b in relevant):
        score-=1
        basis.append('использованные отчёты устарели или их актуальность не установлена')
    if ins.get('bank_view') and any(fresh(b) for b in relevant):
        score += 1
        basis.append('есть актуальная позиция Банка')
    if any(f.get('kind') == 'fact' for f in result.get('calculations', [])):
        score += 1
        basis.append('цифры рассчитаны по данным компании')
    claims = _claims(ins)
    share_if = sum(bool(IF.search(t)) for t in claims) / len(claims) if claims else 0
    if share_if > 0.4:  # the writer always adds 3-4 hypotheses by design; reasoning built on «если» is the real gap
        score -= 2 if share_if > 0.6 else 1
        basis.append(f'{share_if:.0%} выводов — с допущением «если» о деталях бизнеса, которых нет в данных')
    if len(ins.get('hypotheses', [])) >= 5:
        score -= 1
        basis.append(f'{len(ins["hypotheses"])} гипотез требуют проверки')
    if isinstance(plaus,(int,float)) and isfinite(plaus) and plaus<0.4:
        score-=1
        basis.append('ограниченное подтверждение события по публикациям')
    if not basis: basis.append('в сохранённых материалах недостаточно данных для высокой оценки')
    level = 'high' if score >= 3 else 'medium' if score >= 1 else 'low'
    return level, [b for b in basis if b]


def bank_product(ins: dict, profile: dict) -> tuple[str, str] | None:
    """(offer, why) for the card's topic, skipping what the company already uses; None when nothing fits."""
    card = profile.get('card') or {}
    if not card: return None
    has = set(card.get('products') or [])
    # what the card is about: title and «why it matters», not every risk and action (nearly all of them mention the rate)
    text = ' '.join([ins.get('title', ''), *(c.get('text', '') for c in ins.get('why_it_matters', [])[:2])])
    primary=set(topics_of(ins.get('channel',''),top=1,min_hits=1)) & {'fx','rate','labor','fuel','deposit','leasing'}
    topics = primary or set(topics_of(ins.get('title',''),top=1,min_hits=1)) or set(topics_of(text,top=1))
    sup = ((profile.get('links') or {}).get('buyers') or {})
    wanted = []  # the card's own topic first, then what its text says about money flows
    if 'fx' in topics and (card.get('fts_countries') or card.get('fts_flag')):
        wanted.append(('fx', 'инсайт о курсе; в профиле есть сведения о внешней торговле. Применимо при валютных расчётах'))
    if 'rate' in topics or 'leasing' in topics:
        wanted.append(('fixed_rate' if 'loan' in has else 'credit_line', 'стоимость денег для компании зависит от ставки'))
    if 'labor' in topics:
        wanted.append(('salary', 'инсайт о зарплатах и рынке труда'))
    if PROCUREMENT.search(text):
        wanted.append(('guarantee', 'инсайт о госзакупках'))
    if 'fuel' in topics and 'leasing' not in has and re.search(r'парк|техник|транспорт|автомоб',text,re.I):
        wanted.append(('leasing', 'обновление парка под рост стоимости эксплуатации'))
    free_cash=bool(re.search(r'свободн\w* (средств|остат)|избыт\w* ликвидност|размест\w* остат',text,re.I))
    recommendation=' '.join([ins.get('ai_conclusion',''),*ins.get('actions',[])])
    rejects_deposit=bool(re.search(r'не\s+(?:на\s+)?(?:коротк\w*\s+)?депозит|вместо[^.!?]{0,100}депозит',recommendation,re.I))
    if not rejects_deposit and (free_cash or ('deposit' in topics and re.search(r'размещен|размести|депозит|вклад',text,re.I))):
        wanted.append(('deposit', 'в инсайте речь о размещении свободных средств'))
    if RECEIVABLES.search(text) or (LIQUIDITY.search(text) and not free_cash and (sup.get('top5_share') or 0) >= 0.6):
        why = f'на пять крупнейших покупателей приходится {sup["top5_share"]:.0%} поступлений' if sup.get('top5_share') else \
            'деньги компании зависят от сроков оплаты покупателей'
        wanted.append(('factoring', why))
    for key, why in wanted:
        offer, needs_missing = PRODUCTS[key]
        if needs_missing and needs_missing in has:
            continue
        return offer, why
    return None


def vague(product: str) -> bool:
    """«кредитные продукты», «лизинг»: a bare category name without what it does for the company."""
    return product.strip().lower().rstrip('.') in {'','кредитные продукты','кредит','кредиты','лизинг','депозит','депозиты','рко','банковская гарантия','зарплатный проект','факторинг'}


def enrich(result: dict, company_id: str) -> dict:
    ins = result.get('insight')
    if not isinstance(ins, dict) or not ins.get('title'):
        return result
    profile = _profile(company_id)
    if not result.get('credibility'):
        from common.credibility import from_saved
        if (cred:=from_saved(result)) is not None: result=dict(result,credibility=cred)
    level, basis = confidence(result, ins)
    out = dict(result, insight=dict(ins, confidence=level))
    previous=result.get('business_value') or {}
    original=previous.get('writer_bank_product',ins.get('bank_product',''))
    bv = {'confidence': level, 'confidence_basis': basis, 'writer_confidence': previous.get('writer_confidence',ins.get('confidence')),
          'writer_bank_product':original}
    offer = bank_product(ins, profile) if vague(original) else None
    out['insight']['bank_product']=original
    if offer:
        out['insight']['bank_product'] = offer[0]
        bv['bank_product'] = {'offer': offer[0], 'why': offer[1], 'by': 'code'}
    out['business_value'] = bv
    return out
