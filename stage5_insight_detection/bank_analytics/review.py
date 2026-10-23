"""Bank analyst agent between the code's pick and the brief (task 2): the code takes 15–20 candidates, the agent model
reads them with the company card and the event and decides for each whether it fits, through which channel it reaches
the company and whether the bank's view agrees with the event; nothing fits = «аналитики Банка нет».

Fragments go to the model as untrusted items (render_untrusted), the answer is strict JSON (structured), calls are cached
(FileLLMCache via ChatModels) and metered: `budget` (₽, the proxy's usage.cost) stops before the next call would pass it,
and the review then falls back to the code's pick. Unknown ids in the answer are ignored; a fragment the model did not
answer about is dropped. Also: `bank_positions()` — the bank's own views from the report digests (digest.py) for the
reports of the picked fragments, with the report date, for the brief.
"""
import hashlib, json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from common.llm import ChatModels, PromptLibrary, StructuredOutputError, structured
from common.safety import UNTRUSTED_NOTE, render_untrusted
from stage5_insight_detection.bank_analytics.digest import BudgetExceeded, CostMeter
from stage5_insight_detection.bank_analytics.export_text import TEXT_DIR
from stage5_insight_detection.bank_analytics.match import BankMatch

PROMPTS = PromptLibrary(Path(__file__).with_name('prompts'))
CANDIDATES = 20
KEEP = 8
RESCUE = 2  # strong candidates of the code kept against the agent's rejection, marked as doubtful
STRONG_EVENT = 0.8  # the event part of select() that makes a candidate strong (with a fresh own view of the bank)


class Decision(BaseModel):
    id: str
    fits: bool
    channel: str = ''
    agrees: Literal['совпадает', 'расходится', 'нет позиции'] = 'нет позиции'
    why: str = ''


class BankReview(BaseModel):
    items: list[Decision] = []
    none: bool = Field(False, description='ни один фрагмент не подходит')


class BankReviewer:
    def __init__(self, model=None, budget: float | None = None):
        self.meter, self.budget, self.max_call = CostMeter(), budget, 0.0
        agent = ChatModels().agent if model is None else None
        make = lambda: PROMPTS.chat('bank_review') | structured(model or agent(), BankReview,
                                                               attempts=1).with_config(callbacks=[self.meter])
        # an unparsed answer is asked once more; still nothing parsed -> the code's pick, said so in the record
        # (never read as «нет аналитики»). Default reasoning and the 64k cap (a 2000/5000 cap cut reasoning answers).
        self.chain, self.retry_chain = make(), make()

    def review(self, event, profile, candidates: list[BankMatch], keep: int = KEEP) -> tuple[list[BankMatch], dict]:
        """The candidates the agent kept, in the code's order, with its channel and agreement in parts['review'], and the
        record for the trace: {'verdict', 'decisions', 'cost'}."""
        if not candidates:
            return [], {'verdict': 'аналитики Банка нет', 'decisions': [], 'cost': 0.0, 'by': 'code'}
        if self.budget is not None and self.meter.cost + max(self.max_call, 0.05) > self.budget:
            raise BudgetExceeded(f'spent {self.meter.cost:.2f} of {self.budget} ₽')
        before = self.meter.cost
        fragments = '\n'.join(render_untrusted(m.fragment.id, m.fragment.caption(), m.fragment.brief_text()[:900],
                                               {'отчёт': m.fragment.report, 'дата': m.fragment.date, 'свежесть': m.fragment.freshness,
                                                'позиция Банка': 'да' if m.fragment.own_view else 'цитата/данные'})
                              for m in candidates)
        event_text = render_untrusted(event.id, event.title, ' '.join(event.text.split())[:1200], {'дата': event.date})
        inputs = {'note': UNTRUSTED_NOTE, 'card': profile.card(), 'event': event_text, 'fragments': fragments}
        try:
            answer = self.chain.invoke(inputs)
        except StructuredOutputError:
            answer = None
        self.max_call = max(self.max_call, self.meter.cost - before)
        if not parsed(answer):
            if self.budget is not None and self.meter.cost + max(self.max_call, 0.05) > self.budget:
                raise BudgetExceeded(f'spent {self.meter.cost:.2f} of {self.budget} ₽')
            retry_before = self.meter.cost
            try:
                answer = self.retry_chain.invoke(inputs)
            except StructuredOutputError:
                answer = None
            self.max_call = max(self.max_call, self.meter.cost - retry_before)
        if not parsed(answer):
            return candidates[:5], {'verdict': 'ответ агента не разобран: подбор кода', 'decisions': [],
                                    'cost': round(self.meter.cost - before, 4), 'by': 'code'}
        by_id = {d.id: d for d in answer.items}
        kept = []
        for m in candidates:
            d = by_id.get(m.fragment.id)
            if d and d.fits and not answer.none:
                m.parts['review'] = {'channel': d.channel, 'agrees': d.agrees, 'why': d.why}
                m.reason += f'; агент: {d.channel or "подходит"}' + ('; Банк ждёт иного — нужна оговорка' if d.agrees == 'расходится' else '')
                kept.append(m)
        # the agent filters but cannot empty the pick: a strong candidate of the code it rejected, or left unanswered with a
        # bare «нет», goes to the brief marked as doubtful, with the agent's reason (on the labelled pairs it dropped the
        # bank's own rate and rouble forecasts that way)
        bare_none = answer.none and not answer.items
        doubtful = []
        for m in candidates:
            if len(doubtful) == RESCUE or m in kept or not strong(m):
                continue
            d = by_id.get(m.fragment.id)
            why = 'ответ «нет» без разбора по пунктам' if bare_none or d is None else d.why or 'отклонён без причины'
            m.parts['review'] = {'doubt': True, 'why': why}
            m.reason += f'; агент сомневается: {why}'
            doubtful.append(m)
        picked = kept[:keep - len(doubtful)] + doubtful
        record = {'verdict': 'аналитики Банка нет' if not picked else f'{len(picked)} фрагментов' +
                  (f', из них {len(doubtful)} под сомнением агента' if doubtful else ''),
                  'decisions': [d.model_dump() for d in answer.items if d.id in {m.fragment.id for m in candidates}],
                  'doubtful': [m.fragment.id for m in doubtful], 'cost': round(self.meter.cost - before, 4), 'by': 'agent'}
        return picked, record


def strong(m: BankMatch) -> bool:
    """A candidate the code is sure of: close to the event and the bank's own view in a fresh report."""
    return m.parts.get('event', 0.0) >= STRONG_EVENT and m.fragment.freshness == 'fresh' and m.fragment.own_view


def parsed(answer) -> bool:
    """A usable answer: decisions, or an explicit «none»; an empty object is a cut or broken answer, not «нет»."""
    return answer is not None and (bool(answer.items) or answer.none)


def bank_positions(matches: list[BankMatch], folder: Path = TEXT_DIR, per_report: int = 4) -> list[str]:
    """The bank's own views (bank_views of the report digest, checked against their pages) of the reports the picked
    fragments come from, freshest report first: «Отчёт (дата): показатель — значение, срок (стр. N)»."""
    reports, out = {}, []
    for m in matches:
        reports.setdefault(Path(m.fragment.file).stem, (m.fragment.report, m.fragment.date or ''))
    for stem, (title, when) in sorted(reports.items(), key=lambda kv: kv[1][1], reverse=True):
        path = folder / f'{stem}.digest.json'
        source = folder / f'{stem}.md'
        if not path.exists() or not source.exists():
            continue
        from stage5_insight_detection.bank_analytics.digest import ReportDigest, split_pages, verify
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
            md = source.read_text(encoding='utf-8')
            if payload.get('source_sha256') and payload['source_sha256'] != hashlib.sha256(md.encode()).hexdigest():
                continue
            checked = verify(ReportDigest.model_validate(payload), split_pages(md)[1])
            current_pages = {}
            for m in matches:
                if Path(m.fragment.file).stem == stem:
                    current_pages[m.fragment.page] = current_pages.get(m.fragment.page, '') + '\n' + m.fragment.text
            checked = verify(checked, current_pages)
        except (ValueError, TypeError):
            continue  # optional broken/legacy summaries cannot stop the main pipeline
        pages = {m.fragment.page for m in matches if Path(m.fragment.file).stem == stem and m.fragment.own_view and not m.fragment.stale}
        views = [v.model_dump() for v in checked.bank_views if not v.unverified and v.page in pages]
        for v in views[:per_report]:
            out.append(f'{title} ({when}): {v["indicator"]} — {v["value"]} {v.get("unit") or ""}'.rstrip() +
                       (f', {v["horizon"]}' if v.get('horizon') else '') + f' (стр. {v.get("page")})')
    return out
