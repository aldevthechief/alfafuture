"""Step 5, second source: insights from the bank's own analytics (no news). In news 90% is noise, so general candidate
generators and a judge filter it; in the reports everything is useful to a business, the job is to find what matters for
THIS company and role. So a separate `ToolLoop` agent reads the reports (search with ±2 chunks, whole pages, a look at the
PDF page through the vision model, the same topic across reports), takes the company facts (products come from the
`bank` facts: flags of the bank's table) and records up to `max_candidates` candidates, then calls `finish`. Stops at
the step limit too. Not run with a real model yet (needs LLM_API_KEY); design: bank_analytics/analytics_agent_design.md.
"""
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.llm import ChatModels, PromptLibrary, ToolLoop, structured
from common.roles import role_context
from stage2_company_facts.profile.context import render_context, source_facts
from stage5_insight_detection.bank_analytics.number_check import numbers
from stage5_insight_detection.bank_analytics.quotes import locate, normalize, verify
from stage5_insight_detection.bank_analytics.search import BankAnalytics
from stage5_insight_detection.bank_analytics.tools import BankTools

PROMPTS = PromptLibrary(Path(__file__).with_name('prompts'))


class Evidence(BaseModel):
    """Доказательство: дословная цитата со страницы отчёта. Поля «система» заполняет код, не заполняй их."""
    report: str = Field(description='название отчёта как в результатах поиска')
    date: str = Field('', description='дата отчёта YYYY-MM-DD (если не знаешь, код подставит по названию отчёта)')
    page: int = Field(description='страница, на которой стоит цитата')
    quote: str = Field(description='дословный фрагмент текста страницы из read_page или результата поиска (распознавание можно исправить, числа и смысл менять нельзя); '
                                   'для kind=table — строка таблицы словами и числами; для kind=picture — что прочитано по картинке')
    kind: Literal['text', 'table', 'picture'] = Field('text', description='text — абзац; table — строка таблицы; picture — значение прочитано по картинке (look_at_page), приблизительно')
    status: str = Field('', description='система: exact | fuzzy | relocated | table_row | picture | other_report | not_found')
    cited_page: int | None = Field(None, description='система: страница, которую назвал агент (если код исправил страницу)')
    source_excerpt: str = Field('', description='система: текст страницы на месте цитаты, как он распознан')
    start: int | None = Field(None, description='система: смещение начала в тексте страницы')
    end: int | None = Field(None, description='система: смещение конца в тексте страницы')


class BankClaim(BaseModel):
    """Одно утверждение Банка со своими цитатами."""
    text: str = Field(description='позиция Банка близко к тексту отчёта, с числами и периодом (одно утверждение)')
    evidence: list[Evidence] = Field(description='цитаты, на которых стоит это утверждение; значение с графика через look_at_page — kind=picture')


class Action(BaseModel):
    text: str = Field(description='что сделать читателю роли и к какому сроку')
    because: str = Field(description='почему: какое число или утверждение Банка это обосновывает и что оно значит для компании')


class Assumption(BaseModel):
    claim: str = Field(description='проверяемая гипотеза о компании, которой нет в фактах: с числом, порогом или выбором из вариантов')
    check: str = Field(description='какие данные клиента её подтвердят')
    supports: list[str] = Field([], description="на чём она держится: 'bank_claim:N' (утверждение №N), 'link', 'mechanism', 'action:N', 'impact'")


class ImpactEstimate(BaseModel):
    level: Literal['low', 'mid', 'high'] = Field(description='low — фон; mid — заметно для компании; high — прямо бьёт по деньгам, нужна оценка в estimate и выручка в basis')
    criteria: str = Field(description='по какому правилу этот уровень: через что и насколько затрагивает деньги компании')
    estimate: str = Field('', description='оценка словами и числами: на единицу («на каждые 100 млн ₽ …») или долей выручки; только числа из фактов и отчётов, суммы по компании не выдумывать')
    basis: list[str] = Field([], description="на чём оценка: 'fact:N', 'bank_claim:N'")


class AcceptedWarning(BaseModel):
    """Замечание проверки, которое агент проверил и не считает нужным учитывать; причина попадает в карточку для человека."""
    id: str = Field(description='номер замечания из ответа add_candidate, например W1')
    reason: str = Field(description='почему замечание не нужно учитывать')


class AnalyticsCandidate(BaseModel):
    """Записать один инсайт из аналитики Банка для этой компании и роли; он подаётся как «аналитика и прогнозы экспертов Альфа-Банка»."""
    model_config = ConfigDict(title='add_candidate')
    kind: Literal['forecast', 'view_change', 'recommendation', 'calendar', 'product'] = Field(
        description='forecast — прогноз показателя; view_change — банк изменил взгляд между отчётами; recommendation — рекомендация банка; '
                    'calendar — датированное событие, к которому стоит подготовиться; product — продукт банка под условия из аналитики')
    title: str = Field(description='одна строка: что банк говорит и почему это важно этой компании')
    conclusion: str = Field(description='одно предложение: итоговый вывод для читателя роли')
    channel: str = Field(description='канал влияния: ставка, курс, инфляция, потребление, инвестиции и стройка, топливо, торговая маржа, депозиты…')
    effect: Literal['risk', 'opportunity', 'mixed'] = Field(description='для компании механизм — риск, возможность или то и другое')
    role_fit: int | None = Field(None, ge=0, le=3, description='насколько важно этой роли: 0 — нет, 3 — прямо её решения')
    confidence: Literal['low', 'medium', 'high'] = Field('medium', description='уверенность в инсайте')
    event_date: str = Field('', description='дата события YYYY-MM-DD (заседание ЦБ и т. п.); обязательна для kind=calendar, для остальных — если есть')
    product: str = Field('', description='продукт Альфа-Банка из фактов компании (категория bank) или явно недостающий; пусто, если связь натянута')
    bank_claims: list[BankClaim] = Field(description='утверждения Банка, на которых стоит инсайт; у каждого свои цитаты')
    company_facts: list[int] = Field(description='номера фактов о компании (fact:N из company_facts), через которые это её касается; текст фактов подставит код')
    link_inference: str = Field(description='вывод: как эти факты связывают позицию Банка с компанией (рассуждение, не факт; фактов и сумм, которых нет в company_facts, не вписывать)')
    mechanism: str = Field(description='причина → следствие → итог для денег компании, простыми словами, с числами из отчёта')
    actions: list[Action] = Field([], description='1–3 действия читателя роли, каждое с «потому что»')
    assumptions: list[Assumption] = Field([], description='гипотезы о компании, которых нет в фактах, и на чём они держатся')
    impact: ImpactEstimate | None = Field(None, description='оценка влияния на компанию')
    fact_quotes: list[str] = Field([], description='система: тексты фактов')
    warnings: list[str] = Field([], description='заполняет система при проверках (цитаты, числа, факты, оценка, дата события), не заполняй')
    accepted_warnings: list[AcceptedWarning] = Field([], description='необязательно: замечания, которые ты проверил и не считаешь нужным учитывать, с причиной')


class Finish(BaseModel):
    """Закончить: список кандидатов окончателен. Всё, что ты узнал и не вошло в кандидатов, запиши здесь: оно сохраняется."""
    model_config = ConfigDict(title='finish')
    summary: str = Field(description='что прочитал и к какому выводу пришёл для этой компании и роли; почему остановился (набрал нужное или важного больше нет)')
    covered: list[str] = Field([], description='каналы и темы, которые проверил: «канал — что нашёл» или «канал — позиции Банка нет / не касается компании, потому что…»')
    rejected: list[str] = Field([], description='темы, которые видел и не записал, с причиной: «тема — почему отброшена» (повторяет механизм, нет числа, натянуто)')
    open_questions: list[str] = Field([], description='какие данные клиента или отчёты Банка уточнили бы выводы: «что нужно — на какой вывод повлияет»')


def progress(calls: int, found: int, wanted: int, max_steps: int) -> str:
    """The footer of every tool answer: a reasoning model does not see the step limit and keeps exploring (at reasoning_effort=high:
    25 searches, no candidate, limit hit)."""
    note = f'[вызовов инструментов: {calls}; записано кандидатов: {found} из {wanted}; лимит шагов: {max_steps}]'
    if calls >= 8 and found == 0:
        note += ' Ты уже изучил достаточно: запиши найденное через add_candidate сейчас, остальное доищешь потом.'
    elif calls >= 0.7 * max_steps:
        note += ' Лимит шагов близко: допиши кандидатов и вызови finish.'
    return chr(10) + chr(10) + note


STUCK_LIMIT = 3  # consecutive calls that change nothing (the same call again, or a corrected call refused) end the loop


class LoopStuck(Exception):
    """The model repeats calls that change nothing: a reasoning model obeys every advice to the letter and the advice cannot be satisfied."""


def calendar_warnings(cand: AnalyticsCandidate) -> list[str]:
    """`calendar` is a dated event to prepare for: it needs a real date in event_date (advice, the candidate is kept as it is)."""
    if cand.event_date:
        try:
            date.fromisoformat(cand.event_date)
        except ValueError:
            return [f'event_date «{cand.event_date}» не в формате YYYY-MM-DD или такой даты нет: дату стоит поправить'
                    + (' или подобрать другой тип' if cand.kind == 'calendar' else '')]
    elif cand.kind == 'calendar':
        return ['у типа calendar нужна дата события в event_date (YYYY-MM-DD, например заседание ЦБ 2026-10-23); '
                'если даты нет, подойдёт другой тип: recommendation или forecast']
    return []


class AnalyticsTools:
    def __init__(self, facts: list[dict], bank: BankAnalytics, vision: BaseChatModel | None, max_candidates: int):
        self.facts, self.max_candidates, self.candidates = source_facts(facts), max_candidates, []
        self.warn_ids, self.warn_seen = {}, {}  # title -> {warning text: id}, title -> ids already shown to the agent
        self.calls = 0  # tool calls so far, shown to the agent
        self.stuck, self.last_call = 0, None
        self.bank_tools = BankTools(bank, vision)

    def as_tools(self) -> list[BaseTool]:
        b = self.bank_tools
        tools = [StructuredTool.from_function(f) for f in (b.search_bank_analytics, b.read_page, b.look_at_page, b.compare_reports, self.company_facts)]
        add = StructuredTool.from_function(self.add_candidate, name='add_candidate', description=AnalyticsCandidate.__doc__, args_schema=AnalyticsCandidate)
        return tools + [add]

    def claims_text(self, cand: AnalyticsCandidate) -> str:
        return ' '.join(c.text for c in cand.bank_claims)

    def check_quotes(self, cand: AnalyticsCandidate) -> list[str]:
        """Every quote of every claim against its page (quotes.verify): the system fields of the evidence are filled, a wrong page is corrected."""
        out = []
        if not cand.bank_claims:
            return ['нет ни одного утверждения Банка (bank_claims): инсайт стоит на позиции Банка с цитатой и страницей отчёта']
        for ci, claim in enumerate(cand.bank_claims, 1):
            if not claim.evidence:
                out.append(f'у утверждения {ci} нет цитаты: нужна дословная цитата со страницей отчёта')
            for ei, e in enumerate(claim.evidence, 1):
                e.date = e.date or next((x['date'] for x in self.bank_tools.reports if x['title'].lower() in e.report.lower() or e.report.lower() in x['title'].lower()), '')
                r = verify(self.bank_tools.bank.rows, self.bank_tools.reports, e.model_dump()) if self.bank_tools.bank else {'advice': [], 'status': ''}
                e.status = r['status']
                if r['status'] == 'relocated':
                    e.cited_page, e.page = e.page, r['page']
                e.source_excerpt, e.start, e.end = r.get('source_excerpt', ''), r.get('start'), r.get('end')
                out += [f'утверждение {ci}, доказательство {ei}: {a}' for a in r['advice']]
        return out

    def known_numbers(self, cand: AnalyticsCandidate, facts_text: str) -> dict:
        """Numbers a conclusion may stand on: the company facts named, the bank's claims and their quotes, the mechanism."""
        quotes = [e.quote for c in cand.bank_claims for e in c.evidence]
        return numbers(' '.join([facts_text, self.claims_text(cand), *quotes, cand.mechanism]), keep_small=True)

    def check_facts(self, cand: AnalyticsCandidate) -> list[str]:
        """company_facts must name real facts (the code puts their text in), and link_inference must not bring in figures of the company."""
        by_id = {i: f for i, f in enumerate(self.facts, 1)}
        bad = [i for i in cand.company_facts if i not in by_id]
        cand.fact_quotes = [f'fact:{i} [{by_id[i]["cat"]}] {by_id[i]["text"]}' for i in cand.company_facts if i in by_id]
        if not cand.company_facts:
            return ['company_facts пуст: можно указать номера фактов о компании (fact:N из company_facts), через которые это её касается']
        if bad:
            return [f'fact:{", ".join(map(str, bad))} нет среди фактов компании: проверь номера в company_facts']
        known = self.known_numbers(cand, ' '.join(by_id[i]['text'] for i in cand.company_facts))
        out = []
        for c, raw in numbers(cand.link_inference, keep_small=True).items():
            if c not in known:
                out.append(f'число {raw} из link_inference нет ни в названных фактах, ни в позиции Банка, ни в цитатах: суммы и доли по компании нельзя придумывать или вычислять; '
                           'если это предположение, его можно вынести в assumptions как гипотезу')
        return out

    SUPPORTS = re.compile(r'^(bank_claim:(\d+)|link|mechanism|action:(\d+)|impact)$')

    def check_assumptions(self, cand: AnalyticsCandidate) -> list[str]:
        out = []
        for n, a in enumerate(cand.assumptions, 1):
            for t in a.supports:
                m = self.SUPPORTS.match(t.strip())
                limit = len(cand.bank_claims) if t.startswith('bank_claim') else len(cand.actions) if t.startswith('action') else None
                if not m or (limit is not None and not 1 <= int(m.group(2) or m.group(3)) <= limit):
                    out.append(f"гипотеза {n}: supports «{t}» не распознан; допустимо bank_claim:N, link, mechanism, action:N, impact (N в пределах списка)")
        if not cand.actions:
            out.append('нет действий (actions): нужно 1–3 шага читателя роли с «потому что»')
        if cand.role_fit is None:
            out.append('не указан role_fit (0–3): насколько это важно этой роли')
        return out

    PER_UNIT = re.compile(r'на кажд|пример|на единицу', re.I)

    def check_impact(self, cand: AnalyticsCandidate) -> list[str]:
        """The level `high` needs an estimate and a basis (the rule of the analyst loop: no number, no higher than mid); figures of the
        estimate must come from the facts and the bank, except a per-unit example."""
        out, imp = [], cand.impact
        if imp is None:
            return ['нет оценки влияния (impact): нужны level, criteria и, если есть основание, estimate с числами из фактов и отчётов']
        if imp.level == 'high' and not (imp.estimate and imp.basis):
            imp.level = 'mid'
            out.append('оценка влияния high без estimate и basis понижена до mid: для high нужны оценка словами и числами и основание (fact:N, bank_claim:N)')
        if imp.estimate:
            by_id = {i: f for i, f in enumerate(self.facts, 1)}
            known = self.known_numbers(cand, ' '.join(by_id[i]['text'] for i in cand.company_facts if i in by_id))
            for sentence in re.split(r'(?<=[;!?])\s+|(?<=[а-яa-z0-9)%])\.\s+(?=[А-ЯA-Z])', imp.estimate):  # not after «п.п.»
                if self.PER_UNIT.search(sentence):
                    continue
                for c, raw in numbers(sentence, keep_small=True).items():
                    if c not in known:
                        out.append(f'число {raw} из estimate нет в фактах, позиции Банка и цитатах: сумм и долей по компании выдумывать нельзя, оценку можно дать на единицу')
        return out

    def company_facts(self, category: str = '') -> str:
        """Facts about the company from the bank's data; category prefix filters them (profile, scale, rank, activity,
        geo, ownership, bank, trade, payments, persons, group...). Bank products of the company: category bank."""
        rows = [(i, f) for i, f in enumerate(self.facts, 1) if f['cat'].startswith(category)]
        return '\n'.join(f'[fact:{i}][{f["cat"]}] {f["text"]}' for i, f in rows) or 'no facts in this category'

    def add_candidate(self, **fields) -> str:
        cand = AnalyticsCandidate(**fields)
        same = next((i for i, c in enumerate(self.candidates) if c.title == cand.title), None)  # a corrected call replaces its earlier record
        if same is None and len(self.candidates) >= self.max_candidates:
            return f'already {self.max_candidates} candidates: call finish'
        texts = (self.check_quotes(cand) + self.check_facts(cand) + self.check_assumptions(cand) + self.check_impact(cand)
                 + self.bank_tools.check_candidate(self.claims_text(cand), [f'{e.report}, {e.date}, стр. {e.page}' for c in cand.bank_claims for e in c.evidence],
                                                   ' '.join(e.quote for c in cand.bank_claims for e in c.evidence if e.kind == 'picture')) + calendar_warnings(cand))
        ids = self.warn_ids.setdefault(cand.title, {})  # a warning keeps its id while its text stays the same
        for t in texts:
            ids.setdefault(t, f'W{len(ids) + 1}')
        cand.warnings = [f'{ids[t]}: {t}' for t in texts]
        have = {ids[t] for t in texts}
        cand.accepted_warnings = [w for w in cand.accepted_warnings if w.id in have]
        seen, accepted = self.warn_seen.setdefault(cand.title, set()), {w.id for w in cand.accepted_warnings}
        new = [w for t, w in zip(texts, cand.warnings) if ids[t] not in seen and ids[t] not in accepted]  # each warning is shown once
        seen.update(have)
        if same is None:
            self.candidates.append(cand)  # always recorded: the checks advise, they do not refuse
        else:
            self.candidates[same] = cand
        left = self.max_candidates - len(self.candidates)
        out = (f'recorded #{len(self.candidates)}' if same is None else f'replaced #{same + 1}') + (
            f'; {left} more allowed' if left else '; limit reached: call finish')
        if new:
            out += (chr(10) + 'Замечания проверки (кандидат записан, делать ничего не обязательно). Каждое показано один раз. Если замечание верное, можно поправить '
                    'и вызвать add_candidate с тем же title; если ты проверил и уверен, ничего не меняй, при желании укажи причину в accepted_warnings [{id, reason}]:'
                    + chr(10) + chr(10).join(f'- {w}' for w in new))
        return out


class AnalyticsInsightAgent:
    def __init__(self, model: BaseChatModel | None = None, bank: BankAnalytics | None = None, vision: BaseChatModel | None = None,
                 max_steps: int = 30, max_candidates: int = 5, key_facts: int = 25, observe=None):
        models = ChatModels() if model is None or vision is None else None
        self.model = model or models.agent()
        # a page question has a short answer; the 64k default would let a looping vision model burn the whole cap
        self.vision = vision or models.vision(max_tokens=8000)
        self.bank = bank or BankAnalytics()
        self.max_steps, self.max_candidates, self.key_facts, self.observe = max_steps, max_candidates, key_facts, observe
        self.finish = None  # the agent's closing note (summary, covered, rejected, open questions); None when the step limit came first
        self.error = None  # set when the loop dies (e.g. a budget guard in `observe`); candidates recorded so far are still returned

    def run(self, company_id: str, facts: list[dict], role: str, today: str) -> list[AnalyticsCandidate]:
        tools = AnalyticsTools(facts, self.bank, self.vision, self.max_candidates)
        system = PROMPTS.render('analytics_system', max_candidates=self.max_candidates, max_steps=self.max_steps)
        def observe(name: str, args: dict, text: str) -> str:
            tools.calls += 1
            text = self.observe(name, args, text) if self.observe else text
            key = (name, json.dumps(args, sort_keys=True, ensure_ascii=False, default=str))
            tools.stuck = tools.stuck + 1 if key == tools.last_call or False else 0
            tools.last_call = key
            if tools.stuck >= STUCK_LIMIT:
                raise LoopStuck(f'{name} repeated without change {tools.stuck} times')
            return text + progress(tools.calls, len(tools.candidates), self.max_candidates, self.max_steps)
        loop = ToolLoop(self.model, tools.as_tools(), Finish, system, self.max_steps, observe,
                        nudge='Вызови инструменты, add_candidate или finish.')
        selected = [(i, f) for i, f in enumerate(tools.facts, 1) if f['depth'] == 1][:self.key_facts]
        top = [f'[fact:{i}][{f["cat"]}] {f["text"]}' for i, f in selected]
        task = PROMPTS.render('analytics_task', role=role, **role_context(role), company_id=company_id, today=today, facts='\n'.join(top),
                              profile_context=render_context(facts, visible_facts=[f for _, f in selected]),
                              reports='\n'.join(self.bank.available))
        try:
            self.finish = loop.run(task)
        except LoopStuck as e:  # the candidates are kept; the closing note is asked for once, without tools
            self.error = f'LoopStuck: {e}'
            self.finish = self.close(task, tools.candidates)
        except Exception as e:
            self.error = f'{type(e).__name__}: {e}'
        return tools.candidates

    def close(self, task: str, candidates: list[AnalyticsCandidate]) -> Finish | None:
        """The closing note after a loop that had to be stopped: one structured call over the task and the recorded candidates."""
        titles = chr(10).join(f'- {c.kind}: {c.title}' for c in candidates) or '(кандидатов нет)'
        prompt = (task + chr(10) * 2 + 'Работа остановлена: вызовы повторялись без изменений. Записанные кандидаты:' + chr(10) + titles + chr(10) * 2
                  + 'Подведи итог без инструментов: summary (начни словами «Остановлено из-за повторов вызовов»), covered, rejected, open_questions.')
        try:
            return structured(self.model, Finish).invoke(prompt)
        except Exception as e:
            self.error += f'; close failed: {type(e).__name__}: {e}'
            return None
