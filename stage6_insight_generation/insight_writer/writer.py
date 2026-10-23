"""Step 6: turn a positive stage-5 decision into the insight for a given reader role, then self-check it.

write (structured Insight) -> code reminders (one look-again pass) -> check against the evidence (structured Review)
-> up to `revisions` rewrites with the issues. A rejected result keeps the last draft and its issues; every round's
prompt and answer is in data/logs/calls.jsonl. The shared rule files are injected identically into the writer and the
checker, so the checker never demands what the writer was not told; both run at temperature 0 (same draft -> same verdict).
Two plain LangChain chains: no loop with tools is needed here, the evidence is already collected by stage 5.
"""
import re, sys
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.language_models import BaseChatModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.chunk_search import article_context
from common.news_evidence import verify_claim_evidence
from common.llm import ChatModels, PromptLibrary, structured
from common.llm.log import log_tool
from common.roles import role_context
from common.safety import UNTRUSTED_NOTE, render_untrusted
from stage6_insight_generation.insight_writer.insight import Insight, Review
from stage5_insight_detection.analyst_loop.facts import Fact

from stage2_company_facts.profile.context import evidence_facts

PROMPTS = PromptLibrary(Path(__file__).with_name('prompts'))
# shared rules: one wording for the writer and the checker, so the checker never demands what the writer was not told
RULES = '\n\n'.join(PROMPTS.text(name) for name in ('rules_facts', 'rules_layers', 'rules_reader'))


@dataclass
class Evidence:
    """Everything the insight may cite. Ids here are the only valid `sources` values."""
    news: list[dict]  # main news first; {'id', 'date', 'source', 'title', 'text'}
    facts: list[str]  # company facts, cited as fact:<n> (1-based)
    bank: list[dict] = field(default_factory=list)  # BankAnalytics.search hits
    origin: str = 'news'
    calculations: list[dict] = field(default_factory=list)  # Python results and their exact input facts
    full_bank_ids: set[str] = field(default_factory=set)  # already read by judge; preserve complete chunk text

    def render(self) -> str:
        news = [
            render_untrusted(
                f'news:{n["id"]}', n['title'], article_context(n),
                {'дата': n.get('date'), 'источник': n.get('source')},
                body_limit=3500,
            )
            for n in self.news
        ]
        facts = evidence_facts(self.facts)
        bank = [
            render_untrusted(
                b.get('id') or f'bank:{b["report"]}:{b["page"]}',
                b['report'], b['text'],
                {
                    'дата': b.get('date'),
                    'стр.': ', '.join(map(str, b['pages'])) if b.get('pages') else b['page'],
                    'устарел': b['stale'],
                    'позиция Банка': b.get('own_view', False),
                },
                body_limit=len(b['text']) if b.get('id') in self.full_bank_ids else 900,
            )
            for b in self.bank
        ]
        return '\n\n'.join([
            UNTRUSTED_NOTE,
            '## Новости', *news,
            '## Факты о компании', *facts,
            '## Аналитика Банка', *(bank or ['(нет)']),
            *(['## Расчёты кодом (ГИПОТЕЗА означает сценарий, а не установленный эффект компании)',
               *[Fact(**f).render() + f' [источник: {f["source"]}]' for f in self.calculations]] if self.calculations else []),
        ])


# Reminders, not verdicts: code finds what it can see for sure in the reader text and lets the writer re-read it; the writer
# may keep the wording. Content (an assumption without «если», a forecast without its date) stays with the checker.
REMINDERS = [
    (re.compile(r'(?i)(?<![а-яё])(материал\w*|допущени\w*|гипотез\w*|интерфейс\w*)'), 'служебное слово «{}»'),
    (re.compile(r'(?i)(требу\w* проверки|подтвердите|проверьте|сверьте|собственн\w+ (?:данн|статистик|отч[её]тност)\w*)'),
     'похоже на просьбу к читателю что-то проверить: «{}»'),
    (re.compile(r'\([^()]{3,}\)'), 'скобки «{}»'),
    (re.compile(r'\b(?:news|fact|bank):[\w.:-]+'), 'id источника «{}»'),
]


def reader_texts(insight: Insight) -> list[tuple[str, str]]:
    """(field, text) of everything the reader sees; hypotheses and evidence quotes are not reader text."""
    out = [('title', insight.title), ('ai_conclusion', insight.ai_conclusion)]
    for name in ('what_happened', 'why_it_matters', 'risks', 'opportunities', 'bank_view'):
        out += [(f'{name}[{i}]', c.text) for i, c in enumerate(getattr(insight, name), 1)]
    out += [(f'actions[{i}]', a) for i, a in enumerate(insight.actions, 1)]
    if insight.impact:
        out += [('impact', f'{insight.impact.criteria} {insight.impact.estimate}')]
    return out


# heuristics, so the wording asks to look again, never states a violation
STRUCTURE = re.compile(r'(?i)(?<![а-яё])(сет[ьиеюя]\w*|точк\w+|франши\w+|франчайз\w*|закупа\w*|закупк\w*|поставщик\w*|'
                       r'(?:маржа|маржу|затраты|себестоимость) компании)')
FORECAST = re.compile(r'(?i)(?<![а-яё])(прогноз\w*|вырастет|снизится|по итогам года)')
DATED = re.compile(r'(?i)(январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр|квартал|\b20\d\d\b)')
SUPPLY = re.compile(r'(?i)(?<![а-яё])(поставщик\w*|закупа\w*|закупк\w*)')


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r'(?<=[.!?])\s+', text or '') if s]


def reminders(insight: Insight, evidence: 'Evidence | None' = None) -> list[str]:
    found = [f'{field}: {label.format(m.group(0))}' for field, text in reader_texts(insight) for rx, label in REMINDERS
             for m in rx.finditer(text or '')]
    for field, text in reader_texts(insight):
        for s in sentences(text):
            m = STRUCTURE.search(s)
            if m and not field.startswith('what_happened') and 'если' not in s.lower():
                found.append(f'{field}: «{m.group(0)}» без «если» — возможно, это допущение о внутреннем устройстве бизнеса')
            m = FORECAST.search(s)
            if m and not DATED.search(s):
                found.append(f'{field}: «{m.group(0)}» без даты — если это прогноз, стоит указать, когда и кем он сделан')
    if evidence is not None:  # fact ids of incoming payments: money from buyers, not suppliers
        incoming = {f'fact:{i}' for i, t in enumerate(evidence.facts, 1) if t.startswith('[payments_in]')}
        for name in ('why_it_matters', 'risks', 'opportunities', 'hypotheses'):
            for i, c in enumerate(getattr(insight, name), 1):
                refs = sorted(incoming & set(c.sources))
                if refs and (m := SUPPLY.search(c.text)):
                    found.append(f'{name}[{i}]: {", ".join(refs)} — входящие платежи, то есть деньги от покупателей, а в тексте '
                                 f'«{m.group(0)}»')
    for i, h in enumerate(insight.hypotheses, 1):
        if not re.search(r'\d', h.text) and ' или ' not in h.text:
            found.append(f'hypotheses[{i}]: без числа, порога или выбора из вариантов — так её трудно проверить')
    return found


def reminder_note(found: list[str]) -> str:
    return ('Напоминание от проверки кодом — не отказ и не требование: замечено\n'
            + '\n'.join(f'- {f}' for f in found) + '\nПерепроверь эти места по общим правилам; если формулировка '
            'уместна, оставь её как есть.')


class InsightWriter:
    def __init__(self, model: BaseChatModel | None = None, revisions: int = 2):
        model = model or ChatModels().agent()
        if hasattr(model, 'temperature'):  # writer and checker at 0: the same draft gets the same verdict (23.10: 4/6 accepted at 0, 3/5 at 0.2)
            model = model.model_copy(update={'temperature': 0.0})
        self.writer = PROMPTS.chat('write') | structured(model, Insight)
        self.checker = PROMPTS.chat('check') | structured(model, Review)
        self.revisions = revisions

    def write(self, company_id: str, role: str, decision: str, evidence: Evidence) -> tuple[Insight, Review]:
        """Returns the final insight and the review of the last draft that was checked."""
        inputs = {'company_id': company_id, 'role': role, **role_context(role), 'decision': decision, 'evidence': evidence.render(),
                  'rules': RULES}
        insight = self.writer.invoke(inputs | {'feedback': 'нет'})
        if found := self._remind(insight, 0, evidence):  # one look-again pass before the (expensive) checker
            insight = self.writer.invoke(inputs | {'feedback': reminder_note(found)})
        review = self._check(insight, inputs, evidence)
        for n in range(1, self.revisions + 1):  # every rewrite gets the checker's issues in full, plus code reminders
            if review.ok:
                break
            found = self._remind(insight, n, evidence)
            feedback = '\n'.join(f'- {i}' for i in review.issues) + (f'\n\n{reminder_note(found)}' if found else '')
            insight = self.writer.invoke(inputs | {'feedback': feedback})
            review = self._check(insight, inputs, evidence)
        return insight, review

    @staticmethod
    def _remind(insight: Insight, round_no: int, evidence: Evidence) -> list[str]:
        found = reminders(insight, evidence)
        if found:
            log_tool('writer_reminders', {'round': round_no}, '\n'.join(found))
        return found

    def _check(self, insight: Insight, inputs: dict, evidence: Evidence) -> Review:
        # Verbatim quotes get offsets for the card; a missing or paraphrased quote only marks the claim «не подтверждено»
        # (as a blocking issue it rejected 4 of 5 insights on U_68622, 22.10: «(dp, 22.09.2026)» appended, paraphrases).
        verify_claim_evidence(insight, evidence.news, evidence.bank, origin=evidence.origin)
        return self.checker.invoke({'role': inputs['role'], 'role_title': inputs['role_title'], 'role_focus': inputs['role_focus'],
                                    'rules': RULES, 'evidence': inputs['evidence'],
                                    'insight': insight.model_dump_json(indent=1)})
