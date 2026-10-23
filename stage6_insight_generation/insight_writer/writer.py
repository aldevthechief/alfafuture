"""Step 6: turn a positive stage-5 decision into the insight for a given reader role, then self-check it.

write (structured Insight) -> check against the evidence (structured Review) -> up to `revisions` rewrites with the issues.
A rejected result keeps the last draft and its issues; every round's prompt and answer is in data/logs/calls.jsonl.
Two plain LangChain chains: no loop with tools is needed here, the evidence is already collected by stage 5.
Not run with a real model yet (needs LLM_API_KEY).
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.language_models import BaseChatModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.chunk_search import article_context
from common.news_evidence import verify_claim_evidence
from common.llm import ChatModels, PromptLibrary, structured
from common.safety import UNTRUSTED_NOTE, render_untrusted
from stage6_insight_generation.insight_writer.insight import Insight, Review
from stage5_insight_detection.analyst_loop.facts import Fact

from stage2_company_facts.profile.context import evidence_facts

PROMPTS = PromptLibrary(Path(__file__).with_name('prompts'))


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


class InsightWriter:
    def __init__(self, model: BaseChatModel | None = None, revisions: int = 2):
        model = model or ChatModels().agent(temperature=0.3)
        self.writer = PROMPTS.chat('write') | structured(model, Insight)
        self.checker = PROMPTS.chat('check') | structured(model, Review)
        self.revisions = revisions

    def write(self, company_id: str, role: str, decision: str, evidence: Evidence) -> tuple[Insight, Review]:
        """Returns the final insight and the review of the last draft that was checked."""
        inputs = {'company_id': company_id, 'role': role, 'decision': decision, 'evidence': evidence.render()}
        insight = self.writer.invoke(inputs | {'feedback': 'нет'})
        insight.origin = evidence.origin
        review = self._check(insight, inputs, evidence)
        for _ in range(self.revisions):  # every rewrite gets the checker's issues in full
            if review.ok:
                break
            insight = self.writer.invoke(inputs | {'feedback': '\n'.join(f'- {i}' for i in review.issues)})
            insight.origin = evidence.origin
            review = self._check(insight, inputs, evidence)
        return insight, review

    def _check(self, insight: Insight, inputs: dict, evidence: Evidence) -> Review:
        # Verbatim quotes get offsets for the card; a missing or paraphrased quote only marks the claim «не подтверждено»
        # (as a blocking issue it rejected 4 of 5 insights on U_68622, 22.10: «(dp, 22.09.2026)» appended, paraphrases).
        verify_claim_evidence(insight, evidence.news, evidence.bank, origin=evidence.origin)
        return self.checker.invoke({'role': inputs['role'], 'evidence': inputs['evidence'],
                                    'insight': insight.model_dump_json(indent=1)})
