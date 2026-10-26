"""Step 5: does a candidate news carry a business insight for this company and role?

A `ToolLoop` (LangGraph ReAct sub-loop) with tools over the news pool (same-event reports, follow-ups, free search),
the bank analytics and the company facts. It ends with `decide`, a structured `InsightDecision` that stage 6 turns
into the insight text. Not run with a real model yet (needs LLM_API_KEY and the OCR output of bank_analytics/).
"""
import json, sys
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.exceptions import OutputParserException
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic.json_schema import SkipJsonSchema

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.chunk_search import article_context
from common.llm import ChatModels, PromptLibrary, StructuredOutputError, ToolLoop
from stage5_insight_detection.bank_analytics.search import BankAnalytics
from stage5_insight_detection.bank_analytics.tools import BankTools
from stage5_insight_detection.news_enrichment.related_news import RelatedNews
from stage5_insight_detection.insight_judge.calculations import JudgeCalculations

from stage2_company_facts.profile.context import judge_facts

PROMPTS = PromptLibrary(Path(__file__).with_name('prompts'))


class InsightDecision(BaseModel):
    """Final decision: is the news an insight for this company and role."""
    model_config = ConfigDict(title='decide')
    is_insight: bool
    score: int = Field(ge=0, le=3, description='0 — нет; 1 — фон; 2 — инсайт через факт; 3 — сильный инсайт, нужно действие')
    company_link: str = Field(description='через какой факт о компании новость её касается')
    mechanism: str = Field(description='канал, механизм и направление влияния')
    bank_view: str = Field('', description='позиция Банка с отчётом, датой и страницей, или пусто')
    related_news_ids: list[int] = Field([], description='id новостей, которые дополняют или подтверждают')
    plausibility_note: str = Field('', description='подтверждена ли новость, есть ли противоречия')
    missing: str = Field('', description='чего не хватает для уверенного вывода')
    bank_product: str = Field('', description='продукт Альфа-Банка, который помогает компании с этим событием, и почему; пусто, если связь натянута')
    # Filled by code after decide; not part of the model's output schema or the writer's decision prompt.
    calculation_facts: SkipJsonSchema[list[dict]] = Field(default_factory=list, exclude=True)
    calculation_bank: SkipJsonSchema[list[dict]] = Field(default_factory=list, exclude=True)
    bank_evidence: SkipJsonSchema[list[dict]] = Field(default_factory=list, exclude=True)


class JudgeTools:
    """Tools bound to one (news, company) pair, so the model never passes ids it could get wrong."""

    def __init__(self, news_id: int, facts: list[dict], related: RelatedNews, bank: BankAnalytics | None,
                 news: dict | None = None, company_id: str | None = None):
        self.news_id, self.facts, self.related, self.bank = news_id, facts, related, bank
        self.calculations = JudgeCalculations(news or {}, company_id, bank)
        self.bank_tools = BankTools(bank, on_search=self.calculations.observe_bank)

    def as_tools(self) -> list[BaseTool]:
        return [StructuredTool.from_function(f) for f in
                (self.news_text, self.same_event_reports, self.follow_up_news, self.search_news, self.search_bank_analytics,
                 self.company_facts, self.compute)]

    def news_text(self, news_id: int) -> str:
        """Full text of a news by its id (the main news or any id from other tools' results)."""
        if news_id not in self.related.row_of:
            return f'no news with id {news_id}'
        n = self.related.get(news_id)
        return f"id={news_id} {n['date']} {n['source']}: {n['title']}\n{str(n['text'])[:8000]}"

    def same_event_reports(self) -> str:
        """Other reports of the same event in the news pool (other outlets, details, numbers)."""
        return self._news_table(self.related.same_event(self.news_id))

    def follow_up_news(self) -> str:
        """Later news on the same topic that are not the same event: developments, confirmations, reversals."""
        return self._news_table(self.related.follow_ups(self.news_id))

    def search_news(self, query: str) -> str:
        """Free search over the news pool."""
        return self._news_table(self.related.search(query))

    def search_bank_analytics(self, query: str) -> str:
        """The bank's analytics on a topic (parsed reports): up to 3 found fragments with 2 neighbours above and below,
        report, date and pages for every result; stale reports are flagged."""
        return self.bank_tools.search_bank_analytics(query)

    def compute(self, query: str) -> str:
        """Calculate with Python: revenue share in the group, difference between a news indicator and the Bank's,
        a scenario revenue × forecast (%), unit conversion, or a price change on known payments:
        «рост цен на топливо на 10% на платежи поставщикам». Payments cover the available sample,
        not annual costs; pass-through and unchanged volume are assumptions. For bank figures first use search_bank_analytics.
        Returns formula, sources and assumptions, or the reason it cannot calculate. No model or search is called."""
        if not self.calculations.news:
            self.calculations.news = dict(self.related.get(self.news_id))
        return self.calculations.run(query)

    def company_facts(self, category: str = '') -> str:
        """Facts about the company from the bank's data; category prefix filters them (profile, scale, rank, activity,
        geo, ownership, bank, trade, payments, persons, group...)."""
        rows = [f for f in self.facts if f['cat'].startswith(category)]
        return '\n'.join(f'[{f["cat"]}] {f["text"]}' for f in rows) or 'no facts in this category'

    def _news_table(self, df) -> str:
        if not len(df):
            return 'nothing found'
        texts = {r['id']: article_context(r if 'text' in r else self.related.get(r['id']), 2400)
                 for r in df.to_dict('records')}
        return '\n\n'.join(f'id={r.id} {r.date} {r.source}: {r.title}\n{texts[r.id]}' for r in df.itertuples())


class InsightJudge:
    def __init__(self, model: BaseChatModel | None = None, related: RelatedNews | None = None,
                 bank: BankAnalytics | None = None, max_steps: int = 8):
        self.model = model or ChatModels().agent()
        self.related = related or RelatedNews()
        self.bank, self.max_steps = bank, max_steps

    def decide(self, news_id: int, company_id: str, facts: list[dict], role: str, stage4: dict | None = None,
               key_facts: int = 25) -> InsightDecision:
        loop, tools, task = self._setup(news_id, company_id, facts, role, stage4, key_facts)
        for attempt in range(2):
            try:
                if (decision := self._done(loop.run(self._task(task, attempt)), tools)) is not None:
                    return decision
            except (ValidationError, OutputParserException):
                pass
        raise StructuredOutputError(InsightDecision, 2)

    async def adecide(self, news_id: int, company_id: str, facts: list[dict], role: str, stage4: dict | None = None,
                      key_facts: int = 25) -> InsightDecision:
        """decide() on the event loop: the loop's model calls are awaited (no thread waits on the LLM)."""
        loop, tools, task = self._setup(news_id, company_id, facts, role, stage4, key_facts)
        for attempt in range(2):
            try:
                if (decision := self._done(await loop.arun(self._task(task, attempt)), tools)) is not None:
                    return decision
            except (ValidationError, OutputParserException):
                pass
        raise StructuredOutputError(InsightDecision, 2)

    def _setup(self, news_id, company_id, facts, role, stage4, key_facts):
        news = dict(self.related.get(news_id))
        news['matched_fragments'] = (stage4 or {}).get('matched_fragments', [])
        tools = JudgeTools(news_id, facts, self.related, self.bank, news, company_id)
        loop = ToolLoop(self.model, tools.as_tools(), InsightDecision, PROMPTS.text('judge_system'), self.max_steps,
                        nudge='Вызови инструменты или decide.')
        top_facts = judge_facts(facts, key_facts)
        task = PROMPTS.render('judge_task', role=role, company_id=company_id, facts='\n'.join(top_facts), news_id=news_id, date=news['date'], source=news['source'], title=news['title'],
                              text=article_context(news), stage4=json.dumps(stage4 or {}, ensure_ascii=False, default=str),
                              plausibility=self.related.event_plausibility(news_id))
        return loop, tools, task

    @staticmethod
    def _task(task: str, attempt: int) -> str:
        return task if attempt == 0 else task + ('\nПредыдущая попытка не дала корректного decide. '
                                                 'Заверши ответ вызовом decide со всеми обязательными полями.')

    @staticmethod
    def _done(decision, tools) -> 'InsightDecision | None':
        if decision is not None:
            decision.calculation_facts = list(tools.calculations.facts.values())
            decision.calculation_bank = list(tools.calculations.bank_sources.values())
            decision.bank_evidence = list(tools.bank_tools.evidence.values())
        return decision
