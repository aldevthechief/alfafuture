"""Answers to a user's question or comment on an insight, grounded in its frozen evidence and bounded read-only lookups
(ported from the maxon_dialogue branch, aldev). Bare reactions never call a model; every cited id is checked by code."""
from pathlib import Path
import json
import re

from feedback_api.cache import RedisAnswerCache, answer_key
from feedback_api.research import ResearchTools
from feedback_api.schemas import Analysis, AnalysisDetails, FeedbackRecord, InsightRecord, Kind

RULES = {
    Kind.like: ('Спасибо! Будем чаще показывать похожие инсайты.', 'rerank'),
    Kind.useful: ('Спасибо, отметили инсайт как полезный.', 'rerank'),
    Kind.dislike: ('Поняли, похожие инсайты будем показывать реже.', 'rerank'),
    Kind.not_relevant: ('Отметили как нерелевантный: эту новость больше не покажем, похожие — реже.', 'rerank'),
    Kind.already_known: ('Отметили, что это вам уже известно: эту новость больше не покажем.', 'rerank'),
    Kind.emoji: ('Реакция сохранена.', 'none'),
}


class InvalidAnalysis(ValueError):
    pass


class FeedbackAnalyzer:
    def __init__(self, model=None, research_factory=None, max_steps: int = 4, cache=None):
        self.model, self.research_factory, self.max_steps = model, research_factory or ResearchTools, max_steps
        self.cache = cache

    def analyze(self, insight: InsightRecord, feedback: FeedbackRecord, history: list[FeedbackRecord]) -> AnalysisDetails:
        status = 'captured' if insight.evidence is not None else 'missing'
        if not feedback.comment and feedback.kind in RULES:
            response, action = RULES[feedback.kind]
            return AnalysisDetails(intent=feedback.kind, response=response, requested_action=action, evidence_status=status)

        # lazy imports: storing feedback and starting the API never need an LLM key
        from common.llm import ChatModels, PromptLibrary
        from feedback_api.loop import DiscussionLoop
        from langchain_core.utils.function_calling import convert_to_openai_tool

        research = self.research_factory(insight)
        tools = research.as_tools()
        model = self.model if self.model is not None else \
            ChatModels().agent(max_tokens=3000).model_copy(update={'cache': False, 'reasoning_effort': 'low'})
        cache = self.cache if self.cache is not None else RedisAnswerCache.from_env()
        history = [f for f in history if f.id != feedback.id][-8:]
        # Transport UUIDs/timestamps are not semantic context. Stable aliases allow the same context to hit
        # across retries/re-registrations; API responses always get this conversation's real source IDs back.
        aliases = {f'user:{f.id}': f'user:history:{i}' for i, f in enumerate(history)}
        aliases[f'user:{feedback.id}'] = 'user:current'
        restore = {value: key for key, value in aliases.items()}
        allowed = set(restore)
        if insight.evidence is not None:
            allowed |= set(insight.evidence.sources())
        elif insight.card_sources:
            allowed |= set(insight.card_sources)
        else:  # legacy: references of the card itself, not re-read sources
            for block in ('what_happened', 'why_it_matters', 'risks', 'opportunities', 'bank_view', 'hypotheses'):
                for claim in getattr(insight.insight, block):
                    allowed.update(claim.sources)

        def validate(result):
            if not set(result.source_ids) <= allowed | set(research.sources):
                raise InvalidAnalysis('Unknown source IDs')

        def context_message(f):
            data = f.model_dump(mode='json', exclude={'id', 'insight_id', 'request_id', 'created_at'})
            data['date'] = f.created_at[:10]  # relative questions ("today") must not cross calendar days
            return _replace_refs(data, aliases)

        current = context_message(feedback)
        current.pop('analysis', None)
        current['id'] = 'current'
        prompts = PromptLibrary(Path(__file__).with_name('prompts'))
        messages = prompts.chat('analyze').format_messages(
            company_id=insight.company_id, role=insight.insight.role, insight=insight.insight.model_dump_json(exclude={'review'}),
            evidence=insight.evidence.render() if insight.evidence is not None else
            ('Полного снимка нет. Тексты источников, которые показывает карточка (переданы клиентом, сервером не проверены):\n'
             + '\n'.join(f'{k}: {v}' for k, v in insight.card_sources.items())) if insight.card_sources else
            'Снимок исходных материалов отсутствует. Ссылки внутри инсайта не являются проверенными первоисточниками.',
            feedback=json.dumps(current, ensure_ascii=False),
            history='\n'.join(f'{aliases[f"user:{f.id}"]} ({f.created_at[:10]}) {f.kind}: '
                              f'{_replace_refs(f.comment[:1000], aliases)}' +
                              (f'\nОтвет: {_replace_refs(f.analysis.response[:1000], aliases)}' if f.analysis else '')
                              for f in history) or '(нет)')
        key = None
        if cache is not None:
            key = answer_key(dict(user_id=insight.user_id, company_id=insight.company_id,
                role=insight.insight.role, news_id=insight.news_id,
                insight=insight.insight.model_dump(mode='json'),
                evidence=insight.evidence.model_dump(mode='json') if insight.evidence else None,
                card_sources=insight.card_sources, feedback=context_message(feedback),
                history=[context_message(f) for f in history],
                messages=[m.content for m in messages], model=model._get_llm_string(),
                endpoint=str(getattr(model, 'openai_api_base', '') or ''), max_steps=self.max_steps,
                tools=[convert_to_openai_tool(t) for t in tools], schema=AnalysisDetails.model_json_schema()))
            cached = cache.lookup(key)
            if cached is not None:
                if _replay_research(cached, research, tools):
                    try:
                        validate(cached)
                        if cached.evidence_status != status:
                            raise InvalidAnalysis('Evidence status changed')
                    except InvalidAnalysis:
                        pass
                    else:
                        return AnalysisDetails.model_validate(_replace_refs(cached.model_dump(), restore))
                # A stale cache replay must not spend the fresh answer's research budget.
                research = self.research_factory(insight)
                tools = research.as_tools()
        loop = DiscussionLoop(model, tools, Analysis, messages[0].content, max_steps=self.max_steps,
                              validate_answer=validate, nudge='Вызови нужный инструмент или финальный инструмент Analysis.')
        result = loop.run(messages[1].content)
        if result is None:
            raise InvalidAnalysis('Discussion step budget exhausted without a final answer')
        validate(result)
        answer = AnalysisDetails(**result.model_dump(), evidence_status=status,
                                 retrieved_sources=list(research.sources.values()), tool_calls=research.trace)
        if key is not None and all(t.status == 'ok' for t in research.trace):
            cache.put(key, answer)
        return AnalysisDetails.model_validate(_replace_refs(answer.model_dump(), restore))


def _replace_refs(value, refs):
    """Simultaneous replacement, so aliases cannot rewrite each other (history:1 vs history:10)."""
    if isinstance(value, str) and refs:
        pattern = '|'.join(re.escape(k) for k in sorted(refs, key=len, reverse=True))
        return re.sub(pattern + r'(?![\w:-])', lambda match: refs[match.group()], value)
    if isinstance(value, dict):
        return {key: _replace_refs(item, refs) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_refs(item, refs) for item in value]
    return value


def _replay_research(cached, research, tools):
    """A lookup-dependent answer is reusable only if the same read-only lookups still deliver the same data."""
    by_name = {tool.name: tool for tool in tools}
    try:
        for call in cached.tool_calls:
            if call.status != 'ok' or call.name not in by_name:
                return False
            by_name[call.name].invoke(call.arguments)
        actual = [s.model_dump(exclude={'retrieved_at'}) for s in research.sources.values()]
        expected = [s.model_dump(exclude={'retrieved_at'}) for s in cached.retrieved_sources]
        return research.trace == cached.tool_calls and actual == expected
    except Exception:
        return False
