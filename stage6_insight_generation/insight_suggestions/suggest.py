"""Suggested questions for a finished insight: 3 short, distinct prompts the reader can click to ask the chat.

One GLM call per insight (low reasoning, structured output — like every call: cache, Langfuse, call log). Context
is the company profile (deterministic depth-1 facts, no confidential comparisons beyond them) and the shortened
insight. Code checks the shape: exactly 3 unique questions, each a short question mark sentence; one fix attempt
in the same dialog, then None (the UI shows no suggestions — never wrong ones).
"""
import re
import sys
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from langchain_core.messages import AIMessage, HumanMessage

from common.llm import PromptLibrary, StructuredOutputError, structured
from common.llm.log import log_tool

PROMPTS = PromptLibrary(Path(__file__).with_name('prompts'))

MAX_LEN = 60  # characters of one question: a chip in the UI that fits one line, not a sentence with clauses


class SuggestedQuestions(BaseModel):
    questions: list[str] = Field(min_length=3, max_length=3,
                                 description='ровно 3 коротких вопроса читателя по этому инсайту')


def render(insight: dict) -> str:
    """The abbreviated insight: every block trimmed, sources and ids out — the model needs content, not metadata."""
    def text(claim):
        return (claim.get('text', '') if isinstance(claim, dict) else str(claim)).strip()

    def block(key, n, limit=220):
        items = [text(c) for c in (insight.get(key) or []) if text(c)][:n]
        return [t[:limit] + ('…' if len(t) > limit else '') for t in items]

    out = [f'Заголовок: {insight.get("title", "")}']
    for key, name, n in (('what_happened', 'Что произошло', 3), ('why_it_matters', 'Почему важно', 2),
                         ('risks', 'Риски', 2), ('opportunities', 'Возможности', 1),
                         ('bank_view', 'Позиция Банка', 2)):
        if items := block(key, n):
            out += [f'## {name}', *[f'- {t}' for t in items]]
    if insight.get('actions'):
        out += ['## Действия', *[f'- {t}' for t in block('actions', 2, 160)]]
    if insight.get('ai_conclusion'):
        out += ['## Итоговый вывод', str(insight['ai_conclusion'])[:300]]
    return '\n'.join(out)


def profile_lines(facts: list[dict], n: int = 20) -> str:
    """The company profile for the prompt: the deterministic depth-1 facts, most informative first."""
    rows = [f['text'] for f in facts if f.get('depth') == 1 and f.get('cat') != 'profile_context'][:n]
    return '\n'.join(f'- {t}' for t in rows) or '(профиль недоступен)'


def problems(questions: list[str]) -> list[str]:
    """What code can check: three distinct short question sentences."""
    out = []
    if len(questions) != 3:
        out.append(f'вопросов {len(questions)}, а нужно ровно 3')
    for q in questions:
        if len(q) > MAX_LEN:
            out.append(f'слишком длинный вопрос ({len(q)} > {MAX_LEN} символов): «{q}»')
        if not q.rstrip().endswith('?'):
            out.append(f'не оканчивается знаком вопроса: «{q}»')
    if len({re.sub(r'\s+', ' ', q.lower()).strip('?!. ') for q in questions}) != len(questions):
        out.append('вопросы повторяют друг друга — нужны три разных угла')
    return out


class QuestionSuggester:
    """suggest -> code check -> if issues, ONE fix in the same dialog; still failing -> None (no chips in the UI)."""
    def __init__(self, model):
        self.chain = structured(model, SuggestedQuestions)

    async def asuggest(self, insight: dict, profile: str) -> list[str] | None:
        dialog = [HumanMessage(PROMPTS.render('suggest', profile=profile, insight=render(insight)))]
        for attempt in range(2):
            try:
                draft = await self.chain.ainvoke(dialog)
            except StructuredOutputError as e:
                log_tool('suggest_no_schema', {'title': insight.get('title')}, str(e))
                return None
            if not (issues := problems(draft.questions)):
                return draft.questions
            log_tool('suggest_issues', {'title': insight.get('title'), 'attempt': attempt}, '\n'.join(issues))
            dialog += [AIMessage(draft.model_dump_json()), HumanMessage(
                'Исправь и верни ответ целиком:\n' + '\n'.join(f'- {i}' for i in issues))]
        return None


if __name__ == '__main__':  # smoke: python -X utf8 -m stage6_insight_generation.insight_suggestions.suggest <results.json> [n]
    import asyncio, json
    from common.llm import ChatModels
    from stage2_company_facts.profile.integration import load_company

    rows = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    if isinstance(rows, dict):
        rows = rows.get('results') or rows.get('rows') or []
    rows = [r for r in rows if r.get('status') == 'ready' and r.get('insight')]
    picked = [rows[int(i)] for i in sys.argv[2:]] if len(sys.argv) > 2 and sys.argv[2].isdigit() else rows[:3]
    model = ChatModels().agent(temperature=0.4, max_tokens=8000).model_copy(update={'reasoning_effort': 'low'})
    suggester = QuestionSuggester(model)
    profiles = {}

    async def one(r):
        company_id = r['insight'].get('company_id')
        if company_id not in profiles:
            profiles[company_id] = profile_lines(load_company(company_id).facts) if company_id else '(профиль недоступен)'
        return r, await suggester.asuggest(r['insight'], profiles[company_id])

    async def all_rows():
        # one loop for all rows: the model's async client binds its connections to the first running loop
        return await asyncio.gather(*(one(r) for r in picked))

    for r, questions in asyncio.run(all_rows()):
        print(f"\n### [{r['insight'].get('company_id')} {r['insight'].get('role', '')}] {r['insight'].get('title', '')}")
        for q in questions or ['(не удалось сгенерировать)']:
            print(f'  - {q}')
