"""Approach D: batch news scoring where the model returns only scores, never facts.

News go to the cheap model numbered, 25 per call: `[3] <untrusted_item>title (source, date)</untrusted_item>`. The answer is
a typed list `NewsScore(index, category, sentiment, importance 1-5, affected_roles)` relative to one company card.
Title, date, source and url are restored by the code from the original row by index, so the model cannot invent or
swap a source. Indices the model skipped are asked again (up to 3 tries), the first score is never overwritten, and a
news item that is still unscored is dropped with a trace record instead of being guessed. A missing or unparsable date
stays None and fails the freshness filter. Idea after FinRobot (engine/analysis/news_classifier.py); our own code.
"""
import json, re, sys
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.llm import ChatModels, PromptLibrary
from common.safety import UNTRUSTED_NOTE, render_untrusted, suspicious
from common.trace import AgentTrace
from common.chunk_search import article_context

PROMPTS = PromptLibrary(Path(__file__).with_name('prompts'))
Category = Literal['ставка_и_кредит', 'курс_и_ВЭД', 'налоги_и_регулирование', 'отрасль', 'контрагенты_и_цепочки', 'регион',
                   'санкции', 'сырьё_и_цены', 'прочее']
Role = Literal['owner', 'ceo', 'cfo', 'accountant', 'treasury']


class NewsScore(BaseModel):
    index: int = Field(description='номер новости из входа')
    category: Category
    sentiment: Literal['positive', 'negative', 'neutral'] = Field(description='для этой компании, не для рынка')
    importance: int = Field(ge=1, le=5, description='1 шум ... 5 само по себе меняет положение компании (редко)')
    affected_roles: list[Role] = []


class NewsScores(BaseModel):
    """Scores for the numbered news, one item per index."""
    model_config = ConfigDict(title='scores')
    items: list[NewsScore]


def parse_date(value) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


class NewsClassifier:
    def __init__(self, model: BaseChatModel | None = None, batch: int = 25, tries: int = 3, trace: AgentTrace | None = None):
        model = model or ChatModels().cheap(max_tokens=1500)
        self.chain = PROMPTS.chat('classify') | model.with_structured_output(NewsScores, method='function_calling',
                                                                             include_raw=True)
        self.batch, self.tries, self.trace = batch, tries, trace or AgentTrace(folder=None)

    def classify(self, news: list[dict], company_card: str) -> list[dict]:
        """news: {'id', 'title', 'date', 'source', 'url', ...}. Returns the scored rows in input order (unscored dropped)."""
        scored: dict[int, NewsScore] = {}
        for start in range(0, len(news), self.batch):
            idx = list(range(start, min(start + self.batch, len(news))))
            for attempt in range(1, self.tries + 1):
                todo = [i for i in idx if i not in scored]
                if not todo:
                    break
                retry = f' Попытка {attempt}: верни валидные оценки для всех указанных индексов.' if attempt > 1 else ''
                answer = self.chain.invoke({'card': company_card, 'note': UNTRUSTED_NOTE + retry,
                                            'items': '\n'.join(self._item(i, news[i]) for i in todo)})
                for s in scores_of(answer):
                    if s.index in todo:
                        scored.setdefault(s.index, s)  # first score wins, a retry never overwrites it
                self.trace.log('classify', attempt=attempt, asked=len(todo), got=sum(i in scored for i in todo))
        out = []
        for i, row in enumerate(news):
            if i not in scored:
                self.trace.log('classify_dropped', news_id=row.get('id'), reason='модель не вернула оценку за 3 попытки')
                continue
            s = scored[i]
            out.append({'id': row.get('id'), 'title': row.get('title'), 'source': row.get('source'), 'url': row.get('url'),
                        'date': parse_date(row.get('date')), 'category': s.category, 'sentiment': s.sentiment,
                        'importance': s.importance, 'affected_roles': s.affected_roles})
        return out

    def _item(self, i: int, row: dict) -> str:
        if suspicious(row.get('title', '')) or suspicious(row.get('text', '')):
            self.trace.log('injection_suspected', news_id=row.get('id'))
        return render_untrusted(str(i), row.get('title', ''), article_context(row, 2400),
                                meta={'источник': row.get('source'), 'дата': row.get('date')}, body_limit=2400)


def scores_of(answer: dict) -> list[NewsScore]:
    """Tool call if the model made one, else the JSON objects of its text answer (some proxied models ignore tool_choice
    and print ```json```); every object is validated alone, so a truncated or partly broken list still gives its valid items."""
    if answer.get('parsed') is not None:
        return answer['parsed'].items
    out = []
    for chunk in re.findall(r'\{[^{}]*\}', str(answer['raw'].content)):
        try:
            out.append(NewsScore.model_validate(json.loads(chunk)))
        except (ValueError, ValidationError):
            continue
    return out


def fresh(rows: list[dict], today: date, days: int = 14) -> list[dict]:
    """Freshness filter: a row without a date never passes."""
    return [r for r in rows if r['date'] is not None and 0 <= (today - r['date']).days <= days]
