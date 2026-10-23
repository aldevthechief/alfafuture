"""Task 4: requests for missing data go back into the shared agent loop instead of being guessed.

Any step can raise a `DataRequest`: the code (gap rules of grounds.py: no bank view on a macro channel, conflicting
numbers, no link to the company) or the model (the `missing` list of the assessment). The broker dispatches each
request to the tool that can answer it, at most once per (kind, query) and within a budget. Answers change the dossier
(new bank fragments, confirmations, company facts), and the loop recomputes consequences and grounds. A request no
tool could answer stays open: it is returned to the caller (pipeline.py puts it into the result) so the next run or a
human can close it, and the verdict names it as the reason for a refusal when it was blocking.
"""
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from common.trace import AgentTrace

Kind = Literal['bank_analytics', 'news', 'company_facts', 'computation']


class DataRequest(BaseModel):
    kind: Kind = Field(description='bank_analytics — позиция Банка; news — подтверждение или свежие данные из пула новостей; '
                                   'company_facts — факт о компании; computation — расчёт, которого нет')
    query: str = Field(description='что именно найти, коротко')
    reason: str = Field('', description='зачем это нужно для вывода')
    blocking: bool = Field(False, description='без ответа вывод нельзя показывать')
    origin: str = 'code'


@dataclass
class Answer:
    request: DataRequest
    ok: bool
    summary: str
    payload: Any = None


Handler = Callable[[DataRequest], Answer]


@dataclass
class RequestBroker:
    handlers: dict[str, Handler]
    trace: AgentTrace
    budget: int = 6
    asked: set = field(default_factory=set)
    open: list[DataRequest] = field(default_factory=list)

    def dispatch(self, requests: list[DataRequest]) -> list[Answer]:
        answers = []
        for r in requests:
            key = (r.kind, r.query.strip().lower())
            if key in self.asked:
                for pending in self.open:
                    if (pending.kind, pending.query.strip().lower()) == key and r.blocking:
                        pending.blocking = True
                continue
            self.asked.add(key)
            if self.budget <= 0 or r.kind not in self.handlers:
                self.open.append(r)
                self.trace.log('request_open', **r.model_dump(), why='нет бюджета' if self.budget <= 0 else 'нет инструмента')
                continue
            self.budget -= 1
            try:
                a = self.handlers[r.kind](r)
            except Exception as e:  # a failing tool leaves the request open, never crashes the loop
                a = Answer(r, False, f'ошибка инструмента: {type(e).__name__}: {e}')
            if not a.ok:
                self.open.append(r)
            self.trace.log('request', **r.model_dump(), ok=a.ok, answer=a.summary[:300])
            answers.append(a)
        return answers

    def blocking_open(self) -> list[DataRequest]:
        return [r for r in self.open if r.blocking]
