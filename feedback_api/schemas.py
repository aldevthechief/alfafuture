"""API contract of the insight discussion: reactions, questions, answers grounded in the insight's evidence.
Ported from the maxon_dialogue branch (aldev) to insight contract v2 and extended with like / dislike / emoji reactions."""
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from common.evidence import EvidenceSnapshot
from stage6_insight_generation.insight_writer.insight_v2 import Insight


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)


class Kind(StrEnum):
    like = 'like'                    # 👍
    dislike = 'dislike'              # 👎
    emoji = 'emoji'                  # any other emoji: kept, neutral for ranking unless mapped in EMOJI
    useful = 'useful'
    not_relevant = 'not_relevant'
    already_known = 'already_known'
    incorrect = 'incorrect'
    question = 'question'
    comment = 'comment'


# emoji with an unambiguous meaning count as like / dislike; the rest stay neutral
EMOJI = {'👍': 'like', '❤️': 'like', '🔥': 'like', '💡': 'like', '👏': 'like',
         '👎': 'dislike', '😴': 'dislike', '🙄': 'dislike', '🤔': 'emoji', '😮': 'emoji'}
REACTIONS = {Kind.like, Kind.dislike, Kind.emoji, Kind.useful, Kind.not_relevant, Kind.already_known}


class Scope(StrEnum):
    insight = 'insight'   # only this insight's discussion
    future = 'future'     # also the next selections for this (user, company, role)


class RegisterInsight(StrictModel):
    company_id: str = Field(min_length=1, max_length=100)
    registration_key: str | None = Field(default=None, min_length=1, max_length=200)
    news_id: int | None = Field(default=None, ge=0)
    insight: Insight
    evidence: EvidenceSnapshot | None = None
    card_sources: dict[str, str] = Field(default_factory=dict, max_length=80,
                                         description='id -> text of the sources the card shows; used only without `evidence`, '
                                                     'given by the client, not verified by the server')

    @model_validator(mode='after')
    def consistent(self):
        if any(len(k) > 200 or len(v) > 4000 for k, v in self.card_sources.items()):
            raise ValueError('card_sources: ids up to 200 and texts up to 4000 characters')
        if not self.insight.role.strip() or len(self.insight.role) > 100:
            raise ValueError('insight.role must contain 1-100 characters')
        if len(self.insight.model_dump_json()) > 200_000:
            raise ValueError('insight is too large')
        if self.evidence is not None and (self.evidence.company_id, self.evidence.role) != (self.company_id, self.insight.role):
            raise ValueError('Evidence company and role must match the insight')
        return self


class FeedbackInput(StrictModel):
    kind: Kind = Kind.comment
    comment: str = Field(default='', max_length=4000)
    emoji: str = Field(default='', max_length=16)
    scope: Scope | None = Field(default=None, description='default: future for reactions, insight for questions and comments')
    request_id: str = Field(min_length=1, max_length=100,
                            description='client-generated unique id: a retry with the same id does not duplicate feedback')

    @model_validator(mode='after')
    def normalise(self):
        if self.kind == Kind.emoji:
            if not self.emoji:
                raise ValueError('emoji is required for kind=emoji')
            self.kind = Kind(EMOJI.get(self.emoji, 'emoji'))
        if self.kind not in REACTIONS and not self.comment:
            raise ValueError('A comment is required for this feedback kind')
        if self.scope is None:
            self.scope = Scope.future if self.kind in REACTIONS else Scope.insight
        return self


class Question(StrictModel):
    question: str = Field(min_length=1, max_length=2000)
    request_id: str = Field(min_length=1, max_length=100)
    remember: bool = Field(False, description='allow using preferences stated in the question in future selections')


class ContextNote(StrictModel):
    kind: Literal['preference', 'constraint', 'known_information', 'factual_dispute']
    text: str = Field(min_length=1, max_length=1000)
    polarity: Literal['prefer', 'avoid', 'unspecified'] = 'unspecified'


class Analysis(StrictModel):
    intent: Kind
    response: str = Field(min_length=1, max_length=4000)
    requested_action: Literal['none', 'rerank', 'recheck', 'rewrite', 'answer', 'clarify']
    needs_clarification: bool = False
    clarification_question: str = Field(default='', max_length=1000)
    context_notes: list[ContextNote] = Field(default_factory=list, max_length=8)
    source_ids: list[str] = Field(default_factory=list, max_length=30)

    @model_validator(mode='after')
    def require_question(self):
        if self.needs_clarification and not self.clarification_question.strip():
            raise ValueError('A clarification question is required')
        return self


class RetrievedSource(StrictModel):
    id: str
    origin: Literal['news_pool', 'company_data', 'bank_analytics']
    text: str = Field(max_length=6000)
    retrieved_at: str
    company_id: str | None = None


class ToolTrace(StrictModel):
    name: str
    arguments: dict[str, str]
    status: Literal['ok', 'unavailable', 'budget_exceeded']
    source_ids: list[str] = Field(default_factory=list)


class AnalysisDetails(Analysis):
    evidence_status: Literal['captured', 'missing'] = 'missing'
    retrieved_sources: list[RetrievedSource] = Field(default_factory=list)
    tool_calls: list[ToolTrace] = Field(default_factory=list)


class InsightRecord(StrictModel):
    id: str
    user_id: str
    company_id: str
    news_id: int | None
    insight: Insight
    evidence: EvidenceSnapshot | None = None
    card_sources: dict[str, str] = Field(default_factory=dict)
    created_at: str


class FeedbackRecord(StrictModel):
    id: str
    insight_id: str
    kind: Kind
    comment: str
    emoji: str = ''
    scope: Scope
    request_id: str
    analysis: AnalysisDetails | None = None
    created_at: str


class ReactionSummary(StrictModel):
    insight_id: str
    counts: dict[str, int]
    mine: list[str] = Field(description='kinds of this user\'s reactions on the insight')
