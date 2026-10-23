"""Output contract of the pipeline: the insight JSON shown in the Alfa-Business UI.

Layers are separate fields, as the case requires: facts from news / the bank's view / AI conclusions,
and every claim carries the ids of its sources (news:<id>, bank:<report>:<page>, fact:<n>).
"""
import json
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from pydantic.json_schema import SkipJsonSchema


class SourceExcerpt(BaseModel):
    source: str = Field(description='news:<id> статьи; для события origin=bank — id исходного фрагмента bank:…')
    quote: str = Field(description='дословный непрерывный фрагмент текста статьи, подтверждающий факт')
    start: int | None = None
    end: int | None = None
    url: str = ''


class Claim(BaseModel):
    text: str
    sources: list[str] = Field(description='news:<id> | bank:<отчёт>:<стр> | fact:<номер>')
    evidence: list[SourceExcerpt] = Field(default_factory=list, description='Для каждого what_happened — подтверждающие цитаты из статей')
    kind: Literal['fact', 'bank', 'ai', 'hypothesis'] = Field(
        'ai', description='факт из новости или данных / аналитика Банка / вывод AI / гипотеза (допущение, не проверено)')
    unconfirmed: bool = Field(False, description='ставит код аудита: нет источника или неизвестная ссылка')


class Impact(BaseModel):
    """Impact level with fixed criteria (analyst_loop/impact.py); for 'high' a number computed by the code is required."""
    level: Literal['low', 'mid', 'high']
    criteria: str = Field(description='по какому правилу присвоен уровень')
    estimate: str = Field('', description='количественная оценка, обязательна для high: число, формула, допущения')
    sources: list[str] = []


class Insight(BaseModel):
    """Insight for one company and reader role."""
    title: str = Field(description='одна строка: суть для этой компании, не заголовок новости')
    role: str
    what_happened: list[Claim] = Field(description='ФАКТЫ: из новости; для явно указанного события-тезиса Банка — из исходного bank:…')
    why_it_matters: list[Claim] = Field(description='ВЫВОД AI: связь с конкретными фактами о компании')
    risks: list[Claim] = []
    opportunities: list[Claim] = []
    actions: list[str] = Field(description='1-3 конкретных шага для роли читателя')
    bank_view: list[Claim] = Field([], description='ПОЗИЦИЯ БАНКА: только из аналитики Банка, с отчётом и датой')
    ai_conclusion: str = Field(description='итоговый вывод AI одним абзацем')
    confidence: Literal['high', 'medium', 'low']
    bank_product: str = Field('', description='продукт Банка, который помогает с действием, или пусто')
    impact: Impact | None = None
    hypotheses: list[Claim] = Field([], description='ГИПОТЕЗЫ: допущения, которые не проверены данными')
    not_bank_position: bool = Field(False, description='аналитики Банка по теме нет: вывод AI, не официальная позиция Банка')
    disclaimer: str = 'Не является инвестиционной рекомендацией.'
    origin: SkipJsonSchema[Literal['news', 'bank']] = 'news'  # assigned by code, not chosen by the model

    # GLM answers in shapes close to the schema: an action as a claim object (when asked for its «because»), the nested
    # impact as a JSON string; both are normalised here instead of failing the whole insight (U_68622, 23.10: 12 of 17 drafts)
    @field_validator('actions', mode='before')
    @classmethod
    def _action_text(cls, v):
        return [a.get('text', '') if isinstance(a, dict) else a for a in v] if isinstance(v, list) else v

    @field_validator('hypotheses', mode='before')
    @classmethod
    def _hypothesis_claims(cls, v):  # GLM sends a hypothesis as a bare string when its sources are not obvious
        return [{'text': h, 'sources': [], 'kind': 'hypothesis'} if isinstance(h, str) else h for h in v] if isinstance(v, list) else v

    @field_validator('impact', mode='before')
    @classmethod
    def _impact_json(cls, v):
        return json.loads(v) if isinstance(v, str) and v.strip().startswith('{') else v


class Review(BaseModel):
    ok: bool
    issues: list[str] = []
