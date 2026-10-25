"""Insight judge calibrated against the human labeller (insight_judge_calibration): the same four axes — relevance,
significance, clarity, grounding — and the overall winner.
- `judge_pair`: A/B/= per axis. Each pair is judged in both orders and the verdicts are merged: a verdict stands only when
  both orders agree, otherwise «=» (an LLM judge prefers a position; `consistency` shows how often it did).
- `score_card`: 1–5 per axis for one card, the production gate (a pair is not available for a single insight).
Calibration (kappa per axis against the human, Bradley–Terry) — calibrate.py. The judge sees the card text only."""
import asyncio
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.llm import PromptLibrary, structured
from evals.insight_judge.pairs import AXES

PROMPTS = PromptLibrary(Path(__file__).with_name('prompts'))
Verdict = Literal['A', 'B', '=']
FLIP = {'A': 'B', 'B': 'A', '=': '='}


class PairVerdict(BaseModel):
    relevance: Verdict
    significance: Verdict
    clarity: Verdict
    grounding: Verdict
    winner: Verdict
    confidence: int = Field(ge=1, le=3)
    factual_error: bool = False
    note: str = ''


class CardScore(BaseModel):
    relevance: int = Field(ge=1, le=5)
    significance: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    grounding: int = Field(ge=1, le=5)
    factual_error: bool = False
    note: str = ''

    @property
    def overall(self) -> float:
        return sum(getattr(self, a) for a in AXES) / len(AXES)


def merge(first: PairVerdict, swapped: PairVerdict) -> dict:
    """`first` judged (A, B), `swapped` judged (B, A): its verdicts are flipped back before comparing."""
    out, agree = {}, {}
    for key in (*AXES, 'winner'):
        x, y = getattr(first, key), FLIP[getattr(swapped, key)]
        out[key], agree[key] = (x if x == y else '='), x == y
    return dict(out, confidence=min(first.confidence, swapped.confidence), factual_error=first.factual_error or swapped.factual_error,
                agree=agree, raw=[first.model_dump(), swapped.model_dump()])  # raw[1] is in the swapped order


class InsightJudge:
    def __init__(self, model):
        self.pairwise = structured(model, PairVerdict)
        self.pointwise = structured(model, CardScore)

    async def ajudge_pair(self, exposure: str, a: str, b: str) -> dict:
        ask = lambda x, y: self.pairwise.ainvoke(PROMPTS.render('pairwise', exposure=exposure, a=x, b=y))
        first, swapped = await asyncio.gather(ask(a, b), ask(b, a))
        return merge(first, swapped)

    async def ascore_card(self, exposure: str, card: str) -> CardScore:
        return await self.pointwise.ainvoke(PROMPTS.render('pointwise', exposure=exposure, card=card))
