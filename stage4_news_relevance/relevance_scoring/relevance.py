"""Approach C: the cheap LLM scores a (news, role, company facts) triple — only for the stage-3 shortlist
(~50-100 news per company). Not run yet (needs LLM_API_KEY)."""
import sys
from pathlib import Path
from typing import Literal

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.chunk_search import article_context
from common.llm import ChatModels, PromptLibrary, structured

PROMPTS = PromptLibrary(Path(__file__).with_name('prompts'))


class Relevance(BaseModel):
    score: int = Field(ge=0, le=3)
    link: str = Field('', description='через какой факт компании связана новость')
    channel: str = ''
    direction: Literal['+', '-', '?'] = '?'
    role_fit: int = Field(ge=0, le=2)
    why_not: str = Field('', description='если score 0 — почему')


class RelevanceScorer:
    def __init__(self, model: BaseChatModel | None = None):
        self.chain = PROMPTS.chat('relevance') | structured(model or ChatModels().cheap(max_tokens=300), Relevance)

    def score(self, news: dict, facts_text: str, role: str) -> Relevance:
        return self.chain.invoke({'role': role, 'facts': facts_text, 'title': news['title'], 'text': article_context(news)})
