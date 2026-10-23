"""Approach B': HyDE — for a company facet the cheap LLM writes headlines that WOULD affect such a business; they become
extra queries for facet retrieval ("what to look for"). Cached per facet text. Not run yet (needs LLM_API_KEY)."""
import sys
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.llm import ChatModels, JsonlCache, PromptLibrary, structured

PROMPTS = PromptLibrary(Path(__file__).with_name('prompts'))


class Headlines(BaseModel):
    headlines: list[str] = Field(description='5 коротких реалистичных заголовков')


class HydeQueryWriter:
    def __init__(self, model: BaseChatModel | None = None, cache: JsonlCache | None = None):
        model = model or ChatModels().cheap(temperature=0.7, max_tokens=400)
        self.chain = PROMPTS.chat('hyde') | structured(model, Headlines)
        self.cache = cache or JsonlCache('hyde')

    def queries(self, facet: str) -> list[str]:
        return self.cache.get_or_compute(facet, lambda: self.chain.invoke({'facet': facet}).headlines)
