"""One unifying news for a multi-report event: the cheap LLM reads the reports' titles and writes one title + summary
(e.g. «13 октября массовые закрытия аэропортов: Пенза, Саратов, Ярославль»). Lazy: only events with >= min_reports
reports and only when the pipeline asks (after candidate generation). Cached per event (sorted news ids).
Not run yet (needs LLM_API_KEY)."""
import hashlib
import sys
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.llm import ChatModels, JsonlCache, PromptLibrary, structured
from common.news_corpus import article_fingerprint

PROMPTS = PromptLibrary(Path(__file__).with_name('prompts'))


class EventSummary(BaseModel):
    title: str = Field(description='заголовок события одной фразой: что произошло, где, когда')
    summary: str = Field(description='2–3 предложения: суть, масштаб, участники; только то, что есть в сообщениях')


class EventSummarizer:
    def __init__(self, model: BaseChatModel | None = None, cache: JsonlCache | None = None, min_reports: int = 5):
        self.chain = PROMPTS.chat('event_summary') | structured(model or ChatModels().cheap(), EventSummary)
        self.cache = cache or JsonlCache('event_summary')
        self.min_reports = min_reports

    def summarize(self, reports: list[dict]) -> dict | None:
        """reports: rows of news.parquet (id, date, source, title, text) of one event. None for small events:
        their first business-source title already stands for the event."""
        if len(reports) < self.min_reports:
            return None
        signature = PROMPTS.text('event_summary') + ' '.join(sorted(article_fingerprint(r) for r in reports))
        key = hashlib.sha256(signature.encode()).hexdigest()
        ordered = sorted(reports, key=lambda r: (str(r['date']), r['source'], r['title']))
        lines = '\n'.join(f"- {str(r['date'])[:10]} {r['source']}: {r['title']}" for r in ordered)
        return self.cache.get_or_compute(key, lambda: self.chain.invoke({'reports': lines}).model_dump())
