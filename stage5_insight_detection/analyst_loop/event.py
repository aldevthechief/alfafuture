"""Event card: one news item (or a merged story) as the loop sees it. Title, date, source and url come from the parsed
row only; topics, figures and the direction of the indicator are tagged by the same code as bank fragments."""
from common.chunk_search import article_context
from dataclasses import asdict, dataclass, field
from datetime import date

from stage5_insight_detection.bank_analytics.library import Figure, direction, figures_of, sentences, topics_of


@dataclass
class EventCard:
    id: str  # news:<id>
    title: str
    text: str
    date: date | None
    source: str
    url: str = ''
    sources: list[str] = field(default_factory=list)  # all outlets of a merged story (common/dedup.py)
    matched_fragments: list[dict] = field(default_factory=list)
    category: str = ''  # classifier scores (stage 4 news_classifier), optional
    sentiment: str = ''
    importance: int = 3
    plausibility: float | None = None
    topics: list[str] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    moves: dict[str, str] = field(default_factory=dict)  # topic -> up / down / flat, from the title and lead
    origin: str = 'news'  # bank events cite their real fragment, never a synthetic news id

    @classmethod
    def from_thesis(cls, thesis, importance: int = 4) -> 'EventCard':
        return cls(id=thesis.fragment, title=thesis.text(), text=thesis.quote,
                   date=date.fromisoformat(thesis.report_date), source=thesis.report,
                   sources=[thesis.report], importance=importance, plausibility=1.0,
                   topics=[thesis.topic], moves={thesis.topic: thesis.move}, origin='bank')

    @classmethod
    def from_news(cls, row: dict, scores: dict | None = None, plausibility: float | None = None) -> 'EventCard':
        scores = scores or {}
        title, text = str(row.get('title') or ''), str(row.get('text') or '')
        lead = article_context(row) if row.get('matched_fragments') else ' '.join(sentences(text)[:3])
        moves = {}
        for s in [title, *sentences(lead)]:
            d = direction(s)
            for t in topics_of(s):
                if d and t not in moves:
                    moves[t] = d
        when = row.get('date')
        if when is not None and not isinstance(when, date):
            try:
                when = date.fromisoformat(str(when)[:10])
            except ValueError:
                when = None  # unparsable date stays unknown, never "today"
        return cls(id=f'news:{row["id"]}', title=title, text=text, date=when, source=str(row.get('source') or ''),
                   url=str(row.get('url') or ''), sources=list(row.get('sources') or ([row['source']] if row.get('source') else [])),
                   matched_fragments=row.get('matched_fragments', []),
                   category=scores.get('category', ''), sentiment=scores.get('sentiment', ''),
                   importance=int(scores.get('importance', 3)), plausibility=plausibility,
                   topics=topics_of(f'{title}. {lead}'), figures=figures_of(f'{title}. {lead}'), moves=moves)

    @property
    def news_id(self) -> str | None:
        return self.id.split(':', 1)[1] if self.origin == 'news' else None

    def query(self) -> str:
        if self.origin == 'bank':
            return f'{self.title}. {self.text[:2400]}'
        return f'{self.title}. {self.title}. {article_context(dict(id=self.news_id, text=self.text, matched_fragments=self.matched_fragments), 2400)}'

    def dict(self) -> dict:
        d = asdict(self)
        d['date'] = self.date.isoformat() if self.date else None
        d['text'] = self.text[:600]
        return d
