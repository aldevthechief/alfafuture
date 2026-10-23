from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class NewsChunk:
    news_id: str
    chunk_index: int
    text: str
    start: int
    end: int


def split_into_chunks(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[tuple[int, int, str]]:
    """Split article text into overlapping chunks with source offsets."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size")

    if not isinstance(text, str) or not text.strip():
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Prefer ending at a word boundary.
        if end < len(text):
            boundary = text.rfind(" ", start + 1, end)
            if boundary > start + chunk_size // 2:
                end = boundary

        fragment = text[start:end]

        if fragment.strip():
            chunks.append((start, end, fragment))

        if end == len(text):
            break

        # Move forward while retaining part of the previous chunk.
        start = max(start + 1, end - overlap)

    return chunks


def build_news_chunks(
    news: pd.DataFrame,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[NewsChunk]:
    """Build chunks linked to their original news articles."""
    required = {"id", "text"}
    if not required.issubset(news.columns):
        raise ValueError("News must contain id and text columns")

    result = []

    for row in news.itertuples(index=False):
        article_text = row.text

        if not isinstance(article_text, str):
            continue

        fragments = split_into_chunks(
            article_text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for index, (start, end, fragment) in enumerate(fragments):
            result.append(
                NewsChunk(
                    news_id=str(row.id),
                    chunk_index=index,
                    text=fragment,
                    start=start,
                    end=end,
                )
            )

    return result
