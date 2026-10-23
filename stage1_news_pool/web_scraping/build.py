"""Stage 1: clean articles.jsonl -> news.parquet + news.jsonl."""

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd

from discover import DATA, OFFTOPIC
from sources import SOURCES

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.news_corpus import normalize_url, publish_news, stable_id


CYR = re.compile(r"[а-яё]", re.I)
T2_DROP = re.compile(
    r"происшеств|криминал|incident|accident|кино|музык|шоу|звезд",
    re.I,
)

def normalize_text(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def text_fingerprint(text):
    normalized = re.sub(r"\W+", "", text.lower(), flags=re.UNICODE)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def shingles(text, size=5):
    words = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    if len(words) < size:
        return set(words)

    return {
        tuple(words[i:i + size])
        for i in range(len(words) - size + 1)
    }


def remove_near_duplicates(df, threshold=0.92):
    """Compare articles with similar titles using word shingles."""
    if df.empty:
        return df

    kept = []
    groups = {}

    for idx, row in df.iterrows():
        title = normalize_text(row["title"])
        title_words = set(re.findall(r"\w+", title.lower()))
        key = " ".join(sorted(title_words)[:5])

        current = shingles(row["text"])
        duplicate = False

        for previous in groups.get(key, []):
            other = previous["shingles"]
            union = len(current | other)
            similarity = len(current & other) / union if union else 0

            if similarity >= threshold:
                duplicate = True
                break

        if not duplicate:
            kept.append(idx)
            groups.setdefault(key, []).append({
                "shingles": current,
            })

    return df.loc[kept].copy()


def parse_dates(df, days):
    # Page publication date takes priority over RSS/sitemap.
    page = pd.to_datetime(
        df["published_page"], format="mixed", errors="coerce", utc=True
    )
    fallback = pd.to_datetime(
        df["published"], format="mixed", errors="coerce", utc=True
    )

    published = page.fillna(fallback)

    now = pd.Timestamp.now(tz="UTC")
    cutoff = now - pd.Timedelta(days=days)

    # Exclude missing, stale and implausibly future dates.
    valid = published.notna() & published.ge(cutoff)
    valid &= published.le(now + pd.Timedelta(days=1))

    result = df.loc[valid].copy()
    result["date"] = published.loc[valid].dt.date

    return result


def build(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.index, encoding="utf-8") as file:
        index_rows = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    scope = {
        normalize_url(row["url"]): row.get("rss_category")
        for row in index_rows
        if row.get("url")
    }

    df = pd.read_json(args.articles, lines=True)
    initial_count = len(df)

    if df.empty:
        print("No articles to process")
        return

    required = [
        "url", "source", "text", "summary", "title",
        "published", "published_page", "categories",
        "author", "tags", "date_src",
    ]

    for column in required:
        if column not in df.columns:
            df[column] = None

    df["_url"] = df["url"].fillna("").map(normalize_url)
    df = df[
        df["_url"].isin(scope)
        & df["source"].isin(SOURCES)
    ].copy()

    df["text"] = df["text"].fillna("").map(normalize_text)
    df["summary"] = df["summary"].fillna("").map(normalize_text)

    df["text_src"] = "page"

    fallback = (
        df["text"].str.len().lt(args.min_chars)
        & df["summary"].str.len().ge(100)
    )

    df.loc[fallback, "text"] = (
        df.loc[fallback, "summary"]
        .str.replace(r"<[^>]+>", " ", regex=True)
        .map(normalize_text)
    )
    df.loc[fallback, "text_src"] = "summary"

    df = df[
        df["text"].str.len().ge(args.min_chars)
        | df["text_src"].eq("summary")
    ].copy()

    df["_len"] = df["text"].str.len()
    df = (
        df.sort_values("_len", ascending=False)
        .drop_duplicates("_url")
        .copy()
    )

    df = df[
        df["text"].map(
            lambda text: len(CYR.findall(text)) / max(len(text), 1) > 0.4
        )
    ].copy()

    df["tier"] = df["source"].map(
        lambda source: SOURCES[source]["tier"]
    )
    df["topic"] = df["source"].map(
        lambda source: SOURCES[source].get("topic")
    )

    rss_category = df["_url"].map(scope)

    df["rubric"] = (
        df["categories"]
        .replace("", None)
        .fillna(rss_category)
        .replace("", None)
        .fillna(
            df["url"].str.extract(
                r"https?://[^/]+/([a-z_-]{3,})/",
                expand=False,
            )
        )
    )

    meta = df["url"].fillna("") + " " + df["rubric"].fillna("")

    df = df[
        ~meta.str.contains(OFFTOPIC, na=False)
        & ~(
            df["tier"].eq(2)
            & meta.str.contains(T2_DROP, na=False)
        )
    ].copy()

    df = parse_dates(df, args.days)

    # Prefer tier 1 for duplicate content.
    df["text_hash"] = df["text"].map(text_fingerprint)
    df = (
        df.sort_values(["tier", "_len"], ascending=[True, False])
        .drop_duplicates("text_hash")
        .copy()
    )

    if args.near_duplicates:
        before = len(df)
        df = remove_near_duplicates(df, args.similarity)
        print(f"Near duplicates removed: {before - len(df)}")

    df["id"] = df["url"].map(stable_id)

    if df["id"].duplicated().any():
        raise ValueError("Stable news ID collision detected")

    df = df.sort_values(
        ["date", "tier", "source"]
    ).reset_index(drop=True)

    columns = [
        "id", "tier", "topic", "source", "date", "url",
        "title", "summary", "text", "text_src", "rubric",
        "author", "tags", "published", "published_page",
        "date_src",
    ]

    df = df[columns]

    parquet_path = out_dir / "news.parquet"
    jsonl_path = out_dir / "news.jsonl"

    report = publish_news(df, out_dir)

    assert df["url"].is_unique
    assert df["id"].is_unique
    assert df["tier"].isin([1, 2]).all()

    print(f"Raw: {initial_count} -> clean: {len(df)}")
    print(f"Saved: {parquet_path}")
    print(f"Saved: {jsonl_path}")
    print(f"Corpus: {report['corpus_fingerprint']}; {report['events']} events prepared; "
          f"ID migration: {report['migration']}; report: {out_dir / 'rebuild_report.json'}")

    if not df.empty:
        print(f"Median text: {int(df['text'].str.len().median())} chars")
        print(df.groupby("tier").size().to_string())
        print(df["source"].value_counts().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--days", type=int, default=31)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument(
        "--articles",
        default=str(DATA / "articles.jsonl"),
    )
    parser.add_argument(
        "--index",
        default=str(DATA / "index.jsonl"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(DATA),
    )
    parser.add_argument(
        "--near-duplicates",
        action="store_true",
    )
    parser.add_argument(
        "--similarity",
        type=float,
        default=0.92,
    )

    build(parser.parse_args())
