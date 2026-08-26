"""SQLite FTS5/BM25 support for persistent lexical retrieval."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable
from pathlib import Path

LEXICAL_INDEX_SCHEMA_VERSION = 1
LEXICAL_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "release",
        "the",
        "to",
        "what",
        "when",
        "which",
        "with",
    }
)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+", flags=re.IGNORECASE)
TELEQNA_RELEASE_TAG_PATTERN = re.compile(
    r"\[\s*3gpp\s+release\s+\d+\s*\]",
    flags=re.IGNORECASE,
)


def lexical_query(text: str) -> str | None:
    """Return a safe OR query containing only informative unique tokens."""
    terms: list[str] = []
    seen: set[str] = set()
    cleaned = TELEQNA_RELEASE_TAG_PATTERN.sub(" ", text)
    for token in TOKEN_PATTERN.findall(cleaned.casefold()):
        if token in LEXICAL_STOPWORDS or token in seen or len(token) < 2:
            continue
        seen.add(token)
        terms.append(token)
    if not terms:
        return None
    return " OR ".join(f'"{term}"' for term in terms)


class SqliteBm25Index:
    """Read-only BM25 search over a prebuilt FTS5 chunk index."""

    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(
                f"Hybrid retrieval requires {path}. Build it with "
                "`uv run scripts/build_lexical_index.py`."
            )
        self.path = path

    def search(
        self,
        query_text: str,
        selected_series: list[str],
        top_k: int,
    ) -> list[tuple[str, int, float]]:
        if top_k < 1 or not selected_series:
            return []
        query = lexical_query(query_text)
        if query is None:
            return []
        placeholders = ",".join("?" for _ in selected_series)
        sql = f"""
            SELECT series, CAST(metadata_index AS INTEGER), bm25(chunks, 0.0, 0.0, 0.0, 2.0, 1.0)
            FROM chunks
            WHERE chunks MATCH ? AND series IN ({placeholders})
            ORDER BY bm25(chunks, 0.0, 0.0, 0.0, 2.0, 1.0), rowid
            LIMIT ?
        """
        uri = f"file:{self.path.resolve()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            rows = connection.execute(sql, [query, *selected_series, top_k]).fetchall()
        return [(str(series), int(metadata_index), float(score)) for series, metadata_index, score in rows]


def build_bm25_index(
    path: Path,
    rows: Iterable[tuple[str, str, int, str, str]],
) -> int:
    """Build an FTS5 index from `(chunk_id, series, metadata_index, heading, text)` rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing lexical index: {path}")
    inserted = 0
    try:
        with sqlite3.connect(path) as connection:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute(
                """
                CREATE VIRTUAL TABLE chunks USING fts5(
                    chunk_id UNINDEXED,
                    series UNINDEXED,
                    metadata_index UNINDEXED,
                    heading,
                    text,
                    tokenize='unicode61 remove_diacritics 2'
                )
                """
            )
            batch: list[tuple[str, str, int, str, str]] = []
            for row in rows:
                batch.append(row)
                if len(batch) < 1_000:
                    continue
                connection.executemany(
                    "INSERT INTO chunks(chunk_id, series, metadata_index, heading, text) VALUES (?, ?, ?, ?, ?)",
                    batch,
                )
                inserted += len(batch)
                batch.clear()
            if batch:
                connection.executemany(
                    "INSERT INTO chunks(chunk_id, series, metadata_index, heading, text) VALUES (?, ?, ?, ?, ?)",
                    batch,
                )
                inserted += len(batch)
            connection.execute(f"PRAGMA user_version={LEXICAL_INDEX_SCHEMA_VERSION}")
            connection.execute("INSERT INTO chunks(chunks) VALUES ('optimize')")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return inserted
