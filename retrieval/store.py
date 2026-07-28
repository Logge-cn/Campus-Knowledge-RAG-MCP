"""Queries for the local FTS5 RAG index."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from ingestion.index import DEFAULT_OUTPUT, fts_tokens


class RetrievalStore:
    def __init__(self, path: Path = DEFAULT_OUTPUT) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"RAG index does not exist: {path}. Run `uv run python -m ingestion` first.")
        self.connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        self.connection.row_factory = sqlite3.Row

    def close(self) -> None:
        self.connection.close()

    def search(self, query: str, category: str | None = None, top_k: int = 5) -> list[dict[str, Any]]:
        terms = fts_tokens(query)
        if not terms:
            raise ValueError("query must contain Chinese characters, letters, or numbers")
        if not 1 <= top_k <= 10:
            raise ValueError("top_k must be between 1 and 10")
        rows = self.connection.execute(
            """SELECT c.chunk_id, c.doc_id, c.ordinal, c.text, d.title, d.url, d.category,
                      d.published_at, d.crawled_at,
                      -bm25(chunks_fts, 0.0, 5.0, 1.0, 0.5) AS score
               FROM chunks_fts
               JOIN chunks AS c ON c.chunk_id = chunks_fts.chunk_id
               JOIN documents AS d ON d.doc_id = c.doc_id
               WHERE chunks_fts MATCH ? AND (? IS NULL OR d.category = ?)
               ORDER BY score DESC, d.published_at DESC, c.ordinal
               LIMIT ?""",
            (terms, category, category, top_k),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            """SELECT d.*, count(c.chunk_id) AS chunk_count
               FROM documents AS d LEFT JOIN chunks AS c ON c.doc_id = d.doc_id
               WHERE d.doc_id = ? GROUP BY d.doc_id""",
            (doc_id,),
        ).fetchone()
        return dict(row) if row else None

    def status(self) -> dict[str, str | int]:
        metadata = dict(self.connection.execute("SELECT key, value FROM metadata").fetchall())
        return {
            "built_at": metadata.get("built_at", ""),
            "source_database": metadata.get("source_database", ""),
            "document_count": int(metadata.get("document_count", "0")),
            "chunk_count": int(metadata.get("chunk_count", "0")),
            "minimum_text_length": int(metadata.get("minimum_text_length", "0")),
        }
