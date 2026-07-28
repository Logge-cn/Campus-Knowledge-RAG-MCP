"""Create a small, local FTS index from the crawler document store."""

from __future__ import annotations

import argparse
import hashlib
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "crawler.db"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "rag.db"
HAN_CHARACTER = re.compile(r"[\u3400-\u9fff]")


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def normalize_text(text: str) -> str:
    lines = (" ".join(line.split()) for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def fts_tokens(text: str) -> str:
    """Emit space-separated Chinese characters and ASCII words for FTS5."""
    tokens: list[str] = []
    word = ""
    for character in text.lower():
        if HAN_CHARACTER.fullmatch(character):
            if word:
                tokens.append(word)
                word = ""
            tokens.append(character)
        elif character.isascii() and (character.isalnum() or character == "_"):
            word += character
        elif word:
            tokens.append(word)
            word = ""
    if word:
        tokens.append(word)
    return " ".join(tokens)


def split_long_paragraph(paragraph: str, size: int) -> list[str]:
    if len(paragraph) <= size:
        return [paragraph]
    pieces: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[。！？；])", paragraph):
        if not sentence:
            continue
        if current and len(current) + len(sentence) > size:
            pieces.append(current)
            current = ""
        while len(sentence) > size:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:size])
            sentence = sentence[size:]
        current += sentence
    if current:
        pieces.append(current)
    return pieces


def chunk_text(text: str, size: int = 650, overlap: int = 100) -> list[str]:
    paragraphs = [piece for line in text.splitlines() for piece in split_long_paragraph(line, size)]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 1 > size:
            chunks.append(current)
            current = current[-overlap:] if overlap else ""
        current = f"{current}\n{paragraph}".strip()
    if current:
        chunks.append(current)
    return chunks


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            text TEXT NOT NULL,
            url TEXT NOT NULL,
            category TEXT NOT NULL,
            published_at TEXT,
            crawled_at TEXT NOT NULL,
            source_type TEXT NOT NULL,
            content_hash TEXT NOT NULL
        );
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL REFERENCES documents(doc_id),
            ordinal INTEGER NOT NULL,
            text TEXT NOT NULL
        );
        CREATE INDEX chunks_doc_id_idx ON chunks(doc_id);
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            chunk_id UNINDEXED,
            title,
            search_text,
            category,
            tokenize='unicode61'
        );
        """
    )


def build_index(source: Path, output: Path, minimum_text_length: int = 100, force: bool = False) -> tuple[int, int]:
    if not source.is_file():
        raise FileNotFoundError(f"crawler database does not exist: {source}")
    if output.exists() and not force:
        raise FileExistsError(f"index already exists: {output}; rerun with --force to replace it")

    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    source_connection = sqlite3.connect(source)
    output_connection = sqlite3.connect(temporary)
    documents = chunks = 0
    try:
        create_schema(output_connection)
        seen_hashes: set[str] = set()
        rows = source_connection.execute(
            """SELECT doc_id, title, text, url, category, published_at, crawled_at,
                      source_type, content_hash
               FROM documents
               WHERE length(trim(text)) >= ?
               ORDER BY crawled_at DESC, doc_id""",
            (minimum_text_length,),
        )
        for row in rows:
            doc_id, title, text, url, category, published_at, crawled_at, source_type, content_hash = row
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            text = normalize_text(text)
            if len(text) < minimum_text_length:
                continue
            title = normalize_text(title) or url
            output_connection.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (doc_id, title, text, url, category, published_at, crawled_at, source_type, content_hash),
            )
            documents += 1
            for ordinal, chunk in enumerate(chunk_text(text)):
                chunk_id = hashlib.sha256(f"{doc_id}:{ordinal}".encode()).hexdigest()
                output_connection.execute(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?)", (chunk_id, doc_id, ordinal, chunk)
                )
                output_connection.execute(
                    "INSERT INTO chunks_fts VALUES (?, ?, ?, ?)",
                    (chunk_id, title, fts_tokens(f"{title}\n{chunk}"), fts_tokens(category)),
                )
                chunks += 1
        output_connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            [
                ("built_at", now()),
                ("source_database", str(source)),
                ("document_count", str(documents)),
                ("chunk_count", str(chunks)),
                ("minimum_text_length", str(minimum_text_length)),
            ],
        )
        output_connection.commit()
    finally:
        source_connection.close()
        output_connection.close()
    temporary.replace(output)
    return documents, chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-text-length", type=int, default=100)
    parser.add_argument("--force", action="store_true", help="Replace an existing index")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.minimum_text_length < 1:
        raise SystemExit("--minimum-text-length must be positive")
    documents, chunks = build_index(args.source, args.output, args.minimum_text_length, args.force)
    print(f"built {args.output}: {documents} documents, {chunks} chunks")
    return 0
