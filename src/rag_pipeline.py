"""Build and search a local, traceable retrieval index from extracted PDF artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_INDEX_PATH = PROJECT_ROOT / "rag_index" / "knowledge_base.json"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
VECTOR_DIMENSIONS = 512


def _inside_project(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(f"Path must be inside the project: {path}") from error
    return resolved


def _frontmatter_and_content(path: Path) -> tuple[dict[str, str], str]:
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---\n"):
        return {}, raw.strip()
    _, frontmatter, content = raw.split("---\n", 2)
    metadata = {
        key.strip(): value.strip()
        for line in frontmatter.splitlines()
        if ":" in line
        for key, value in [line.split(":", 1)]
    }
    return metadata, content.strip()


def _split_long_unit(unit: str, max_length: int) -> list[str]:
    if len(unit) <= max_length:
        return [unit]
    parts = re.split(r"(?<=[。！？；.!?;])", unit)
    chunks: list[str] = []
    current = ""
    for part in parts:
        if not part:
            continue
        if len(part) > max_length:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(part[index : index + max_length] for index in range(0, len(part), max_length))
        elif current and len(current) + len(part) > max_length:
            chunks.append(current)
            current = part
        else:
            current += part
    if current:
        chunks.append(current)
    return chunks


def split_chunks(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split Markdown at paragraph boundaries while retaining a small text overlap."""
    normalized = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not normalized:
        return []
    units = [part.strip() for part in normalized.split("\n\n") if part.strip()]
    units = [part for unit in units for part in _split_long_unit(unit, size)]
    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = unit if not current else f"{current}\n\n{unit}"
        if current and len(candidate) > size:
            chunks.append(current)
            suffix = current[-overlap:].lstrip()
            current = f"{suffix}\n\n{unit}" if suffix else unit
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _tokens(text: str) -> list[str]:
    raw_tokens = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)*", text.lower())
    chinese = [token for token in raw_tokens if len(token) == 1 and "\u4e00" <= token <= "\u9fff"]
    bigrams = [left + right for left, right in zip(chinese, chinese[1:])]
    return raw_tokens + bigrams


def _bucket(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big") % VECTOR_DIMENSIONS


def _vector(text: str, idf: dict[str, float], default_idf: float) -> list[float]:
    counts = Counter(_tokens(text))
    values = [0.0] * VECTOR_DIMENSIONS
    for token, count in counts.items():
        values[_bucket(token)] += (1 + math.log(count)) * idf.get(token, default_idf)
    length = math.sqrt(sum(value * value for value in values))
    return [round(value / length, 8) for value in values] if length else values


def _source_records(artifacts_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(artifacts_root.glob("**/pages/*.md")) + sorted(artifacts_root.glob("**/tables/*.md")):
        metadata, text = _frontmatter_and_content(path)
        for index, chunk in enumerate(split_chunks(text), 1):
            records.append(
                {
                    "id": f"{path.relative_to(artifacts_root).as_posix()}#{index}",
                    "source_file": metadata.get("source_file", "unknown"),
                    "page": int(metadata.get("page", "0")),
                    "source_type": metadata.get("source_type", "pdf"),
                    "artifact_path": path.relative_to(PROJECT_ROOT).as_posix(),
                    "chunk_index": index,
                    "text": chunk,
                }
            )
    return records


def build_index(artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT, index_path: Path = DEFAULT_INDEX_PATH) -> dict[str, Any]:
    artifacts_root = _inside_project(artifacts_root)
    index_path = _inside_project(index_path)
    records = _source_records(artifacts_root)
    if not records:
        raise ValueError(f"No extracted Markdown files found in {artifacts_root}")
    document_frequency: Counter[str] = Counter()
    for record in records:
        document_frequency.update(set(_tokens(record["text"])))
    count = len(records)
    idf = {token: round(math.log((count + 1) / (frequency + 1)) + 1, 8) for token, frequency in document_frequency.items()}
    default_idf = round(math.log(count + 1) + 1, 8)
    for record in records:
        record["vector"] = _vector(record["text"], idf, default_idf)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "artifacts_root": artifacts_root.relative_to(PROJECT_ROOT).as_posix(),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "vector_dimensions": VECTOR_DIMENSIONS,
        "documents": len({record["source_file"] for record in records}),
        "chunks": records,
        "idf": idf,
        "default_idf": default_idf,
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return {"index_path": index_path.relative_to(PROJECT_ROOT).as_posix(), "documents": payload["documents"], "chunks": len(records)}


def _load_index(index_path: Path = DEFAULT_INDEX_PATH) -> dict[str, Any]:
    index_path = _inside_project(index_path)
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}. Run `python src/rag_pipeline.py build` first.")
    return json.loads(index_path.read_text(encoding="utf-8"))


def search(query: str, limit: int = 5, index_path: Path = DEFAULT_INDEX_PATH) -> list[dict[str, Any]]:
    if not query.strip():
        raise ValueError("query must not be empty")
    if not 1 <= limit <= 10:
        raise ValueError("limit must be between 1 and 10")
    index = _load_index(index_path)
    query_vector = _vector(query, index["idf"], index["default_idf"])
    scored = []
    for record in index["chunks"]:
        score = sum(left * right for left, right in zip(query_vector, record["vector"]))
        if score > 0:
            scored.append((score, record))
    return [
        {
            "score": round(score, 4),
            "source_file": record["source_file"],
            "page": record["page"],
            "source_type": record["source_type"],
            "artifact_path": record["artifact_path"],
            "chunk_index": record["chunk_index"],
            "text": record["text"],
        }
        for score, record in sorted(scored, key=lambda item: item[0], reverse=True)[:limit]
    ]


def status(index_path: Path = DEFAULT_INDEX_PATH) -> dict[str, Any]:
    index = _load_index(index_path)
    return {
        "index_path": _inside_project(index_path).relative_to(PROJECT_ROOT).as_posix(),
        "created_at": index["created_at"],
        "documents": index["documents"],
        "chunks": len(index["chunks"]),
        "vector_dimensions": index["vector_dimensions"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "search", "status"))
    parser.add_argument("query", nargs="?")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    if args.command == "build":
        result = build_index()
    elif args.command == "search":
        if not args.query:
            parser.error("search requires a query")
        result = search(args.query, args.limit)
    else:
        result = status()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
