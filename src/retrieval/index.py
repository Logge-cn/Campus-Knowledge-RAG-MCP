"""Build, persist and validate the hybrid retrieval index."""

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from retrieval.bm25 import build_bm25
from retrieval.chunking import split_chunks, split_table_chunks
from retrieval.config import (
    BM25_B,
    BM25_K1,
    BM25_PATH_NAME,
    CANDIDATE_LIMIT,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNK_TARGET_TOKENS,
    CHUNKS_PATH_NAME,
    DEFAULT_ARTIFACTS_ROOT,
    DEFAULT_INDEX_PATH,
    EMBEDDING_MODEL,
    EMBEDDINGS_PATH_NAME,
    PROJECT_ROOT,
    RRF_K,
    SCHEMA_VERSION,
    TOKENIZER_VERSION,
    inside_project,
)
from retrieval.embeddings import encode_documents


def _index_paths(index_path: Path) -> dict[str, Path]:
    metadata_path = inside_project(index_path)
    index_root = metadata_path.parent
    return {
        "metadata": metadata_path,
        "chunks": index_root / CHUNKS_PATH_NAME,
        "bm25": index_root / BM25_PATH_NAME,
        "embeddings": index_root / EMBEDDINGS_PATH_NAME,
    }


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


def _source_records(artifacts_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    paths = sorted(artifacts_root.glob("**/pages/*.md")) + sorted(artifacts_root.glob("**/tables/*.md"))
    for path in paths:
        metadata, text = _frontmatter_and_content(path)
        source_type = metadata.get("source_type", "pdf")
        table_chunks = split_table_chunks(text) if source_type == "table" else []
        chunks = table_chunks or [{"text": chunk} for chunk in split_chunks(text)]
        for chunk_index, chunk_data in enumerate(chunks, 1):
            chunk = str(chunk_data["text"])
            normalized = re.sub(r"\s+", " ", chunk).strip()
            if normalized in seen_text:
                continue
            seen_text.add(normalized)
            record = {
                "chunk_id": f"{path.relative_to(artifacts_root).as_posix()}#{chunk_index}",
                "source_file": metadata.get("source_file", "unknown"),
                "page": int(metadata.get("page", "0")),
                "source_type": source_type,
                "artifact_path": path.relative_to(PROJECT_ROOT).as_posix(),
                "chunk_index": chunk_index,
                "text": chunk,
            }
            for key in (
                "source_sha256",
                "content_sha256",
                "imported_at",
                "quality_warnings",
                "table_id",
                "table_title",
                "extraction_method",
                "extraction_score",
                "confidence",
                "low_confidence",
                "processing_note",
            ):
                if key in metadata:
                    value: Any = metadata[key]
                    if key in {"extraction_score", "confidence"}:
                        value = float(value)
                    elif key == "low_confidence":
                        value = value.lower() == "true"
                    elif key == "quality_warnings":
                        value = [] if value == "none" else value.split(",")
                    record[key] = value
            for key in ("row_start", "row_end", "oversize_row"):
                if key in chunk_data:
                    record[key] = chunk_data[key]
            records.append(record)
    return records


def _source_digest(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _existing_summary(records: list[dict[str, Any]], artifacts_root: Path, index_path: Path) -> dict[str, Any] | None:
    paths = _index_paths(index_path)
    if not all(path.exists() for path in paths.values()):
        return None
    try:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        embeddings = np.load(paths["embeddings"], mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if (
        metadata.get("schema_version") != SCHEMA_VERSION
        or metadata.get("artifacts_root") != artifacts_root.relative_to(PROJECT_ROOT).as_posix()
        or metadata.get("embedding_model") != EMBEDDING_MODEL
        or metadata.get("bm25_k1") != BM25_K1
        or metadata.get("bm25_b") != BM25_B
        or metadata.get("tokenizer_version") != TOKENIZER_VERSION
        or metadata.get("rrf_k") != RRF_K
        or metadata.get("source_digest") != _source_digest(records)
        or metadata.get("chunk_target_tokens") != CHUNK_TARGET_TOKENS
        or metadata.get("chunks") != len(records)
        or embeddings.ndim != 2
        or embeddings.shape[0] != len(records)
    ):
        return None
    return {
        "index_path": paths["metadata"].relative_to(PROJECT_ROOT).as_posix(),
        "documents": metadata["documents"],
        "chunks": metadata["chunks"],
        "reused": True,
    }


def build_index(
    artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT,
    index_path: Path = DEFAULT_INDEX_PATH,
    *,
    force: bool = False,
) -> dict[str, Any]:
    artifacts_root = inside_project(artifacts_root)
    paths = _index_paths(index_path)
    records = _source_records(artifacts_root)
    if not records:
        raise ValueError(f"No extracted Markdown files found in {artifacts_root}")
    if not force and (summary := _existing_summary(records, artifacts_root, index_path)) is not None:
        return summary

    bm25 = build_bm25(records)
    embeddings = encode_documents(records)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(records):
        raise ValueError("Embedding model returned a vector count that does not match the chunks")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "artifacts_root": artifacts_root.relative_to(PROJECT_ROOT).as_posix(),
        "chunk_size": CHUNK_SIZE,
        "chunk_target_tokens": CHUNK_TARGET_TOKENS,
        "chunk_overlap": CHUNK_OVERLAP,
        "documents": len({record["source_file"] for record in records}),
        "chunks": len(records),
        "source_digest": _source_digest(records),
        "retrieval_mode": "hybrid_bm25_embedding_rrf",
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": int(embeddings.shape[1]),
        "embedding_normalization": "l2",
        "bm25_k1": BM25_K1,
        "bm25_b": BM25_B,
        "tokenizer_version": TOKENIZER_VERSION,
        "candidate_limit": CANDIDATE_LIMIT,
        "rrf_k": RRF_K,
        "chunks_file": CHUNKS_PATH_NAME,
        "bm25_file": BM25_PATH_NAME,
        "embeddings_file": EMBEDDINGS_PATH_NAME,
    }
    paths["metadata"].parent.mkdir(parents=True, exist_ok=True)
    paths["chunks"].write_text(json.dumps(records, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    paths["bm25"].write_text(json.dumps(bm25, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    np.save(paths["embeddings"], embeddings, allow_pickle=False)
    paths["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "index_path": paths["metadata"].relative_to(PROJECT_ROOT).as_posix(),
        "documents": metadata["documents"],
        "chunks": metadata["chunks"],
        "reused": False,
    }


def load_index(index_path: Path = DEFAULT_INDEX_PATH) -> dict[str, Any]:
    paths = _index_paths(index_path)
    missing = [path.name for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Hybrid index files missing: {', '.join(missing)}. Run the build command first.")
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Hybrid index schema is incompatible. Rebuild the index.")
    chunks = json.loads(paths["chunks"].read_text(encoding="utf-8"))
    bm25 = json.loads(paths["bm25"].read_text(encoding="utf-8"))
    embeddings = np.load(paths["embeddings"], mmap_mode="r", allow_pickle=False)
    count = metadata.get("chunks")
    if (
        count != len(chunks)
        or count != len(bm25.get("document_lengths", []))
        or count != len(bm25.get("term_frequencies", []))
        or embeddings.ndim != 2
        or count != embeddings.shape[0]
        or metadata.get("embedding_dimensions") != embeddings.shape[1]
    ):
        raise ValueError("Hybrid index files are inconsistent. Rebuild the index.")
    return {"metadata": metadata, "chunks": chunks, "bm25": bm25, "embeddings": embeddings}
