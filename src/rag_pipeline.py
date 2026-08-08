"""Build and search a local, traceable hybrid retrieval index."""

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

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_INDEX_ROOT = PROJECT_ROOT / "rag_index"
DEFAULT_INDEX_PATH = DEFAULT_INDEX_ROOT / "metadata.json"
DEFAULT_MODEL_PATH = DEFAULT_INDEX_ROOT / "models" / "bge-base-zh-v1.5"
CHUNKS_PATH_NAME = "chunks.json"
BM25_PATH_NAME = "bm25.json"
EMBEDDINGS_PATH_NAME = "embeddings.npy"
SCHEMA_VERSION = 2
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
BM25_K1 = 1.5
BM25_B = 0.75
CANDIDATE_LIMIT = 20
RRF_K = 1
EMBEDDING_MODEL = "BAAI/bge-base-zh-v1.5"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
TOKENIZER_VERSION = "char-bigram-v1"

_MODEL: Any | None = None


def _inside_project(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(f"Path must be inside the project: {path}") from error
    return resolved


def _index_paths(index_path: Path) -> dict[str, Path]:
    metadata_path = _inside_project(index_path)
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


def _source_records(artifacts_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    paths = sorted(artifacts_root.glob("**/pages/*.md")) + sorted(artifacts_root.glob("**/tables/*.md"))
    for path in paths:
        metadata, text = _frontmatter_and_content(path)
        for chunk_index, chunk in enumerate(split_chunks(text), 1):
            chunk_id = f"{path.relative_to(artifacts_root).as_posix()}#{chunk_index}"
            records.append(
                {
                    "chunk_id": chunk_id,
                    "source_file": metadata.get("source_file", "unknown"),
                    "page": int(metadata.get("page", "0")),
                    "source_type": metadata.get("source_type", "pdf"),
                    "artifact_path": path.relative_to(PROJECT_ROOT).as_posix(),
                    "chunk_index": chunk_index,
                    "text": chunk,
                }
            )
    return records


def _source_digest(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record["chunk_id"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(record["text"].encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _build_bm25(records: list[dict[str, Any]]) -> dict[str, Any]:
    term_frequencies: list[dict[str, int]] = []
    document_frequency: Counter[str] = Counter()
    document_lengths: list[int] = []
    for record in records:
        counts = Counter(_tokens(record["text"]))
        term_frequencies.append(dict(counts))
        document_frequency.update(counts.keys())
        document_lengths.append(sum(counts.values()))
    return {
        "k1": BM25_K1,
        "b": BM25_B,
        "tokenizer_version": TOKENIZER_VERSION,
        "average_document_length": sum(document_lengths) / len(document_lengths),
        "document_lengths": document_lengths,
        "document_frequency": dict(document_frequency),
        "term_frequencies": term_frequencies,
    }


def _load_model() -> Any:
    global _MODEL
    if _MODEL is None:
        model_path = _inside_project(DEFAULT_MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Embedding model not found: {model_path}. Download {EMBEDDING_MODEL} into this project first."
            )
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(str(model_path), local_files_only=True)
    return _MODEL


def _embedding_text(record: dict[str, Any]) -> str:
    return f"{record['source_file']}\n{record['text']}"


def _encode_documents(records: list[dict[str, Any]]) -> np.ndarray:
    values = _load_model().encode(
        [_embedding_text(record) for record in records],
        batch_size=16,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(values, dtype=np.float32)


def _encode_query(query: str) -> np.ndarray:
    values = _load_model().encode(
        [f"{QUERY_INSTRUCTION}{query}"],
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(values[0], dtype=np.float32)


def _existing_summary(records: list[dict[str, Any]], index_path: Path) -> dict[str, Any] | None:
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
        or metadata.get("embedding_model") != EMBEDDING_MODEL
        or metadata.get("bm25_k1") != BM25_K1
        or metadata.get("bm25_b") != BM25_B
        or metadata.get("tokenizer_version") != TOKENIZER_VERSION
        or metadata.get("rrf_k") != RRF_K
        or metadata.get("source_digest") != _source_digest(records)
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
    artifacts_root = _inside_project(artifacts_root)
    paths = _index_paths(index_path)
    records = _source_records(artifacts_root)
    if not records:
        raise ValueError(f"No extracted Markdown files found in {artifacts_root}")
    if not force and (summary := _existing_summary(records, index_path)) is not None:
        return summary

    bm25 = _build_bm25(records)
    embeddings = _encode_documents(records)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(records):
        raise ValueError("Embedding model returned a vector count that does not match the chunks")

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "artifacts_root": artifacts_root.relative_to(PROJECT_ROOT).as_posix(),
        "chunk_size": CHUNK_SIZE,
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


def _load_index(index_path: Path = DEFAULT_INDEX_PATH) -> dict[str, Any]:
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


def _ranked_result(record_index: int, score: float) -> dict[str, Any]:
    return {"record_index": record_index, "score": float(score)}


def bm25_search(query: str, candidate_limit: int = CANDIDATE_LIMIT, *, index: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if not query.strip():
        raise ValueError("query must not be empty")
    loaded = index or _load_index()
    bm25 = loaded["bm25"]
    query_tokens = _tokens(query)
    count = len(loaded["chunks"])
    average_length = bm25["average_document_length"] or 1.0
    k1 = bm25["k1"]
    b = bm25["b"]
    scored: list[tuple[float, int]] = []
    for record_index, frequencies in enumerate(bm25["term_frequencies"]):
        document_length = bm25["document_lengths"][record_index]
        score = 0.0
        for token in query_tokens:
            frequency = frequencies.get(token, 0)
            if not frequency:
                continue
            document_frequency = bm25["document_frequency"].get(token, 0)
            idf = math.log(1 + (count - document_frequency + 0.5) / (document_frequency + 0.5))
            denominator = frequency + k1 * (1 - b + b * document_length / average_length)
            score += idf * frequency * (k1 + 1) / denominator
        if score > 0:
            scored.append((score, record_index))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [_ranked_result(record_index, score) for score, record_index in scored[:candidate_limit]]


def vector_search(query: str, candidate_limit: int = CANDIDATE_LIMIT, *, index: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if not query.strip():
        raise ValueError("query must not be empty")
    loaded = index or _load_index()
    query_vector = _encode_query(query)
    embeddings = loaded["embeddings"]
    if query_vector.shape[0] != embeddings.shape[1]:
        raise ValueError("Query vector dimension does not match the index. Rebuild the index.")
    scores = np.asarray(embeddings @ query_vector)
    order = np.argsort(-scores, kind="stable")[:candidate_limit]
    return [_ranked_result(int(record_index), float(scores[record_index])) for record_index in order]


def rrf_fuse(
    bm25_results: list[dict[str, Any]],
    vector_results: list[dict[str, Any]],
    limit: int,
    *,
    rrf_k: int = RRF_K,
) -> list[dict[str, Any]]:
    fused: dict[int, dict[str, Any]] = {}
    for method, results in (("bm25", bm25_results), ("vector", vector_results)):
        for rank, result in enumerate(results, 1):
            record_index = result["record_index"]
            item = fused.setdefault(
                record_index,
                {"record_index": record_index, "score": 0.0, "matched_by": []},
            )
            item["score"] += 1 / (rrf_k + rank)
            item["matched_by"].append(method)
            item[f"{method}_rank"] = rank
            item[f"{method}_score"] = result["score"]
    ranked = sorted(fused.values(), key=lambda item: (-item["score"], item["record_index"]))
    return ranked[:limit]


def search(query: str, limit: int = 5, index_path: Path = DEFAULT_INDEX_PATH) -> list[dict[str, Any]]:
    if not query.strip():
        raise ValueError("query must not be empty")
    if not 1 <= limit <= 10:
        raise ValueError("limit must be between 1 and 10")
    index = _load_index(index_path)
    candidate_limit = min(CANDIDATE_LIMIT, len(index["chunks"]))
    bm25_results = bm25_search(query, candidate_limit, index=index)
    vector_results = vector_search(query, candidate_limit, index=index)
    fused = rrf_fuse(bm25_results, vector_results, limit)
    results: list[dict[str, Any]] = []
    for item in fused:
        record = index["chunks"][item["record_index"]]
        result = {
            "score": round(item["score"], 6),
            "score_type": "rrf",
            "matched_by": item["matched_by"],
            "bm25_rank": item.get("bm25_rank"),
            "vector_rank": item.get("vector_rank"),
            "source_file": record["source_file"],
            "page": record["page"],
            "source_type": record["source_type"],
            "artifact_path": record["artifact_path"],
            "chunk_index": record["chunk_index"],
            "text": record["text"],
        }
        results.append(result)
    return results


def status(index_path: Path = DEFAULT_INDEX_PATH) -> dict[str, Any]:
    index = _load_index(index_path)
    metadata = index["metadata"]
    return {
        "index_path": _inside_project(index_path).relative_to(PROJECT_ROOT).as_posix(),
        "schema_version": metadata["schema_version"],
        "created_at": metadata["created_at"],
        "documents": metadata["documents"],
        "chunks": metadata["chunks"],
        "retrieval_mode": metadata["retrieval_mode"],
        "embedding_model": metadata["embedding_model"],
        "embedding_dimensions": metadata["embedding_dimensions"],
        "bm25_k1": metadata["bm25_k1"],
        "bm25_b": metadata["bm25_b"],
        "candidate_limit": metadata["candidate_limit"],
        "rrf_k": metadata["rrf_k"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "search", "status"))
    parser.add_argument("query", nargs="?")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        result = build_index(force=args.force)
    elif args.command == "search":
        if not args.query:
            parser.error("search requires a query")
        result = search(args.query, args.limit)
    else:
        result = status()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
