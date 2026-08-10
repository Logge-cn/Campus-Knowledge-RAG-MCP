"""Hybrid BM25 and embedding retrieval with RRF fusion."""

from pathlib import Path
from typing import Any

from retrieval.bm25 import bm25_search as _bm25_search
from retrieval.config import (
    CANDIDATE_LIMIT,
    CHUNK_MAX_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_TARGET_TOKENS,
    DEFAULT_INDEX_PATH,
    EMBEDDING_MODEL,
    PROJECT_ROOT,
    RRF_BM25_WEIGHT,
    RRF_K,
    RRF_VECTOR_WEIGHT,
    inside_project,
)
from retrieval.embeddings import vector_search as _vector_search
from retrieval.index import load_index


def bm25_search(query: str, candidate_limit: int = CANDIDATE_LIMIT, *, index: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    return _bm25_search(query, candidate_limit, index=index or load_index())


def vector_search(query: str, candidate_limit: int = CANDIDATE_LIMIT, *, index: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    return _vector_search(query, candidate_limit, index=index or load_index())


def rrf_fuse(
    bm25_results: list[dict[str, Any]],
    vector_results: list[dict[str, Any]],
    limit: int,
    *,
    rrf_k: int = RRF_K,
    bm25_weight: float = 1.0,
    vector_weight: float = 1.0,
) -> list[dict[str, Any]]:
    fused: dict[int, dict[str, Any]] = {}
    for method, results, weight in (
        ("bm25", bm25_results, bm25_weight),
        ("vector", vector_results, vector_weight),
    ):
        for rank, result in enumerate(results, 1):
            record_index = result["record_index"]
            item = fused.setdefault(
                record_index,
                {"record_index": record_index, "score": 0.0, "matched_by": []},
            )
            item["score"] += weight / (rrf_k + rank)
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
    index = load_index(index_path)
    candidate_limit = min(CANDIDATE_LIMIT, len(index["chunks"]))
    fused = rrf_fuse(
        _bm25_search(query, candidate_limit, index=index),
        _vector_search(query, candidate_limit, index=index),
        limit,
        bm25_weight=RRF_BM25_WEIGHT,
        vector_weight=RRF_VECTOR_WEIGHT,
    )
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
            "row_start",
            "row_end",
            "oversize_row",
        ):
            if key in record:
                result[key] = record[key]
        results.append(result)
    return results


def status(index_path: Path = DEFAULT_INDEX_PATH) -> dict[str, Any]:
    metadata = load_index(index_path)["metadata"]
    return {
        "index_path": inside_project(index_path).relative_to(PROJECT_ROOT).as_posix(),
        "schema_version": metadata["schema_version"],
        "created_at": metadata["created_at"],
        "documents": metadata["documents"],
        "chunks": metadata["chunks"],
        "chunk_target_tokens": CHUNK_TARGET_TOKENS,
        "chunk_max_tokens": CHUNK_MAX_TOKENS,
        "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
        "retrieval_mode": metadata["retrieval_mode"],
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": metadata["embedding_dimensions"],
        "bm25_k1": metadata["bm25_k1"],
        "bm25_b": metadata["bm25_b"],
        "candidate_limit": metadata["candidate_limit"],
        "rrf_k": metadata["rrf_k"],
        "rrf_bm25_weight": metadata["rrf_bm25_weight"],
        "rrf_vector_weight": metadata["rrf_vector_weight"],
    }
