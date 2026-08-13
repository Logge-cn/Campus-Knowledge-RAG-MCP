"""Hybrid BM25 and embedding retrieval with RRF fusion."""

import json
import time
from functools import lru_cache
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
    RERANK_CANDIDATE_LIMIT,
    RERANKER_MODEL,
    RERANK_RRF_PRIOR_METHOD,
    RERANK_RRF_PRIOR_WEIGHT,
    RERANK_SCORE_NORMALIZATION,
    RERANK_TOP5_PROTECTION_SCORE,
    RRF_BM25_WEIGHT,
    RRF_K,
    RRF_VECTOR_WEIGHT,
    inside_project,
    relative_asset_path,
)
from retrieval.embeddings import vector_search as _vector_search
from retrieval.evidence import assess_evidence
from retrieval.index import index_freshness, load_index
from retrieval.reranker import rerank


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


def _active_candidates(
    items: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Compatibility helper for filtering already-ranked candidates."""
    return [item for item in items if chunks[item["record_index"]].get("active", True)][:limit]


def _search_uncached_with_timings(
    query: str,
    limit: int,
    index_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if not query.strip():
        raise ValueError("query must not be empty")
    if not 1 <= limit <= 10:
        raise ValueError("limit must be between 1 and 10")
    total_started = time.perf_counter()
    phase_started = time.perf_counter()
    index = load_index(index_path)
    index_ms = (time.perf_counter() - phase_started) * 1000
    active_indices = [
        record_index
        for record_index, chunk in enumerate(index["chunks"])
        if chunk.get("active", True)
    ]
    candidate_limit = min(CANDIDATE_LIMIT, len(active_indices))
    phase_started = time.perf_counter()
    bm25 = _bm25_search(query, candidate_limit, index=index, record_indices=active_indices)
    bm25_ms = (time.perf_counter() - phase_started) * 1000
    phase_started = time.perf_counter()
    vector = _vector_search(query, candidate_limit, index=index, record_indices=active_indices)
    vector_ms = (time.perf_counter() - phase_started) * 1000
    phase_started = time.perf_counter()
    fused = rrf_fuse(
        bm25,
        vector,
        min(RERANK_CANDIDATE_LIMIT, candidate_limit),
        bm25_weight=RRF_BM25_WEIGHT,
        vector_weight=RRF_VECTOR_WEIGHT,
    )
    fusion_ms = (time.perf_counter() - phase_started) * 1000
    phase_started = time.perf_counter()
    reranked = rerank(query, fused, index["chunks"])
    reranker_ms = (time.perf_counter() - phase_started) * 1000
    phase_started = time.perf_counter()
    results: list[dict[str, Any]] = []
    for item in reranked[:limit]:
        record = index["chunks"][item["record_index"]]
        result = {
            "chunk_id": record["chunk_id"],
            "score": round(item["score"], 6),
            "score_type": "cross_encoder_rank_rrf_blend",
            "reranker_score": round(item["reranker_score"], 6),
            "rrf_score": round(item["rrf_score"], 6),
            "retrieval_rank": item["retrieval_rank"],
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
            "document_id",
            "version",
            "effective_date",
            "active",
        ):
            if key in record:
                result[key] = record[key]
        results.append(result)
    formatting_ms = (time.perf_counter() - phase_started) * 1000
    return results, {
        "index_ms": round(index_ms, 2),
        "bm25_ms": round(bm25_ms, 2),
        "vector_ms": round(vector_ms, 2),
        "fusion_ms": round(fusion_ms, 2),
        "reranker_ms": round(reranker_ms, 2),
        "formatting_ms": round(formatting_ms, 2),
        "total_ms": round((time.perf_counter() - total_started) * 1000, 2),
    }


def _search_uncached(query: str, limit: int, index_path: Path) -> list[dict[str, Any]]:
    return _search_uncached_with_timings(query, limit, index_path)[0]


def profile_search(query: str, limit: int = 5, index_path: Path = DEFAULT_INDEX_PATH) -> dict[str, Any]:
    """Run an uncached query and expose phase-level timings for benchmarks."""
    results, timings = _search_uncached_with_timings(query, limit, inside_project(index_path))
    return {"results": results, "timings": timings}


def _index_version(index_path: Path) -> tuple[int, int]:
    metadata_path = inside_project(index_path)
    stat = metadata_path.stat()
    return stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=128)
def _cached_search(query: str, limit: int, index_path: str, version: tuple[int, int]) -> str:
    del version
    results = _search_uncached(query, limit, Path(index_path))
    return json.dumps(results, ensure_ascii=False, separators=(",", ":"))


def clear_search_cache() -> None:
    _cached_search.cache_clear()


def search(query: str, limit: int = 5, index_path: Path = DEFAULT_INDEX_PATH) -> list[dict[str, Any]]:
    if not query.strip():
        raise ValueError("query must not be empty")
    if not 1 <= limit <= 10:
        raise ValueError("limit must be between 1 and 10")
    resolved = inside_project(index_path)
    payload = _cached_search(query, limit, str(resolved), _index_version(resolved))
    return json.loads(payload)


def retrieve(query: str, limit: int = 5, index_path: Path = DEFAULT_INDEX_PATH) -> dict[str, Any]:
    """Return evidence together with an explicit sufficiency assessment."""
    before = _cached_search.cache_info()
    started = time.perf_counter()
    results = search(query, limit, index_path)
    retrieval_ms = (time.perf_counter() - started) * 1000
    after = _cached_search.cache_info()
    assessment_started = time.perf_counter()
    assessment = assess_evidence(query, results)
    assessment_ms = (time.perf_counter() - assessment_started) * 1000
    return {
        "query": query,
        "evidence_sufficient": assessment["evidence_sufficient"],
        "confidence": assessment["confidence"],
        "reason": assessment["reason"],
        "assessment": assessment,
        "diagnostics": {
            "cache_hit": after.hits > before.hits,
            "retrieval_ms": round(retrieval_ms, 2),
            "assessment_ms": round(assessment_ms, 2),
            "total_ms": round(retrieval_ms + assessment_ms, 2),
        },
        "results": results,
    }


def status(index_path: Path = DEFAULT_INDEX_PATH) -> dict[str, Any]:
    metadata = load_index(index_path)["metadata"]
    result = {
        "index_path": relative_asset_path(inside_project(index_path)).as_posix(),
        "schema_version": metadata["schema_version"],
        "created_at": metadata["created_at"],
        "documents": metadata["documents"],
        "chunks": metadata["chunks"],
        "chunk_target_tokens": CHUNK_TARGET_TOKENS,
        "chunk_max_tokens": CHUNK_MAX_TOKENS,
        "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
        "retrieval_mode": "hybrid_rrf_cross_encoder_rerank",
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": metadata["embedding_dimensions"],
        "bm25_k1": metadata["bm25_k1"],
        "bm25_b": metadata["bm25_b"],
        "candidate_limit": CANDIDATE_LIMIT,
        "rrf_k": metadata["rrf_k"],
        "rrf_bm25_weight": metadata["rrf_bm25_weight"],
        "rrf_vector_weight": metadata["rrf_vector_weight"],
        "reranker_model": RERANKER_MODEL,
        "rerank_candidate_limit": RERANK_CANDIDATE_LIMIT,
        "rerank_score_normalization": RERANK_SCORE_NORMALIZATION,
        "rerank_rrf_prior_method": RERANK_RRF_PRIOR_METHOD,
        "rerank_rrf_prior_weight": RERANK_RRF_PRIOR_WEIGHT,
        "rerank_top5_protection_score": RERANK_TOP5_PROTECTION_SCORE,
        "answer_generation": "mcp_client",
        "evidence_sufficiency": "multi_signal_retrieval_diagnostic",
        "search_cache": _cached_search.cache_info()._asdict(),
        "document_versions": metadata.get("document_versions", []),
    }
    try:
        result["freshness"] = index_freshness(index_path)
    except (OSError, ValueError, KeyError) as error:
        result["freshness"] = {
            "pending_update": None,
            "error": f"{type(error).__name__}: {error}",
        }
    return result
