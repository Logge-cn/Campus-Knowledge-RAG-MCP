"""Local cross-encoder reranking for retrieved RAG evidence."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from retrieval.config import (
    DEFAULT_RERANKER_PATH,
    RERANK_BATCH_SIZE,
    RERANK_RRF_PRIOR_WEIGHT,
    RERANK_TOP5_PROTECTION_SCORE,
    inside_project,
)
from retrieval.query_expansion import expand_query


_MODEL: Any | None = None


def _is_table_query(query: str, candidates: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> bool:
    top_candidates = candidates[:5]
    if any(chunks[item["record_index"]].get("source_type") == "table" for item in top_candidates):
        return True
    return bool(
        re.search(
            r"分别|对应|排名|区间|比例|占比|折算|表中|列表|明细|"
            r"各.+(?:多少|是什么|哪些)|地址.+(?:电话|联系电话)|(?:编号|学时|学分).*(?:学期|分别)",
            query,
        )
    )


def protect_hybrid_top_five(
    query: str,
    candidate_record_indices: list[int],
    ranked_record_indices: list[int],
    chunks: list[dict[str, Any]],
    reranker_scores: dict[int, float],
    score_threshold: float = RERANK_TOP5_PROTECTION_SCORE,
) -> list[int]:
    """Keep high-confidence or table Hybrid Top-5 evidence without freezing other ranks."""
    if len(candidate_record_indices) <= 5:
        return ranked_record_indices
    candidates = [{"record_index": record_index} for record_index in candidate_record_indices]
    table_query = _is_table_query(query, candidates, chunks)
    protected_ids = {
        record_index
        for record_index in candidate_record_indices[:5]
        if table_query and chunks[record_index].get("source_type") == "table"
        or reranker_scores[record_index] >= score_threshold
    }
    top_five = ranked_record_indices[:5]
    missing = [record_index for record_index in ranked_record_indices[5:] if record_index in protected_ids]
    for protected_id in missing:
        replacement = next(
            (index for index in range(len(top_five) - 1, -1, -1) if top_five[index] not in protected_ids),
            None,
        )
        if replacement is None:
            break
        top_five[replacement] = protected_id
    selected = set(top_five)
    return top_five + [record_index for record_index in ranked_record_indices if record_index not in selected]


def load_reranker() -> Any:
    global _MODEL
    if _MODEL is None:
        model_path = inside_project(DEFAULT_RERANKER_PATH)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Reranker model not found: {model_path}. Run `python src/prepare_model.py --reranker` first."
            )
        from sentence_transformers import CrossEncoder

        _MODEL = CrossEncoder(str(model_path), local_files_only=True)
    return _MODEL


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    *,
    model: Any | None = None,
    batch_size: int = RERANK_BATCH_SIZE,
) -> list[dict[str, Any]]:
    """Score query-chunk pairs and return a stable descending relevance order."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not candidates:
        return []
    active = candidates
    reranker_query = expand_query(query)
    pairs = [(reranker_query, chunks[item["record_index"]]["text"]) for item in active]
    scores = np.asarray((model or load_reranker()).predict(pairs, batch_size=batch_size, show_progress_bar=False))
    return rerank_from_scores(query, active, chunks, scores)


def rerank_from_scores(
    query: str,
    candidates: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    scores: list[float] | np.ndarray,
) -> list[dict[str, Any]]:
    """Apply the production ranking policy to precomputed cross-encoder scores."""
    if not candidates:
        return []
    if len(candidates) != len(scores):
        raise ValueError("scores must contain one value for every candidate")
    active = candidates
    scores = np.asarray(scores)
    score_order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
    score_ranks = [0] * len(scores)
    for score_rank, index in enumerate(score_order, 1):
        score_ranks[index] = score_rank
    score_denominator = max(1, len(scores) - 1)
    reranked = []
    for rank, (item, score, score_rank) in enumerate(zip(active, scores, score_ranks), 1):
        reranker_score = float(score)
        normalized_reranker_score = 1.0 - (score_rank - 1) / score_denominator
        retrieval_prior = 1.0 / rank
        combined_score = normalized_reranker_score + RERANK_RRF_PRIOR_WEIGHT * retrieval_prior
        reranked.append(
            {
                **item,
                "score": combined_score,
                "rrf_score": item["score"],
                "reranker_score": reranker_score,
                "normalized_reranker_score": normalized_reranker_score,
                "retrieval_rank": rank,
            }
        )
    ranked = sorted(
        reranked,
        key=lambda item: (-item["score"], item["retrieval_rank"], item["record_index"]),
    )
    scores_by_record = {item["record_index"]: item["reranker_score"] for item in ranked}
    protected_order = protect_hybrid_top_five(
        query,
        [item["record_index"] for item in active],
        [item["record_index"] for item in ranked],
        chunks,
        scores_by_record,
    )
    items_by_record = {item["record_index"]: item for item in ranked}
    return [items_by_record[record_index] for record_index in protected_order]
