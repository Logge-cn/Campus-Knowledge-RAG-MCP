"""Local cross-encoder reranking for retrieved RAG evidence."""

from __future__ import annotations

from typing import Any

import numpy as np

from retrieval.config import DEFAULT_RERANKER_PATH, RERANK_BATCH_SIZE, RERANK_RRF_PRIOR_WEIGHT, inside_project
from retrieval.query_expansion import expand_query


_MODEL: Any | None = None


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
    rrf_scores = np.asarray([item["score"] for item in active], dtype=float)
    rrf_minimum, rrf_maximum = float(rrf_scores.min()), float(rrf_scores.max())
    rrf_scale = rrf_maximum - rrf_minimum or 1.0
    reranked = []
    for rank, (item, score) in enumerate(zip(active, scores), 1):
        reranker_score = float(score)
        normalized_rrf_score = (float(item["score"]) - rrf_minimum) / rrf_scale
        combined_score = reranker_score + RERANK_RRF_PRIOR_WEIGHT * normalized_rrf_score
        reranked.append(
            {
                **item,
                "score": combined_score,
                "rrf_score": item["score"],
                "reranker_score": reranker_score,
                "retrieval_rank": rank,
            }
        )
    return sorted(
        reranked,
        key=lambda item: (-item["score"], item["retrieval_rank"], item["record_index"]),
    )
