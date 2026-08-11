"""Local cross-encoder reranking for retrieved RAG evidence."""

from __future__ import annotations

from typing import Any

import numpy as np

from retrieval.config import DEFAULT_RERANKER_PATH, RERANK_BATCH_SIZE, RERANK_RRF_PRIOR_WEIGHT, inside_project


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
) -> list[dict[str, Any]]:
    """Score query-chunk pairs and return a stable descending relevance order."""
    if not candidates:
        return []
    active = candidates
    pairs = [(query, chunks[item["record_index"]]["text"]) for item in active]
    scores = np.asarray((model or load_reranker()).predict(pairs, batch_size=RERANK_BATCH_SIZE, show_progress_bar=False))
    minimum, maximum = float(scores.min()), float(scores.max())
    scale = maximum - minimum or 1.0
    reranked = []
    for rank, (item, score) in enumerate(zip(active, scores), 1):
        reranker_score = float(score)
        combined_score = (reranker_score - minimum) / scale + RERANK_RRF_PRIOR_WEIGHT / rank
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
