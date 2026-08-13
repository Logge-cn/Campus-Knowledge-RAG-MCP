"""Runtime initialization helpers for long-lived retrieval services."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from retrieval.embeddings import load_model
from retrieval.config import DEFAULT_INDEX_PATH
from retrieval.index import load_index
from retrieval.reranker import load_reranker


def warmup(index_path: Path = DEFAULT_INDEX_PATH) -> dict[str, Any]:
    """Load the index and both local models before the first user query."""
    started = time.perf_counter()
    index_started = time.perf_counter()
    index = load_index(index_path)
    index_ms = (time.perf_counter() - index_started) * 1000
    embedding_started = time.perf_counter()
    load_model()
    embedding_ms = (time.perf_counter() - embedding_started) * 1000
    reranker_started = time.perf_counter()
    load_reranker()
    reranker_ms = (time.perf_counter() - reranker_started) * 1000
    return {
        "documents": index["metadata"]["documents"],
        "chunks": index["metadata"]["chunks"],
        "index_ms": round(index_ms, 2),
        "embedding_model_ms": round(embedding_ms, 2),
        "reranker_model_ms": round(reranker_ms, 2),
        "total_ms": round((time.perf_counter() - started) * 1000, 2),
    }
