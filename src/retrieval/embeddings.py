"""Local Chinese embedding model loading and vector retrieval."""

from typing import Any

import numpy as np

from retrieval.config import CANDIDATE_LIMIT, DEFAULT_MODEL_PATH, EMBEDDING_MODEL, QUERY_INSTRUCTION, inside_project


_MODEL: Any | None = None


def load_model() -> Any:
    global _MODEL
    if _MODEL is None:
        model_path = inside_project(DEFAULT_MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Embedding model not found: {model_path}. Download {EMBEDDING_MODEL} into this project first."
            )
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(str(model_path), local_files_only=True)
    return _MODEL


def encode_documents(records: list[dict[str, Any]]) -> np.ndarray:
    texts = [f"{record['source_file']}\n{record['text']}" for record in records]
    values = load_model().encode(
        texts,
        batch_size=16,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(values, dtype=np.float32)


def encode_query(query: str) -> np.ndarray:
    values = load_model().encode(
        [f"{QUERY_INSTRUCTION}{query}"],
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(values[0], dtype=np.float32)


def vector_search(
    query: str,
    candidate_limit: int = CANDIDATE_LIMIT,
    *,
    index: dict[str, Any],
) -> list[dict[str, Any]]:
    if not query.strip():
        raise ValueError("query must not be empty")
    query_vector = encode_query(query)
    embeddings = index["embeddings"]
    if query_vector.shape[0] != embeddings.shape[1]:
        raise ValueError("Query vector dimension does not match the index. Rebuild the index.")
    scores = np.asarray(embeddings @ query_vector)
    order = np.argsort(-scores, kind="stable")[:candidate_limit]
    return [
        {"record_index": int(record_index), "score": float(scores[record_index])}
        for record_index in order
    ]
