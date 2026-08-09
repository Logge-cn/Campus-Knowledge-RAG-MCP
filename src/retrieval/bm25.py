"""BM25 index construction and lexical retrieval."""

import math
from collections import Counter
from typing import Any

from retrieval.chunking import tokenize
from retrieval.config import BM25_B, BM25_K1, CANDIDATE_LIMIT, TOKENIZER_VERSION


def build_bm25(records: list[dict[str, Any]]) -> dict[str, Any]:
    term_frequencies: list[dict[str, int]] = []
    document_frequency: Counter[str] = Counter()
    document_lengths: list[int] = []
    for record in records:
        counts = Counter(tokenize(record["text"]))
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


def bm25_search(
    query: str,
    candidate_limit: int = CANDIDATE_LIMIT,
    *,
    index: dict[str, Any],
) -> list[dict[str, Any]]:
    if not query.strip():
        raise ValueError("query must not be empty")
    bm25 = index["bm25"]
    query_tokens = tokenize(query)
    count = len(index["chunks"])
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
    return [
        {"record_index": record_index, "score": float(score)}
        for score, record_index in scored[:candidate_limit]
    ]
