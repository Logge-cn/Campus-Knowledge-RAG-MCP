"""BM25 index construction and lexical retrieval."""

import math
from collections import Counter
from typing import Any, Iterable

from retrieval.chunking import tokenize
from retrieval.config import BM25_B, BM25_K1, CANDIDATE_LIMIT, QUERY_EXPANSION_WEIGHT, TOKENIZER_VERSION
from retrieval.query_expansion import expand_query


def build_bm25(records: list[dict[str, Any]]) -> dict[str, Any]:
    term_frequencies: list[dict[str, int]] = []
    document_frequency: Counter[str] = Counter()
    document_lengths: list[int] = []
    for record in records:
        counts = Counter(tokenize(record.get("retrieval_text", record["text"])))
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
    record_indices: Iterable[int] | None = None,
) -> list[dict[str, Any]]:
    if not query.strip():
        raise ValueError("query must not be empty")
    bm25 = index["bm25"]
    expanded_query = expand_query(query)
    query_tokens = tokenize(query)
    if expanded_query != query:
        expansion = expanded_query[len(query) :].strip()
        query_tokens.extend(tokenize(expansion) * QUERY_EXPANSION_WEIGHT)
    indices = list(record_indices) if record_indices is not None else list(range(len(index["chunks"])))
    count = len(indices)
    if not count:
        return []
    average_length = sum(bm25["document_lengths"][record_index] for record_index in indices) / count or 1.0
    k1 = bm25["k1"]
    b = bm25["b"]
    scored: list[tuple[float, int]] = []
    document_frequency = {
        token: sum(token in bm25["term_frequencies"][record_index] for record_index in indices)
        for token in set(query_tokens)
    }
    for record_index in indices:
        frequencies = bm25["term_frequencies"][record_index]
        document_length = bm25["document_lengths"][record_index]
        score = 0.0
        for token in query_tokens:
            frequency = frequencies.get(token, 0)
            if not frequency:
                continue
            frequency_count = document_frequency.get(token, 0)
            idf = math.log(1 + (count - frequency_count + 0.5) / (frequency_count + 0.5))
            denominator = frequency + k1 * (1 - b + b * document_length / average_length)
            score += idf * frequency * (k1 + 1) / denominator
        if score > 0:
            scored.append((score, record_index))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [
        {"record_index": record_index, "score": float(score)}
        for score, record_index in scored[:candidate_limit]
    ]
