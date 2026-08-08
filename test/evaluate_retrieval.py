"""Compare legacy lexical, BM25, embedding and hybrid retrieval on the fixed evaluation set."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_pipeline import _load_index, _tokens, bm25_search, rrf_fuse, vector_search


EVALUATION_PATH = PROJECT_ROOT / "test" / "retrieval_evaluation.json"
REPORT_PATH = PROJECT_ROOT / "artifacts" / "retrieval_evaluation_report.json"
LEGACY_DIMENSIONS = 512
TOP_K = 5


def _bucket(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % LEGACY_DIMENSIONS


def _legacy_rank(query: str, chunks: list[dict], limit: int) -> list[int]:
    document_frequency: Counter[str] = Counter()
    for chunk in chunks:
        document_frequency.update(set(_tokens(chunk["text"])))
    count = len(chunks)
    idf = {token: math.log((count + 1) / (frequency + 1)) + 1 for token, frequency in document_frequency.items()}
    default_idf = math.log(count + 1) + 1

    def vector(text: str) -> list[float]:
        values = [0.0] * LEGACY_DIMENSIONS
        for token, frequency in Counter(_tokens(text)).items():
            values[_bucket(token)] += (1 + math.log(frequency)) * idf.get(token, default_idf)
        length = math.sqrt(sum(value * value for value in values))
        return [value / length for value in values] if length else values

    query_vector = vector(query)
    scored = []
    for index, chunk in enumerate(chunks):
        score = sum(left * right for left, right in zip(query_vector, vector(chunk["text"])))
        if score > 0:
            scored.append((score, index))
    return [index for _, index in sorted(scored, key=lambda item: (-item[0], item[1]))[:limit]]


def _is_relevant(case: dict, chunk: dict) -> bool:
    return case["source_file"] == chunk["source_file"] and chunk["page"] in case["pages"]


def _metrics(cases: list[dict], rankings: dict[str, list[list[int]]], chunks: list[dict], timings: dict[str, list[float]]) -> dict:
    answerable = [index for index, case in enumerate(cases) if case["source_file"] is not None]
    output = {}
    for method, method_rankings in rankings.items():
        hits = 0
        reciprocals = []
        for case_index in answerable:
            case = cases[case_index]
            relevant_ranks = [
                rank
                for rank, chunk_index in enumerate(method_rankings[case_index], 1)
                if _is_relevant(case, chunks[chunk_index])
            ]
            if relevant_ranks:
                hits += 1
                reciprocals.append(1 / min(relevant_ranks))
            else:
                reciprocals.append(0.0)
        ordered_timings = sorted(timings[method])
        p95_index = max(0, math.ceil(len(ordered_timings) * 0.95) - 1)
        output[method] = {
            "recall_at_5": round(hits / len(answerable), 4),
            "mrr": round(mean(reciprocals), 4),
            "mean_query_ms": round(mean(timings[method]), 2),
            "p95_query_ms": round(ordered_timings[p95_index], 2),
        }
    return output


def main() -> None:
    cases = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    if not 30 <= len(cases) <= 50:
        raise ValueError("The evaluation set must contain 30 to 50 cases")
    index = _load_index()
    chunks = index["chunks"]
    rankings = {method: [] for method in ("legacy", "bm25", "vector", "hybrid")}
    timings = {method: [] for method in rankings}
    details = []

    for case in cases:
        started = time.perf_counter()
        legacy = _legacy_rank(case["query"], chunks, TOP_K)
        timings["legacy"].append((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        bm25 = bm25_search(case["query"], candidate_limit=20, index=index)
        timings["bm25"].append((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        vector = vector_search(case["query"], candidate_limit=20, index=index)
        timings["vector"].append((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        hybrid = rrf_fuse(bm25, vector, TOP_K)
        fusion_ms = (time.perf_counter() - started) * 1000
        timings["hybrid"].append(timings["bm25"][-1] + timings["vector"][-1] + fusion_ms)

        method_indices = {
            "legacy": legacy,
            "bm25": [item["record_index"] for item in bm25[:TOP_K]],
            "vector": [item["record_index"] for item in vector[:TOP_K]],
            "hybrid": [item["record_index"] for item in hybrid],
        }
        for method, indices in method_indices.items():
            rankings[method].append(indices)
        details.append(
            {
                "query": case["query"],
                "category": case["category"],
                "expected": {"source_file": case["source_file"], "pages": case["pages"]},
                "hits": {
                    method: any(_is_relevant(case, chunks[index_value]) for index_value in indices)
                    if case["source_file"] is not None
                    else None
                    for method, indices in method_indices.items()
                },
            }
        )

    metrics = _metrics(cases, rankings, chunks, timings)
    report = {
        "evaluation_cases": len(cases),
        "answerable_cases": sum(case["source_file"] is not None for case in cases),
        "no_answer_cases": sum(case["source_file"] is None for case in cases),
        "top_k": TOP_K,
        "metrics": metrics,
        "acceptance": {
            "hybrid_recall_not_below_single_retrievers": metrics["hybrid"]["recall_at_5"]
            >= max(metrics["bm25"]["recall_at_5"], metrics["vector"]["recall_at_5"]),
            "semantic_upgrade_improves_over_legacy": metrics["hybrid"]["recall_at_5"]
            > metrics["legacy"]["recall_at_5"],
        },
        "note": "No-answer cases are labeled but refusal is not scored because this phase implements retrieval, not answer generation or a refusal threshold.",
        "details": details,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, ensure_ascii=False, indent=2))
    if not all(report["acceptance"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
