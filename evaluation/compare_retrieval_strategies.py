"""Compare the frozen legacy and optimized retrieval policies on one labeled dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.evaluate import _first_relevant_rank, _ranking_metrics, _source_type_by_file
from retrieval.bm25 import bm25_search, build_bm25
from retrieval.config import (
    CANDIDATE_LIMIT,
    DEFAULT_INDEX_PATH,
    RERANK_BATCH_SIZE,
    RERANK_CANDIDATE_LIMIT,
    RRF_BM25_WEIGHT,
    RRF_VECTOR_WEIGHT,
)
from retrieval.embeddings import vector_search
from retrieval.hybrid import rrf_fuse
from retrieval.index import load_index
from retrieval.query_expansion import expand_query
from retrieval.reranker import load_reranker, rerank_from_scores


BASELINE_RERANK_RRF_PRIOR_WEIGHT = 0.05
TOP_K = 5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--channel-limit", type=int, default=CANDIDATE_LIMIT)
    parser.add_argument("--candidate-limit", type=int, default=RERANK_CANDIDATE_LIMIT)
    parser.add_argument("--rerank-batch-size", type=int, default=RERANK_BATCH_SIZE)
    parser.add_argument("--no-enforce", action="store_true")
    args = parser.parse_args()
    if args.channel_limit < TOP_K:
        parser.error(f"--channel-limit must be at least {TOP_K}")
    if not TOP_K <= args.candidate_limit <= args.channel_limit * 2:
        parser.error(f"--candidate-limit must be between {TOP_K} and the fused candidate count")
    if args.rerank_batch_size < 1:
        parser.error("--rerank-batch-size must be at least 1")
    return args


def _baseline_rerank_from_scores(
    candidates: list[dict[str, Any]],
    scores: list[float] | np.ndarray,
) -> list[dict[str, Any]]:
    """Reproduce the pre-optimization cross-encoder plus min-max RRF ranking."""
    if not candidates:
        return []
    if len(candidates) != len(scores):
        raise ValueError("scores must contain one value for every candidate")
    rrf_scores = np.asarray([item["score"] for item in candidates], dtype=float)
    minimum, maximum = float(rrf_scores.min()), float(rrf_scores.max())
    scale = maximum - minimum or 1.0
    reranked = []
    for retrieval_rank, (item, raw_score) in enumerate(zip(candidates, scores), 1):
        normalized_rrf_score = (float(item["score"]) - minimum) / scale
        reranked.append(
            {
                **item,
                "score": float(raw_score) + BASELINE_RERANK_RRF_PRIOR_WEIGHT * normalized_rrf_score,
                "rrf_score": item["score"],
                "reranker_score": float(raw_score),
                "retrieval_rank": retrieval_rank,
            }
        )
    return sorted(
        reranked,
        key=lambda item: (-item["score"], item["retrieval_rank"], item["record_index"]),
    )


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def _metric_delta(baseline: dict[str, Any], optimized: dict[str, Any]) -> dict[str, float]:
    return {
        metric: round(float(optimized[metric]) - float(baseline[metric]), 4)
        for metric in ("recall_at_1", "recall_at_3", "recall_at_5", "mrr_at_5", "ndcg_at_5")
    }


def _group_metrics(
    cases: list[dict[str, Any]],
    rankings: dict[str, list[list[int]]],
    chunks: list[dict[str, Any]],
    group_values: list[str],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for method, method_rankings in rankings.items():
        output[method] = {}
        for group in sorted(set(group_values)):
            indices = [index for index, value in enumerate(group_values) if value == group]
            output[method][group] = _ranking_metrics(cases, indices, method_rankings, chunks, oracle_k=None)
    return output


def _validate_locked_cases(cases: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> None:
    if not cases:
        raise ValueError("The acceptance dataset must not be empty")
    chunk_ids = {chunk["chunk_id"] for chunk in chunks}
    source_files = {chunk["source_file"] for chunk in chunks}
    for case in cases:
        if not case.get("source_file"):
            raise ValueError(f"Acceptance case {case.get('id')} must be answerable")
        if case["source_file"] not in source_files:
            raise ValueError(f"Acceptance source is absent from the index: {case['source_file']}")
        qrels = case.get("relevant_chunk_ids") or []
        if not qrels:
            raise ValueError(f"Acceptance case {case.get('id')} requires exact chunk qrels")
        missing = [chunk_id for chunk_id in qrels if chunk_id not in chunk_ids]
        if missing:
            raise ValueError(f"Acceptance case {case.get('id')} has missing qrels: {missing}")


def main() -> None:
    args = _parse_args()
    dataset_path = args.dataset.resolve()
    report_path = args.report.resolve()
    dataset_path.relative_to(PROJECT_ROOT)
    report_path.relative_to(PROJECT_ROOT)

    cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    index = load_index(args.index_path)
    chunks = index["chunks"]
    _validate_locked_cases(cases, chunks)

    baseline_chunks = [{**chunk, "retrieval_text": chunk["text"]} for chunk in chunks]
    baseline_index = {**index, "bm25": build_bm25(baseline_chunks)}
    source_types = _source_type_by_file(chunks)
    model = load_reranker()

    rankings = {
        method: []
        for method in ("baseline_hybrid", "baseline_reranker", "optimized_hybrid", "optimized_reranker")
    }
    timings = {method: [] for method in ("baseline_retrieval", "optimized_retrieval", "shared_reranker")}
    details = []

    for case in cases:
        query = case["query"]
        vector_started = time.perf_counter()
        vector = vector_search(query, candidate_limit=args.channel_limit, index=index)
        vector_ms = (time.perf_counter() - vector_started) * 1000

        baseline_started = time.perf_counter()
        baseline_bm25 = bm25_search(query, candidate_limit=args.channel_limit, index=baseline_index)
        baseline_hybrid = rrf_fuse(
            baseline_bm25,
            vector,
            args.channel_limit,
            bm25_weight=RRF_BM25_WEIGHT,
            vector_weight=RRF_VECTOR_WEIGHT,
        )
        timings["baseline_retrieval"].append(vector_ms + (time.perf_counter() - baseline_started) * 1000)

        optimized_started = time.perf_counter()
        optimized_bm25 = bm25_search(query, candidate_limit=args.channel_limit, index=index)
        optimized_hybrid = rrf_fuse(
            optimized_bm25,
            vector,
            args.channel_limit,
            bm25_weight=RRF_BM25_WEIGHT,
            vector_weight=RRF_VECTOR_WEIGHT,
        )
        timings["optimized_retrieval"].append(vector_ms + (time.perf_counter() - optimized_started) * 1000)

        baseline_candidates = baseline_hybrid[: args.candidate_limit]
        optimized_candidates = optimized_hybrid[: args.candidate_limit]
        union_record_indices = list(
            dict.fromkeys(
                item["record_index"]
                for item in [*baseline_candidates, *optimized_candidates]
            )
        )
        reranker_query = expand_query(query)
        pairs = [(reranker_query, chunks[record_index]["text"]) for record_index in union_record_indices]
        reranker_started = time.perf_counter()
        raw_scores = model.predict(pairs, batch_size=args.rerank_batch_size, show_progress_bar=False)
        timings["shared_reranker"].append((time.perf_counter() - reranker_started) * 1000)
        scores_by_record = dict(zip(union_record_indices, map(float, raw_scores)))

        baseline_reranked = _baseline_rerank_from_scores(
            baseline_candidates,
            [scores_by_record[item["record_index"]] for item in baseline_candidates],
        )
        optimized_reranked = rerank_from_scores(
            query,
            optimized_candidates,
            chunks,
            [scores_by_record[item["record_index"]] for item in optimized_candidates],
        )

        method_rankings = {
            "baseline_hybrid": [item["record_index"] for item in baseline_hybrid],
            "baseline_reranker": [item["record_index"] for item in baseline_reranked],
            "optimized_hybrid": [item["record_index"] for item in optimized_hybrid],
            "optimized_reranker": [item["record_index"] for item in optimized_reranked],
        }
        for method, ranking in method_rankings.items():
            rankings[method].append(ranking)
        ranks = {
            method: _first_relevant_rank(case, ranking, chunks)
            for method, ranking in method_rankings.items()
        }
        details.append(
            {
                "id": case.get("id"),
                "query": query,
                "category": case["category"],
                "source_file": case["source_file"],
                "source_type": source_types[case["source_file"]],
                "relevant_chunk_ids": case["relevant_chunk_ids"],
                "ranks": ranks,
                "reranker_rank_change": (
                    None
                    if ranks["baseline_reranker"] is None or ranks["optimized_reranker"] is None
                    else ranks["baseline_reranker"] - ranks["optimized_reranker"]
                ),
            }
        )

    all_indices = list(range(len(cases)))
    metrics = {
        method: _ranking_metrics(cases, all_indices, method_rankings, chunks, oracle_k=None)
        for method, method_rankings in rankings.items()
    }
    comparison = {
        "hybrid": _metric_delta(metrics["baseline_hybrid"], metrics["optimized_hybrid"]),
        "reranker": _metric_delta(metrics["baseline_reranker"], metrics["optimized_reranker"]),
    }
    reranker_delta = comparison["reranker"]
    acceptance = {
        "optimized_recall_at_1_improves": reranker_delta["recall_at_1"] > 0,
        "optimized_mrr_at_5_improves": reranker_delta["mrr_at_5"] > 0,
        "optimized_recall_at_5_not_lower": reranker_delta["recall_at_5"] >= 0,
        "all_cases_have_exact_chunk_qrels": all(bool(case.get("relevant_chunk_ids")) for case in cases),
    }
    category_values = [case["category"] for case in cases]
    pdf_type_values = [source_types[case["source_file"]] for case in cases]
    report = {
        "evaluation_cases": len(cases),
        "dataset": dataset_path.relative_to(PROJECT_ROOT).as_posix(),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "index_path": args.index_path.resolve().relative_to(PROJECT_ROOT).as_posix(),
        "index_source_digest": index["metadata"]["source_digest"],
        "channel_limit": args.channel_limit,
        "candidate_limit": args.candidate_limit,
        "rerank_batch_size": args.rerank_batch_size,
        "baseline_policy": {
            "bm25_text": "original evidence text",
            "rerank_formula": "raw cross-encoder score + 0.05 * min-max-normalized RRF score",
            "top5_protection": False,
        },
        "optimized_policy": {
            "bm25_text": "table field-value retrieval text with a lossless wide-table fallback",
            "rerank_formula": "rank-normalized cross-encoder score + 0.16 / hybrid rank",
            "top5_protection": "high-confidence and table evidence",
        },
        "metrics": metrics,
        "comparison": comparison,
        "category_metrics": _group_metrics(cases, rankings, chunks, category_values),
        "pdf_type_metrics": _group_metrics(cases, rankings, chunks, pdf_type_values),
        "timings": {
            method: {"mean_ms": round(mean(values), 2), "p95_ms": round(_p95(values), 2)}
            for method, values in timings.items()
        },
        "acceptance": acceptance,
        "details": details,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, ensure_ascii=False, indent=2))
    if not args.no_enforce and not all(acceptance.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
