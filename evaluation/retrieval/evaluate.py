"""Compare legacy lexical, BM25, embedding and hybrid retrieval on the fixed evaluation set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval import bm25_search, load_index, rerank, rrf_fuse, tokenize, vector_search
from retrieval.config import (
    CANDIDATE_LIMIT,
    DEFAULT_INDEX_PATH,
    RERANK_BATCH_SIZE,
    RERANK_CANDIDATE_LIMIT,
    RRF_BM25_WEIGHT,
    RRF_VECTOR_WEIGHT,
)


EVALUATION_PATH = PROJECT_ROOT / "evaluation" / "evidence" / "development.json"
REPORT_PATH = PROJECT_ROOT / "runtime" / "reports" / "retrieval" / "latest.json"
LEGACY_DIMENSIONS = 512
TOP_K = 5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        action="append",
        help="Evaluation JSON array. Repeat to evaluate several compatible datasets as one split.",
    )
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--channel-limit", type=int, default=CANDIDATE_LIMIT)
    parser.add_argument("--candidate-limit", type=int, default=RERANK_CANDIDATE_LIMIT)
    parser.add_argument("--rerank-batch-size", type=int, default=RERANK_BATCH_SIZE)
    refusal = parser.add_mutually_exclusive_group()
    refusal.add_argument("--calibrate-refusal", action="store_true")
    refusal.add_argument("--refusal-threshold", type=float)
    parser.add_argument("--no-enforce", action="store_true")
    args = parser.parse_args()
    if args.channel_limit < TOP_K:
        parser.error(f"--channel-limit must be at least {TOP_K}")
    if args.candidate_limit < TOP_K:
        parser.error(f"--candidate-limit must be at least {TOP_K}")
    if args.candidate_limit > args.channel_limit * 2:
        parser.error("--candidate-limit cannot exceed the union of both retrieval channels")
    if args.rerank_batch_size < 1:
        parser.error("--rerank-batch-size must be at least 1")
    if not args.dataset:
        args.dataset = [EVALUATION_PATH]
    return args


def _bucket(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % LEGACY_DIMENSIONS


def _legacy_rank(query: str, chunks: list[dict], limit: int) -> list[int]:
    document_frequency: Counter[str] = Counter()
    for chunk in chunks:
        document_frequency.update(set(tokenize(chunk["text"])))
    count = len(chunks)
    idf = {token: math.log((count + 1) / (frequency + 1)) + 1 for token, frequency in document_frequency.items()}
    default_idf = math.log(count + 1) + 1

    def vector(text: str) -> list[float]:
        values = [0.0] * LEGACY_DIMENSIONS
        for token, frequency in Counter(tokenize(text)).items():
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
    if case.get("relevant_chunk_ids"):
        return chunk["chunk_id"] in case["relevant_chunk_ids"]
    return case["source_file"] == chunk["source_file"] and chunk["page"] in case["pages"]


def _source_type_by_file(chunks: list[dict]) -> dict[str, str]:
    source_types: dict[str, str] = {}
    for chunk in chunks:
        source_file = chunk["source_file"]
        current = source_types.get(source_file, "native")
        source_types[source_file] = "scanned" if chunk["source_type"] == "ocr" else current
    return source_types


def _first_relevant_rank(case: dict, ranking: list[int], chunks: list[dict], limit: int | None = None) -> int | None:
    active = ranking if limit is None else ranking[:limit]
    return next(
        (rank for rank, chunk_index in enumerate(active, 1) if _is_relevant(case, chunks[chunk_index])),
        None,
    )


def _ndcg_at_k(case: dict, ranking: list[int], chunks: list[dict], k: int = TOP_K) -> float:
    gains = [1.0 if _is_relevant(case, chunks[index]) else 0.0 for index in ranking[:k]]
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))
    relevant_count = sum(_is_relevant(case, chunk) for chunk in chunks)
    ideal_count = min(relevant_count, k)
    if ideal_count == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / idcg


def _ranking_metrics(
    cases: list[dict],
    case_indices: list[int],
    rankings: list[list[int]],
    chunks: list[dict],
    *,
    oracle_k: int | None = 20,
) -> dict:
    ranks = [_first_relevant_rank(cases[index], rankings[index], chunks) for index in case_indices]
    ndcg_values = [_ndcg_at_k(cases[index], rankings[index], chunks) for index in case_indices]
    count = len(case_indices)
    if count == 0:
        output = {
            "cases": 0,
            "hits_at_1": 0,
            "hits_at_5": 0,
            "recall_at_1": None,
            "recall_at_3": None,
            "recall_at_5": None,
            "mrr": None,
            "mrr_at_5": None,
            "ndcg_at_5": None,
            "candidate_recall": None,
        }
        if oracle_k is not None:
            output[f"oracle_recall_at_{oracle_k}"] = None
        return output
    mrr_at_5 = round(mean(1 / rank if rank is not None and rank <= TOP_K else 0.0 for rank in ranks), 4)
    output = {
        "cases": count,
        "hits_at_1": sum(rank == 1 for rank in ranks),
        "hits_at_5": sum(rank is not None and rank <= TOP_K for rank in ranks),
        "recall_at_1": round(sum(rank == 1 for rank in ranks) / count, 4),
        "recall_at_3": round(sum(rank is not None and rank <= 3 for rank in ranks) / count, 4),
        "recall_at_5": round(sum(rank is not None and rank <= TOP_K for rank in ranks) / count, 4),
        "mrr": mrr_at_5,
        "mrr_at_5": mrr_at_5,
        "ndcg_at_5": round(mean(ndcg_values), 4),
        "candidate_recall": round(sum(rank is not None for rank in ranks) / count, 4),
    }
    if oracle_k is not None:
        output[f"oracle_recall_at_{oracle_k}"] = round(
            sum(rank is not None and rank <= oracle_k for rank in ranks) / count,
            4,
        )
    return output


def _metrics(cases: list[dict], rankings: dict[str, list[list[int]]], chunks: list[dict], timings: dict[str, list[float]]) -> dict:
    answerable = [index for index, case in enumerate(cases) if case["source_file"] is not None]
    output = {}
    for method, method_rankings in rankings.items():
        method_metrics = _ranking_metrics(
            cases,
            answerable,
            method_rankings,
            chunks,
            oracle_k=None if method == "reranker" else 20,
        )
        ordered_timings = sorted(timings[method])
        p95_index = max(0, math.ceil(len(ordered_timings) * 0.95) - 1)
        method_metrics.update(
            {
                "mean_query_ms": round(mean(timings[method]), 2),
                "p95_query_ms": round(ordered_timings[p95_index], 2),
            }
        )
        output[method] = method_metrics
    return output


def _category_metrics(cases: list[dict], rankings: dict[str, list[list[int]]], chunks: list[dict]) -> dict:
    categories = sorted({case["category"] for case in cases if case["source_file"] is not None})
    output = {}
    for method, method_rankings in rankings.items():
        output[method] = {}
        for category in categories:
            case_indices = [
                index
                for index, case in enumerate(cases)
                if case["source_file"] is not None and case["category"] == category
            ]
            output[method][category] = _ranking_metrics(
                cases,
                case_indices,
                method_rankings,
                chunks,
                oracle_k=None if method == "reranker" else 20,
            )
    return output


def _pdf_type_metrics(cases: list[dict], rankings: dict[str, list[list[int]]], chunks: list[dict]) -> dict:
    source_types = _source_type_by_file(chunks)
    output = {}
    for method, method_rankings in rankings.items():
        output[method] = {}
        for pdf_type in ("native", "scanned"):
            case_indices = [
                index
                for index, case in enumerate(cases)
                if case["source_file"] is not None and source_types[case["source_file"]] == pdf_type
            ]
            output[method][pdf_type] = _ranking_metrics(
                cases,
                case_indices,
                method_rankings,
                chunks,
                oracle_k=None if method == "reranker" else 20,
            )
    return output


def _ranking_diagnostics(details: list[dict]) -> dict:
    output = {}
    for method in ("bm25", "vector", "hybrid", "reranker"):
        ranks = [detail["ranks"][method] for detail in details if detail["expected"]["source_file"] is not None]
        diagnostics = {
            "not_recalled_in_candidates": sum(rank is None for rank in ranks),
            "recalled_in_candidates_but_missed_at_5": sum(rank is not None and rank > TOP_K for rank in ranks),
            "hit_at_5_but_not_rank_1": sum(rank is not None and 1 < rank <= TOP_K for rank in ranks),
        }
        if method != "reranker":
            diagnostics["not_recalled_at_20"] = diagnostics["not_recalled_in_candidates"]
            diagnostics["recalled_at_20_but_missed_at_5"] = diagnostics[
                "recalled_in_candidates_but_missed_at_5"
            ]
        output[method] = diagnostics
    return output


def _top1_diagnostics(details: list[dict]) -> dict:
    answerable = [detail for detail in details if detail["expected"]["source_file"] is not None]
    output: dict[str, Any] = {"by_pdf_type": {}, "worst_cases": []}
    for pdf_type in ("native", "scanned"):
        selected = [detail for detail in answerable if detail.get("source_type") == pdf_type]
        misses = [detail for detail in selected if detail["ranks"]["reranker"] != 1]
        categories = Counter(detail["category"] for detail in misses)
        output["by_pdf_type"][pdf_type] = {
            "cases": len(selected),
            "hits_at_1": len(selected) - len(misses),
            "recall_at_1": round((len(selected) - len(misses)) / len(selected), 4) if selected else None,
            "top1_misses": len(misses),
            "misses_by_category": dict(sorted(categories.items())),
        }
    worst = sorted(
        (detail for detail in answerable if detail["ranks"]["reranker"] != 1),
        key=lambda detail: (-(detail["ranks"]["reranker"] or 10**9), detail["query"]),
    )
    output["worst_cases"] = [
        {
            "id": detail.get("id"),
            "query": detail["query"],
            "source_type": detail.get("source_type"),
            "category": detail["category"],
            "hybrid_rank": detail["ranks"]["hybrid"],
            "reranker_rank": detail["ranks"]["reranker"],
        }
        for detail in worst[:20]
    ]
    return output


def _score_summary(scores: list[float]) -> dict:
    if not scores:
        return {"cases": 0, "minimum": None, "mean": None, "maximum": None}
    return {
        "cases": len(scores),
        "minimum": round(min(scores), 6),
        "mean": round(mean(scores), 6),
        "maximum": round(max(scores), 6),
    }


def _refusal_metrics(details: list[dict], threshold: float) -> dict:
    scored = [detail for detail in details if detail.get("top_reranker_score") is not None]
    answerable = [detail for detail in scored if detail["expected"]["source_file"] is not None]
    no_answer = [detail for detail in scored if detail["expected"]["source_file"] is None]
    true_answer = sum(detail["top_reranker_score"] >= threshold for detail in answerable)
    false_refusal = len(answerable) - true_answer
    false_answer = sum(detail["top_reranker_score"] >= threshold for detail in no_answer)
    true_refusal = len(no_answer) - false_answer
    answerable_accept_rate = true_answer / len(answerable) if answerable else 0.0
    refusal_recall = true_refusal / len(no_answer) if no_answer else 0.0
    balanced_accuracy = mean((answerable_accept_rate, refusal_recall)) if answerable and no_answer else 0.0
    total = len(scored)
    return {
        "threshold": threshold,
        "cases": total,
        "true_answer": true_answer,
        "false_refusal": false_refusal,
        "true_refusal": true_refusal,
        "false_answer": false_answer,
        "answerable_accept_rate": round(answerable_accept_rate, 4),
        "refusal_recall": round(refusal_recall, 4),
        "false_answer_rate": round(false_answer / len(no_answer), 4) if no_answer else None,
        "overall_accuracy": round((true_answer + true_refusal) / total, 4) if total else None,
        "balanced_accuracy": round(balanced_accuracy, 4),
    }


def _calibrate_refusal_threshold(details: list[dict]) -> dict:
    scores = sorted({detail["top_reranker_score"] for detail in details if detail.get("top_reranker_score") is not None})
    if not scores:
        raise ValueError("Cannot calibrate refusal without reranker scores")
    epsilon = 1e-9
    thresholds = [scores[0] - epsilon, *scores, scores[-1] + epsilon]
    candidates = [_refusal_metrics(details, threshold) for threshold in thresholds]
    return max(
        candidates,
        key=lambda item: (
            item["balanced_accuracy"],
            item["overall_accuracy"],
            item["refusal_recall"],
            item["threshold"],
        ),
    )


def _refusal_breakdown(details: list[dict], threshold: float) -> dict:
    answerable = [
        detail
        for detail in details
        if detail["expected"]["source_file"] is not None and detail.get("top_reranker_score") is not None
    ]
    no_answer = [
        detail
        for detail in details
        if detail["expected"]["source_file"] is None and detail.get("top_reranker_score") is not None
    ]
    by_pdf_type = {}
    for pdf_type in sorted({detail.get("source_type") or "unknown" for detail in answerable}):
        selected = [detail for detail in answerable if (detail.get("source_type") or "unknown") == pdf_type]
        accepted = sum(detail["top_reranker_score"] >= threshold for detail in selected)
        by_pdf_type[pdf_type] = {
            "cases": len(selected),
            "answerable_accept_rate": round(accepted / len(selected), 4),
            "false_refusal_rate": round((len(selected) - accepted) / len(selected), 4),
        }
    by_no_answer_type = {}
    for no_answer_type in sorted({detail.get("no_answer_type") or "unspecified" for detail in no_answer}):
        selected = [
            detail for detail in no_answer if (detail.get("no_answer_type") or "unspecified") == no_answer_type
        ]
        false_answers = sum(detail["top_reranker_score"] >= threshold for detail in selected)
        by_no_answer_type[no_answer_type] = {
            "cases": len(selected),
            "refusal_recall": round((len(selected) - false_answers) / len(selected), 4),
            "false_answer_rate": round(false_answers / len(selected), 4),
        }
    return {"answerable_by_pdf_type": by_pdf_type, "no_answer_by_type": by_no_answer_type}


def _refusal_report(details: list[dict], *, calibrate: bool, threshold: float | None) -> dict:
    answerable_scores = [
        detail["top_reranker_score"]
        for detail in details
        if detail["expected"]["source_file"] is not None and detail.get("top_reranker_score") is not None
    ]
    no_answer_scores = [
        detail["top_reranker_score"]
        for detail in details
        if detail["expected"]["source_file"] is None and detail.get("top_reranker_score") is not None
    ]
    report = {
        "score_summary": {
            "answerable": _score_summary(answerable_scores),
            "no_answer": _score_summary(no_answer_scores),
        }
    }
    if calibrate:
        metrics = _calibrate_refusal_threshold(details)
        report.update(
            {
                "mode": "calibrated_on_this_dataset",
                "metrics": metrics,
                "breakdown": _refusal_breakdown(details, metrics["threshold"]),
            }
        )
    elif threshold is not None:
        report.update(
            {
                "mode": "fixed_threshold",
                "metrics": _refusal_metrics(details, threshold),
                "breakdown": _refusal_breakdown(details, threshold),
            }
        )
    else:
        report.update(
            {
                "mode": "not_scored",
                "metrics": None,
                "note": "Use --calibrate-refusal on development data, then --refusal-threshold on locked data.",
            }
        )
    return report


def main() -> None:
    args = _parse_args()
    dataset_paths = [path.resolve() for path in args.dataset]
    report_path = args.report.resolve()
    for dataset_path in dataset_paths:
        dataset_path.relative_to(PROJECT_ROOT)
    report_path.relative_to(PROJECT_ROOT)
    cases = [
        case
        for dataset_path in dataset_paths
        for case in json.loads(dataset_path.read_text(encoding="utf-8"))
    ]
    if not cases:
        raise ValueError("The evaluation set must not be empty")
    index = load_index(args.index_path)
    chunks = index["chunks"]
    source_types = _source_type_by_file(chunks)
    rankings = {method: [] for method in ("legacy", "bm25", "vector", "hybrid", "reranker")}
    timings = {method: [] for method in rankings}
    details = []

    for case in cases:
        started = time.perf_counter()
        legacy = _legacy_rank(case["query"], chunks, args.channel_limit)
        timings["legacy"].append((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        bm25 = bm25_search(case["query"], candidate_limit=args.channel_limit, index=index)
        timings["bm25"].append((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        vector = vector_search(case["query"], candidate_limit=args.channel_limit, index=index)
        timings["vector"].append((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        hybrid = rrf_fuse(
            bm25,
            vector,
            args.channel_limit,
            bm25_weight=RRF_BM25_WEIGHT,
            vector_weight=RRF_VECTOR_WEIGHT,
        )
        fusion_ms = (time.perf_counter() - started) * 1000
        timings["hybrid"].append(timings["bm25"][-1] + timings["vector"][-1] + fusion_ms)

        started = time.perf_counter()
        reranked = rerank(
            case["query"],
            hybrid[: args.candidate_limit],
            chunks,
            batch_size=args.rerank_batch_size,
        )
        timings["reranker"].append(timings["hybrid"][-1] + (time.perf_counter() - started) * 1000)

        method_candidates = {
            "legacy": legacy,
            "bm25": [item["record_index"] for item in bm25],
            "vector": [item["record_index"] for item in vector],
            "hybrid": [item["record_index"] for item in hybrid],
            "reranker": [item["record_index"] for item in reranked],
        }
        for method, indices in method_candidates.items():
            rankings[method].append(indices)
        ranks = {
            method: _first_relevant_rank(case, indices, chunks)
            if case["source_file"] is not None
            else None
            for method, indices in method_candidates.items()
        }
        details.append(
            {
                "id": case.get("id"),
                "query": case["query"],
                "category": case["category"],
                "source_type": source_types.get(case["source_file"]) if case["source_file"] is not None else None,
                "no_answer_type": case.get("no_answer_type"),
                "expected": {
                    "source_file": case["source_file"],
                    "pages": case["pages"],
                    "relevant_chunk_ids": case.get("relevant_chunk_ids"),
                },
                "hits": {
                    method: any(_is_relevant(case, chunks[index_value]) for index_value in indices[:TOP_K])
                    if case["source_file"] is not None
                    else None
                    for method, indices in method_candidates.items()
                },
                "ranks": ranks,
                "top_reranker_score": reranked[0]["reranker_score"] if reranked else None,
            }
        )

    metrics = _metrics(cases, rankings, chunks, timings)
    pdf_type_metrics = _pdf_type_metrics(cases, rankings, chunks)
    dataset_records = [
        {
            "path": dataset_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        }
        for dataset_path in dataset_paths
    ]
    combined_dataset_digest = hashlib.sha256()
    for record in dataset_records:
        combined_dataset_digest.update(record["path"].encode("utf-8"))
        combined_dataset_digest.update(b"\0")
        combined_dataset_digest.update(record["sha256"].encode("ascii"))
        combined_dataset_digest.update(b"\0")
    report = {
        "evaluation_cases": len(cases),
        "dataset": dataset_records[0]["path"] if len(dataset_records) == 1 else None,
        "datasets": dataset_records,
        "index_path": args.index_path.resolve().relative_to(PROJECT_ROOT).as_posix(),
        "dataset_sha256": (
            dataset_records[0]["sha256"] if len(dataset_records) == 1 else combined_dataset_digest.hexdigest()
        ),
        "answerable_cases": sum(case["source_file"] is not None for case in cases),
        "no_answer_cases": sum(case["source_file"] is None for case in cases),
        "top_k": TOP_K,
        "channel_limit": args.channel_limit,
        "candidate_limit": args.candidate_limit,
        "rerank_batch_size": args.rerank_batch_size,
        "metrics": metrics,
        "category_metrics": _category_metrics(cases, rankings, chunks),
        "pdf_type_metrics": pdf_type_metrics,
        "ranking_diagnostics": _ranking_diagnostics(details),
        "top1_diagnostics": _top1_diagnostics(details),
        "refusal": _refusal_report(
            details,
            calibrate=args.calibrate_refusal,
            threshold=args.refusal_threshold,
        ),
        "label_coverage": {
            "answerable_with_relevant_chunk_ids": sum(
                case["source_file"] is not None and bool(case.get("relevant_chunk_ids")) for case in cases
            ),
            "answerable_with_gold_answer": sum(
                case["source_file"] is not None and bool(case.get("gold_answer")) for case in cases
            ),
            "answerable_with_required_facts": sum(
                case["source_file"] is not None and bool(case.get("required_facts")) for case in cases
            ),
        },
        "index": index["metadata"],
        "acceptance": {
            "hybrid_recall_not_below_single_retrievers": metrics["hybrid"]["recall_at_5"]
            >= max(metrics["bm25"]["recall_at_5"], metrics["vector"]["recall_at_5"]),
            "semantic_upgrade_improves_over_legacy": metrics["hybrid"]["recall_at_5"]
            > metrics["legacy"]["recall_at_5"],
            "reranker_recall_not_below_hybrid": metrics["reranker"]["recall_at_5"] >= metrics["hybrid"]["recall_at_5"],
            "reranker_mrr_improves_hybrid": metrics["reranker"]["mrr"] > metrics["hybrid"]["mrr"],
            "reranker_ndcg_not_below_hybrid": metrics["reranker"]["ndcg_at_5"]
            >= metrics["hybrid"]["ndcg_at_5"],
            "reranker_recall_not_below_hybrid_by_pdf_type": all(
                pdf_type_metrics["reranker"][pdf_type]["recall_at_5"]
                >= pdf_type_metrics["hybrid"][pdf_type]["recall_at_5"]
                for pdf_type in ("native", "scanned")
            ),
            "reranker_mrr_improves_hybrid_by_pdf_type": all(
                pdf_type_metrics["reranker"][pdf_type]["mrr"] > pdf_type_metrics["hybrid"][pdf_type]["mrr"]
                for pdf_type in ("native", "scanned")
            ),
        },
        "note": (
            "Refusal metrics use the top reranker score only. Calibrate on development data and apply the fixed "
            "threshold to locked data; retrieval metrics still exclude no-answer cases."
        ),
        "details": details,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, ensure_ascii=False, indent=2))
    if not args.no_enforce and not all(report["acceptance"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
