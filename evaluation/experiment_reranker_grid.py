"""Cache reranker scores once and compare score blending strategies cheaply."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval import bm25_search, load_index, rrf_fuse, vector_search
from retrieval.config import RERANK_BATCH_SIZE, RRF_BM25_WEIGHT, RRF_K, RRF_VECTOR_WEIGHT
from retrieval.query_expansion import expand_query
from retrieval.reranker import load_reranker, protect_hybrid_top_five


REPORTS_ROOT = PROJECT_ROOT / "evaluation" / "reports"
TOP_K = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, action="append")
    parser.add_argument("--index-path", type=Path)
    parser.add_argument("--name", default="development")
    parser.add_argument("--candidate-limit", type=int, default=20)
    parser.add_argument("--channel-limit", type=int, default=20)
    parser.add_argument("--input-variant", choices=("plain", "source"), default="plain")
    parser.add_argument("--query-variant", choices=("plain", "expanded"), default="plain")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--prior-weights",
        type=float,
        nargs="+",
        default=(0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5),
    )
    parser.add_argument(
        "--prior-methods",
        nargs="+",
        choices=("reciprocal_rank", "minmax_rrf", "raw_rrf"),
        default=("reciprocal_rank", "minmax_rrf", "raw_rrf"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset_paths(args: argparse.Namespace) -> list[Path]:
    return args.dataset or [PROJECT_ROOT / "evaluation" / "dataset.json"]


def load_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        case
        for path in dataset_paths(args)
        for case in json.loads(path.read_text(encoding="utf-8"))
        if case["source_file"] is not None
    ]


def dataset_sha256(args: argparse.Namespace) -> str:
    digest = hashlib.sha256()
    for path in dataset_paths(args):
        digest.update(path.resolve().as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def cache_path(args: argparse.Namespace) -> Path:
    query_suffix = "" if args.query_variant == "plain" else f"-q{args.query_variant}"
    return REPORTS_ROOT / (
        f"reranker-cache-{args.name}-{args.input_variant}{query_suffix}-c{args.candidate_limit}-r{args.channel_limit}.json"
    )


def report_path(args: argparse.Namespace) -> Path:
    query_suffix = "" if args.query_variant == "plain" else f"-q{args.query_variant}"
    return REPORTS_ROOT / (
        f"reranker-grid-{args.name}-{args.input_variant}{query_suffix}-c{args.candidate_limit}-r{args.channel_limit}.json"
    )


def input_text(record: dict[str, Any], variant: str) -> str:
    if variant == "plain":
        return record["text"]
    source_name = Path(record["source_file"]).stem
    source_type = {"pdf": "原生正文", "ocr": "OCR 正文", "table": "表格"}.get(
        record["source_type"], record["source_type"]
    )
    return f"文档：{source_name}\n内容类型：{source_type}\n{record['text']}"


def build_cache(args: argparse.Namespace) -> dict[str, Any]:
    if args.candidate_limit > args.channel_limit * 2:
        raise ValueError("candidate-limit cannot exceed the union of both channels")
    cases = load_cases(args)
    index = load_index(args.index_path) if args.index_path else load_index()
    chunks = index["chunks"]
    model = load_reranker()
    entries = []
    for case_index, case in enumerate(cases, 1):
        bm25 = bm25_search(case["query"], candidate_limit=args.channel_limit, index=index)
        vector = vector_search(case["query"], candidate_limit=args.channel_limit, index=index)
        fused = rrf_fuse(
            bm25,
            vector,
            args.candidate_limit,
            rrf_k=RRF_K,
            bm25_weight=RRF_BM25_WEIGHT,
            vector_weight=RRF_VECTOR_WEIGHT,
        )
        reranker_query = expand_query(case["query"]) if args.query_variant == "expanded" else case["query"]
        pairs = [(reranker_query, input_text(chunks[item["record_index"]], args.input_variant)) for item in fused]
        scores = model.predict(pairs, batch_size=RERANK_BATCH_SIZE, show_progress_bar=False).tolist()
        if len(scores) != len(fused):
            raise ValueError(f"Reranker returned {len(scores)} scores for {len(fused)} candidates")
        entries.append(
            {
                "query": case["query"],
                "reranker_query": reranker_query,
                "candidates": [
                    {
                        "record_index": item["record_index"],
                        "rrf_rank": rank,
                        "rrf_score": item["score"],
                        "reranker_score": float(score),
                    }
                    for rank, (item, score) in enumerate(zip(fused, scores), 1)
                ],
            }
        )
        print(f"scored {case_index}/{len(cases)}", flush=True)
    cache = {
        "metadata": {
            "dataset_sha256": dataset_sha256(args),
            "datasets": [path.resolve().as_posix() for path in dataset_paths(args)],
            "index_source_digest": index["metadata"]["source_digest"],
            "candidate_limit": args.candidate_limit,
            "channel_limit": args.channel_limit,
            "input_variant": args.input_variant,
            "query_variant": args.query_variant,
            "rrf_k": RRF_K,
            "rrf_bm25_weight": RRF_BM25_WEIGHT,
            "rrf_vector_weight": RRF_VECTOR_WEIGHT,
        },
        "entries": entries,
    }
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    cache_path(args).write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache


def load_cache(args: argparse.Namespace) -> dict[str, Any]:
    path = cache_path(args)
    if args.rebuild_cache or not path.exists():
        return build_cache(args)
    cache = json.loads(path.read_text(encoding="utf-8"))
    if cache["metadata"]["dataset_sha256"] != dataset_sha256(args):
        cases = load_cases(args)
        if [entry["query"] for entry in cache["entries"]] != [case["query"] for case in cases]:
            raise ValueError("Dataset queries changed; rebuild the reranker cache")
    index = load_index(args.index_path) if args.index_path else load_index()
    if cache["metadata"]["index_source_digest"] != index["metadata"]["source_digest"]:
        raise ValueError("Index changed; rebuild the reranker cache")
    return cache


def normalize(candidates: list[dict[str, Any]], method: str) -> list[float]:
    scores = [item["reranker_score"] for item in candidates]
    if method == "identity":
        return scores
    if method == "sqrt":
        return [math.sqrt(max(0.0, score)) for score in scores]
    if method == "log1p":
        return [math.log1p(max(0.0, score)) for score in scores]
    if method == "minmax":
        minimum, maximum = min(scores), max(scores)
        scale = maximum - minimum or 1.0
        return [(score - minimum) / scale for score in scores]
    if method == "sigmoid":
        return [1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, score)))) for score in scores]
    if method == "zscore_sigmoid":
        center = mean(scores)
        scale = pstdev(scores) or 1.0
        return [1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, (score - center) / scale)))) for score in scores]
    if method == "rank":
        order = sorted(range(len(scores)), key=lambda index: (-scores[index], candidates[index]["rrf_rank"]))
        ranks = [0] * len(scores)
        for rank, index in enumerate(order, 1):
            ranks[index] = rank
        denominator = max(1, len(scores) - 1)
        return [1.0 - (rank - 1) / denominator for rank in ranks]
    raise ValueError(f"Unknown normalization: {method}")


def prior_scores(candidates: list[dict[str, Any]], method: str) -> list[float]:
    if method == "reciprocal_rank":
        return [1.0 / item["rrf_rank"] for item in candidates]
    scores = [item["rrf_score"] for item in candidates]
    if method == "raw_rrf":
        return scores
    if method == "minmax_rrf":
        minimum, maximum = min(scores), max(scores)
        scale = maximum - minimum or 1.0
        return [(score - minimum) / scale for score in scores]
    raise ValueError(f"Unknown prior method: {method}")


def rank_candidates(
    candidates: list[dict[str, Any]],
    normalization: str,
    prior_weight: float,
    prior_method: str = "reciprocal_rank",
) -> list[int]:
    normalized = normalize(candidates, normalization)
    priors = prior_scores(candidates, prior_method)
    ranked = sorted(
        zip(candidates, normalized, priors),
        key=lambda pair: (
            -(pair[1] + prior_weight * pair[2]),
            pair[0]["rrf_rank"],
            pair[0]["record_index"],
        ),
    )
    return [item["record_index"] for item, _, _ in ranked]


def relevant_rank(case: dict[str, Any], ranking: list[int], chunks: list[dict[str, Any]]) -> int | None:
    return next(
        (
            rank
            for rank, record_index in enumerate(ranking, 1)
            if (
                chunks[record_index]["chunk_id"] in case["relevant_chunk_ids"]
                if case.get("relevant_chunk_ids")
                else chunks[record_index]["source_file"] == case["source_file"]
                and chunks[record_index]["page"] in case["pages"]
            )
        ),
        None,
    )


def metrics(ranks: list[int | None]) -> dict[str, float | int]:
    count = len(ranks)
    return {
        "cases": count,
        "hits_at_1": sum(rank == 1 for rank in ranks),
        "recall_at_1": round(sum(rank == 1 for rank in ranks) / count, 4),
        "recall_at_3": round(sum(rank is not None and rank <= 3 for rank in ranks) / count, 4),
        "recall_at_5": round(sum(rank is not None and rank <= TOP_K for rank in ranks) / count, 4),
        "mrr_at_5": round(mean(1 / rank if rank is not None and rank <= TOP_K else 0.0 for rank in ranks), 4),
        "oracle_recall": round(sum(rank is not None for rank in ranks) / count, 4),
    }


def evaluate(cache: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cases = load_cases(args)
    index = load_index(args.index_path) if args.index_path else load_index()
    chunks = index["chunks"]
    answerable = [index for index, case in enumerate(cases) if case["source_file"] is not None]
    source_types: dict[str, str] = {}
    for chunk in chunks:
        current = source_types.get(chunk["source_file"], "native")
        source_types[chunk["source_file"]] = "scanned" if chunk["source_type"] == "ocr" else current

    configurations = []
    for normalization in ("identity", "sqrt", "log1p", "minmax", "sigmoid", "zscore_sigmoid", "rank"):
        for prior_method, prior_weight in product(args.prior_methods, args.prior_weights):
            rankings = [
                rank_candidates(
                    cache["entries"][case_index]["candidates"],
                    normalization,
                    prior_weight,
                    prior_method,
                )
                for case_index in range(len(cases))
            ]
            rankings = [
                protect_hybrid_top_five(
                    cases[case_index]["query"],
                    [item["record_index"] for item in cache["entries"][case_index]["candidates"]],
                    ranking,
                    chunks,
                    {
                        item["record_index"]: item["reranker_score"]
                        for item in cache["entries"][case_index]["candidates"]
                    },
                )
                for case_index, ranking in enumerate(rankings)
            ]
            ranks = {
                case_index: relevant_rank(cases[case_index], rankings[case_index], chunks)
                for case_index in answerable
            }
            groups: dict[str, list[int | None]] = defaultdict(list)
            for case_index in answerable:
                case = cases[case_index]
                rank = ranks[case_index]
                groups["overall"].append(rank)
                groups[f"category:{case['category']}"] .append(rank)
                groups[f"source_type:{source_types[case['source_file']]}"] .append(rank)
                groups[
                    "label_type:exact_chunk" if case.get("relevant_chunk_ids") else "label_type:page_fallback"
                ].append(rank)
            hybrid_ranks = {
                case_index: relevant_rank(
                    cases[case_index],
                    [item["record_index"] for item in cache["entries"][case_index]["candidates"]],
                    chunks,
                )
                for case_index in answerable
            }
            configurations.append(
                {
                    "normalization": normalization,
                    "prior_method": prior_method,
                    "prior_weight": prior_weight,
                    "metrics": {name: metrics(values) for name, values in sorted(groups.items())},
                    "movements": {
                        "promoted_to_1": sum(hybrid_ranks[index] != 1 and ranks[index] == 1 for index in answerable),
                        "demoted_from_1": sum(hybrid_ranks[index] == 1 and ranks[index] != 1 for index in answerable),
                        "entered_top5": sum(
                            (hybrid_ranks[index] is None or hybrid_ranks[index] > TOP_K)
                            and ranks[index] is not None
                            and ranks[index] <= TOP_K
                            for index in answerable
                        ),
                        "dropped_from_top5": sum(
                            hybrid_ranks[index] is not None
                            and hybrid_ranks[index] <= TOP_K
                            and (ranks[index] is None or ranks[index] > TOP_K)
                            for index in answerable
                        ),
                    },
                }
            )
    configurations.sort(
        key=lambda item: (
            -item["metrics"]["overall"]["recall_at_5"],
            -item["metrics"]["overall"]["mrr_at_5"],
            -item["metrics"]["overall"]["recall_at_1"],
        )
    )
    best = configurations[0]
    best_rankings = [
        rank_candidates(
            entry["candidates"],
            best["normalization"],
            best["prior_weight"],
            best["prior_method"],
        )
        for entry in cache["entries"]
    ]
    best_rankings = [
        protect_hybrid_top_five(
            cases[case_index]["query"],
            [item["record_index"] for item in cache["entries"][case_index]["candidates"]],
            ranking,
            chunks,
            {
                item["record_index"]: item["reranker_score"]
                for item in cache["entries"][case_index]["candidates"]
            },
        )
        for case_index, ranking in enumerate(best_rankings)
    ]
    best_details = [
        {
            "id": cases[case_index].get("id"),
            "query": cases[case_index]["query"],
            "category": cases[case_index]["category"],
            "source_file": cases[case_index]["source_file"],
            "pages": cases[case_index]["pages"],
            "hybrid_rank": relevant_rank(
                cases[case_index],
                [item["record_index"] for item in cache["entries"][case_index]["candidates"]],
                chunks,
            ),
            "reranker_rank": relevant_rank(cases[case_index], best_rankings[case_index], chunks),
        }
        for case_index in answerable
    ]
    report = {
        "cache_metadata": cache["metadata"],
        "evaluation_dataset_sha256": dataset_sha256(args),
        "configurations": configurations,
        "best_details": best_details,
    }
    output = report_path(args)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(configurations[:10], ensure_ascii=False, indent=2))
    print(f"wrote {output.relative_to(PROJECT_ROOT)}")
    return report


def main() -> None:
    args = parse_args()
    cache = load_cache(args)
    evaluate(cache, args)


if __name__ == "__main__":
    main()
