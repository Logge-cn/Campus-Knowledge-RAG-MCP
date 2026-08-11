"""Compare cross-encoder and RRF-prior blending on the fixed evaluation set."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval import bm25_search, load_index, rerank, rrf_fuse, vector_search
from retrieval.config import RRF_BM25_WEIGHT, RRF_VECTOR_WEIGHT


CASES = json.loads((PROJECT_ROOT / "evaluation" / "dataset.json").read_text(encoding="utf-8"))
TOP_K = 5


def relevant(case: dict, chunk: dict) -> bool:
    return case["source_file"] == chunk["source_file"] and chunk["page"] in case["pages"]


def metrics(rankings: list[list[dict]], chunks: list[dict]) -> dict:
    groups = {
        "mixed": [index for index, case in enumerate(CASES) if case["source_file"]],
        "native": [
            index
            for index, case in enumerate(CASES)
            if case["source_file"] and any(chunk["source_file"] == case["source_file"] and chunk["source_type"] != "ocr" for chunk in chunks)
        ],
        "scanned": [
            index
            for index, case in enumerate(CASES)
            if case["source_file"] and all(chunk["source_file"] != case["source_file"] or chunk["source_type"] == "ocr" for chunk in chunks)
        ],
    }
    output = {}
    for group, indices in groups.items():
        ranks = [
            next((rank for rank, item in enumerate(rankings[index][:TOP_K], 1) if relevant(CASES[index], chunks[item["record_index"]])), None)
            for index in indices
        ]
        output[group] = {
            "recall_at_5": round(sum(rank is not None for rank in ranks) / len(ranks), 4),
            "mrr": round(sum(1 / rank if rank else 0 for rank in ranks) / len(ranks), 4),
        }
    return output


def blend(items: list[dict], rrf_weight: float) -> list[dict]:
    scores = [item["reranker_score"] for item in items]
    minimum, maximum = min(scores), max(scores)
    scale = maximum - minimum or 1.0
    return sorted(
        items,
        key=lambda item: (
            -((item["reranker_score"] - minimum) / scale + rrf_weight / item["retrieval_rank"]),
            item["retrieval_rank"],
        ),
    )


def main() -> None:
    index = load_index()
    all_reranked = []
    for case in CASES:
        bm25 = bm25_search(case["query"], 20, index=index)
        vector = vector_search(case["query"], 20, index=index)
        fused = rrf_fuse(bm25, vector, 20, bm25_weight=RRF_BM25_WEIGHT, vector_weight=RRF_VECTOR_WEIGHT)
        all_reranked.append(rerank(case["query"], fused, index["chunks"]))
    for weight in (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
        rankings = [blend(items, weight) for items in all_reranked]
        print(json.dumps({"rrf_prior_weight": weight, "metrics": metrics(rankings, index["chunks"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
