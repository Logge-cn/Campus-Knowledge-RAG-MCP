"""Measure retrieval fusion settings on the fixed evaluation set."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval import bm25_search, load_index, rrf_fuse, vector_search


CASES = json.loads((PROJECT_ROOT / "evaluation" / "dataset.json").read_text(encoding="utf-8"))
TOP_K = 5


def relevant(case: dict, chunk: dict) -> bool:
    return case["source_file"] == chunk["source_file"] and chunk["page"] in case["pages"]


def score(rankings: list[list[int]], chunks: list[dict]) -> tuple[float, float]:
    answerable = [(case, ranking) for case, ranking in zip(CASES, rankings) if case["source_file"]]
    ranks = [
        next((rank for rank, index in enumerate(ranking[:TOP_K], 1) if relevant(case, chunks[index])), None)
        for case, ranking in answerable
    ]
    return (
        round(sum(rank is not None for rank in ranks) / len(ranks), 4),
        round(sum(1 / rank if rank else 0 for rank in ranks) / len(ranks), 4),
    )


def main() -> None:
    index = load_index()
    chunks = index["chunks"]
    lexical = [bm25_search(case["query"], candidate_limit=20, index=index) for case in CASES]
    semantic = [vector_search(case["query"], candidate_limit=20, index=index) for case in CASES]
    settings = [(1, 1, 1), (1, 2, 1), (1, 3, 1), (1, 4, 1), (1, 5, 1), (1, 10, 1), (1, 20, 1), (5, 1, 1), (10, 1, 1), (20, 1, 1)]
    for rrf_k, lexical_weight, semantic_weight in settings:
        rankings = [
            [item["record_index"] for item in rrf_fuse(left, right, 20, rrf_k=rrf_k, bm25_weight=lexical_weight, vector_weight=semantic_weight)]
            for left, right in zip(lexical, semantic)
        ]
        recall, mrr = score(rankings, chunks)
        print(json.dumps({"rrf_k": rrf_k, "bm25_weight": lexical_weight, "vector_weight": semantic_weight, "recall_at_5": recall, "mrr": mrr}, ensure_ascii=False))


if __name__ == "__main__":
    main()
