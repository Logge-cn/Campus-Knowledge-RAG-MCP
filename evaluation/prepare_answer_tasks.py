"""Prepare reproducible MCP-client answer tasks with retrieved evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval import retrieve, status


def prepare_tasks(
    cases: list[dict[str, Any]],
    limit: int,
    retrieve_fn: Callable[[str, int], dict[str, Any]] = retrieve,
) -> list[dict[str, Any]]:
    """Convert labeled questions into a client-neutral answer task bundle."""
    tasks = []
    for case in cases:
        payload = retrieve_fn(case["query"], limit)
        tasks.append(
            {
                "id": case["id"],
                "query": case["query"],
                "evidence_sufficient": payload["evidence_sufficient"],
                "confidence": payload["confidence"],
                "reason": payload["reason"],
                "evidence": [
                    {
                        "chunk_id": item["chunk_id"],
                        "source_file": item["source_file"],
                        "page": item["page"],
                        "text": item["text"],
                    }
                    for item in payload["results"]
                ],
                "prediction_contract": {
                    "answer": "string",
                    "refused": "boolean",
                    "cited_chunk_ids": "list[string]",
                    "review": {"correct": None, "complete": None, "citation_supported": None},
                },
            }
        )
    return tasks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--client", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-version", required=True)
    parser.add_argument("--limit", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_paths = [path.resolve() for path in args.dataset]
    output = args.output.resolve()
    for path in [*dataset_paths, output]:
        path.relative_to(PROJECT_ROOT)
    cases = [case for path in dataset_paths for case in json.loads(path.read_text(encoding="utf-8"))]
    tasks = prepare_tasks(cases, args.limit)
    manifest = {
        "schema_version": 1,
        "client": args.client,
        "model": args.model,
        "prompt_version": args.prompt_version,
        "retrieval_status": status(),
        "datasets": [
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in dataset_paths
        ],
        "tasks": tasks,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": output.relative_to(PROJECT_ROOT).as_posix(), "tasks": len(tasks)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
