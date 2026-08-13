"""Benchmark cold initialization, uncached retrieval and cached retrieval."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import psutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval import clear_search_cache, retrieve, status
from retrieval.config import DEFAULT_INDEX_PATH, inside_project
from retrieval.hybrid import profile_search
from retrieval.runtime import warmup


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * quantile) - 1)]


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": round(mean(values), 2),
        "p50_ms": round(percentile(values, 0.50), 2),
        "p95_ms": round(percentile(values, 0.95), 2),
        "maximum_ms": round(max(values), 2),
    }


def hardware_info() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or "unknown",
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "concurrency": 1,
    }


def index_size_bytes(index_path: Path) -> int:
    root = inside_project(index_path).parent
    return sum(path.stat().st_size for path in root.iterdir() if path.is_file())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--limit", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = args.output.resolve()
    output.relative_to(PROJECT_ROOT)
    index_path = inside_project(args.index_path)
    process = psutil.Process()
    memory_before = process.memory_info()
    cold_started = time.perf_counter()
    warmup_result = warmup(index_path)
    cold_ms = (time.perf_counter() - cold_started) * 1000
    uncached: list[float] = []
    cached: list[float] = []
    phases: dict[str, list[float]] = {}
    for query in args.query:
        clear_search_cache()
        profile = profile_search(query, args.limit, index_path)
        uncached.append(profile["timings"]["total_ms"])
        for name, value in profile["timings"].items():
            if name != "total_ms":
                phases.setdefault(name, []).append(value)
        retrieve(query, args.limit, index_path)
        started = time.perf_counter()
        payload = retrieve(query, args.limit, index_path)
        cached.append((time.perf_counter() - started) * 1000)
        if not payload["diagnostics"]["cache_hit"]:
            raise RuntimeError("Repeated benchmark query did not use the search cache")
    index_status = status(index_path)
    memory_after = process.memory_info()
    report: dict[str, Any] = {
        "schema_version": 1,
        "hardware": hardware_info(),
        "queries": args.query,
        "index": {
            "documents": index_status["documents"],
            "chunks": index_status["chunks"],
            "schema_version": index_status["schema_version"],
            "bytes": index_size_bytes(index_path),
        },
        "memory": {
            "rss_before_warmup_mb": round(memory_before.rss / 1024 / 1024, 2),
            "rss_after_benchmark_mb": round(memory_after.rss / 1024 / 1024, 2),
            "peak_working_set_mb": round(getattr(memory_after, "peak_wset", memory_after.rss) / 1024 / 1024, 2),
        },
        "cold_initialization_ms": round(cold_ms, 2),
        "warmup": warmup_result,
        "uncached": _summary(uncached),
        "uncached_phases": {name: _summary(values) for name, values in phases.items()},
        "cached": _summary(cached),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
