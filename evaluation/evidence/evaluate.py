"""Evaluate or calibrate retrieval evidence sufficiency."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval import retrieve
from retrieval.evidence import MIN_LEXICAL_COVERAGE, MIN_TOP_RERANKER_SCORE


def classification_metrics(records: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    answerable = [record for record in records if record["answerable"]]
    no_answer = [record for record in records if not record["answerable"]]

    def accepted(record: dict[str, Any]) -> bool:
        return bool(record["hard_rules_passed"] and record["confidence"] >= threshold)

    false_refusals = sum(not accepted(record) for record in answerable)
    false_answers = sum(accepted(record) for record in no_answer)
    answerable_accept_rate = 1 - false_refusals / len(answerable) if answerable else None
    refusal_recall = 1 - false_answers / len(no_answer) if no_answer else None
    rates = [rate for rate in (answerable_accept_rate, refusal_recall) if rate is not None]
    return {
        "threshold": round(threshold, 6),
        "answerable_cases": len(answerable),
        "no_answer_cases": len(no_answer),
        "false_refusals": false_refusals,
        "false_answers": false_answers,
        "false_refusal_rate": round(false_refusals / len(answerable), 4) if answerable else None,
        "false_answer_rate": round(false_answers / len(no_answer), 4) if no_answer else None,
        "answerable_accept_rate": round(answerable_accept_rate, 4) if answerable_accept_rate is not None else None,
        "refusal_recall": round(refusal_recall, 4) if refusal_recall is not None else None,
        "balanced_accuracy": round(mean(rates), 4) if rates else None,
    }


def select_threshold(records: list[dict[str, Any]], maximum_false_answer_rate: float = 0.05) -> dict[str, Any]:
    confidences = sorted({float(record["confidence"]) for record in records})
    candidates = [0.0, *confidences, 1.0 + 1e-9]
    metrics = [classification_metrics(records, threshold) for threshold in candidates]
    eligible = [
        item
        for item in metrics
        if item["false_answer_rate"] is None or item["false_answer_rate"] <= maximum_false_answer_rate
    ]
    if not eligible:
        raise ValueError("No threshold satisfies the requested false-answer limit")
    return max(
        eligible,
        key=lambda item: (
            item["answerable_accept_rate"] if item["answerable_accept_rate"] is not None else 0.0,
            item["balanced_accuracy"] if item["balanced_accuracy"] is not None else 0.0,
            item["threshold"],
        ),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--maximum-false-answer-rate", type=float, default=0.05)
    parser.add_argument("--maximum-false-refusal-rate", type=float, default=0.35)
    parser.add_argument("--enforce", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.calibrate == (args.threshold is not None):
        raise SystemExit("Use exactly one of --calibrate or --threshold")
    paths = [path.resolve() for path in args.dataset]
    report_path = args.report.resolve()
    for path in [*paths, report_path]:
        path.relative_to(PROJECT_ROOT)
    cases = [case for path in paths for case in json.loads(path.read_text(encoding="utf-8"))]
    records = []
    for case in cases:
        payload = retrieve(case["query"], args.limit)
        signals = payload["assessment"]["signals"]
        records.append(
            {
                "id": case.get("id"),
                "query": case["query"],
                "answerable": case.get("source_file") is not None,
                "source_type": case.get("source_type"),
                "no_answer_type": case.get("no_answer_type"),
                "confidence": payload["confidence"],
                "top_reranker_score": signals["top_reranker_score"],
                "hard_rules_passed": (
                    (signals["top_reranker_score"] or 0.0) >= MIN_TOP_RERANKER_SCORE
                    and signals["lexical_coverage"] >= MIN_LEXICAL_COVERAGE
                    and (not signals["time_sensitive_query"] or signals["current_evidence"])
                    and not signals["wrong_school_entity"]
                    and not signals["missing_query_cues"]
                ),
                "signals": signals,
                "retrieved_source_files": [item["source_file"] for item in payload["results"]],
            }
        )
    metrics = (
        select_threshold(records, args.maximum_false_answer_rate)
        if args.calibrate
        else classification_metrics(records, args.threshold)
    )
    acceptance = {
        "false_answer_rate_within_limit": (
            metrics["false_answer_rate"] is None
            or metrics["false_answer_rate"] <= args.maximum_false_answer_rate
        ),
        "false_refusal_rate_within_limit": (
            metrics["false_refusal_rate"] is None
            or metrics["false_refusal_rate"] < args.maximum_false_refusal_rate
        ),
    }
    acceptance["passed"] = all(acceptance.values())
    report = {
        "schema_version": 1,
        "mode": "calibrated_on_development" if args.calibrate else "fixed_threshold",
        "datasets": [
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in paths
        ],
        "minimum_top_reranker_score": MIN_TOP_RERANKER_SCORE,
        "minimum_lexical_coverage": MIN_LEXICAL_COVERAGE,
        "metrics": metrics,
        "acceptance": acceptance,
        "details": records,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, ensure_ascii=False, indent=2))
    if args.enforce and not acceptance["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
