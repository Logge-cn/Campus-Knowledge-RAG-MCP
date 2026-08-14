"""Evaluate generated answers, refusals, required facts and cited evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _normalize(text: str) -> str:
    return re.sub(r"[^\w]+", "", text).lower()


def _average(values: list[float]) -> float | None:
    return round(mean(values), 4) if values else None


def _character_ngram_recall(reference: str, answer: str, size: int = 2) -> float:
    reference_text = _normalize(reference)
    answer_text = _normalize(answer)
    if not reference_text or not answer_text:
        return 0.0
    if len(reference_text) < size:
        return float(reference_text in answer_text)
    reference_ngrams = Counter(reference_text[index : index + size] for index in range(len(reference_text) - size + 1))
    answer_ngrams = Counter(answer_text[index : index + size] for index in range(len(answer_text) - size + 1))
    return sum((reference_ngrams & answer_ngrams).values()) / sum(reference_ngrams.values())


def _index_unique(items: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(items):
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id.strip():
            raise ValueError(f"{label} item {index} has no non-empty id")
        if item_id in output:
            raise ValueError(f"Duplicate {label} id: {item_id}")
        output[item_id] = item
    return output


def _retrieved_chunk_ids(prediction: dict[str, Any]) -> set[str] | None:
    results = prediction.get("retrieval_results")
    if not isinstance(results, list):
        return None
    chunk_ids = set()
    for result in results:
        if not isinstance(result, dict) or not isinstance(result.get("chunk_id"), str):
            raise ValueError("retrieval_results must contain objects with string chunk_id fields")
        chunk_ids.add(result["chunk_id"])
    return chunk_ids


def _citation_metadata_complete(prediction: dict[str, Any], cited_chunk_ids: list[str]) -> bool:
    citations = prediction.get("citations")
    if not isinstance(citations, list) or not citations:
        return False
    citation_ids = set()
    for citation in citations:
        if not isinstance(citation, dict):
            return False
        if not isinstance(citation.get("chunk_id"), str):
            return False
        if not isinstance(citation.get("source_file"), str) or not citation["source_file"]:
            return False
        if not isinstance(citation.get("page"), int):
            return False
        citation_ids.add(citation["chunk_id"])
    return citation_ids == set(cited_chunk_ids)


def _failure_stage(
    case: dict[str, Any],
    prediction: dict[str, Any],
    refused: bool,
    cited_chunk_ids: list[str],
) -> str:
    retrieved = _retrieved_chunk_ids(prediction)
    evidence_sufficient = prediction.get("evidence_sufficient")
    if retrieved is None or not isinstance(evidence_sufficient, bool):
        return "unclassified_missing_trace"

    is_answerable = case.get("source_file") is not None
    if not is_answerable:
        if evidence_sufficient:
            return "evidence_judgment_error"
        return "passed" if refused else "generation_error"

    relevant = set(case.get("relevant_chunk_ids", []))
    if not relevant.intersection(retrieved):
        return "retrieval_error"
    if not evidence_sufficient:
        return "evidence_judgment_error"
    if refused or not relevant.intersection(cited_chunk_ids):
        return "generation_error"

    review = prediction.get("review")
    if not isinstance(review, dict):
        return "pending_manual_review"
    required_review = ("correct", "complete", "citation_supported", "uses_model_memory_or_guess")
    if not all(isinstance(review.get(name), bool) for name in required_review):
        return "pending_manual_review"
    if (
        not review["correct"]
        or not review["complete"]
        or not review["citation_supported"]
        or review["uses_model_memory_or_guess"]
    ):
        return "generation_error"
    return "passed"


def evaluate_answers(cases: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    case_by_id = _index_unique(cases, "dataset")
    prediction_by_id = _index_unique(predictions, "prediction")
    missing = sorted(set(case_by_id) - set(prediction_by_id))
    unexpected = sorted(set(prediction_by_id) - set(case_by_id))
    if missing or unexpected:
        raise ValueError(f"Prediction ids do not match the dataset: missing={missing}, unexpected={unexpected}")

    details = []
    fact_recalls: list[float] = []
    fact_ngram_recalls: list[float] = []
    citation_precisions: list[float] = []
    citation_recalls: list[float] = []
    citation_supported: list[float] = []
    citation_metadata_complete: list[float] = []
    answerable_not_refused: list[float] = []
    no_answer_refused: list[float] = []
    evidence_false_refused: list[float] = []
    answered_review_correct: list[float] = []
    review_values: dict[str, list[float]] = {
        "correct": [],
        "complete": [],
        "citation_supported": [],
        "uses_model_memory_or_guess": [],
    }
    reviewed_answerable = 0
    failure_stages: Counter[str] = Counter()

    for case_id, case in case_by_id.items():
        prediction = prediction_by_id[case_id]
        refused = prediction.get("refused")
        if not isinstance(refused, bool):
            raise ValueError(f"Prediction {case_id} must have a boolean refused field")
        answer = prediction.get("answer", "")
        if not isinstance(answer, str):
            raise ValueError(f"Prediction {case_id} answer must be a string")
        cited_chunk_ids = prediction.get("cited_chunk_ids", [])
        if not isinstance(cited_chunk_ids, list) or not all(isinstance(value, str) for value in cited_chunk_ids):
            raise ValueError(f"Prediction {case_id} cited_chunk_ids must be a list of strings")

        evidence_sufficient = prediction.get("evidence_sufficient")
        if isinstance(evidence_sufficient, bool) and not evidence_sufficient:
            evidence_false_refused.append(float(refused))

        is_answerable = case.get("source_file") is not None
        stage = _failure_stage(case, prediction, refused, cited_chunk_ids)
        failure_stages[stage] += 1
        detail: dict[str, Any] = {
            "id": case_id,
            "answerable": is_answerable,
            "refused": refused,
            "failure_stage": stage,
        }
        if not is_answerable:
            no_answer_refused.append(float(refused))
            detail["refusal_correct"] = refused
            details.append(detail)
            continue

        answerable_not_refused.append(float(not refused))
        required_facts = case.get("required_facts")
        if not isinstance(required_facts, list) or not required_facts:
            raise ValueError(f"Answerable case {case_id} has no required_facts")
        relevant_chunk_ids = case.get("relevant_chunk_ids")
        if not isinstance(relevant_chunk_ids, list) or not relevant_chunk_ids:
            raise ValueError(f"Answerable case {case_id} has no relevant_chunk_ids")

        normalized_answer = _normalize(answer) if not refused else ""
        fact_hits = [bool(_normalize(fact) and _normalize(fact) in normalized_answer) for fact in required_facts]
        fact_ngram_scores = [_character_ngram_recall(fact, answer) if not refused else 0.0 for fact in required_facts]
        fact_recall = sum(fact_hits) / len(fact_hits)
        fact_ngram_recall = mean(fact_ngram_scores)
        relevant = set(relevant_chunk_ids)
        cited = set(cited_chunk_ids)
        citation_overlap = cited & relevant
        citation_precision = len(citation_overlap) / len(cited) if cited else 0.0
        citation_recall = len(citation_overlap) / len(relevant)
        has_supported_citation = bool(citation_overlap)
        fact_recalls.append(fact_recall)
        fact_ngram_recalls.append(fact_ngram_recall)
        citation_precisions.append(citation_precision)
        citation_recalls.append(citation_recall)
        citation_supported.append(float(has_supported_citation))
        citation_metadata_complete.append(float(_citation_metadata_complete(prediction, cited_chunk_ids)))
        detail.update(
            {
                "required_fact_lexical_recall": round(fact_recall, 4),
                "required_fact_character_bigram_recall": round(fact_ngram_recall, 4),
                "citation_precision": round(citation_precision, 4),
                "citation_recall": round(citation_recall, 4),
                "has_supported_citation": has_supported_citation,
            }
        )

        review = prediction.get("review")
        if isinstance(review, dict) and any(isinstance(review.get(name), bool) for name in review_values):
            detail["review"] = {}
            for name in review_values:
                value = review.get(name)
                if isinstance(value, bool):
                    review_values[name].append(float(value))
                    detail["review"][name] = value
            if all(isinstance(review.get(name), bool) for name in review_values):
                reviewed_answerable += 1
            if not refused and isinstance(review.get("correct"), bool):
                answered_review_correct.append(float(review["correct"]))
        details.append(detail)

    answerable_count = sum(case.get("source_file") is not None for case in cases)
    no_answer_count = len(cases) - answerable_count
    return {
        "cases": len(cases),
        "answerable_cases": answerable_count,
        "no_answer_cases": no_answer_count,
        "metrics": {
            "answerable_answer_rate": _average(answerable_not_refused),
            "no_answer_refusal_accuracy": _average(no_answer_refused),
            "required_fact_lexical_recall": _average(fact_recalls),
            "required_fact_character_bigram_recall": _average(fact_ngram_recalls),
            "citation_precision": _average(citation_precisions),
            "citation_recall": _average(citation_recalls),
            "answer_with_supported_citation_rate": _average(citation_supported),
            "citation_metadata_complete_rate": _average(citation_metadata_complete),
            "incorrect_answer_rate": (
                round(1 - mean(answered_review_correct), 4) if answered_review_correct else None
            ),
            "false_answer_rate": (
                round(1 - mean(no_answer_refused), 4) if no_answer_refused else None
            ),
            "false_refusal_rate": (
                round(1 - mean(answerable_not_refused), 4) if answerable_not_refused else None
            ),
            "evidence_insufficient_refusal_compliance": _average(evidence_false_refused),
        },
        "manual_review": {
            "reviewed_answerable_cases": reviewed_answerable,
            "coverage": round(reviewed_answerable / answerable_count, 4) if answerable_count else None,
            **{f"{name}_rate": _average(values) for name, values in review_values.items()},
        },
        "failure_stage_counts": dict(sorted(failure_stages.items())),
        "note": (
            "Required-fact lexical and character-bigram coverage are deterministic diagnostics, not semantic "
            "correctness judges. Use the manual review fields for final correctness, completeness and "
            "evidence-support claims."
        ),
        "details": details,
    }


def _inside_project(path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(PROJECT_ROOT)
    return resolved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        action="append",
        required=True,
        help="Labeled JSON array. Repeat to evaluate several compatible files as one split.",
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_paths = [_inside_project(path) for path in args.dataset]
    predictions_path = _inside_project(args.predictions)
    report_path = _inside_project(args.report)
    cases = []
    dataset_records = []
    combined_dataset_digest = hashlib.sha256()
    for dataset_path in dataset_paths:
        loaded = json.loads(dataset_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError(f"Dataset must be a JSON array: {dataset_path}")
        cases.extend(loaded)
        record = {
            "path": dataset_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        }
        dataset_records.append(record)
        combined_dataset_digest.update(record["path"].encode("utf-8"))
        combined_dataset_digest.update(b"\0")
        combined_dataset_digest.update(record["sha256"].encode("ascii"))
        combined_dataset_digest.update(b"\0")
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise ValueError(f"Predictions must be a JSON array: {predictions_path}")
    report = {
        "datasets": dataset_records,
        "dataset_sha256": (
            dataset_records[0]["sha256"] if len(dataset_records) == 1 else combined_dataset_digest.hexdigest()
        ),
        "predictions": {
            "path": predictions_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        },
        **evaluate_answers(cases, predictions),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
