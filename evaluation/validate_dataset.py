"""Validate expanded development and locked RAG evaluation datasets."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEVELOPMENT_PATH = PROJECT_ROOT / "evaluation" / "dataset.json"
DEFAULT_LOCKED_PATH = PROJECT_ROOT / "evaluation" / "dataset_holdout.json"
DEFAULT_POLICY_PATH = PROJECT_ROOT / "evaluation" / "dataset_expansion_policy.json"
DEFAULT_EXPANSION_PLAN_PATH = PROJECT_ROOT / "evaluation" / "dataset_expansion_plan.json"
DEFAULT_DOCUMENT_MANIFEST_PATH = PROJECT_ROOT / "evaluation" / "document_manifest.json"
DEFAULT_CONFIG_FREEZE_PATH = PROJECT_ROOT / "evaluation" / "config_freeze.json"
DEFAULT_DEVELOPMENT_NEW_PATH = PROJECT_ROOT / "evaluation" / "dataset_expanded_development_new.json"
DEFAULT_LOCKED_ANSWERABLE_PATH = PROJECT_ROOT / "evaluation" / "dataset_expanded_locked_answerable.json"
DEFAULT_LOCKED_NO_ANSWER_PATH = PROJECT_ROOT / "evaluation" / "dataset_expanded_locked_no_answer.json"
DEFAULT_DEVELOPMENT_CHUNKS_PATH = (
    PROJECT_ROOT / "storage" / "evaluation_splits" / "development" / "index" / "chunks.json"
)
DEFAULT_LOCKED_CHUNKS_PATH = (
    PROJECT_ROOT / "storage" / "evaluation_splits" / "locked_test" / "index" / "chunks.json"
)
PDF_TYPES = ("native", "scanned")
PORTABLE_TEXT_HASH_SUFFIXES = {".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _load_json(path: Path) -> Any:
    resolved = path.resolve()
    resolved.relative_to(PROJECT_ROOT)
    return json.loads(resolved.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_frozen_file(path: Path) -> str:
    """Hash frozen text independently of the checkout's line-ending policy."""
    if path.suffix.lower() not in PORTABLE_TEXT_HASH_SUFFIXES:
        return _sha256_file(path)
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def validate_config_freeze(freeze: dict[str, Any]) -> dict[str, Any]:
    """Verify frozen retrieval code, development labels, sealed labels, and reranker bytes."""
    errors: list[dict[str, Any]] = []
    verified: list[str] = []
    portable_text_hashes = freeze.get("portable_text_sha256", {})
    if not isinstance(portable_text_hashes, dict):
        errors.append({"code": "invalid_portable_text_hash_map"})
        portable_text_hashes = {}

    def verify_relative_file(relative_path: Any, expected_sha256: Any, *, expected_bytes: Any = None) -> None:
        if not _non_empty_string(relative_path):
            errors.append({"code": "invalid_frozen_path", "path": relative_path})
            return
        path = (PROJECT_ROOT / relative_path).resolve()
        try:
            path.relative_to(PROJECT_ROOT)
        except ValueError:
            errors.append({"code": "frozen_path_outside_project", "path": relative_path})
            return
        if not path.is_file():
            errors.append({"code": "frozen_file_missing", "path": relative_path})
            return
        if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            errors.append({"code": "invalid_frozen_sha256", "path": relative_path})
            return
        if expected_bytes is not None and path.stat().st_size != expected_bytes:
            errors.append(
                {
                    "code": "frozen_file_size_mismatch",
                    "path": relative_path,
                    "expected": expected_bytes,
                    "actual": path.stat().st_size,
                }
            )
        portable_sha256 = portable_text_hashes.get(relative_path)
        if portable_sha256 is not None and not (
            isinstance(portable_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", portable_sha256)
        ):
            errors.append({"code": "invalid_portable_text_sha256", "path": relative_path})
            return
        comparison_sha256 = portable_sha256 or expected_sha256
        actual_sha256 = _sha256_frozen_file(path) if portable_sha256 else _sha256_file(path)
        if actual_sha256 != comparison_sha256:
            errors.append(
                {
                    "code": "frozen_file_sha256_mismatch",
                    "path": relative_path,
                    "expected": comparison_sha256,
                    "actual": actual_sha256,
                }
            )
        else:
            verified.append(relative_path)

    for field in ("frozen_file_sha256", "sealed_evaluation_artifacts"):
        file_hashes = freeze.get(field, {})
        if not isinstance(file_hashes, dict) or not file_hashes:
            errors.append({"code": "missing_frozen_file_map", "field": field})
            continue
        for relative_path, expected_sha256 in file_hashes.items():
            verify_relative_file(relative_path, expected_sha256)

    reranker = freeze.get("reranker_artifact")
    if not isinstance(reranker, dict):
        errors.append({"code": "missing_frozen_reranker_artifact"})
    else:
        verify_relative_file(reranker.get("path"), reranker.get("sha256"), expected_bytes=reranker.get("bytes"))

    return {
        "valid": not errors,
        "frozen_at": freeze.get("frozen_at"),
        "locked_labels_sealed_at": freeze.get("locked_labels_sealed_at"),
        "verified_files": sorted(verified),
        "errors": errors,
    }


def _split_stats(cases: list[dict[str, Any]], source_type_hints: dict[str, str] | None = None) -> dict[str, Any]:
    source_type_hints = source_type_hints or {}
    answerable = [case for case in cases if case.get("source_file") is not None]
    no_answer = [case for case in cases if case.get("source_file") is None]
    documents: dict[str, set[str]] = defaultdict(set)
    type_counts: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    document_case_counts: Counter[str] = Counter()
    for case in answerable:
        source_file = case.get("source_file")
        pdf_type = case.get("source_type") or source_type_hints.get(source_file)
        if pdf_type in PDF_TYPES and _non_empty_string(source_file):
            documents[pdf_type].add(source_file)
            type_counts[pdf_type] += 1
            document_case_counts[source_file] += 1
        if _non_empty_string(case.get("category")):
            categories[case["category"]] += 1
    return {
        "cases": len(cases),
        "answerable_cases": len(answerable),
        "no_answer_cases": len(no_answer),
        "documents": {pdf_type: len(documents[pdf_type]) for pdf_type in PDF_TYPES},
        "answerable_by_pdf_type": {pdf_type: type_counts[pdf_type] for pdf_type in PDF_TYPES},
        "answerable_by_document": dict(sorted(document_case_counts.items())),
        "answerable_by_category": dict(sorted(categories.items())),
    }


def _quota_gaps(stats: dict[str, Any], split_policy: dict[str, Any], categories: set[str]) -> dict[str, Any]:
    document_gaps = {
        pdf_type: max(0, split_policy["minimum_documents"][pdf_type] - stats["documents"][pdf_type])
        for pdf_type in PDF_TYPES
    }
    answerable_gaps = {
        pdf_type: max(
            split_policy["minimum_answerable_cases"][pdf_type]
            - stats["answerable_by_pdf_type"][pdf_type],
            document_gaps[pdf_type] * split_policy.get("minimum_answerable_cases_per_document", 0),
        )
        for pdf_type in PDF_TYPES
    }
    no_answer_gap = max(0, split_policy["minimum_no_answer_cases"] - stats["no_answer_cases"])
    return {
        "documents_to_add": document_gaps,
        "answerable_cases_to_add": answerable_gaps,
        "no_answer_cases_to_add": no_answer_gap,
        "category_cases_to_add": {
            category: max(
                0,
                split_policy["minimum_cases_per_category"]
                - stats["answerable_by_category"].get(category, 0),
            )
            for category in sorted(categories)
        },
        "minimum_new_cases": sum(answerable_gaps.values()) + no_answer_gap,
    }


def validate_expansion_plan(plan: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    """Validate that the explicit expansion plan is internally consistent."""
    errors: list[dict[str, Any]] = []
    slots = plan.get("document_slots")
    matrix = plan.get("answerable_case_matrix")
    required_categories = set(policy["required_categories"])
    if not isinstance(slots, list) or not slots:
        return {"valid": False, "errors": [{"code": "missing_document_slots"}]}
    if not isinstance(matrix, dict):
        return {"valid": False, "errors": [{"code": "missing_case_matrix"}]}

    slots_by_name: dict[str, dict[str, Any]] = {}
    for slot in slots:
        slot_name = slot.get("slot")
        if not _non_empty_string(slot_name) or slot_name in slots_by_name:
            errors.append({"code": "invalid_or_duplicate_slot", "slot": slot_name})
            continue
        if slot.get("split") not in policy["splits"]:
            errors.append({"code": "invalid_slot_split", "slot": slot_name})
        if slot.get("source_type") not in PDF_TYPES:
            errors.append({"code": "invalid_slot_source_type", "slot": slot_name})
        if not isinstance(slot.get("minimum_answerable_cases"), int) or slot["minimum_answerable_cases"] <= 0:
            errors.append({"code": "invalid_slot_case_count", "slot": slot_name})
        slots_by_name[slot_name] = slot

    if set(matrix) != set(slots_by_name):
        errors.append(
            {
                "code": "case_matrix_slot_mismatch",
                "missing": sorted(set(slots_by_name) - set(matrix)),
                "unexpected": sorted(set(matrix) - set(slots_by_name)),
            }
        )

    answerable_total = 0
    for slot_name, category_counts in matrix.items():
        if slot_name not in slots_by_name or not isinstance(category_counts, dict):
            continue
        if set(category_counts) != required_categories:
            errors.append({"code": "case_matrix_category_mismatch", "slot": slot_name})
            continue
        if not all(isinstance(count, int) and count >= 0 for count in category_counts.values()):
            errors.append({"code": "invalid_category_case_count", "slot": slot_name})
            continue
        row_total = sum(category_counts.values())
        answerable_total += row_total
        if row_total != slots_by_name[slot_name]["minimum_answerable_cases"]:
            errors.append(
                {
                    "code": "slot_case_total_mismatch",
                    "slot": slot_name,
                    "declared": slots_by_name[slot_name]["minimum_answerable_cases"],
                    "matrix_total": row_total,
                }
            )

    no_answer = plan.get("new_no_answer_cases", {})
    no_answer_types = no_answer.get("types", {})
    no_answer_total = no_answer.get("total")
    if (
        not isinstance(no_answer_total, int)
        or not isinstance(no_answer_types, dict)
        or not all(isinstance(count, int) and count >= 0 for count in no_answer_types.values())
        or no_answer_total != sum(no_answer_types.values())
    ):
        errors.append({"code": "no_answer_total_mismatch"})

    declared = plan.get("minimum_new_data", {})
    expected = {
        "pdfs": len(slots_by_name),
        "answerable_cases": answerable_total,
        "no_answer_cases": no_answer_total,
        "total_cases": answerable_total + no_answer_total if isinstance(no_answer_total, int) else None,
    }
    for key, value in expected.items():
        if declared.get(key) != value:
            errors.append(
                {"code": "minimum_new_data_mismatch", "field": key, "declared": declared.get(key), "actual": value}
            )
    return {"valid": not errors, "totals": expected, "errors": errors}


def validate_document_manifest(manifest: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    """Validate document quotas and content-hash isolation before case labeling."""
    documents = manifest.get("documents")
    if not isinstance(documents, list):
        return {"valid": False, "errors": [{"code": "missing_manifest_documents"}]}

    errors: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_files: set[str] = set()
    documents_by_split_type: dict[str, Counter[str]] = defaultdict(Counter)
    fingerprints_by_split: dict[str, set[str]] = defaultdict(set)
    for index, document in enumerate(documents):
        document_id = document.get("document_id")
        source_file = document.get("source_file")
        split = document.get("split")
        source_type = document.get("source_type")
        location = {"document_index": index, "document_id": document_id}
        if not _non_empty_string(document_id) or document_id in seen_ids:
            errors.append({"code": "invalid_or_duplicate_document_id", **location})
        else:
            seen_ids.add(document_id)
        if not _non_empty_string(source_file) or source_file in seen_files:
            errors.append({"code": "invalid_or_duplicate_source_file", **location})
        else:
            seen_files.add(source_file)
        if split not in policy["splits"]:
            errors.append({"code": "invalid_document_split", **location})
            continue
        if source_type not in PDF_TYPES:
            errors.append({"code": "invalid_document_source_type", **location})
            continue
        documents_by_split_type[split][source_type] += 1
        if not isinstance(document.get("pages"), int) or document["pages"] <= 0:
            errors.append({"code": "invalid_document_page_count", **location})
        extraction = document.get("extraction")
        if not isinstance(extraction, dict) or extraction.get("pages_extracted") != document.get("pages"):
            errors.append({"code": "incomplete_document_extraction", **location})
        sha256_value = document.get("sha256")
        if not isinstance(sha256_value, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256_value):
            errors.append({"code": "invalid_document_sha256", **location})
        else:
            fingerprints_by_split[split].add(sha256_value)
        derived = document.get("derived_from")
        if derived is not None:
            source_sha256 = derived.get("source_sha256") if isinstance(derived, dict) else None
            if not isinstance(source_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
                errors.append({"code": "invalid_derived_source_sha256", **location})
            else:
                fingerprints_by_split[split].add(source_sha256)
        if not _non_empty_string(document.get("layout_profile")):
            errors.append({"code": "missing_layout_profile", **location})
        if not _non_empty_string(document.get("topic")):
            errors.append({"code": "missing_document_topic", **location})

    for split, split_policy in policy["splits"].items():
        for source_type in PDF_TYPES:
            required = split_policy["minimum_documents"][source_type]
            actual = documents_by_split_type[split][source_type]
            if actual < required:
                errors.append(
                    {
                        "code": "manifest_insufficient_documents",
                        "split": split,
                        "source_type": source_type,
                        "required": required,
                        "actual": actual,
                    }
                )
    fingerprint_overlap = sorted(
        fingerprints_by_split["development"] & fingerprints_by_split["locked_test"]
    )
    if fingerprint_overlap:
        errors.append({"code": "manifest_content_hash_overlap", "sha256": fingerprint_overlap})
    return {
        "valid": not errors,
        "documents": len(documents),
        "documents_by_split_type": {
            split: {source_type: documents_by_split_type[split][source_type] for source_type in PDF_TYPES}
            for split in policy["splits"]
        },
        "content_hash_overlap": fingerprint_overlap,
        "errors": errors,
    }


def validate_labeled_expansion(
    development_new: list[dict[str, Any]],
    locked_answerable: list[dict[str, Any]],
    locked_no_answer: list[dict[str, Any]],
    plan: dict[str, Any],
    manifest: dict[str, Any],
    policy: dict[str, Any],
    *,
    chunks_by_split: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Validate that labeled cases exactly implement the expansion plan and qrels."""
    component_policy = copy.deepcopy(policy)
    planned_slots = plan.get("document_slots", [])
    component_policy["splits"] = {}
    for split in policy["splits"]:
        split_slots = [slot for slot in planned_slots if slot.get("split") == split]
        per_document = [
            slot.get("minimum_answerable_cases")
            for slot in split_slots
            if isinstance(slot.get("minimum_answerable_cases"), int)
        ]
        component_policy["splits"][split] = {
            "minimum_documents": {
                source_type: sum(slot.get("source_type") == source_type for slot in split_slots)
                for source_type in PDF_TYPES
            },
            "minimum_answerable_cases": {
                source_type: sum(
                    slot.get("minimum_answerable_cases", 0)
                    for slot in split_slots
                    if slot.get("source_type") == source_type
                )
                for source_type in PDF_TYPES
            },
            "minimum_no_answer_cases": (
                plan.get("new_no_answer_cases", {}).get("total", 0) if split == "locked_test" else 0
            ),
            "minimum_cases_per_category": 0,
            "minimum_answerable_cases_per_document": min(per_document, default=0),
        }
    locked_cases = locked_answerable + locked_no_answer
    base_report = validate_datasets(development_new, locked_cases, component_policy)
    errors = list(base_report["errors"])
    warnings = list(base_report["warnings"])

    def error(code: str, **context: Any) -> None:
        errors.append({"code": code, **context})

    documents = manifest.get("documents", [])
    documents_by_file = {
        document.get("source_file"): document
        for document in documents
        if _non_empty_string(document.get("source_file"))
    }
    slots = {slot.get("slot"): slot for slot in plan.get("document_slots", [])}
    expected_matrix = plan.get("answerable_case_matrix", {})
    actual_matrix: dict[str, Counter[str]] = defaultdict(Counter)
    all_answerable = development_new + locked_answerable
    seen_queries: dict[str, str] = {}
    seen_topics: dict[str, str] = {}

    for case in all_answerable:
        case_id = case.get("id")
        source_file = case.get("source_file")
        document = documents_by_file.get(source_file)
        if document is None:
            error("case_source_missing_from_manifest", id=case_id, source_file=source_file)
            continue
        slot_name = document.get("document_id")
        slot = slots.get(slot_name)
        if slot is None:
            error("case_source_not_in_expansion_plan", id=case_id, document_id=slot_name)
            continue
        if not _non_empty_string(case_id) or not case_id.startswith(f"{slot_name}-"):
            error("case_id_slot_mismatch", id=case_id, document_id=slot_name)
        for field in ("split", "source_type"):
            if case.get(field) != document.get(field) or case.get(field) != slot.get(field):
                error(
                    "case_manifest_slot_mismatch",
                    id=case_id,
                    field=field,
                    case_value=case.get(field),
                    manifest_value=document.get(field),
                    slot_value=slot.get(field),
                )
        category = case.get("category")
        actual_matrix[slot_name][category] += 1
        normalized_query = re.sub(r"\s+", "", str(case.get("query", ""))).casefold()
        if normalized_query in seen_queries:
            error("duplicate_expansion_query", id=case_id, first_id=seen_queries[normalized_query])
        else:
            seen_queries[normalized_query] = str(case_id)
        topic_group = case.get("topic_group")
        if _non_empty_string(topic_group):
            if topic_group in seen_topics:
                error("duplicate_expansion_topic_group", id=case_id, first_id=seen_topics[topic_group])
            else:
                seen_topics[topic_group] = str(case_id)

    for slot_name, expected_counts in expected_matrix.items():
        actual_counts = {category: actual_matrix[slot_name][category] for category in expected_counts}
        if actual_counts != expected_counts:
            error(
                "labeled_case_matrix_mismatch",
                slot=slot_name,
                expected=expected_counts,
                actual=actual_counts,
            )

    expected_no_answer = plan.get("new_no_answer_cases", {})
    actual_no_answer_types = Counter(case.get("no_answer_type") for case in locked_no_answer)
    expected_no_answer_types = expected_no_answer.get("types", {})
    if len(locked_no_answer) != expected_no_answer.get("total"):
        error(
            "labeled_no_answer_total_mismatch",
            expected=expected_no_answer.get("total"),
            actual=len(locked_no_answer),
        )
    if {key: actual_no_answer_types[key] for key in expected_no_answer_types} != expected_no_answer_types:
        error(
            "labeled_no_answer_type_mismatch",
            expected=expected_no_answer_types,
            actual=dict(sorted(actual_no_answer_types.items(), key=lambda item: str(item[0]))),
        )
    for case in locked_no_answer:
        if case.get("no_answer_type") not in expected_no_answer_types:
            error("invalid_no_answer_type", id=case.get("id"), value=case.get("no_answer_type"))

    qrel_stats = {"checked": False, "chunk_ids": 0}
    if chunks_by_split is None:
        warnings.append(
            {
                "code": "chunk_qrel_validation_skipped",
                "message": "Provide both split chunk files to verify every relevant_chunk_id against its source and page.",
            }
        )
    else:
        qrel_stats["checked"] = True
        chunk_maps = {
            split: {chunk.get("chunk_id"): chunk for chunk in chunks}
            for split, chunks in chunks_by_split.items()
        }
        for case in all_answerable:
            split = case.get("split")
            for chunk_id in case.get("relevant_chunk_ids", []):
                qrel_stats["chunk_ids"] += 1
                chunk = chunk_maps.get(split, {}).get(chunk_id)
                if chunk is None:
                    error("relevant_chunk_not_found", id=case.get("id"), chunk_id=chunk_id, split=split)
                    continue
                if chunk.get("source_file") != case.get("source_file"):
                    error(
                        "relevant_chunk_source_mismatch",
                        id=case.get("id"),
                        chunk_id=chunk_id,
                        expected=case.get("source_file"),
                        actual=chunk.get("source_file"),
                    )
                if chunk.get("page") not in case.get("pages", []):
                    error(
                        "relevant_chunk_page_mismatch",
                        id=case.get("id"),
                        chunk_id=chunk_id,
                        chunk_page=chunk.get("page"),
                        labeled_pages=case.get("pages"),
                    )

    return {
        "valid": not errors,
        "answerable_cases": len(all_answerable),
        "development_answerable_cases": len(development_new),
        "locked_answerable_cases": len(locked_answerable),
        "locked_no_answer_cases": len(locked_no_answer),
        "documents": len({case.get("source_file") for case in all_answerable}),
        "case_matrix": {
            slot_name: {category: actual_matrix[slot_name][category] for category in expected_matrix[slot_name]}
            for slot_name in expected_matrix
        },
        "no_answer_types": dict(sorted(actual_no_answer_types.items(), key=lambda item: str(item[0]))),
        "qrels": qrel_stats,
        "errors": errors,
        "warnings": warnings,
    }


def validate_datasets(
    development: list[dict[str, Any]],
    locked_test: list[dict[str, Any]],
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Return machine-readable validation errors and dataset statistics."""
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    datasets = {"development": development, "locked_test": locked_test}
    required_categories = set(policy["required_categories"])
    seen_ids: dict[str, str] = {}
    documents_by_split: dict[str, set[str]] = defaultdict(set)
    source_type_hints = {
        case["source_file"]: case["source_type"]
        for cases in datasets.values()
        for case in cases
        if _non_empty_string(case.get("source_file")) and case.get("source_type") in PDF_TYPES
    }

    def error(code: str, message: str, **context: Any) -> None:
        errors.append({"code": code, "message": message, **context})

    for split_name, cases in datasets.items():
        expected_split = split_name
        for index, case in enumerate(cases):
            location = {"split": split_name, "case_index": index}
            case_id = case.get("id")
            if not _non_empty_string(case_id):
                error("missing_id", "Every case must have a non-empty id.", **location)
            elif case_id in seen_ids:
                error(
                    "duplicate_id",
                    "Case ids must be unique across both splits.",
                    id=case_id,
                    first_split=seen_ids[case_id],
                    **location,
                )
            else:
                seen_ids[case_id] = split_name

            if not _non_empty_string(case.get("query")):
                error("missing_query", "Every case must have a non-empty query.", **location)
            if case.get("split") != expected_split:
                error(
                    "invalid_split",
                    f"Case split must be {expected_split!r}.",
                    actual=case.get("split"),
                    **location,
                )

            source_file = case.get("source_file")
            if source_file is None:
                if case.get("category") != "no_answer":
                    error("invalid_no_answer_category", "No-answer cases must use category 'no_answer'.", **location)
                if case.get("pages") not in (None, []):
                    error("invalid_no_answer_pages", "No-answer cases must not label source pages.", **location)
                continue

            if not _non_empty_string(source_file):
                error("invalid_source_file", "Answerable cases must have a non-empty source_file.", **location)
            else:
                documents_by_split[split_name].add(source_file)
            pages = case.get("pages")
            if not isinstance(pages, list) or not pages or not all(isinstance(page, int) and page > 0 for page in pages):
                error("invalid_pages", "Answerable cases must have one or more positive integer pages.", **location)
            if case.get("source_type") not in PDF_TYPES:
                error("invalid_source_type", "source_type must be 'native' or 'scanned'.", **location)
            if case.get("category") not in required_categories:
                error("invalid_category", "Answerable case category is not allowed by the policy.", **location)
            if not _non_empty_string(case.get("topic_group")):
                error("missing_topic_group", "Answerable cases must have a topic_group.", **location)
            if policy.get("require_relevant_chunk_ids"):
                chunk_ids = case.get("relevant_chunk_ids")
                if not isinstance(chunk_ids, list) or not chunk_ids or not all(_non_empty_string(value) for value in chunk_ids):
                    error(
                        "missing_relevant_chunk_ids",
                        "Answerable cases must have human-verified relevant_chunk_ids.",
                        **location,
                    )
            if policy.get("require_gold_answer") and not _non_empty_string(case.get("gold_answer")):
                error("missing_gold_answer", "Answerable cases must have a non-empty gold_answer.", **location)
            if policy.get("require_required_facts"):
                required_facts = case.get("required_facts")
                if (
                    not isinstance(required_facts, list)
                    or not required_facts
                    or not all(_non_empty_string(value) for value in required_facts)
                ):
                    error(
                        "missing_required_facts",
                        "Answerable cases must list the facts required for a complete answer.",
                        **location,
                    )

        split_policy = policy["splits"][split_name]
        stats = _split_stats(cases, source_type_hints)
        for pdf_type in PDF_TYPES:
            minimum_documents = split_policy["minimum_documents"][pdf_type]
            actual_documents = stats["documents"][pdf_type]
            if actual_documents < minimum_documents:
                error(
                    "insufficient_documents",
                    f"{split_name} requires at least {minimum_documents} {pdf_type} documents.",
                    split=split_name,
                    pdf_type=pdf_type,
                    required=minimum_documents,
                    actual=actual_documents,
                )
            minimum_cases = split_policy["minimum_answerable_cases"][pdf_type]
            actual_cases = stats["answerable_by_pdf_type"][pdf_type]
            if actual_cases < minimum_cases:
                error(
                    "insufficient_answerable_cases",
                    f"{split_name} requires at least {minimum_cases} answerable {pdf_type} cases.",
                    split=split_name,
                    pdf_type=pdf_type,
                    required=minimum_cases,
                    actual=actual_cases,
                )
        minimum_no_answer = split_policy["minimum_no_answer_cases"]
        if stats["no_answer_cases"] < minimum_no_answer:
            error(
                "insufficient_no_answer_cases",
                f"{split_name} requires at least {minimum_no_answer} no-answer cases.",
                split=split_name,
                required=minimum_no_answer,
                actual=stats["no_answer_cases"],
            )
        minimum_per_category = split_policy["minimum_cases_per_category"]
        for category in sorted(required_categories):
            actual = stats["answerable_by_category"].get(category, 0)
            if actual < minimum_per_category:
                error(
                    "insufficient_category_cases",
                    f"{split_name} requires at least {minimum_per_category} {category} cases.",
                    split=split_name,
                    category=category,
                    required=minimum_per_category,
                    actual=actual,
                )
        minimum_per_document = split_policy.get("minimum_answerable_cases_per_document", 0)
        for source_file, actual in stats["answerable_by_document"].items():
            if actual < minimum_per_document:
                error(
                    "insufficient_document_cases",
                    f"{split_name} requires at least {minimum_per_document} cases for each source document.",
                    split=split_name,
                    source_file=source_file,
                    required=minimum_per_document,
                    actual=actual,
                )

    overlap = sorted(documents_by_split["development"] & documents_by_split["locked_test"])
    if policy.get("require_document_disjoint_splits") and overlap:
        error(
            "document_split_overlap",
            "Development and locked-test source documents must be disjoint.",
            documents=overlap,
        )
    if not overlap:
        warnings.append(
            {
                "code": "locked_split_usage",
                "message": "Keep the locked-test dataset sealed until the final configuration is selected.",
            }
        )

    stats_by_split = {name: _split_stats(cases, source_type_hints) for name, cases in datasets.items()}
    eligible_locked_cases = [
        case
        for case in locked_test
        if case.get("source_file") is None or case.get("source_file") not in set(overlap)
    ]
    quota_stats = {
        "development": stats_by_split["development"],
        "locked_test": _split_stats(eligible_locked_cases, source_type_hints),
    }
    expansion_gaps = {
        "assumption": (
            "Keep development documents in development. Locked-test answerable cases from overlapping documents "
            "do not count toward expansion quotas."
        ),
        "overlapping_documents": overlap,
        "splits": {
            name: _quota_gaps(quota_stats[name], policy["splits"][name], required_categories)
            for name in datasets
        },
    }
    expansion_gaps["minimum_new_documents"] = sum(
        sum(gap["documents_to_add"].values()) for gap in expansion_gaps["splits"].values()
    )
    expansion_gaps["minimum_new_cases"] = sum(
        gap["minimum_new_cases"] for gap in expansion_gaps["splits"].values()
    )
    recommended_quota_stats = {
        "development": _split_stats(development + locked_test, source_type_hints),
        "locked_test": _split_stats([], source_type_hints),
    }
    recommended_split_gaps = {
        name: _quota_gaps(recommended_quota_stats[name], policy["splits"][name], required_categories)
        for name in datasets
    }
    expansion_gaps["recommended_rebuild_locked"] = {
        "assumption": (
            "Move the already-inspected holdout into the development diagnostic pool and build a completely new "
            "locked test from unseen documents."
        ),
        "splits": recommended_split_gaps,
        "minimum_new_documents": sum(
            sum(gap["documents_to_add"].values()) for gap in recommended_split_gaps.values()
        ),
        "minimum_new_cases": sum(gap["minimum_new_cases"] for gap in recommended_split_gaps.values()),
    }

    return {
        "valid": not errors,
        "schema_version": policy["schema_version"],
        "stats": stats_by_split,
        "expansion_gaps": expansion_gaps,
        "errors": errors,
        "warnings": warnings,
    }


def summarize_report(report: dict[str, Any], max_examples: int = 2) -> dict[str, Any]:
    error_counts = Counter(error["code"] for error in report["errors"])
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for error in report["errors"]:
        if len(examples[error["code"]]) < max_examples:
            examples[error["code"]].append(error)
    return {
        "valid": report["valid"],
        "schema_version": report["schema_version"],
        "stats": report["stats"],
        "expansion_gaps": report["expansion_gaps"],
        "error_count": len(report["errors"]),
        "error_counts": dict(sorted(error_counts.items())),
        "error_examples": dict(sorted(examples.items())),
        "warnings": report["warnings"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT_PATH)
    parser.add_argument("--locked-test", type=Path, default=DEFAULT_LOCKED_PATH)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--expansion-plan", type=Path, default=DEFAULT_EXPANSION_PLAN_PATH)
    parser.add_argument("--document-manifest", type=Path, default=DEFAULT_DOCUMENT_MANIFEST_PATH)
    parser.add_argument("--config-freeze", type=Path, default=DEFAULT_CONFIG_FREEZE_PATH)
    parser.add_argument("--development-new", type=Path, default=DEFAULT_DEVELOPMENT_NEW_PATH)
    parser.add_argument("--locked-answerable", type=Path, default=DEFAULT_LOCKED_ANSWERABLE_PATH)
    parser.add_argument("--locked-no-answer", type=Path, default=DEFAULT_LOCKED_NO_ANSWER_PATH)
    parser.add_argument("--development-chunks", type=Path, default=DEFAULT_DEVELOPMENT_CHUNKS_PATH)
    parser.add_argument("--locked-chunks", type=Path, default=DEFAULT_LOCKED_CHUNKS_PATH)
    parser.add_argument(
        "--expansion-only",
        action="store_true",
        help="Validate the 140 newly labeled cases without requiring legacy datasets to use the v2 schema.",
    )
    parser.add_argument(
        "--require-chunks",
        action="store_true",
        help="Fail when split indexes are unavailable instead of skipping qrel-to-index validation.",
    )
    parser.add_argument("--full", action="store_true", help="Print every case-level error instead of a summary")
    parser.add_argument("--max-examples", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    policy = _load_json(args.policy)
    plan = _load_json(args.expansion_plan)
    manifest = _load_json(args.document_manifest)
    freeze_report = validate_config_freeze(_load_json(args.config_freeze))
    plan_report = validate_expansion_plan(plan, policy)
    manifest_report = validate_document_manifest(manifest, policy)
    chunk_paths = {
        "development": args.development_chunks,
        "locked_test": args.locked_chunks,
    }
    available_chunks = all(path.is_file() for path in chunk_paths.values())
    chunks_by_split = (
        {split: _load_json(path) for split, path in chunk_paths.items()}
        if available_chunks
        else ({split: [] for split in chunk_paths} if args.require_chunks else None)
    )
    label_report = validate_labeled_expansion(
        _load_json(args.development_new),
        _load_json(args.locked_answerable),
        _load_json(args.locked_no_answer),
        plan,
        manifest,
        policy,
        chunks_by_split=chunks_by_split,
    )
    if args.expansion_only:
        report = None
        output = {"valid": label_report["valid"]}
    else:
        report = validate_datasets(
            _load_json(args.development),
            _load_json(args.locked_test),
            policy,
        )
        output = report if args.full else summarize_report(report, max(0, args.max_examples))
    output["expansion_plan_validation"] = plan_report
    output["document_manifest_validation"] = manifest_report
    output["labeled_expansion_validation"] = label_report
    output["config_freeze_validation"] = freeze_report
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if (
        (report is not None and not report["valid"])
        or not plan_report["valid"]
        or not manifest_report["valid"]
        or not label_report["valid"]
        or not freeze_report["valid"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
