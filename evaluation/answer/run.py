"""Prepare and finalize the fixed answer suite run by isolated Codex subagents."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_ROOT = PROJECT_ROOT / "runtime" / "reports" / "answer"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mcp_server import SEARCH_TOOL_DESCRIPTION, SEARCH_TOOL_DESCRIPTION_VERSION
from retrieval import status

from evaluation.answer.evaluate import evaluate_answers


CLIENT = "codex-subagents"
DEFAULT_MODEL = "gpt-5.6-sol"
PROMPT_VERSION = "answer-eval-v1"
SUBAGENT_PROTOCOL_VERSION = "answer-eval-subagent-v1"
DEFAULT_MAX_CONCURRENCY = 3
DEFAULT_LIMIT = 5
SAFE_CASE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def _inside_project(path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(PROJECT_ROOT)
    return resolved


def _inside_reports(path: Path) -> Path:
    resolved = _inside_project(path)
    resolved.relative_to(REPORTS_ROOT.resolve())
    return resolved


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_prompt(template: str, cases: list[dict[str, Any]], limit: int) -> str:
    public_cases = [{"id": case["id"], "query": case["query"]} for case in cases]
    return template.replace("{{LIMIT}}", str(limit)).replace(
        "{{CASES_JSON}}", json.dumps(public_cases, ensure_ascii=False, indent=2)
    )


def build_agent_prompt(
    template: str,
    case: dict[str, Any],
    limit: int,
    model: str,
    result_path: str,
) -> str:
    evaluation_prompt = build_prompt(template, [case], limit)
    envelope = {
        "protocol_version": SUBAGENT_PROTOCOL_VERSION,
        "id": case["id"],
        "query": case["query"],
        "tool_call": {
            "name": "search_knowledge_base",
            "arguments": {"query": case["query"], "limit": limit},
            "result": "<copy the complete MCP result object without modification>",
        },
        "prediction": {
            "id": case["id"],
            "answer": "<answer or fixed refusal text>",
            "refused": "<boolean>",
            "cited_chunk_ids": ["<cited chunk id>"],
            "citations": [
                {"chunk_id": "<chunk id>", "source_file": "<source file>", "page": "<integer>"}
            ],
        },
    }
    return (
        f"{evaluation_prompt}\n\n"
        "子 agent 执行协议（本段只替代上面第 2、5、6 条中的产物写入和最终回复要求，"
        "不改变证据与拒答规则）：\n"
        f"- 这是一个全新上下文中的单题任务，固定模型为 {model}。\n"
        "- 除一次 njupt-rag.search_knowledge_base 调用外，只允许使用 apply_patch 写入下述结果文件；"
        "不得读取仓库文件、执行 shell、搜索网络或调用其他工具。\n"
        f"- 将完整结果写入 `{result_path}`，文件必须是一个 JSON 对象，不要使用 Markdown。\n"
        "- tool_call.result 必须逐字段复制本次 MCP 返回的完整 JSON 对象，不得摘要、删除或改写。\n"
        "- prediction 必须只依据 tool_call.result 生成；cited_chunk_ids、citations 与检索结果必须一致。\n"
        "- 拒答时 answer 固定写为“知识库证据不足，无法回答”，两个引用字段都为空数组。\n"
        "- 成功写入后，最终回复只写 case id 和 completed，不要再次输出答案。\n\n"
        "结果文件结构：\n"
        f"{json.dumps(envelope, ensure_ascii=False, indent=2)}"
    )


def build_agent_tasks(
    template: str,
    cases: list[dict[str, Any]],
    limit: int,
    model: str,
    output_dir: Path,
) -> list[dict[str, Any]]:
    tasks = []
    for case in cases:
        result_path = (output_dir / "case-results" / f"{case['id']}.json").relative_to(PROJECT_ROOT).as_posix()
        tasks.append(
            {
                "id": case["id"],
                "query": case["query"],
                "model": model,
                "fork_turns": "none",
                "limit": limit,
                "result_path": result_path,
                "prompt": build_agent_prompt(template, case, limit, model, result_path),
            }
        )
    return tasks


def _require_type(value: Any, expected: type | tuple[type, ...], message: str) -> None:
    if not isinstance(value, expected):
        raise ValueError(message)


def _require_exact_keys(value: dict[str, Any], expected: set[str], message: str) -> None:
    if set(value) != expected:
        raise ValueError(message)


def validate_case_result(task: dict[str, Any], result: dict[str, Any]) -> None:
    case_id = task["id"]
    _require_type(result, dict, f"Case {case_id} result must be an object")
    _require_exact_keys(
        result,
        {"protocol_version", "id", "query", "tool_call", "prediction"},
        f"Case {case_id} result fields do not match the protocol",
    )
    if result.get("protocol_version") != SUBAGENT_PROTOCOL_VERSION:
        raise ValueError(f"Case {case_id} has an invalid protocol_version")
    if result.get("id") != case_id or result.get("query") != task["query"]:
        raise ValueError(f"Case {case_id} identity does not match its task")

    tool_call = result.get("tool_call")
    _require_type(tool_call, dict, f"Case {case_id} has no tool_call object")
    _require_exact_keys(tool_call, {"name", "arguments", "result"}, f"Case {case_id} tool_call fields are invalid")
    if tool_call.get("name") != "search_knowledge_base":
        raise ValueError(f"Case {case_id} used an unexpected tool")
    arguments = tool_call.get("arguments")
    _require_type(arguments, dict, f"Case {case_id} has no tool arguments object")
    _require_exact_keys(arguments, {"query", "limit"}, f"Case {case_id} tool argument fields are invalid")
    expected_arguments = {"query": task["query"], "limit": task["limit"]}
    if arguments != expected_arguments:
        raise ValueError(f"Case {case_id} tool arguments do not match its task")

    payload = tool_call.get("result")
    _require_type(payload, dict, f"Case {case_id} has no MCP result object")
    if payload.get("query") != task["query"]:
        raise ValueError(f"Case {case_id} MCP result query does not match")
    _require_type(payload.get("evidence_sufficient"), bool, f"Case {case_id} has invalid evidence_sufficient")
    confidence = payload.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError(f"Case {case_id} has invalid confidence")
    _require_type(payload.get("reason"), str, f"Case {case_id} has invalid reason")
    retrieval_results = payload.get("results")
    _require_type(retrieval_results, list, f"Case {case_id} has invalid retrieval results")
    if len(retrieval_results) > task["limit"]:
        raise ValueError(f"Case {case_id} returned more than {task['limit']} retrieval results")

    retrieval_by_id: dict[str, dict[str, Any]] = {}
    for item in retrieval_results:
        _require_type(item, dict, f"Case {case_id} has a non-object retrieval result")
        for field, expected_type in (("chunk_id", str), ("source_file", str), ("page", int), ("text", str)):
            _require_type(item.get(field), expected_type, f"Case {case_id} retrieval result has invalid {field}")
        if isinstance(item["page"], bool):
            raise ValueError(f"Case {case_id} retrieval result has invalid page")
        if item["chunk_id"] in retrieval_by_id:
            raise ValueError(f"Case {case_id} has duplicate retrieval chunk ids")
        retrieval_by_id[item["chunk_id"]] = item

    prediction = result.get("prediction")
    _require_type(prediction, dict, f"Case {case_id} has no prediction object")
    _require_exact_keys(
        prediction,
        {"id", "answer", "refused", "cited_chunk_ids", "citations"},
        f"Case {case_id} prediction fields do not match the protocol",
    )
    if prediction.get("id") != case_id:
        raise ValueError(f"Case {case_id} prediction id does not match")
    _require_type(prediction.get("answer"), str, f"Case {case_id} has invalid answer")
    _require_type(prediction.get("refused"), bool, f"Case {case_id} has invalid refused flag")
    cited_chunk_ids = prediction.get("cited_chunk_ids")
    citations = prediction.get("citations")
    _require_type(cited_chunk_ids, list, f"Case {case_id} has invalid cited_chunk_ids")
    _require_type(citations, list, f"Case {case_id} has invalid citations")
    if not all(isinstance(chunk_id, str) and chunk_id for chunk_id in cited_chunk_ids):
        raise ValueError(f"Case {case_id} has an invalid cited chunk id")
    if len(cited_chunk_ids) != len(set(cited_chunk_ids)):
        raise ValueError(f"Case {case_id} has duplicate cited chunk ids")

    citation_ids = []
    for citation in citations:
        _require_type(citation, dict, f"Case {case_id} has a non-object citation")
        _require_exact_keys(
            citation,
            {"chunk_id", "source_file", "page"},
            f"Case {case_id} citation fields do not match the protocol",
        )
        chunk_id = citation.get("chunk_id")
        if not isinstance(chunk_id, str) or chunk_id not in retrieval_by_id:
            raise ValueError(f"Case {case_id} cites a chunk that was not retrieved")
        retrieved = retrieval_by_id[chunk_id]
        if citation.get("source_file") != retrieved["source_file"] or citation.get("page") != retrieved["page"]:
            raise ValueError(f"Case {case_id} citation metadata does not match retrieval")
        citation_ids.append(chunk_id)
    if citation_ids != cited_chunk_ids:
        raise ValueError(f"Case {case_id} citation ids do not match cited_chunk_ids")

    if payload["evidence_sufficient"]:
        if prediction["refused"] or not prediction["answer"].strip() or not citations:
            raise ValueError(f"Case {case_id} must answer with at least one citation")
    elif (
        not prediction["refused"]
        or prediction["answer"] != "知识库证据不足，无法回答"
        or cited_chunk_ids
        or citations
    ):
        raise ValueError(f"Case {case_id} must use the fixed insufficient-evidence refusal")


def merge_predictions(tasks: list[dict[str, Any]], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not all(isinstance(result, dict) and isinstance(result.get("id"), str) for result in results):
        raise ValueError("Every case result must be an object with a string id")
    result_ids = [result["id"] for result in results]
    if len(result_ids) != len(set(result_ids)):
        raise ValueError("Case results contain duplicate ids")
    result_by_id = {result["id"]: result for result in results}
    expected_ids = {task["id"] for task in tasks}
    if set(result_by_id) != expected_ids:
        raise ValueError(
            f"Case result ids mismatch: missing={sorted(expected_ids - set(result_by_id))}, "
            f"unexpected={sorted(set(result_by_id) - expected_ids)}"
        )

    predictions = []
    for task in tasks:
        result = result_by_id[task["id"]]
        validate_case_result(task, result)
        payload = result["tool_call"]["result"]
        prediction = result["prediction"]
        predictions.append(
            {
                "id": task["id"],
                "query": task["query"],
                "evidence_sufficient": payload["evidence_sufficient"],
                "confidence": payload["confidence"],
                "reason": payload["reason"],
                "retrieval_results": [
                    {
                        "chunk_id": item["chunk_id"],
                        "source_file": item["source_file"],
                        "page": item["page"],
                        "text": item["text"],
                    }
                    for item in payload["results"]
                ],
                "answer": prediction["answer"],
                "refused": prediction["refused"],
                "cited_chunk_ids": prediction["cited_chunk_ids"],
                "citations": prediction["citations"],
                "review": {
                    "correct": None,
                    "complete": None,
                    "citation_supported": None,
                    "uses_model_memory_or_guess": None,
                },
            }
        )
    return predictions


def _git_value(*args: str) -> str | None:
    completed = subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def prepare_run(args: argparse.Namespace) -> Path:
    dataset_path = _inside_project(args.dataset)
    prompt_path = _inside_project(args.prompt)
    prediction_schema_path = _inside_project(args.prediction_schema)
    case_result_schema_path = _inside_project(args.case_result_schema)
    output_dir = _inside_reports(
        args.output_dir
        or REPORTS_ROOT / f"answer-eval-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "case-results").mkdir()

    cases = _load_json(dataset_path)
    if not isinstance(cases, list) or not cases:
        raise ValueError("Answer evaluation dataset must be a non-empty JSON array")
    case_ids = [case.get("id") for case in cases]
    if not all(isinstance(case_id, str) and case_id for case_id in case_ids) or len(case_ids) != len(set(case_ids)):
        raise ValueError("Answer evaluation dataset must contain unique string ids")
    if not all(SAFE_CASE_ID.fullmatch(case_id) for case_id in case_ids):
        raise ValueError("Answer evaluation case ids may contain only letters, numbers, dot, underscore, and hyphen")

    template = prompt_path.read_text(encoding="utf-8")
    tasks = build_agent_tasks(template, cases, args.limit, args.model, output_dir)
    tasks_path = output_dir / "agent-tasks.json"
    _write_json(tasks_path, tasks)

    manifest = {
        "schema_version": 2,
        "status": "prepared",
        "started_at": _utc_now(),
        "finished_at": None,
        "client": CLIENT,
        "execution_mode": "one-case-per-fresh-subagent",
        "trace_provenance": "subagent-submitted-mcp-result",
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": _sha256(prompt_path),
        "subagent_protocol_version": SUBAGENT_PROTOCOL_VERSION,
        "prediction_schema_sha256": _sha256(prediction_schema_path),
        "case_result_schema_sha256": _sha256(case_result_schema_path),
        "tool_description_version": SEARCH_TOOL_DESCRIPTION_VERSION,
        "tool_description_sha256": hashlib.sha256(SEARCH_TOOL_DESCRIPTION.encode("utf-8")).hexdigest(),
        "retrieval_limit": args.limit,
        "retrieval_status": status(),
        "subagents": {
            "fork_turns": "none",
            "cases_per_agent": 1,
            "max_concurrency": args.max_concurrency,
            "expected": len(cases),
            "validated": 0,
        },
        "agent_tasks_sha256": _sha256(tasks_path),
        "dataset": {
            "path": dataset_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": _sha256(dataset_path),
            "cases": len(cases),
        },
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_status": _git_value("status", "--short"),
    }
    _write_json(output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "output_dir": output_dir.relative_to(PROJECT_ROOT).as_posix(),
                "tasks": len(tasks),
                "max_concurrency": args.max_concurrency,
                "task_manifest": tasks_path.relative_to(PROJECT_ROOT).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return output_dir


def _load_run(output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _load_json(output_dir / "manifest.json")
    tasks_path = output_dir / "agent-tasks.json"
    tasks = _load_json(tasks_path)
    if manifest.get("schema_version") != 2 or manifest.get("execution_mode") != "one-case-per-fresh-subagent":
        raise ValueError("Output directory is not a subagent answer-evaluation run")
    if (
        manifest.get("client") != CLIENT
        or manifest.get("model") != DEFAULT_MODEL
        or manifest.get("prompt_version") != PROMPT_VERSION
        or manifest.get("subagent_protocol_version") != SUBAGENT_PROTOCOL_VERSION
        or manifest.get("retrieval_limit") != DEFAULT_LIMIT
    ):
        raise ValueError("Run manifest changed the frozen evaluation boundary")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Run has no agent tasks")
    if manifest.get("agent_tasks_sha256") != _sha256(tasks_path):
        raise ValueError("Agent task manifest changed after preparation")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict) or not isinstance(dataset.get("path"), str):
        raise ValueError("Run manifest has invalid dataset metadata")
    dataset_path = _inside_project(PROJECT_ROOT / dataset["path"])
    if dataset.get("sha256") != _sha256(dataset_path):
        raise ValueError("Evaluation dataset changed after preparation")
    task_ids = [task.get("id") for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Agent task manifest has duplicate ids")
    for task in tasks:
        if (
            task.get("model") != DEFAULT_MODEL
            or task.get("fork_turns") != "none"
            or task.get("limit") != DEFAULT_LIMIT
        ):
            raise ValueError(f"Agent task {task.get('id')} changed the frozen execution boundary")
        expected_result_path = (output_dir / "case-results" / f"{task['id']}.json").relative_to(
            PROJECT_ROOT
        ).as_posix()
        if task.get("result_path") != expected_result_path:
            raise ValueError(f"Agent task {task['id']} has an invalid result path")
    return manifest, tasks


def validate_result_file(output_dir: Path, case_id: str) -> Path:
    _, tasks = _load_run(output_dir)
    task_by_id = {task["id"]: task for task in tasks}
    if case_id not in task_by_id:
        raise ValueError(f"Unknown case id: {case_id}")
    result_path = output_dir / "case-results" / f"{case_id}.json"
    validate_case_result(task_by_id[case_id], _load_json(result_path))
    print(json.dumps({"case_id": case_id, "status": "valid"}, ensure_ascii=False))
    return result_path


def finalize_run(output_dir: Path) -> None:
    manifest_path = output_dir / "manifest.json"
    manifest, tasks = _load_run(output_dir)
    try:
        result_dir = output_dir / "case-results"
        expected_paths = {result_dir / f"{task['id']}.json" for task in tasks}
        actual_paths = set(result_dir.glob("*.json"))
        if actual_paths != expected_paths:
            raise ValueError(
                f"Case result files mismatch: missing={sorted(path.name for path in expected_paths - actual_paths)}, "
                f"unexpected={sorted(path.name for path in actual_paths - expected_paths)}"
            )
        results = [_load_json(result_dir / f"{task['id']}.json") for task in tasks]
        predictions = merge_predictions(tasks, results)
        cases = _load_json(PROJECT_ROOT / manifest["dataset"]["path"])
        predictions_path = output_dir / "predictions.json"
        _write_json(predictions_path, predictions)
        manifest["status"] = "completed"
        manifest["finished_at"] = _utc_now()
        manifest["subagents"]["validated"] = len(results)
        manifest["artifacts"] = {
            "agent_tasks": _sha256(output_dir / "agent-tasks.json"),
            "predictions": _sha256(predictions_path),
        }
        _write_json(manifest_path, manifest)
        report = {"run_manifest": manifest, **evaluate_answers(cases, predictions)}
        _write_json(output_dir / "report.json", report)
    except Exception as exc:
        manifest["status"] = "validation_failed"
        manifest["finished_at"] = _utc_now()
        manifest["validation_error"] = str(exc)
        _write_json(manifest_path, manifest)
        raise

    print(
        json.dumps(
            {
                "output_dir": output_dir.relative_to(PROJECT_ROOT).as_posix(),
                "cases": len(cases),
                "failure_stage_counts": report["failure_stage_counts"],
                "manual_review_coverage": report["manual_review"]["coverage"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _add_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "evaluation" / "answer" / "dataset.json")
    parser.add_argument("--prompt", type=Path, default=PROJECT_ROOT / "evaluation" / "answer" / "prompt_v1.md")
    parser.add_argument(
        "--prediction-schema", type=Path, default=PROJECT_ROOT / "evaluation" / "answer" / "output_schema.json"
    )
    parser.add_argument(
        "--case-result-schema",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "answer" / "case_result_schema.json",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare", help="create a new isolated-subagent run")
    _add_prepare_arguments(prepare_parser)
    validate_parser = subparsers.add_parser("validate-case", help="validate one completed case result")
    validate_parser.add_argument("--output-dir", type=Path, required=True)
    validate_parser.add_argument("--case-id", required=True)
    finalize_parser = subparsers.add_parser("finalize", help="validate all cases and generate reports")
    finalize_parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "prepare":
        if args.model != DEFAULT_MODEL:
            raise ValueError(f"Answer evaluation model must remain {DEFAULT_MODEL}")
        if args.limit != DEFAULT_LIMIT:
            raise ValueError(f"Answer evaluation retrieval limit must remain {DEFAULT_LIMIT}")
        if not 1 <= args.max_concurrency <= DEFAULT_MAX_CONCURRENCY:
            raise ValueError(f"Max concurrency must be between 1 and {DEFAULT_MAX_CONCURRENCY}")
        prepare_run(args)
        return
    output_dir = _inside_reports(args.output_dir)
    if args.command == "validate-case":
        validate_result_file(output_dir, args.case_id)
        return
    finalize_run(output_dir)


if __name__ == "__main__":
    main()
