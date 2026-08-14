"""Run the fixed answer suite through Codex CLI and the real local MCP server."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mcp_server import SEARCH_TOOL_DESCRIPTION, SEARCH_TOOL_DESCRIPTION_VERSION
from retrieval import status

from evaluation.evaluate_answers import evaluate_answers


CLIENT = "codex-cli"
DEFAULT_MODEL = "gpt-5.6-sol"
PROMPT_VERSION = "answer-eval-v1"


def _inside_project(path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(PROJECT_ROOT)
    return resolved


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_prompt(template: str, cases: list[dict[str, Any]], limit: int) -> str:
    public_cases = [{"id": case["id"], "query": case["query"]} for case in cases]
    return template.replace("{{LIMIT}}", str(limit)).replace(
        "{{CASES_JSON}}", json.dumps(public_cases, ensure_ascii=False, indent=2)
    )


def build_codex_command(
    codex: str,
    model: str,
    schema_path: Path,
    final_message_path: Path,
) -> list[str]:
    root = PROJECT_ROOT.as_posix()
    python = Path(sys.executable).resolve().as_posix()
    configs = [
        f'mcp_servers.njupt-rag.command={json.dumps(python)}',
        'mcp_servers.njupt-rag.args=["src/mcp_server.py"]',
        f'mcp_servers.njupt-rag.cwd={json.dumps(root)}',
        'mcp_servers.njupt-rag.env.RAG_PREWARM="0"',
        'mcp_servers.njupt-rag.env.PYTHONUTF8="1"',
        'mcp_servers.njupt-rag.env.PYTHONIOENCODING="utf-8"',
        "mcp_servers.njupt-rag.required=true",
        'mcp_servers.njupt-rag.enabled_tools=["search_knowledge_base"]',
        'mcp_servers.njupt-rag.default_tools_approval_mode="approve"',
    ]
    command = [
        codex,
        "exec",
        "-C",
        str(PROJECT_ROOT),
        "--ignore-user-config",
        "--model",
        model,
        "--sandbox",
        "read-only",
        "--ephemeral",
        "--json",
        "--color",
        "never",
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(final_message_path),
    ]
    for config in configs:
        command.extend(("-c", config))
    command.append("-")
    return command


def parse_search_traces(raw_events: str, expected_limit: int) -> dict[str, dict[str, Any]]:
    traces: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(raw_events.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Codex event line {line_number} is not valid JSON") from exc
        item = event.get("item")
        if event.get("type") != "item.completed" or not isinstance(item, dict):
            continue
        if item.get("type") != "mcp_tool_call" or item.get("tool") != "search_knowledge_base":
            continue
        if item.get("status") != "completed" or item.get("error") is not None:
            raise ValueError(f"MCP search call failed: {item.get('error')}")
        arguments = item.get("arguments")
        if not isinstance(arguments, dict) or not isinstance(arguments.get("query"), str):
            raise ValueError("MCP search event has invalid arguments")
        if arguments.get("limit") != expected_limit:
            raise ValueError(f"MCP search used limit={arguments.get('limit')}, expected {expected_limit}")
        query = arguments["query"]
        if query in traces:
            raise ValueError(f"MCP search was called more than once for query: {query}")
        result = item.get("result")
        content = result.get("content") if isinstance(result, dict) else None
        if not isinstance(content, list) or not content or not isinstance(content[0], dict):
            raise ValueError(f"MCP search returned no text result for query: {query}")
        text = content[0].get("text")
        if not isinstance(text, str):
            raise ValueError(f"MCP search returned invalid text for query: {query}")
        payload = json.loads(text)
        if payload.get("query") != query:
            raise ValueError(f"MCP result query mismatch: {query}")
        traces[query] = payload
    return traces


def merge_predictions(
    cases: list[dict[str, Any]],
    client_output: dict[str, Any],
    traces: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    answers = client_output.get("predictions")
    if not isinstance(answers, list):
        raise ValueError("Client output has no predictions array")
    answer_by_id = {}
    for answer in answers:
        if not isinstance(answer, dict) or not isinstance(answer.get("id"), str):
            raise ValueError("Client prediction has no string id")
        if answer["id"] in answer_by_id:
            raise ValueError(f"Duplicate client prediction id: {answer['id']}")
        answer_by_id[answer["id"]] = answer

    expected_ids = {case["id"] for case in cases}
    if set(answer_by_id) != expected_ids:
        raise ValueError(
            f"Client prediction ids mismatch: missing={sorted(expected_ids - set(answer_by_id))}, "
            f"unexpected={sorted(set(answer_by_id) - expected_ids)}"
        )
    expected_queries = {case["query"] for case in cases}
    if set(traces) != expected_queries:
        raise ValueError(
            f"MCP trace queries mismatch: missing={sorted(expected_queries - set(traces))}, "
            f"unexpected={sorted(set(traces) - expected_queries)}"
        )

    predictions = []
    for case in cases:
        answer = answer_by_id[case["id"]]
        payload = traces[case["query"]]
        citations = answer.get("citations")
        cited_chunk_ids = answer.get("cited_chunk_ids")
        if not isinstance(citations, list) or not isinstance(cited_chunk_ids, list):
            raise ValueError(f"Client prediction {case['id']} has invalid citations")
        if {citation.get("chunk_id") for citation in citations if isinstance(citation, dict)} != set(cited_chunk_ids):
            raise ValueError(f"Client prediction {case['id']} citation ids do not match")
        predictions.append(
            {
                "id": case["id"],
                "query": case["query"],
                "evidence_sufficient": payload["evidence_sufficient"],
                "confidence": payload["confidence"],
                "reason": payload["reason"],
                "retrieval_results": [
                    {
                        "chunk_id": result["chunk_id"],
                        "source_file": result["source_file"],
                        "page": result["page"],
                        "text": result["text"],
                    }
                    for result in payload["results"]
                ],
                "answer": answer["answer"],
                "refused": answer["refused"],
                "cited_chunk_ids": cited_chunk_ids,
                "citations": citations,
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "evaluation" / "answer_eval_dataset.json")
    parser.add_argument("--prompt", type=Path, default=PROJECT_ROOT / "evaluation" / "answer_eval_prompt_v1.md")
    parser.add_argument(
        "--schema", type=Path, default=PROJECT_ROOT / "evaluation" / "answer_eval_output_schema.json"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_path = _inside_project(args.dataset)
    prompt_path = _inside_project(args.prompt)
    schema_path = _inside_project(args.schema)
    output_dir = _inside_project(
        args.output_dir
        or PROJECT_ROOT
        / "evaluation"
        / "reports"
        / f"answer-eval-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    cases = _load_json(dataset_path)
    if not isinstance(cases, list) or not cases:
        raise ValueError("Answer evaluation dataset must be a non-empty JSON array")
    prompt = build_prompt(prompt_path.read_text(encoding="utf-8"), cases, args.limit)
    codex = shutil.which("codex")
    if codex is None:
        raise RuntimeError("codex CLI is not available on PATH")

    raw_events_path = output_dir / "raw-events.jsonl"
    stderr_path = output_dir / "client-stderr.log"
    client_output_path = output_dir / "client-output.json"
    predictions_path = output_dir / "predictions.json"
    report_path = output_dir / "report.json"
    manifest_path = output_dir / "manifest.json"
    started_at = _utc_now()
    command = build_codex_command(codex, args.model, schema_path, client_output_path)
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    finished_at = _utc_now()
    raw_events_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")

    version = subprocess.run(
        [codex, "--version"], capture_output=True, text=True, encoding="utf-8", errors="replace"
    ).stdout.strip()
    manifest = {
        "schema_version": 1,
        "status": "completed" if completed.returncode == 0 else "client_failed",
        "started_at": started_at,
        "finished_at": finished_at,
        "client": CLIENT,
        "client_version": version,
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": _sha256(prompt_path),
        "tool_description_version": SEARCH_TOOL_DESCRIPTION_VERSION,
        "tool_description_sha256": hashlib.sha256(SEARCH_TOOL_DESCRIPTION.encode("utf-8")).hexdigest(),
        "retrieval_limit": args.limit,
        "retrieval_status": status(),
        "dataset": {
            "path": dataset_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": _sha256(dataset_path),
            "cases": len(cases),
        },
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_status": _git_value("status", "--short"),
        "client_exit_code": completed.returncode,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Codex client failed; see {stderr_path.relative_to(PROJECT_ROOT)}")

    client_output = _load_json(client_output_path)
    traces = parse_search_traces(completed.stdout, args.limit)
    predictions = merge_predictions(cases, client_output, traces)
    predictions_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {"run_manifest": manifest, **evaluate_answers(cases, predictions)}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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


if __name__ == "__main__":
    main()
