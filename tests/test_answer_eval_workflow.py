import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


from evaluation.run_answer_evaluation import (
    SUBAGENT_PROTOCOL_VERSION,
    build_agent_tasks,
    build_prompt,
    finalize_run,
    merge_predictions,
    prepare_run,
    validate_case_result,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_task(case_id: str = "case-1", query: str = "question") -> dict:
    return {
        "id": case_id,
        "query": query,
        "model": "gpt-5.6-sol",
        "fork_turns": "none",
        "limit": 5,
        "result_path": f"evaluation/reports/test/case-results/{case_id}.json",
        "prompt": "prompt",
    }


def make_result(*, evidence_sufficient: bool = True) -> dict:
    prediction = {
        "id": "case-1",
        "answer": "answer",
        "refused": False,
        "cited_chunk_ids": ["chunk-1"],
        "citations": [{"chunk_id": "chunk-1", "source_file": "source.pdf", "page": 2}],
    }
    if not evidence_sufficient:
        prediction = {
            "id": "case-1",
            "answer": "知识库证据不足，无法回答",
            "refused": True,
            "cited_chunk_ids": [],
            "citations": [],
        }
    return {
        "protocol_version": SUBAGENT_PROTOCOL_VERSION,
        "id": "case-1",
        "query": "question",
        "tool_call": {
            "name": "search_knowledge_base",
            "arguments": {"query": "question", "limit": 5},
            "result": {
                "query": "question",
                "evidence_sufficient": evidence_sufficient,
                "confidence": 0.9,
                "reason": "supported" if evidence_sufficient else "insufficient_retrieval_evidence",
                "results": [
                    {"chunk_id": "chunk-1", "source_file": "source.pdf", "page": 2, "text": "evidence"}
                ],
            },
        },
        "prediction": prediction,
    }


class AnswerEvaluationWorkflowTests(unittest.TestCase):
    def test_fixed_dataset_covers_required_scenarios(self):
        cases = json.loads((PROJECT_ROOT / "evaluation" / "answer_eval_dataset.json").read_text(encoding="utf-8"))

        self.assertEqual(len({case["id"] for case in cases}), len(cases))
        self.assertTrue(
            {"ordinary_text", "table", "scanned_pdf", "outside_corpus", "wrong_school", "time_sensitive"}
            <= {case["scenario"] for case in cases}
        )
        self.assertTrue(any(case.get("source_type") == "native" for case in cases))
        self.assertTrue(any(case.get("source_type") == "scanned" for case in cases))
        for case in cases:
            if case["source_file"] is not None:
                self.assertTrue(case["relevant_chunk_ids"])
                self.assertTrue(case["required_facts"])

    def test_prompt_exposes_questions_but_not_gold_answers(self):
        cases = [{"id": "case-1", "query": "question", "gold_answer": "secret gold"}]

        prompt = build_prompt("limit={{LIMIT}}\n{{CASES_JSON}}", cases, 5)

        self.assertIn('"query": "question"', prompt)
        self.assertNotIn("secret gold", prompt)

    def test_builds_one_fresh_subagent_task_per_case_without_cross_case_context(self):
        cases = [
            {"id": "case-1", "query": "question one", "gold_answer": "secret one"},
            {"id": "case-2", "query": "question two", "gold_answer": "secret two"},
        ]

        tasks = build_agent_tasks(
            "limit={{LIMIT}}\n{{CASES_JSON}}",
            cases,
            5,
            "gpt-5.6-sol",
            PROJECT_ROOT / "evaluation" / "reports" / "test-run",
        )

        self.assertEqual([task["id"] for task in tasks], ["case-1", "case-2"])
        self.assertTrue(all(task["fork_turns"] == "none" for task in tasks))
        self.assertIn("question one", tasks[0]["prompt"])
        self.assertNotIn("question two", tasks[0]["prompt"])
        self.assertNotIn("secret one", tasks[0]["prompt"])
        self.assertNotEqual(tasks[0]["result_path"], tasks[1]["result_path"])

    def test_validates_subagent_mcp_result_and_merges_prediction(self):
        task = make_task()
        result = make_result()

        validate_case_result(task, result)
        predictions = merge_predictions([task], [result])

        self.assertEqual(predictions[0]["confidence"], 0.9)
        self.assertEqual(predictions[0]["retrieval_results"][0]["text"], "evidence")
        self.assertIsNone(predictions[0]["review"]["correct"])

    def test_rejects_citation_metadata_not_present_in_retrieval(self):
        result = make_result()
        result["prediction"]["citations"][0]["page"] = 3

        with self.assertRaisesRegex(ValueError, "citation metadata"):
            validate_case_result(make_task(), result)

    def test_requires_fixed_refusal_when_evidence_is_insufficient(self):
        result = make_result(evidence_sufficient=False)
        validate_case_result(make_task(), result)
        result["prediction"]["answer"] = "guess"

        with self.assertRaisesRegex(ValueError, "fixed insufficient-evidence refusal"):
            validate_case_result(make_task(), result)

    def test_prepares_and_finalizes_a_subagent_run_without_codex_cli(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "evaluation" / "reports") as temp_dir:
            temp_root = Path(temp_dir)
            dataset_path = temp_root / "dataset.json"
            dataset_path.write_text(
                json.dumps([{"id": "case-1", "query": "question", "source_file": None}]),
                encoding="utf-8",
            )
            output_dir = temp_root / "run"
            args = argparse.Namespace(
                dataset=dataset_path,
                prompt=PROJECT_ROOT / "evaluation" / "answer_eval_prompt_v1.md",
                prediction_schema=PROJECT_ROOT / "evaluation" / "answer_eval_output_schema.json",
                case_result_schema=PROJECT_ROOT / "evaluation" / "answer_eval_case_result_schema.json",
                output_dir=output_dir,
                model="gpt-5.6-sol",
                limit=5,
                max_concurrency=3,
            )
            with patch("evaluation.run_answer_evaluation.status", return_value={"schema_version": 4}):
                prepare_run(args)

            tasks = json.loads((output_dir / "agent-tasks.json").read_text(encoding="utf-8"))
            result = make_result(evidence_sufficient=False)
            (output_dir / "case-results" / "case-1.json").write_text(
                json.dumps(result, ensure_ascii=False),
                encoding="utf-8",
            )
            finalize_run(output_dir)

            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            predictions = json.loads((output_dir / "predictions.json").read_text(encoding="utf-8"))
            self.assertEqual(len(tasks), 1)
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["subagents"]["validated"], 1)
            self.assertTrue(predictions[0]["refused"])


if __name__ == "__main__":
    unittest.main()
