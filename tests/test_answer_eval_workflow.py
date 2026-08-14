import json
import unittest
from pathlib import Path


from evaluation.run_answer_evaluation import build_prompt, merge_predictions, parse_search_traces


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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

    def test_parses_authoritative_mcp_trace_and_merges_client_answer(self):
        payload = {
            "query": "question",
            "evidence_sufficient": True,
            "confidence": 0.9,
            "reason": "supported",
            "results": [
                {"chunk_id": "chunk-1", "source_file": "source.pdf", "page": 2, "text": "evidence"}
            ],
        }
        event = {
            "type": "item.completed",
            "item": {
                "type": "mcp_tool_call",
                "tool": "search_knowledge_base",
                "status": "completed",
                "error": None,
                "arguments": {"query": "question", "limit": 5},
                "result": {"content": [{"type": "text", "text": json.dumps(payload)}]},
            },
        }
        traces = parse_search_traces(json.dumps(event), 5)
        cases = [{"id": "case-1", "query": "question"}]
        output = {
            "predictions": [
                {
                    "id": "case-1",
                    "answer": "answer",
                    "refused": False,
                    "cited_chunk_ids": ["chunk-1"],
                    "citations": [{"chunk_id": "chunk-1", "source_file": "source.pdf", "page": 2}],
                }
            ]
        }

        predictions = merge_predictions(cases, output, traces)

        self.assertEqual(predictions[0]["confidence"], 0.9)
        self.assertEqual(predictions[0]["retrieval_results"][0]["text"], "evidence")
        self.assertIsNone(predictions[0]["review"]["correct"])


if __name__ == "__main__":
    unittest.main()
