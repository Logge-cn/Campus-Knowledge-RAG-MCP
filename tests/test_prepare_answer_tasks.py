import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.prepare_answer_tasks import prepare_tasks


class PrepareAnswerTasksTests(unittest.TestCase):
    def test_preserves_retrieval_decision_evidence_and_prediction_contract(self):
        calls = []

        def fake_retrieve(query, limit):
            calls.append((query, limit))
            return {
                "evidence_sufficient": False,
                "confidence": 0.2,
                "reason": "insufficient_retrieval_evidence",
                "results": [
                    {
                        "chunk_id": "chunk-1",
                        "source_file": "source.pdf",
                        "page": 2,
                        "text": "evidence",
                    }
                ],
            }

        tasks = prepare_tasks([{"id": "case-1", "query": "question"}], 3, fake_retrieve)

        self.assertEqual(calls, [("question", 3)])
        self.assertFalse(tasks[0]["evidence_sufficient"])
        self.assertEqual(tasks[0]["evidence"][0]["chunk_id"], "chunk-1")
        self.assertEqual(tasks[0]["prediction_contract"]["refused"], "boolean")


if __name__ == "__main__":
    unittest.main()
