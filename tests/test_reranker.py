import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from retrieval.reranker import rerank


class FakeCrossEncoder:
    def predict(self, pairs, **kwargs):
        self.pairs = pairs
        self.kwargs = kwargs
        return [0.4, 0.9, 0.9][: len(pairs)]


class RerankerTests(unittest.TestCase):
    def test_reranker_scores_every_candidate_and_keeps_ties_stable(self):
        candidates = [
            {"record_index": 0, "score": 0.3},
            {"record_index": 1, "score": 0.2},
            {"record_index": 2, "score": 0.1},
        ]
        chunks = [{"text": "甲"}, {"text": "乙"}, {"text": "丙"}]
        model = FakeCrossEncoder()

        results = rerank("问题", candidates, chunks, model=model)

        self.assertEqual(model.pairs, [("问题", "甲"), ("问题", "乙"), ("问题", "丙")])
        self.assertEqual([item["record_index"] for item in results], [1, 2, 0])
        self.assertEqual(results[0]["retrieval_rank"], 2)
        self.assertEqual(results[0]["rrf_score"], 0.2)
        self.assertEqual(results[0]["reranker_score"], 0.9)
        self.assertAlmostEqual(results[0]["score"], 0.925)

    def test_reranker_reuses_query_expansion_terms(self):
        candidates = [{"record_index": 0, "score": 1.0}]
        chunks = [{"text": "学生可按规定提出书面申诉。"}]
        model = FakeCrossEncoder()

        rerank("学生对学校处分有异议可以申诉吗", candidates, chunks, model=model)

        expanded_query, _ = model.pairs[0]
        self.assertIn("学校处分异议申诉", expanded_query)
        self.assertIn("提出申诉", expanded_query)


if __name__ == "__main__":
    unittest.main()
