import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from retrieval.reranker import rerank, rerank_from_scores


class FakeCrossEncoder:
    def predict(self, pairs, **kwargs):
        self.pairs = pairs
        self.kwargs = kwargs
        return [0.4, 0.9, 0.9][: len(pairs)]


class RerankerTests(unittest.TestCase):
    def test_precomputed_scores_use_the_same_production_policy(self):
        candidates = [
            {"record_index": 0, "score": 0.3},
            {"record_index": 1, "score": 0.2},
            {"record_index": 2, "score": 0.1},
        ]
        chunks = [{"text": "甲"}, {"text": "乙"}, {"text": "丙"}]

        results = rerank_from_scores("问题", candidates, chunks, [0.4, 0.9, 0.9])

        self.assertEqual([item["record_index"] for item in results], [1, 2, 0])

    def test_precomputed_scores_require_one_score_per_candidate(self):
        with self.assertRaisesRegex(ValueError, "one value"):
            rerank_from_scores(
                "问题",
                [{"record_index": 0, "score": 1.0}],
                [{"text": "甲"}],
                [],
            )

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
        self.assertEqual(results[0]["normalized_reranker_score"], 1.0)
        self.assertAlmostEqual(results[0]["score"], 1.08)
        self.assertEqual(model.kwargs["batch_size"], 16)

    def test_reranker_reuses_query_expansion_terms(self):
        candidates = [{"record_index": 0, "score": 1.0}]
        chunks = [{"text": "学生可按规定提出书面申诉。"}]
        model = FakeCrossEncoder()

        rerank("学生对学校处分有异议可以申诉吗", candidates, chunks, model=model)

        expanded_query, _ = model.pairs[0]
        self.assertIn("学校处分异议申诉", expanded_query)
        self.assertIn("提出申诉", expanded_query)

    def test_reranker_scores_original_evidence_instead_of_retrieval_projection(self):
        candidates = [{"record_index": 0, "score": 1.0}]
        chunks = [{"text": "原始 Markdown 表格", "retrieval_text": "检索专用字段值文本"}]
        model = FakeCrossEncoder()

        rerank("问题", candidates, chunks, model=model)

        self.assertEqual(model.pairs, [("问题", "原始 Markdown 表格")])

    def test_reranker_accepts_an_experimental_batch_size(self):
        candidates = [{"record_index": 0, "score": 1.0}]
        chunks = [{"text": "测试内容"}]
        model = FakeCrossEncoder()

        rerank("问题", candidates, chunks, model=model, batch_size=32)

        self.assertEqual(model.kwargs["batch_size"], 32)

    def test_reranker_rejects_invalid_batch_size(self):
        with self.assertRaisesRegex(ValueError, "batch_size"):
            rerank("问题", [], [], batch_size=0)

    def test_table_query_keeps_hybrid_top_five_inside_final_top_five(self):
        candidates = [
            {"record_index": index, "score": 1.0 - index / 10}
            for index in range(6)
        ]
        chunks = [
            {"text": f"证据{index}", "source_type": "table" if index == 4 else "pdf"}
            for index in range(6)
        ]

        class TableCrossEncoder:
            def predict(self, pairs, **kwargs):
                return [0.9, 0.8, 0.7, 0.6, 0.1, 1.0]

        results = rerank(
            "百分制成绩各区间对应的绩点范围是什么？",
            candidates,
            chunks,
            model=TableCrossEncoder(),
        )

        self.assertIn(4, {item["record_index"] for item in results[:5]})
        self.assertEqual(results[0]["record_index"], 5)

    def test_high_confidence_hybrid_top_five_candidate_is_not_dropped(self):
        candidates = [
            {"record_index": index, "score": 1.0 - index / 10}
            for index in range(7)
        ]
        chunks = [{"text": f"证据{index}", "source_type": "pdf"} for index in range(7)]

        class ConfidenceCrossEncoder:
            def predict(self, pairs, **kwargs):
                return [0.999, 0.995, 0.986, 0.973, 0.995, 1.0, 0.998]

        results = rerank("普通问题", candidates, chunks, model=ConfidenceCrossEncoder())

        self.assertIn(3, {item["record_index"] for item in results[:5]})


if __name__ == "__main__":
    unittest.main()
