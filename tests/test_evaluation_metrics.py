import unittest


from evaluation.evaluate import (
    _calibrate_refusal_threshold,
    _is_relevant,
    _metrics,
    _refusal_breakdown,
    _refusal_metrics,
    _top1_diagnostics,
)
from evaluation.experiment_reranker_grid import relevant_rank as grid_relevant_rank
from evaluation.compare_retrieval_strategies import _baseline_rerank_from_scores, _metric_delta


class EvaluationMetricsTests(unittest.TestCase):
    def test_frozen_baseline_reranker_reproduces_raw_score_plus_rrf_prior(self):
        candidates = [
            {"record_index": 0, "score": 0.4},
            {"record_index": 1, "score": 0.2},
        ]

        result = _baseline_rerank_from_scores(candidates, [0.5, 0.54])

        self.assertEqual([item["record_index"] for item in result], [0, 1])
        self.assertAlmostEqual(result[0]["score"], 0.55)
        self.assertAlmostEqual(result[1]["score"], 0.54)

    def test_strategy_comparison_reports_optimized_minus_baseline(self):
        baseline = {
            "recall_at_1": 0.5,
            "recall_at_3": 0.7,
            "recall_at_5": 0.9,
            "mrr_at_5": 0.65,
            "ndcg_at_5": 0.6,
        }
        optimized = {
            "recall_at_1": 0.6,
            "recall_at_3": 0.75,
            "recall_at_5": 0.9,
            "mrr_at_5": 0.7,
            "ndcg_at_5": 0.64,
        }

        self.assertEqual(
            _metric_delta(baseline, optimized),
            {
                "recall_at_1": 0.1,
                "recall_at_3": 0.05,
                "recall_at_5": 0.0,
                "mrr_at_5": 0.05,
                "ndcg_at_5": 0.04,
            },
        )

    def test_chunk_labels_take_precedence_over_page_labels(self):
        case = {
            "source_file": "guide.pdf",
            "pages": [1],
            "relevant_chunk_ids": ["target"],
        }

        self.assertFalse(_is_relevant(case, {"chunk_id": "other", "source_file": "guide.pdf", "page": 1}))
        self.assertTrue(_is_relevant(case, {"chunk_id": "target", "source_file": "guide.pdf", "page": 2}))

    def test_reranker_grid_uses_exact_chunk_labels_when_available(self):
        case = {
            "source_file": "guide.pdf",
            "pages": [1],
            "relevant_chunk_ids": ["target"],
        }
        chunks = [
            {"chunk_id": "same-page-noise", "source_file": "guide.pdf", "page": 1},
            {"chunk_id": "target", "source_file": "guide.pdf", "page": 2},
        ]

        self.assertEqual(grid_relevant_rank(case, [0, 1], chunks), 2)

    def test_metrics_report_recall_at_1_3_5_and_mrr_at_5(self):
        cases = [
            {"source_file": "guide.pdf", "pages": [1]},
            {"source_file": "rules.pdf", "pages": [2]},
            {"source_file": None, "pages": []},
        ]
        chunks = [
            {"chunk_id": "noise", "source_file": "other.pdf", "page": 1},
            {"chunk_id": "guide", "source_file": "guide.pdf", "page": 1},
            {"chunk_id": "rules", "source_file": "rules.pdf", "page": 2},
        ]
        rankings = {"method": [[0, 2, 1], [0, 2, 1], [0, 1, 2]]}
        timings = {"method": [1.0, 2.0, 3.0]}

        result = _metrics(cases, rankings, chunks, timings)["method"]

        self.assertEqual(result["hits_at_1"], 0)
        self.assertEqual(result["recall_at_1"], 0.0)
        self.assertEqual(result["recall_at_3"], 1.0)
        self.assertEqual(result["recall_at_5"], 1.0)
        self.assertEqual(result["mrr"], 0.4167)
        self.assertEqual(result["mrr_at_5"], 0.4167)
        self.assertEqual(result["ndcg_at_5"], 0.5655)
        self.assertEqual(result["candidate_recall"], 1.0)
        self.assertEqual(result["oracle_recall_at_20"], 1.0)

    def test_refusal_threshold_reports_both_error_types(self):
        details = [
            {"expected": {"source_file": "a.pdf"}, "top_reranker_score": 0.9},
            {"expected": {"source_file": "b.pdf"}, "top_reranker_score": 0.7},
            {"expected": {"source_file": None}, "top_reranker_score": 0.8},
            {"expected": {"source_file": None}, "top_reranker_score": 0.2},
        ]

        result = _refusal_metrics(details, 0.75)

        self.assertEqual(result["true_answer"], 1)
        self.assertEqual(result["false_refusal"], 1)
        self.assertEqual(result["true_refusal"], 1)
        self.assertEqual(result["false_answer"], 1)
        self.assertEqual(result["balanced_accuracy"], 0.5)

    def test_refusal_threshold_is_calibrated_on_labeled_scores(self):
        details = [
            {"expected": {"source_file": "a.pdf"}, "top_reranker_score": 0.9},
            {"expected": {"source_file": "b.pdf"}, "top_reranker_score": 0.8},
            {"expected": {"source_file": None}, "top_reranker_score": 0.2},
            {"expected": {"source_file": None}, "top_reranker_score": 0.1},
        ]

        result = _calibrate_refusal_threshold(details)

        self.assertEqual(result["threshold"], 0.8)
        self.assertEqual(result["balanced_accuracy"], 1.0)

    def test_refusal_breakdown_reports_pdf_and_no_answer_types(self):
        details = [
            {
                "expected": {"source_file": "a.pdf"},
                "source_type": "native",
                "top_reranker_score": 0.9,
            },
            {
                "expected": {"source_file": "b.pdf"},
                "source_type": "native",
                "top_reranker_score": 0.4,
            },
            {
                "expected": {"source_file": None},
                "no_answer_type": "outside_corpus",
                "top_reranker_score": 0.8,
            },
            {
                "expected": {"source_file": None},
                "no_answer_type": "outside_corpus",
                "top_reranker_score": 0.2,
            },
        ]

        result = _refusal_breakdown(details, 0.75)

        self.assertEqual(result["answerable_by_pdf_type"]["native"]["false_refusal_rate"], 0.5)
        self.assertEqual(result["no_answer_by_type"]["outside_corpus"]["false_answer_rate"], 0.5)

    def test_top1_diagnostics_group_misses_by_pdf_type_and_category(self):
        details = [
            {
                "id": "a",
                "query": "问题一",
                "category": "procedure",
                "source_type": "native",
                "expected": {"source_file": "a.pdf"},
                "ranks": {"hybrid": 3, "reranker": 2},
            },
            {
                "id": "b",
                "query": "问题二",
                "category": "exact_term",
                "source_type": "scanned",
                "expected": {"source_file": "b.pdf"},
                "ranks": {"hybrid": 2, "reranker": 1},
            },
        ]

        result = _top1_diagnostics(details)

        self.assertEqual(result["by_pdf_type"]["native"]["top1_misses"], 1)
        self.assertEqual(result["by_pdf_type"]["native"]["misses_by_category"], {"procedure": 1})
        self.assertEqual(result["by_pdf_type"]["scanned"]["recall_at_1"], 1.0)


if __name__ == "__main__":
    unittest.main()
