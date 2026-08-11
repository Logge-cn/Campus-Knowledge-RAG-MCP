import unittest


from evaluation.evaluate import _is_relevant, _metrics


class EvaluationMetricsTests(unittest.TestCase):
    def test_chunk_labels_take_precedence_over_page_labels(self):
        case = {
            "source_file": "guide.pdf",
            "pages": [1],
            "relevant_chunk_ids": ["target"],
        }

        self.assertFalse(_is_relevant(case, {"chunk_id": "other", "source_file": "guide.pdf", "page": 1}))
        self.assertTrue(_is_relevant(case, {"chunk_id": "target", "source_file": "guide.pdf", "page": 2}))

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


if __name__ == "__main__":
    unittest.main()
