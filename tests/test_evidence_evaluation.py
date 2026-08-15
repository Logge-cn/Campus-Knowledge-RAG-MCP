import unittest


from evaluation.evidence.evaluate import classification_metrics, select_threshold


class EvidenceEvaluationTests(unittest.TestCase):
    def test_classification_metrics_report_both_error_types(self):
        records = [
            {"answerable": True, "confidence": 0.9, "hard_rules_passed": True},
            {"answerable": True, "confidence": 0.2, "hard_rules_passed": True},
            {"answerable": False, "confidence": 0.8, "hard_rules_passed": True},
            {"answerable": False, "confidence": 0.1, "hard_rules_passed": True},
        ]

        metrics = classification_metrics(records, 0.5)

        self.assertEqual(metrics["false_answer_rate"], 0.5)
        self.assertEqual(metrics["false_refusal_rate"], 0.5)

    def test_select_threshold_respects_false_answer_limit(self):
        records = [
            {"answerable": True, "confidence": 0.9, "hard_rules_passed": True},
            {"answerable": True, "confidence": 0.7, "hard_rules_passed": True},
            {"answerable": False, "confidence": 0.6, "hard_rules_passed": True},
            {"answerable": False, "confidence": 0.1, "hard_rules_passed": True},
        ]

        selected = select_threshold(records, maximum_false_answer_rate=0.0)

        self.assertGreater(selected["threshold"], 0.6)
        self.assertEqual(selected["false_answer_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
