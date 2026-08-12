import unittest


from evaluation.evaluate_answers import _character_ngram_recall, evaluate_answers


class AnswerEvaluationTests(unittest.TestCase):
    def test_character_bigram_recall_tolerates_small_wording_changes(self):
        score = _character_ngram_recall("一般在学年限为2.5至3年", "在学年限一般为2.5至3年")

        self.assertGreater(score, 0.7)

    def test_scores_required_facts_citations_refusal_and_manual_review(self):
        cases = [
            {
                "id": "answerable",
                "source_file": "guide.pdf",
                "required_facts": ["金额为一千元", "比例为百分之十"],
                "relevant_chunk_ids": ["chunk-a", "chunk-b"],
            },
            {"id": "no-answer", "source_file": None},
        ]
        predictions = [
            {
                "id": "answerable",
                "answer": "金额为一千元，比例为百分之十。",
                "refused": False,
                "cited_chunk_ids": ["chunk-a", "noise"],
                "review": {"correct": True, "complete": True, "citation_supported": True},
            },
            {"id": "no-answer", "answer": "", "refused": True, "cited_chunk_ids": []},
        ]

        report = evaluate_answers(cases, predictions)

        self.assertEqual(report["metrics"]["required_fact_lexical_recall"], 1.0)
        self.assertEqual(report["metrics"]["required_fact_character_bigram_recall"], 1.0)
        self.assertEqual(report["metrics"]["citation_precision"], 0.5)
        self.assertEqual(report["metrics"]["citation_recall"], 0.5)
        self.assertEqual(report["metrics"]["no_answer_refusal_accuracy"], 1.0)
        self.assertEqual(report["manual_review"]["correct_rate"], 1.0)

    def test_prediction_ids_must_match_the_dataset(self):
        cases = [{"id": "expected", "source_file": None}]
        predictions = [{"id": "unexpected", "answer": "", "refused": True}]

        with self.assertRaisesRegex(ValueError, "missing=.*expected"):
            evaluate_answers(cases, predictions)


if __name__ == "__main__":
    unittest.main()
