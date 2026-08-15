import unittest


from evaluation.answer.evaluate import _character_ngram_recall, evaluate_answers


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
                "evidence_sufficient": True,
                "retrieval_results": [{"chunk_id": "chunk-a"}],
                "cited_chunk_ids": ["chunk-a", "noise"],
                "citations": [
                    {"chunk_id": "chunk-a", "source_file": "guide.pdf", "page": 1},
                    {"chunk_id": "noise", "source_file": "guide.pdf", "page": 2},
                ],
                "review": {
                    "correct": True,
                    "complete": True,
                    "citation_supported": True,
                    "uses_model_memory_or_guess": False,
                },
            },
            {
                "id": "no-answer",
                "answer": "",
                "refused": True,
                "evidence_sufficient": False,
                "retrieval_results": [{"chunk_id": "noise"}],
                "cited_chunk_ids": [],
            },
        ]

        report = evaluate_answers(cases, predictions)

        self.assertEqual(report["metrics"]["required_fact_lexical_recall"], 1.0)
        self.assertEqual(report["metrics"]["required_fact_character_bigram_recall"], 1.0)
        self.assertEqual(report["metrics"]["citation_precision"], 0.5)
        self.assertEqual(report["metrics"]["citation_recall"], 0.5)
        self.assertEqual(report["metrics"]["no_answer_refusal_accuracy"], 1.0)
        self.assertEqual(report["metrics"]["citation_metadata_complete_rate"], 1.0)
        self.assertEqual(report["metrics"]["evidence_insufficient_refusal_compliance"], 1.0)
        self.assertEqual(report["manual_review"]["correct_rate"], 1.0)
        self.assertEqual(report["manual_review"]["uses_model_memory_or_guess_rate"], 0.0)
        self.assertEqual(report["failure_stage_counts"], {"passed": 2})

    def test_distinguishes_retrieval_evidence_and_generation_errors(self):
        cases = [
            {
                "id": "retrieval",
                "source_file": "guide.pdf",
                "required_facts": ["fact"],
                "relevant_chunk_ids": ["gold"],
            },
            {
                "id": "evidence",
                "source_file": "guide.pdf",
                "required_facts": ["fact"],
                "relevant_chunk_ids": ["gold"],
            },
            {
                "id": "generation",
                "source_file": "guide.pdf",
                "required_facts": ["fact"],
                "relevant_chunk_ids": ["gold"],
            },
        ]
        predictions = [
            {
                "id": "retrieval",
                "answer": "",
                "refused": True,
                "evidence_sufficient": False,
                "retrieval_results": [{"chunk_id": "noise"}],
                "cited_chunk_ids": [],
            },
            {
                "id": "evidence",
                "answer": "",
                "refused": True,
                "evidence_sufficient": False,
                "retrieval_results": [{"chunk_id": "gold"}],
                "cited_chunk_ids": [],
            },
            {
                "id": "generation",
                "answer": "unsupported",
                "refused": False,
                "evidence_sufficient": True,
                "retrieval_results": [{"chunk_id": "gold"}],
                "cited_chunk_ids": [],
            },
        ]

        report = evaluate_answers(cases, predictions)

        self.assertEqual(
            report["failure_stage_counts"],
            {"evidence_judgment_error": 1, "generation_error": 1, "retrieval_error": 1},
        )

    def test_prediction_ids_must_match_the_dataset(self):
        cases = [{"id": "expected", "source_file": None}]
        predictions = [{"id": "unexpected", "answer": "", "refused": True}]

        with self.assertRaisesRegex(ValueError, "missing=.*expected"):
            evaluate_answers(cases, predictions)


if __name__ == "__main__":
    unittest.main()
