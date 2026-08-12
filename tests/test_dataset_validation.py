import copy
import unittest


from evaluation.validate_dataset import (
    summarize_report,
    validate_datasets,
    validate_document_manifest,
    validate_expansion_plan,
    validate_labeled_expansion,
)


POLICY = {
    "schema_version": 1,
    "required_categories": ["exact_term"],
    "require_document_disjoint_splits": True,
    "require_relevant_chunk_ids": True,
    "require_gold_answer": True,
    "require_required_facts": True,
    "splits": {
        "development": {
            "minimum_documents": {"native": 1, "scanned": 1},
            "minimum_answerable_cases": {"native": 1, "scanned": 1},
            "minimum_no_answer_cases": 1,
            "minimum_cases_per_category": 1,
            "minimum_answerable_cases_per_document": 1,
        },
        "locked_test": {
            "minimum_documents": {"native": 1, "scanned": 1},
            "minimum_answerable_cases": {"native": 1, "scanned": 1},
            "minimum_no_answer_cases": 1,
            "minimum_cases_per_category": 1,
            "minimum_answerable_cases_per_document": 1,
        },
    },
}


def answerable(case_id: str, split: str, source_file: str, source_type: str) -> dict:
    return {
        "id": case_id,
        "query": f"query {case_id}",
        "source_file": source_file,
        "pages": [1],
        "category": "exact_term",
        "source_type": source_type,
        "topic_group": case_id,
        "split": split,
        "relevant_chunk_ids": [f"chunk-{case_id}"],
        "gold_answer": f"answer {case_id}",
        "required_facts": [f"fact {case_id}"],
    }


def no_answer(case_id: str, split: str) -> dict:
    return {
        "id": case_id,
        "query": f"query {case_id}",
        "source_file": None,
        "pages": [],
        "category": "no_answer",
        "split": split,
    }


class DatasetValidationTests(unittest.TestCase):
    def setUp(self):
        self.development = [
            answerable("dev-native", "development", "dev-native.pdf", "native"),
            answerable("dev-scanned", "development", "dev-scanned.pdf", "scanned"),
            no_answer("dev-none", "development"),
        ]
        self.locked = [
            answerable("locked-native", "locked_test", "locked-native.pdf", "native"),
            answerable("locked-scanned", "locked_test", "locked-scanned.pdf", "scanned"),
            no_answer("locked-none", "locked_test"),
        ]

    def test_valid_document_disjoint_datasets_pass(self):
        report = validate_datasets(self.development, self.locked, POLICY)

        self.assertTrue(report["valid"])
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["stats"]["development"]["documents"], {"native": 1, "scanned": 1})

    def test_document_overlap_is_rejected(self):
        locked = copy.deepcopy(self.locked)
        locked[0]["source_file"] = "dev-native.pdf"

        report = validate_datasets(self.development, locked, POLICY)

        self.assertFalse(report["valid"])
        self.assertIn("document_split_overlap", {error["code"] for error in report["errors"]})
        self.assertEqual(report["expansion_gaps"]["splits"]["locked_test"]["documents_to_add"]["native"], 1)
        self.assertEqual(report["expansion_gaps"]["splits"]["locked_test"]["answerable_cases_to_add"]["native"], 1)
        self.assertEqual(report["expansion_gaps"]["recommended_rebuild_locked"]["minimum_new_documents"], 2)
        self.assertEqual(report["expansion_gaps"]["recommended_rebuild_locked"]["minimum_new_cases"], 3)

    def test_answerable_cases_require_chunk_labels_and_gold_answers(self):
        development = copy.deepcopy(self.development)
        development[0].pop("relevant_chunk_ids")
        development[0].pop("gold_answer")
        development[0].pop("required_facts")

        report = validate_datasets(development, self.locked, POLICY)

        codes = {error["code"] for error in report["errors"]}
        self.assertIn("missing_relevant_chunk_ids", codes)
        self.assertIn("missing_gold_answer", codes)
        self.assertIn("missing_required_facts", codes)

    def test_summary_aggregates_repeated_case_errors(self):
        development = copy.deepcopy(self.development)
        development[0].pop("gold_answer")
        development[1].pop("gold_answer")

        summary = summarize_report(validate_datasets(development, self.locked, POLICY), max_examples=1)

        self.assertEqual(summary["error_counts"]["missing_gold_answer"], 2)
        self.assertEqual(len(summary["error_examples"]["missing_gold_answer"]), 1)

    def test_each_new_document_requires_enough_cases(self):
        policy = copy.deepcopy(POLICY)
        policy["splits"]["development"]["minimum_answerable_cases_per_document"] = 2

        report = validate_datasets(self.development, self.locked, policy)

        self.assertIn("insufficient_document_cases", {error["code"] for error in report["errors"]})

    def test_expansion_plan_totals_are_executable_and_consistent(self):
        plan = {
            "minimum_new_data": {
                "pdfs": 2,
                "answerable_cases": 2,
                "no_answer_cases": 1,
                "total_cases": 3,
            },
            "document_slots": [
                {
                    "slot": "development-native-02",
                    "split": "development",
                    "source_type": "native",
                    "minimum_answerable_cases": 1,
                },
                {
                    "slot": "locked-scanned-01",
                    "split": "locked_test",
                    "source_type": "scanned",
                    "minimum_answerable_cases": 1,
                },
            ],
            "answerable_case_matrix": {
                "development-native-02": {"exact_term": 1},
                "locked-scanned-01": {"exact_term": 1},
            },
            "new_no_answer_cases": {"total": 1, "types": {"outside_corpus": 1}},
        }

        report = validate_expansion_plan(plan, POLICY)

        self.assertTrue(report["valid"])
        self.assertEqual(report["totals"]["total_cases"], 3)

        plan["minimum_new_data"]["total_cases"] = 2
        report = validate_expansion_plan(plan, POLICY)
        self.assertFalse(report["valid"])
        self.assertIn("minimum_new_data_mismatch", {error["code"] for error in report["errors"]})

    def test_document_manifest_enforces_hash_isolation_and_document_quotas(self):
        documents = []
        for index, (split, source_type) in enumerate(
            [
                ("development", "native"),
                ("development", "scanned"),
                ("locked_test", "native"),
                ("locked_test", "scanned"),
            ],
            1,
        ):
            documents.append(
                {
                    "document_id": f"document-{index}",
                    "split": split,
                    "source_type": source_type,
                    "source_file": f"document-{index}.pdf",
                    "pages": 1,
                    "sha256": f"{index:064x}",
                    "layout_profile": "single_column",
                    "topic": f"topic-{index}",
                    "extraction": {"pages_extracted": 1},
                }
            )
        manifest = {"documents": documents}

        report = validate_document_manifest(manifest, POLICY)
        self.assertTrue(report["valid"])
        self.assertEqual(report["documents_by_split_type"]["locked_test"], {"native": 1, "scanned": 1})

        manifest["documents"][-1]["derived_from"] = {"source_sha256": f"{1:064x}"}
        report = validate_document_manifest(manifest, POLICY)
        self.assertFalse(report["valid"])
        self.assertIn("manifest_content_hash_overlap", {error["code"] for error in report["errors"]})

    def test_labeled_expansion_matches_plan_manifest_and_chunks(self):
        development = [
            answerable(
                "development-native-02-001",
                "development",
                "new-development.pdf",
                "native",
            )
        ]
        locked_answerable = [
            answerable(
                "locked-scanned-01-001",
                "locked_test",
                "new-locked.pdf",
                "scanned",
            )
        ]
        locked_no_answer = [no_answer("locked-no-answer-001", "locked_test")]
        locked_no_answer[0]["no_answer_type"] = "outside_corpus"
        plan = {
            "document_slots": [
                {
                    "slot": "development-native-02",
                    "split": "development",
                    "source_type": "native",
                    "minimum_answerable_cases": 1,
                },
                {
                    "slot": "locked-scanned-01",
                    "split": "locked_test",
                    "source_type": "scanned",
                    "minimum_answerable_cases": 1,
                },
            ],
            "answerable_case_matrix": {
                "development-native-02": {"exact_term": 1},
                "locked-scanned-01": {"exact_term": 1},
            },
            "new_no_answer_cases": {"total": 1, "types": {"outside_corpus": 1}},
        }
        manifest = {
            "documents": [
                {
                    "document_id": "development-native-02",
                    "source_file": "new-development.pdf",
                    "split": "development",
                    "source_type": "native",
                },
                {
                    "document_id": "locked-scanned-01",
                    "source_file": "new-locked.pdf",
                    "split": "locked_test",
                    "source_type": "scanned",
                },
            ]
        }
        chunks = {
            "development": [
                {
                    "chunk_id": "chunk-development-native-02-001",
                    "source_file": "new-development.pdf",
                    "page": 1,
                }
            ],
            "locked_test": [
                {
                    "chunk_id": "chunk-locked-scanned-01-001",
                    "source_file": "new-locked.pdf",
                    "page": 1,
                }
            ],
        }

        report = validate_labeled_expansion(
            development,
            locked_answerable,
            locked_no_answer,
            plan,
            manifest,
            POLICY,
            chunks_by_split=chunks,
        )
        self.assertTrue(report["valid"])
        self.assertEqual(report["qrels"], {"checked": True, "chunk_ids": 2})

        chunks["locked_test"][0]["page"] = 2
        report = validate_labeled_expansion(
            development,
            locked_answerable,
            locked_no_answer,
            plan,
            manifest,
            POLICY,
            chunks_by_split=chunks,
        )
        self.assertFalse(report["valid"])
        self.assertIn("relevant_chunk_page_mismatch", {error["code"] for error in report["errors"]})


if __name__ == "__main__":
    unittest.main()
