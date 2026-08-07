import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_pipeline import DEFAULT_INDEX_PATH, build_index, search, split_chunks, status


class RAGPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = build_index()

    def test_chunking_retains_overlap(self):
        text = "第一段" * 180 + "\n\n" + "第二段" * 180
        chunks = split_chunks(text, size=200, overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunks))

    def test_index_contains_the_extracted_documents(self):
        self.assertEqual(self.summary["documents"], 2)
        self.assertGreater(self.summary["chunks"], 400)
        self.assertTrue(DEFAULT_INDEX_PATH.exists())

    def test_retrieval_returns_traceable_scholarship_evidence(self):
        results = search("2024年研究生学业奖学金评审实施细则", limit=3)
        self.assertTrue(results)
        self.assertTrue(all(result["page"] > 0 for result in results))
        self.assertTrue(all(result["artifact_path"].startswith("artifacts/") for result in results))
        self.assertTrue(any("奖学金" in result["source_file"] for result in results))

    def test_status_matches_built_index(self):
        self.assertEqual(status()["chunks"], self.summary["chunks"])


if __name__ == "__main__":
    unittest.main()
