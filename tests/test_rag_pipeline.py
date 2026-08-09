import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval import (
    BM25_PATH_NAME,
    CHUNKS_PATH_NAME,
    DEFAULT_INDEX_PATH,
    EMBEDDINGS_PATH_NAME,
    bm25_search,
    build_index,
    rrf_fuse,
    search,
    split_chunks,
    status,
    vector_search,
)


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
        self.assertTrue((DEFAULT_INDEX_PATH.parent / CHUNKS_PATH_NAME).exists())
        self.assertTrue((DEFAULT_INDEX_PATH.parent / BM25_PATH_NAME).exists())
        self.assertTrue((DEFAULT_INDEX_PATH.parent / EMBEDDINGS_PATH_NAME).exists())

    def test_bm25_and_vector_retrievers_return_candidates(self):
        bm25_results = bm25_search("研究生学业奖学金", candidate_limit=5)
        vector_results = vector_search("研究生怎么评奖", candidate_limit=5)
        self.assertEqual(len(bm25_results), 5)
        self.assertEqual(len(vector_results), 5)
        self.assertTrue(all("record_index" in result for result in bm25_results))
        self.assertTrue(all("record_index" in result for result in vector_results))

    def test_rrf_fuses_duplicate_and_single_channel_results(self):
        fused = rrf_fuse(
            [{"record_index": 0, "score": 3.0}, {"record_index": 1, "score": 2.0}],
            [{"record_index": 1, "score": 0.8}, {"record_index": 2, "score": 0.7}],
            limit=3,
        )
        self.assertEqual([item["record_index"] for item in fused], [1, 0, 2])
        self.assertEqual(fused[0]["matched_by"], ["bm25", "vector"])
        self.assertEqual(fused[1]["matched_by"], ["bm25"])
        self.assertEqual(fused[2]["matched_by"], ["vector"])

    def test_retrieval_returns_traceable_scholarship_evidence(self):
        results = search("2024年研究生学业奖学金评审实施细则", limit=3)
        self.assertTrue(results)
        self.assertTrue(all(result["page"] > 0 for result in results))
        self.assertTrue(all(result["artifact_path"].startswith("storage/artifacts/") for result in results))
        self.assertTrue(any("奖学金" in result["source_file"] for result in results))
        self.assertTrue(all(result["score_type"] == "rrf" for result in results))
        self.assertTrue(all(result["matched_by"] for result in results))

    def test_status_matches_built_index(self):
        self.assertEqual(status()["chunks"], self.summary["chunks"])
        self.assertEqual(status()["retrieval_mode"], "hybrid_bm25_embedding_rrf")
        self.assertEqual(status()["embedding_model"], "BAAI/bge-base-zh-v1.5")

    def test_search_rejects_invalid_arguments(self):
        with self.assertRaises(ValueError):
            search("   ")
        with self.assertRaises(ValueError):
            search("奖学金", limit=11)


if __name__ == "__main__":
    unittest.main()
