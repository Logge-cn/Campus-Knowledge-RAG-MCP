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
    load_index,
    rrf_fuse,
    search,
    split_chunks,
    status,
    vector_search,
)
from retrieval.chunking import split_table_chunks, token_count
from retrieval.index import _source_digest
from retrieval.query_expansion import expand_query


class RAGPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = build_index()

    def test_chunking_retains_overlap(self):
        text = "第一段" * 180 + "\n\n" + "第二段" * 180
        chunks = split_chunks(text, size=200, overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunks))
        self.assertTrue(all(token_count(chunk) <= 200 for chunk in chunks))
        self.assertIn(chunks[0][-10:], chunks[1])

    def test_default_chunking_enforces_the_pdf_token_limits(self):
        chunks = split_chunks("第一段。" * 600)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(token_count(chunk) <= 512 for chunk in chunks))

    def test_table_chunking_repeats_headers_and_tracks_row_ranges(self):
        table = "# 成绩表\n\n| 姓名 | 成绩 |\n| --- | --- |\n" + "\n".join(
            f"| 学生{index} | {index} |" for index in range(1, 180)
        )
        chunks = split_table_chunks(table, target_tokens=80, max_tokens=100)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all("| 姓名 | 成绩 |" in chunk["text"] for chunk in chunks))
        self.assertTrue(all(token_count(chunk["text"]) <= 100 for chunk in chunks))
        self.assertEqual(chunks[0]["row_start"], 1)
        self.assertEqual(chunks[-1]["row_end"], 179)
        self.assertTrue(all(left["row_end"] + 1 == right["row_start"] for left, right in zip(chunks, chunks[1:])))

    def test_source_digest_changes_when_traceability_metadata_changes(self):
        record = {"chunk_id": "document/page.md#1", "text": "正文", "low_confidence": False}
        changed = {**record, "low_confidence": True}
        self.assertNotEqual(_source_digest([record]), _source_digest([changed]))

    def test_query_expansion_preserves_the_original_question_and_adds_policy_synonyms(self):
        self.assertEqual(expand_query("普通问题"), "普通问题")
        self.assertIn("办理手续离校", expand_query("休学后需要离校手续吗"))
        self.assertIn("评定资格", expand_query("退学研究生能参加奖学金评选吗"))

    def test_index_contains_the_extracted_documents(self):
        self.assertEqual(self.summary["documents"], 2)
        self.assertGreater(self.summary["chunks"], 400)
        self.assertTrue(DEFAULT_INDEX_PATH.exists())
        self.assertTrue((DEFAULT_INDEX_PATH.parent / CHUNKS_PATH_NAME).exists())
        self.assertTrue((DEFAULT_INDEX_PATH.parent / BM25_PATH_NAME).exists())
        self.assertTrue((DEFAULT_INDEX_PATH.parent / EMBEDDINGS_PATH_NAME).exists())
        records = load_index()["chunks"]
        table = next(record for record in records if record["source_type"] == "table")
        ocr = next(record for record in records if record["source_type"] == "ocr")
        self.assertTrue(table["table_id"])
        self.assertGreaterEqual(table["row_start"], 1)
        self.assertGreaterEqual(table["row_end"], table["row_start"])
        self.assertIsInstance(table["low_confidence"], bool)
        self.assertEqual(len(table["source_sha256"]), 64)
        self.assertIsInstance(ocr["confidence"], float)
        self.assertEqual(len(ocr["content_sha256"]), 64)

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
        self.assertTrue(all(result["score_type"] == "cross_encoder_rrf_blend" for result in results))
        self.assertTrue(all(result["matched_by"] for result in results))
        self.assertTrue(all("reranker_score" in result for result in results))
        self.assertTrue(all("rrf_score" in result for result in results))

    def test_status_matches_built_index(self):
        self.assertEqual(status()["chunks"], self.summary["chunks"])
        self.assertEqual(status()["retrieval_mode"], "hybrid_rrf_cross_encoder_rerank")
        self.assertEqual(status()["candidate_limit"], 20)
        self.assertEqual(status()["rerank_candidate_limit"], 20)
        self.assertEqual(status()["embedding_model"], "BAAI/bge-base-zh-v1.5")
        self.assertEqual(status()["chunk_target_tokens"], 410)
        self.assertEqual(status()["chunk_max_tokens"], 512)
        self.assertEqual(status()["chunk_overlap_tokens"], 41)

    def test_search_rejects_invalid_arguments(self):
        with self.assertRaises(ValueError):
            search("   ")
        with self.assertRaises(ValueError):
            search("奖学金", limit=11)


if __name__ == "__main__":
    unittest.main()
