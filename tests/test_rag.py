from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from ingestion.index import build_index
from retrieval.store import RetrievalStore


class RAGIndexTest(unittest.TestCase):
    @unittest.skip("Crawler-backed indexing was replaced by local reference files.")
    def test_build_and_search_chinese_document(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            crawler = root / "crawler.db"
            connection = sqlite3.connect(crawler)
            connection.execute(
                """CREATE TABLE documents (
                    doc_id TEXT, url TEXT, title TEXT, text TEXT, category TEXT,
                    published_at TEXT, crawled_at TEXT, content_hash TEXT, source_type TEXT, raw_path TEXT
                )"""
            )
            connection.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "scholarship",
                    "https://example.edu/scholarship",
                    "奖学金申请通知",
                    "本学年奖学金申请时间为九月一日至九月十五日，申请人须提交材料。",
                    "学生工作处",
                    "2026-08-20",
                    "2026-08-20T00:00:00+00:00",
                    "one",
                    "html",
                    "raw.html",
                ),
            )
            connection.commit()
            connection.close()
            index = root / "rag.db"
            self.assertEqual(build_index(crawler, index, minimum_text_length=1), (1, 1))
            store = RetrievalStore(index)
            try:
                results = store.search("奖学金申请")
            finally:
                store.close()
            self.assertEqual(results[0]["doc_id"], "scholarship")


if __name__ == "__main__":
    unittest.main()
