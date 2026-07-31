from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import fitz

from ingestion.index import build_index
from retrieval.store import RetrievalStore


class FileIndexTest(unittest.TestCase):
    def test_build_and_sync_pdf_references(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename, text in {
                "scholarship.pdf": "Scholarship application closes on 15 September.",
                "handbook.pdf": "Students must follow the university rules.",
            }.items():
                pdf = fitz.open()
                page = pdf.new_page()
                page.insert_text((72, 72), text)
                pdf.save(root / filename)
                pdf.close()

            index = root / "rag.db"
            self.assertEqual(build_index(root, index, minimum_text_length=1), (2, 2))
            store = RetrievalStore(index)
            try:
                results = store.search("scholarship application")
            finally:
                store.close()
            self.assertEqual(results[0]["title"], "scholarship")
            self.assertEqual(results[0]["page"], 1)

            (root / "handbook.pdf").unlink()
            self.assertEqual(build_index(root, index, minimum_text_length=1), (1, 1))
            store = RetrievalStore(index)
            try:
                self.assertEqual(store.status()["document_count"], 1)
            finally:
                store.close()
