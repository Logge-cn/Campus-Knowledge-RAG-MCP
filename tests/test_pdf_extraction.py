import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from extraction.native_pdf import (
    TableArtifact,
    bbox_overlap,
    calculate_metrics,
    decision,
    detect_tables,
    markdown_table,
    plausible_stream_table,
)


PROJECT_ROOT = Path(__file__).parents[1]
ASSET_ROOT = Path(os.environ.get("RAG_ASSET_ROOT", PROJECT_ROOT)).resolve()
DATA = PROJECT_ROOT / "data"
ARTIFACTS = ASSET_ROOT / "storage" / "artifacts"


def pdf_by_size(size: int) -> Path:
    return next(path for path in DATA.glob("*.pdf") if path.stat().st_size == size)


class NativePdfExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manual = fitz.open(pdf_by_size(2444739))
        cls.scholarship = fitz.open(pdf_by_size(4286200))
        cls.manual_path = Path(cls.manual.name)

    @classmethod
    def tearDownClass(cls):
        cls.manual.close()
        cls.scholarship.close()

    def test_native_and_scanned_pages_are_routed_differently(self):
        self.assertEqual(decision(calculate_metrics(self.manual[0])), "native")
        self.assertEqual(decision(calculate_metrics(self.manual[134])), "native")
        self.assertEqual(decision(calculate_metrics(self.scholarship[0])), "ocr")

    def test_pymupdf_preserves_the_multi_level_header(self):
        tables = detect_tables(self.manual_path, self.manual[57], 58)
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].method, "pymupdf.find_tables")
        rendered = markdown_table(tables[0].rows)
        self.assertIn("肺活量（ml） / 大一 大二", rendered)
        self.assertIn("5040", rendered)

    def test_sparse_pymupdf_table_falls_back_to_camelot(self):
        tables = detect_tables(self.manual_path, self.manual[59], 60)
        self.assertEqual([table.method for table in tables], ["camelot.lattice", "camelot.lattice"])
        self.assertTrue(all(len(table.rows[0]) == 3 for table in tables))
        self.assertEqual(tables[0].title, "表5男生引体向上评分表（单位：次）")
        self.assertEqual(tables[1].title, "表6男生1000米跑评分表（单位：分·秒）")

    def test_table_without_a_header_uses_generic_columns(self):
        rendered = markdown_table([["", "", "10 分"], ["", "学习态度", "不遵守课堂纪律"], ["综合评议", "民主评议", ""]])
        self.assertTrue(rendered.startswith("| 列1 | 列2 | 列3 |"))
        self.assertTrue(rendered.endswith("| 综合评议 | 民主评议 |  |"))

    def test_table_candidates_are_matched_by_page_coordinates(self):
        self.assertEqual(bbox_overlap((0, 0, 100, 100), (10, 10, 90, 90)), 1.0)
        self.assertEqual(bbox_overlap((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_stream_fallback_rejects_prose_like_candidates(self):
        good = TableArtifact(
            rows=[["姓名", "成绩"], ["甲", "90"], ["乙", "80"]],
            bbox=(0, 0, 100, 100),
            method="camelot.stream",
            score=0.9,
            title="",
        )
        prose = TableArtifact(
            rows=[["这是一段很长的普通正文，并不是表格中的短单元格。"], ["第二段正文"], ["第三段正文"]],
            bbox=(0, 0, 100, 100),
            method="camelot.stream",
            score=0.9,
            title="",
        )
        self.assertTrue(plausible_stream_table(good))
        self.assertFalse(plausible_stream_table(prose))

    @patch("extraction.native_pdf.page_may_contain_borderless_table", return_value=True)
    @patch("extraction.native_pdf.camelot_candidates")
    def test_rejected_stream_candidate_is_a_diagnostic_not_a_failure(self, candidates, _may_contain):
        candidates.return_value = [
            TableArtifact(
                rows=[["普通正文"], ["不是表格"], ["仍是正文"]],
                bbox=(0, 0, 100, 100),
                method="camelot.stream",
                score=0.9,
                title="",
            )
        ]
        failures = []
        diagnostics = []

        tables = detect_tables(self.manual_path, self.manual[0], 1, failures, diagnostics)

        self.assertEqual(tables, [])
        self.assertEqual(failures, [])
        self.assertEqual(diagnostics, ["camelot.stream: candidates rejected by table quality checks"])

    def test_generated_artifacts_are_complete(self):
        entries = [(path.parent, json.loads(path.read_text(encoding="utf-8"))) for path in ARTIFACTS.glob("*/metadata.json")]
        manual_dir, manual = next(item for item in entries if len(item[1]["pages"]) == 363)
        scanned_dir, scanned = next(item for item in entries if len(item[1]["pages"]) == 30)
        self.assertEqual(len(list((manual_dir / "pages").glob("*.md"))), 363)
        self.assertEqual(len(list((manual_dir / "tables").glob("*.md"))), 38)
        self.assertEqual(len(list((scanned_dir / "pages").glob("*.md"))), 30)
        self.assertEqual(len(list((scanned_dir / "ocr_raw").glob("*.json"))), 30)
        self.assertEqual(len(list((scanned_dir / "rendered").glob("*.png"))), 30)
        self.assertGreaterEqual(min(page["confidence"] for page in scanned["pages"]), 0.85)
        self.assertIn("学业奖学金", (scanned_dir / "pages" / "page-010.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
