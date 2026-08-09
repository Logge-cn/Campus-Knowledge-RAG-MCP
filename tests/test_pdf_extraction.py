import json
import sys
import unittest
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from extraction.native_pdf import calculate_metrics, decision, detect_tables, markdown_table


DATA = Path(__file__).parents[1] / "data"
ARTIFACTS = Path(__file__).parents[1] / "storage" / "artifacts"


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
