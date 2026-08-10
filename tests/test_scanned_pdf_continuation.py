import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from extraction.scanned_pdf import page_continuation


class ScannedPdfContinuationTests(unittest.TestCase):
    def test_repairs_a_heading_split_across_page_break(self):
        previous = "如出现下列情形之一者，则取消其研究生学业奖学金评"
        self.assertIn("奖学金评", page_continuation(previous, "定资格：\n1、退学研究生"))

    def test_does_not_add_context_to_a_complete_heading(self):
        self.assertEqual(page_continuation("上页结束。", "三、奖励比例和标准"), "")


if __name__ == "__main__":
    unittest.main()
