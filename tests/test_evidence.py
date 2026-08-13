import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval.evidence import assess_evidence


def result(
    text: str,
    *,
    reranker_score: float,
    matched_by: list[str] | None = None,
    source_file: str = "学生手册.pdf",
) -> dict:
    return {
        "text": text,
        "reranker_score": reranker_score,
        "matched_by": matched_by or ["bm25", "vector"],
        "source_file": source_file,
        "page": 1,
        "low_confidence": False,
    }


class EvidenceAssessmentTests(unittest.TestCase):
    def test_accepts_high_scoring_supported_evidence(self):
        assessment = assess_evidence(
            "本科学生国家奖学金奖励标准是多少",
            [
                result("本科学生国家奖学金的奖励标准为每生每年8000元。", reranker_score=0.98),
                result("本科学生国家奖学金评审办法。", reranker_score=0.72),
            ],
        )

        self.assertTrue(assessment["evidence_sufficient"])
        self.assertGreaterEqual(assessment["confidence"], 0.8)
        self.assertEqual(assessment["reason"], "supported_by_retrieved_evidence")

    def test_rejects_unrelated_low_scoring_evidence(self):
        assessment = assess_evidence(
            "学校附近明天会不会下雨",
            [
                result("学生应遵守考场纪律和学校安全管理规定。", reranker_score=0.05),
                result("宿舍内禁止使用违章电器。", reranker_score=0.02),
            ],
        )

        self.assertFalse(assessment["evidence_sufficient"])
        self.assertEqual(assessment["reason"], "current_evidence_required")
        self.assertTrue(assessment["signals"]["time_sensitive_query"])

    def test_time_sensitive_query_requires_current_evidence(self):
        assessment = assess_evidence(
            "仙林校区到三牌楼校区校车时刻表",
            [result("仙林校区和三牌楼校区机动车限速规定。", reranker_score=0.99)],
        )
        self.assertFalse(assessment["evidence_sufficient"])
        self.assertFalse(assessment["signals"]["current_evidence"])

    def test_empty_results_are_not_sufficient(self):
        assessment = assess_evidence("任意问题", [])

        self.assertFalse(assessment["evidence_sufficient"])
        self.assertEqual(assessment["reason"], "no_retrieval_results")

    def test_rejects_query_for_another_university(self):
        assessment = assess_evidence(
            "南京大学仙林校区图书馆本科生能借几本书",
            [result("南京邮电大学图书馆本科生借阅规则。", reranker_score=0.99)],
        )

        self.assertFalse(assessment["evidence_sufficient"])
        self.assertEqual(assessment["reason"], "query_targets_different_institution")
        self.assertEqual(assessment["signals"]["wrong_school_entity"], "南京大学")

    def test_rejects_when_required_action_cue_is_missing(self):
        assessment = assess_evidence(
            "羽毛球场如何预约和收费",
            [result("体育馆开放时间与安全管理规定。", reranker_score=0.99)],
        )

        self.assertFalse(assessment["evidence_sufficient"])
        self.assertEqual(assessment["reason"], "missing_required_evidence_cues")
        self.assertEqual(assessment["signals"]["missing_query_cues"], ["预约", "收费/计费"])

    def test_rejects_generic_match_without_query_target_overlap(self):
        assessment = assess_evidence(
            "校园班车遗失物品在哪里领取",
            [result("学生证遗失后可以申请补办并领取。", reranker_score=0.99)],
        )

        self.assertFalse(assessment["evidence_sufficient"])
        self.assertEqual(assessment["reason"], "insufficient_query_evidence_overlap")


if __name__ == "__main__":
    unittest.main()
