"""Assess whether retrieved chunks are strong enough for an answer."""

from __future__ import annotations

import re
from typing import Any


MIN_TOP_RERANKER_SCORE = 0.10
MIN_EVIDENCE_CONFIDENCE = 0.60
MIN_LEXICAL_COVERAGE = 0.25
_TIME_SENSITIVE_PATTERN = re.compile(
    r"今天|明天|后天|现在|此刻|今晚|本周|实时|最新|天气|下雨|菜单|新闻|当前时间|现任|今年|本学期|时刻表|空座位|空闲座位"
)
_UNIVERSITY_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,10}大学(?!生)")
_LOCAL_UNIVERSITIES = {"南京邮电大学"}
_QUERY_CUE_GROUPS = (
    ("预约", re.compile(r"预约"), re.compile(r"预约")),
    ("遗失物品", re.compile(r"遗失|失物"), re.compile(r"遗失|失物")),
    ("退款/退费", re.compile(r"退还|退款|退费"), re.compile(r"退还|退款|退费")),
    (
        "收费/计费",
        re.compile(r"收费|计费|多少钱"),
        re.compile(r"收费|计费|费用|价格|\d+(?:\.\d+)?\s*元"),
    ),
)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _character_bigrams(text: str) -> set[str]:
    normalized = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", text.lower())
    return {normalized[index : index + 2] for index in range(max(0, len(normalized) - 1))}


def _lexical_coverage(query: str, results: list[dict[str, Any]]) -> float:
    query_bigrams = _character_bigrams(query)
    if not query_bigrams:
        return 0.0
    evidence_bigrams = _character_bigrams(" ".join(str(item.get("text", "")) for item in results[:3]))
    return len(query_bigrams & evidence_bigrams) / len(query_bigrams)


def _wrong_school_entity(query: str) -> str | None:
    for name in _UNIVERSITY_PATTERN.findall(query):
        if name not in _LOCAL_UNIVERSITIES:
            return name
    return None


def _missing_query_cues(query: str, results: list[dict[str, Any]]) -> list[str]:
    evidence = " ".join(str(item.get("text", "")) for item in results[:3])
    return [label for label, query_pattern, evidence_pattern in _QUERY_CUE_GROUPS if query_pattern.search(query) and not evidence_pattern.search(evidence)]


def assess_evidence(query: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a conservative, inspectable evidence-sufficiency decision.

    This is a retrieval diagnostic, not a guarantee that a generated answer is
    correct. The raw cross-encoder score remains a separate signal because the
    final ranking score also contains retrieval-rank priors.
    """
    time_sensitive_query = bool(_TIME_SENSITIVE_PATTERN.search(query))
    wrong_school_entity = _wrong_school_entity(query)
    if not results:
        return {
            "evidence_sufficient": False,
            "confidence": 0.0,
            "reason": "no_retrieval_results",
            "signals": {
                "top_reranker_score": None,
                "reranker_score_margin": None,
                "lexical_coverage": 0.0,
                "top_channel_agreement": False,
                "top_source_consistency": 0.0,
                "low_quality_ratio": 0.0,
                "time_sensitive_query": time_sensitive_query,
                "current_evidence": False,
                "wrong_school_entity": wrong_school_entity,
                "missing_query_cues": [],
            },
        }

    top_score = float(results[0].get("reranker_score", 0.0))
    second_score = float(results[1].get("reranker_score", 0.0)) if len(results) > 1 else 0.0
    score_margin = max(0.0, top_score - second_score)
    lexical_coverage = _lexical_coverage(query, results)
    top_channel_agreement = {"bm25", "vector"}.issubset(set(results[0].get("matched_by", [])))
    source_counts: dict[str, int] = {}
    for item in results[:3]:
        source = str(item.get("source_file", ""))
        source_counts[source] = source_counts.get(source, 0) + 1
    top_source_consistency = max(source_counts.values()) / min(3, len(results))
    low_quality_ratio = sum(bool(item.get("low_confidence")) for item in results[:3]) / min(3, len(results))
    current_evidence = any(item.get("effective_date") and item.get("active", True) for item in results[:3])
    missing_query_cues = _missing_query_cues(query, results)

    confidence = (
        0.65 * _clamp(top_score)
        + 0.10 * _clamp(score_margin * 2)
        + 0.10 * lexical_coverage
        + 0.10 * float(top_channel_agreement)
        + 0.05 * top_source_consistency
        - 0.15 * low_quality_ratio
        - 0.15 * float(time_sensitive_query and top_score < 0.8)
    )
    confidence = round(_clamp(confidence), 4)
    hard_rules_passed = (
        lexical_coverage >= MIN_LEXICAL_COVERAGE
        and not wrong_school_entity
        and not missing_query_cues
        and (not time_sensitive_query or current_evidence)
    )
    sufficient = top_score >= MIN_TOP_RERANKER_SCORE and confidence >= MIN_EVIDENCE_CONFIDENCE and hard_rules_passed

    if wrong_school_entity:
        reason = "query_targets_different_institution"
    elif missing_query_cues:
        reason = "missing_required_evidence_cues"
    elif time_sensitive_query and not current_evidence:
        reason = "current_evidence_required"
    elif lexical_coverage < MIN_LEXICAL_COVERAGE:
        reason = "insufficient_query_evidence_overlap"
    else:
        reason = "supported_by_retrieved_evidence" if sufficient else "insufficient_retrieval_evidence"

    return {
        "evidence_sufficient": sufficient,
        "confidence": confidence,
        "reason": reason,
        "signals": {
            "top_reranker_score": round(top_score, 6),
            "reranker_score_margin": round(score_margin, 6),
            "lexical_coverage": round(lexical_coverage, 4),
            "top_channel_agreement": top_channel_agreement,
            "top_source_consistency": round(top_source_consistency, 4),
            "low_quality_ratio": round(low_quality_ratio, 4),
            "time_sensitive_query": time_sensitive_query,
            "current_evidence": current_evidence,
            "wrong_school_entity": wrong_school_entity,
            "missing_query_cues": missing_query_cues,
        },
    }
