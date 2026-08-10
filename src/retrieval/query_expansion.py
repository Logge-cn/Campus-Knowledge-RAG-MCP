"""Normalize common Chinese policy-document queries for lexical retrieval."""

from __future__ import annotations


def expand_query(query: str) -> str:
    """Append terms that bridge frequent paraphrases without changing the user's query."""
    additions = []
    if "离校" in query and "手续" in query:
        additions.append("办理手续离校")
    if "作弊" in query and "成绩" in query:
        additions.append("课程考核成绩 无效")
    if "处分" in query and "异议" in query:
        additions.append("学校处分异议申诉 提出申诉")
    if "奖学金" in query and ("申请" in query or "评选" in query):
        additions.append("评定资格 不具备")
    if "网络" in query and "禁止" in query:
        additions.append("不得 规定")
    return " ".join([query, *additions])
