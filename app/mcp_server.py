"""A dependency-free stdio MCP server for read-only local RAG retrieval."""

from __future__ import annotations

import json
import sqlite3
import sys
from typing import Any

from retrieval.store import RetrievalStore


TOOLS = [
    {
        "name": "search_school_docs",
        "description": "搜索已入库的南邮公开资料，返回相关原文片段及可核验来源。回答校园通知、制度和办事问题前应先调用它。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "要检索的问题或关键词"},
                "category": {"type": "string", "description": "可选，资料分类的精确名称"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_school_document",
        "description": "按 document id 读取已检索到的完整原文和来源元数据，用于补足上下文。",
        "inputSchema": {
            "type": "object",
            "properties": {"doc_id": {"type": "string", "description": "search_school_docs 返回的 doc_id"}},
            "required": ["doc_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "school_knowledge_status",
        "description": "查看本地知识库索引的构建时间、文档数和文本块数。",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]


def response(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def tool_result(payload: Any, is_error: bool = False) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}], "isError": is_error}


def handle_tool_call(store: RetrievalStore, params: dict[str, Any]) -> dict[str, Any]:
    name = params.get("name")
    arguments = params.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError("tool arguments must be an object")
    if name == "search_school_docs":
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query is required")
        category = arguments.get("category")
        if category is not None and not isinstance(category, str):
            raise ValueError("category must be a string")
        top_k = arguments.get("top_k", 5)
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValueError("top_k must be an integer")
        return tool_result({"query": query, "results": store.search(query, category, top_k)})
    if name == "get_school_document":
        doc_id = arguments.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError("doc_id is required")
        document = store.get_document(doc_id)
        return tool_result({"document": document} if document else {"document": None, "message": "document not found"})
    if name == "school_knowledge_status":
        return tool_result(store.status())
    raise ValueError(f"unknown tool: {name}")


def handle(store: RetrievalStore, request: dict[str, Any]) -> dict[str, Any] | None:
    method = request.get("method")
    request_id = request.get("id")
    if method == "notifications/initialized":
        return None
    if method == "initialize":
        requested = request.get("params", {}).get("protocolVersion", "2024-11-05")
        protocol_version = requested if requested in {"2024-11-05", "2025-03-26", "2025-06-18"} else "2024-11-05"
        return response(
            request_id,
            {
                "protocolVersion": protocol_version,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "njupt-rag", "version": "0.1.0"},
            },
        )
    if method == "tools/list":
        return response(request_id, {"tools": TOOLS})
    if method == "tools/call":
        try:
            return response(request_id, handle_tool_call(store, request.get("params", {})))
        except (ValueError, sqlite3.Error) as exc:
            return response(request_id, tool_result({"message": str(exc)}, is_error=True))
    return error(request_id, -32601, f"method not found: {method}")


def main() -> int:
    try:
        store = RetrievalStore()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                reply = handle(store, request)
                if reply is not None:
                    print(json.dumps(reply, ensure_ascii=False), flush=True)
            except json.JSONDecodeError as exc:
                print(json.dumps(error(None, -32700, f"parse error: {exc.msg}")), flush=True)
    finally:
        store.close()
    return 0
