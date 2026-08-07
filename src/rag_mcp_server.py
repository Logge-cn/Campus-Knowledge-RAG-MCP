"""Expose the local PDF knowledge base through the Model Context Protocol."""

from __future__ import annotations

import asyncio

from mcp.server.mcpserver import MCPServer

from rag_pipeline import search, status


server = MCPServer(
    name="njupt-rag",
    title="南邮文档知识库",
    description="检索本项目中已解析 PDF 的可追溯文本片段。",
)


@server.tool(description="返回当前本地知识库的索引状态与文档、chunk 数量。")
def knowledge_base_status() -> dict:
    return status()


@server.tool(description="按问题检索最相关的 PDF 文本片段，并返回来源文件、页码和原始提取文件路径。")
def search_knowledge_base(query: str, limit: int = 5) -> dict:
    return {"query": query, "results": search(query, limit)}


def main() -> None:
    asyncio.run(server.run_stdio_async())


if __name__ == "__main__":
    main()
