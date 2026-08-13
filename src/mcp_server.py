"""Expose the local PDF knowledge base through the Model Context Protocol."""

import asyncio
import os

from mcp.server.mcpserver import MCPServer

from retrieval import retrieve, status
from retrieval.runtime import warmup


server = MCPServer(
    name="njupt-rag",
    title="南邮文档知识库",
    description="通过 BM25 与中文语义向量融合检索本项目中已解析 PDF 的可追溯文本片段。",
)


@server.tool(description="返回本地知识库的索引状态、混合检索模式、Embedding 模型与文档、chunk 数量。")
def knowledge_base_status() -> dict:
    return status()


@server.tool(description="检索最相关的 PDF 证据，返回证据充分性判断、置信度、chunk ID、来源文件和页码。证据不足时不要据此生成答案。")
def search_knowledge_base(query: str, limit: int = 5) -> dict:
    return retrieve(query, limit)


def main() -> None:
    if os.environ.get("RAG_PREWARM", "1") != "0":
        warmup()
    asyncio.run(server.run_stdio_async())


if __name__ == "__main__":
    main()
