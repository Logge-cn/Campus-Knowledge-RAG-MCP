"""Expose the local PDF knowledge base through the Model Context Protocol."""

import asyncio

from mcp.server.mcpserver import MCPServer

from retrieval import search, status


server = MCPServer(
    name="njupt-rag",
    title="南邮文档知识库",
    description="通过 BM25 与中文语义向量融合检索本项目中已解析 PDF 的可追溯文本片段。",
)


@server.tool(description="返回本地知识库的索引状态、混合检索模式、Embedding 模型与文档、chunk 数量。")
def knowledge_base_status() -> dict:
    return status()


@server.tool(description="使用 BM25 与中文语义向量混合检索最相关的 PDF 文本片段，并返回融合排名、来源文件、页码和原始提取文件路径。")
def search_knowledge_base(query: str, limit: int = 5) -> dict:
    return {"query": query, "results": search(query, limit)}


def main() -> None:
    asyncio.run(server.run_stdio_async())


if __name__ == "__main__":
    main()
