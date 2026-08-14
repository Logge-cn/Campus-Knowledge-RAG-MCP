"""Expose the local PDF knowledge base through the Model Context Protocol."""

import asyncio
import os

from mcp.server.mcpserver import MCPServer

from retrieval import retrieve, status
from retrieval.runtime import warmup


SEARCH_TOOL_DESCRIPTION_VERSION = "answer-eval-v1"
SEARCH_TOOL_DESCRIPTION = (
    "检索南京邮电大学本地 PDF 知识库。"
    "当用户询问学校规章制度、奖助学金、学籍、培养方案等校园文档事实时，应先调用本工具。"
    "仅当 evidence_sufficient=true 时根据 results 回答，并引用 source_file、page 和 chunk_id；"
    "如果 evidence_sufficient=false，应明确说明知识库证据不足，不得使用模型记忆补充或猜测。"
)


server = MCPServer(
    name="njupt-rag",
    title="南邮文档知识库",
    description="通过 BM25 与中文语义向量融合检索本项目中已解析 PDF 的可追溯文本片段。",
)


@server.tool(description="返回本地知识库的索引状态、混合检索模式、Embedding 模型与文档、chunk 数量。")
def knowledge_base_status() -> dict:
    return status()


@server.tool(description=SEARCH_TOOL_DESCRIPTION)
def search_knowledge_base(query: str, limit: int = 5) -> dict:
    return retrieve(query, limit)


def main() -> None:
    if os.environ.get("RAG_PREWARM", "1") != "0":
        warmup()
    asyncio.run(server.run_stdio_async())


if __name__ == "__main__":
    main()
