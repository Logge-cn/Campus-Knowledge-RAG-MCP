import asyncio
import json
import os
import sys
import unittest
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval import build_index


class RAGMCPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        build_index()

    def test_mcp_lists_and_calls_knowledge_base_tools(self):
        async def exercise_server():
            env = os.environ.copy()
            # The uv --with pytest overlay uses a temporary Python executable;
            # test warmup order separately and exercise the MCP process lazily here.
            env["RAG_PREWARM"] = "0"
            parameters = StdioServerParameters(
                command=sys.executable,
                args=["src/mcp_server.py"],
                cwd=PROJECT_ROOT,
                env=env,
            )
            async with stdio_client(parameters) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    self.assertEqual({tool.name for tool in tools.tools}, {"knowledge_base_status", "search_knowledge_base"})
                    search_tool = next(tool for tool in tools.tools if tool.name == "search_knowledge_base")
                    self.assertIn("evidence_sufficient=true", search_tool.description)
                    self.assertIn("source_file、page 和 chunk_id", search_tool.description)
                    self.assertIn("不得使用模型记忆补充或猜测", search_tool.description)
                    status_response = await session.call_tool("knowledge_base_status", {})
                    status_payload = json.loads(status_response.content[0].text)
                    self.assertGreater(status_payload["chunks"], 400)
                    self.assertEqual(status_payload["retrieval_mode"], "hybrid_rrf_cross_encoder_rerank")
                    self.assertEqual(status_payload["embedding_model"], "BAAI/bge-base-zh-v1.5")
                    response = await session.call_tool("search_knowledge_base", {"query": "奖学金", "limit": 2})
                    payload = json.loads(response.content[0].text)
                    self.assertEqual(payload["query"], "奖学金")
                    self.assertIn("evidence_sufficient", payload)
                    self.assertIn("confidence", payload)
                    self.assertIn("reason", payload)
                    self.assertTrue(payload["results"])
                    self.assertTrue(all("page" in result for result in payload["results"]))
                    self.assertTrue(
                        all(
                            result["score_type"] == "cross_encoder_rank_rrf_blend"
                            for result in payload["results"]
                        )
                    )
                    self.assertTrue(all("reranker_score" in result for result in payload["results"]))
                    self.assertTrue(all("chunk_id" in result for result in payload["results"]))
                    self.assertTrue(all(result["matched_by"] for result in payload["results"]))

        asyncio.run(exercise_server())


if __name__ == "__main__":
    unittest.main()
