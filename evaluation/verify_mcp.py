"""Verify MCP tool discovery and the knowledge-base status call over STDIO."""

from __future__ import annotations

import asyncio
import json
import os
import tomllib
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TOOLS = {"knowledge_base_status", "search_knowledge_base"}


async def verify() -> dict:
    config = tomllib.loads((PROJECT_ROOT / ".codex" / "config.toml").read_text(encoding="utf-8"))
    server_config = config["mcp_servers"]["njupt-rag"]
    env = os.environ.copy()
    env.update(server_config.get("env", {}))
    parameters = StdioServerParameters(
        command=server_config["command"],
        args=server_config.get("args", []),
        cwd=(PROJECT_ROOT / server_config.get("cwd", ".")).resolve(),
        env=env,
    )
    async with stdio_client(parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {tool.name for tool in tools.tools}
            if names != EXPECTED_TOOLS:
                raise RuntimeError(f"Unexpected MCP tools: {sorted(names)}")
            response = await session.call_tool("knowledge_base_status", {})
            status = json.loads(response.content[0].text)
            if int(status.get("chunks", 0)) <= 0:
                raise RuntimeError("knowledge_base_status reported an empty index")
            return {"tools": sorted(names), "knowledge_base_status": status}


def main() -> None:
    print(json.dumps(asyncio.run(verify()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
