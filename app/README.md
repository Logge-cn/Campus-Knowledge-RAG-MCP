# MCP RAG service

This package exposes the local school-document index as a read-only stdio MCP
server. It does not call a chat or embedding API.

## Build the local index

```powershell
uv run python -m ingestion
```

The command reads `data/crawler.db`, removes duplicate and very short pages,
splits the remaining text into paragraph-aware chunks, and writes
`data/rag.db`. To rebuild it after a crawl:

```powershell
uv run python -m ingestion --force
```

## Use from Codex CLI

From the repository root, register the stdio server once:

```powershell
codex.cmd mcp add njupt-rag -- uv --directory E:\logge\Projects\rag run python -m app
```

Restart Codex after registration. The available tools are:

- `search_school_docs`: retrieves cited chunks from the local index.
- `get_school_document`: reads a retrieved source document.
- `school_knowledge_status`: reports index coverage and build time.

For a quick terminal-only retrieval check:

```powershell
uv run python -m retrieval "奖学金申请"
```
