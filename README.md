# 南京邮电大学文档 RAG

一个面向南京邮电大学文档的本地 RAG 项目。它从原生或扫描版 PDF 中提取可追溯的 Markdown 内容，构建本地检索索引，并通过命令行或 MCP 服务提供查询。

## Installation

需要 Python 3.12 或更高版本以及 [uv](https://docs.astral.sh/uv/)。

```powershell
uv sync
```

## Run

将 PDF 放入 `data/` 后，根据文件类型提取内容：

```powershell
uv run python src/extract_native_pdf.py "data/文档.pdf"
uv run python src/extract_scanned_pdf.py "data/扫描文档.pdf"
```

首次使用时，将固定的中文 Embedding 模型下载到本项目的 `rag_index/models/`。模型与下载缓存均保存在本项目目录：

```powershell
uv run python src/prepare_embedding_model.py
```

构建 BM25 与中文 Embedding 混合索引，并执行检索：

```powershell
uv run python src/rag_pipeline.py build
uv run python src/rag_pipeline.py search "查询内容"
uv run python src/rag_pipeline.py status
```

检索时分别取得 BM25 和语义向量候选，再使用 RRF 融合排名；结果保留来源文件、页码和原始提取文件路径。索引文件保存在 `rag_index/`，内容未变化时会复用现有索引；需要强制重建时为 `build` 增加 `--force`。

启动 MCP stdio 服务：

```powershell
uv run python src/rag_mcp_server.py
```

## Structure

```text
rag/
├── data/          # 待处理的 PDF 文档
├── artifacts/     # 提取后的页面、表格和 OCR 结果
├── rag_index/     # BM25、Embedding、元数据和项目内模型
├── src/           # PDF 提取、索引检索和 MCP 服务源码
├── test/          # 自动化测试
├── pyproject.toml # Python 版本与依赖声明
└── uv.lock        # 锁定的依赖版本
```
