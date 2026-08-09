# 南京邮电大学文档 RAG

一个面向校内 PDF 文档的本地混合检索项目。系统提取原生或扫描 PDF，构建 BM25 与中文 Embedding 索引，通过 RRF 融合排序，并以命令行或 MCP 返回带文件名和页码的证据。

## Installation

需要 Python 3.12 或更高版本以及 [uv](https://docs.astral.sh/uv/)。

```powershell
$env:UV_CACHE_DIR="$PWD\.cache\uv"
uv sync
```

## Run

首次使用时，将固定的中文 Embedding 模型下载到项目内的 `models/`：

```powershell
uv run python src/prepare_model.py
```

将 PDF 放入 `data/`，根据文件类型提取内容：

```powershell
uv run python src/extraction/native_pdf.py "data/文档.pdf"
uv run python src/extraction/scanned_pdf.py "data/扫描文档.pdf"
```

构建索引、检索和查看状态：

```powershell
uv run python src/cli.py build
uv run python src/cli.py search "查询内容"
uv run python src/cli.py status
```

启动 MCP stdio 服务：

```powershell
uv run python src/mcp_server.py
```

运行自动化测试和检索评测：

```powershell
uv run python -m unittest discover -s tests -v
uv run python evaluation/evaluate.py
```

## Structure

```text
rag/
├── data/               # 原始 PDF 知识源
├── models/             # 本地 Embedding 模型；一个模型一个子目录
├── storage/
│   ├── artifacts/      # PDF 页面、表格、OCR 与元数据
│   └── index/          # Chunk、BM25、Embedding 与索引元数据
├── src/
│   ├── extraction/     # 原生 PDF 和扫描 PDF 提取
│   ├── retrieval/      # 切块、BM25、Embedding、索引与 RRF
│   ├── cli.py          # build、search、status 命令
│   ├── mcp_server.py   # MCP 服务入口
│   └── prepare_model.py
├── tests/              # 自动化正确性测试
├── evaluation/         # 固定评测集、评测程序与生成报告
├── docs/               # 设计与处理规范
├── slides/             # 浏览器演示
├── pyproject.toml
└── uv.lock
```

`models/`、`storage/` 和 `evaluation/reports/` 是本地生成内容，不提交 Git。`data/` 是受控知识源，也默认不提交。
