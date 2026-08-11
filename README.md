# 南京邮电大学文档 RAG

一个面向校内 PDF 文档的本地两阶段检索项目。系统提取原生或扫描 PDF，构建 BM25 与中文 Embedding 索引；先通过加权 RRF 召回候选，再以本地 BGE cross-encoder 精排，并以命令行或 MCP 返回带文件名和页码的证据。

## Installation

需要 Python 3.12 或更高版本以及 [uv](https://docs.astral.sh/uv/)。

```powershell
$env:UV_CACHE_DIR="$PWD\.cache\uv"
uv sync
```

## Run

首次使用时，将固定的中文 Embedding 模型和 reranker 下载到项目内的 `models/`：

```powershell
uv run python src/prepare_model.py
uv run python src/prepare_model.py --reranker
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

默认检索路径为“BM25 + Embedding → 加权 RRF → BGE reranker”。RRF 为两路各取前 20 个候选生成统一排名，reranker 对候选以 `BAAI/bge-reranker-base` 做本地 cross-encoder 打分后返回前 K 条。检索结果保留 `rrf_score`、`reranker_score` 与 `retrieval_rank`，状态接口的 `retrieval_mode` 为 `hybrid_rrf_cross_encoder_rerank`。

当前固定评测集包含 100 条问题（其中 90 条可回答）。reranker 的 Recall@5 为 98.89%，与 RRF 基线持平；MRR 从 71.33% 提升到 75.63%。在 CPU 环境下，平均查询耗时为 2656.15 ms、P95 为 3566.35 ms；这是逐对 cross-encoder 精排的代价。

## Structure

```text
rag/
├── data/               # 原始 PDF 知识源
├── models/             # 本地 Embedding 与 reranker 模型；一个模型一个子目录
├── storage/
│   ├── artifacts/      # PDF 页面、表格、OCR 与元数据
│   └── index/          # Chunk、BM25、Embedding 与索引元数据
├── src/
│   ├── extraction/     # 原生 PDF 和扫描 PDF 提取
│   ├── retrieval/      # 切块、BM25、Embedding、索引、RRF 与 reranker
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
