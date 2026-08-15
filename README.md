# 校园文档 RAG-MCP

校园文档 RAG-MCP 是一个面向本地 PDF 的证据检索服务。它支持原生 PDF 文字与表格提取、扫描件 OCR、BM25 与中文向量混合召回、本地 Cross-Encoder 重排，并通过 CLI 或 MCP 返回带文件名、页码和 chunk ID 的证据。

项目只负责检索与证据充分性判断；最终自然语言答案由 Codex 或其他 MCP 客户端生成。检索 Recall 不等于最终答案正确率。

## 当前状态

- 当前索引 schema 为 4，包含 2 份 PDF、393 页和 731 个 chunks。
- v2 历史验收集的 Recall@1 / Recall@5 / MRR@5 为 90.00% / 100.00% / 94.37%。
- 2026-08-15 的干净环境复现已完成：160 DPI 重建、MCP、冻结周期 96 项测试、两组证据门禁和性能基准均通过。
- 当前答案级评测基础设施已经完成；真实 12 题答案生成和人工复核尚未执行。
- 历次优化、动机与效果见 [项目路线总结](docs/项目路线总结.md)，待办与验收标准见 [未来优化方案](docs/未来优化方案.md)。
- 离线网页演示位于 [data/demo/index.html](data/demo/index.html)。

## 安装

需要 Python 3.12+ 和 [uv](https://docs.astral.sh/uv/)。

```powershell
$env:UV_CACHE_DIR="$PWD\.cache\uv"
uv sync --frozen --all-groups
```

准备运行所需的四个本地模型：

```powershell
uv run --frozen python src/prepare_model.py
uv run --frozen python src/prepare_model.py --reranker
uv run --frozen python src/prepare_model.py --ocr
```

默认运行资产目录为 `runtime/`：

| 用途 | 模型 | 目录 |
| --- | --- | --- |
| Embedding | `BAAI/bge-base-zh-v1.5` | `runtime/models/bge-base-zh-v1.5/` |
| reranker | `BAAI/bge-reranker-base` | `runtime/models/bge-reranker-base/` |
| OCR 检测 | `PP-OCRv6_medium_det` | `runtime/models/paddlex/official_models/PP-OCRv6_medium_det/` |
| OCR 识别 | `PP-OCRv6_medium_rec` | `runtime/models/paddlex/official_models/PP-OCRv6_medium_rec/` |

需要隔离复现资产时，可以在运行命令前设置一个明确的非根目录：

```powershell
$env:RAG_ASSET_ROOT="$PWD\runtime-reproduction"
```

该目录内部仍使用 `models/` 和 `storage/` 两个子目录。

## 导入 PDF

仓库跟踪两份正式输入：

```text
data/sources/南京邮电大学学生手册（2023版）+.pdf
data/sources/奖学金细则.pdf
```

先检查类型，再正式构建解析产物和索引：

```powershell
uv run --frozen python src/ingest.py `
  "data/sources/南京邮电大学学生手册（2023版）+.pdf" `
  "data/sources/奖学金细则.pdf" `
  --data-root "data/sources" `
  --dry-run

uv run --frozen python src/ingest.py `
  "data/sources/南京邮电大学学生手册（2023版）+.pdf" `
  "data/sources/奖学金细则.pdf" `
  --data-root "data/sources" `
  --render-dpi 160
```

导入会在临时目录完成提取、质量检查和新索引构建，全部成功后才切换正式产物。失败时旧索引保持可用。日常高质量 OCR 默认使用 300 DPI；固定复现使用 160 DPI 控制 CPU 成本。

## 检索与 MCP

```powershell
uv run --frozen python src/cli.py search "本科学生国家奖学金奖励标准是多少"
uv run --frozen python src/cli.py retrieve "学校附近明天会不会下雨"
uv run --frozen python src/cli.py status
uv run --frozen python src/mcp_server.py
```

`retrieve` 和 MCP 返回：

- `evidence_sufficient`、`confidence` 和 `reason`；
- 可审计的证据判断 signals；
- 缓存命中和分阶段耗时；
- 每条证据的文件名、页码、chunk ID 和正文。

当 `evidence_sufficient=false` 时，客户端不应依据返回的调试候选生成事实答案。

项目级 MCP 配置位于 `.codex/config.toml`。验证真实 STDIO 工具发现和状态读取：

```powershell
uv run --frozen python evaluation/release/verify_mcp.py
```

## 测试与评测

运行完整测试：

```powershell
uv run --frozen python -m pytest -q -p no:cacheprovider
```

证据充分性回归：

```powershell
uv run --frozen python evaluation/evidence/evaluate.py `
  --dataset evaluation/evidence/development.json `
  --report runtime/reports/evidence-development.json `
  --threshold 0.60 --enforce
```

准备答案级评测：

```powershell
uv run --frozen python evaluation/answer/run.py prepare `
  --output-dir runtime/reports/answer/answer-eval-v1
```

每题结果由隔离子 agent 写入后，依次校验并汇总：

```powershell
uv run --frozen python evaluation/answer/run.py validate-case `
  --output-dir runtime/reports/answer/answer-eval-v1 `
  --case-id <case-id>

uv run --frozen python evaluation/answer/run.py finalize `
  --output-dir runtime/reports/answer/answer-eval-v1
```

人工复核后重新生成答案报告：

```powershell
uv run --frozen python evaluation/answer/evaluate.py `
  --dataset evaluation/answer/dataset.json `
  --predictions runtime/reports/answer/answer-eval-v1/predictions.json `
  --report runtime/reports/answer/answer-eval-v1/report-reviewed.json
```

运行时基准：

```powershell
uv run --frozen python evaluation/retrieval/benchmark.py `
  --query "本科学生国家奖学金奖励标准是多少" `
  --query "学校附近明天会不会下雨" `
  --output runtime/reports/runtime.json
```

## 发布与复现

生成并校验当前发布清单：

```powershell
uv run --frozen python evaluation/release/manifest.py create `
  --output evaluation/release/manifest.json `
  --tracked

uv run --frozen python evaluation/release/manifest.py verify `
  evaluation/release/manifest.json
```

只读检查模型、PDF、发布清单和复现计划：

```powershell
uv run --frozen python evaluation/release/reproduce.py
```

完整复现会重建所指定资产目录中的 `storage/`：

```powershell
uv run --frozen python evaluation/release/reproduce.py `
  --asset-root "$PWD\runtime-reproduction" `
  --execute `
  --report runtime/reports/reproduction.json
```

首次执行前必须先在该资产目录准备四个固定模型。两份 PDF 始终从 Git 工作区的 `data/sources/` 读取。

## 目录结构

```text
rag/
├── data/
│   ├── sources/              # 两份受版本控制的原始 PDF
│   └── demo/                 # 可离线打开的 HTML 演示
├── docs/                     # 两份 PDF 规范、路线总结、未来方案
├── evaluation/
│   ├── answer/               # 答案级评测
│   ├── evidence/             # 证据充分性评测
│   ├── retrieval/            # 检索指标与性能基准
│   └── release/              # 发布清单和复现工具
├── runtime/                  # 本地模型、索引和报告，不提交 Git
│   ├── models/
│   ├── storage/
│   └── reports/
├── src/
│   ├── extraction/           # 原生/扫描 PDF 提取
│   └── retrieval/            # 切块、召回、重排和证据判断
├── tests/                    # pytest 测试
├── pyproject.toml
└── uv.lock
```

PDF 处理边界分别见 [原生内容规范](docs/PDF原生内容处理规范.md) 和 [扫描件规范](docs/PDF扫描件处理规范.md)。
