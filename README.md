# 校园文档RAG-MCP

校园文档RAG-MCP 是一个面向校内 PDF 文档的本地两阶段检索项目。系统提取原生或扫描 PDF，构建 BM25 与中文 Embedding 索引；先通过加权 RRF 召回候选，再以本地 BGE cross-encoder 精排，并通过命令行或 MCP 返回带文件名和页码的证据。

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

本轮已在 8 份新增 PDF 上补充 120 条可回答问题和 20 条无答案问题。开发集新增 40 条可回答问题；锁定集包含 80 条可回答问题和 20 条无答案问题。每条可回答问题都标注了金标答案、必需事实和精确 chunk qrels，开发/锁定文档无内容哈希重叠。下面的校验同时检查题型配额、文档隔离、qrels、标签及冻结文件哈希：

```powershell
uv run python evaluation/validate_dataset.py --expansion-only --require-chunks
```

开发语料和锁定语料使用独立 artifacts 与索引。拒答阈值只在开发集校准；锁定集必须使用 `evaluation/config_freeze.json` 固定的检索配置和开发集阈值，且结果不能反向用于本轮调参：

```powershell
uv run python evaluation/evaluate.py `
  --dataset evaluation/dataset.json `
  --dataset evaluation/dataset_holdout.json `
  --dataset evaluation/dataset_expanded_development_new.json `
  --index-path storage/evaluation_splits/development/index/metadata.json `
  --report evaluation/reports/expanded-development.json `
  --calibrate-refusal --no-enforce

$threshold = 0.9738160385756657
uv run python evaluation/evaluate.py `
  --dataset evaluation/dataset_expanded_locked_answerable.json `
  --dataset evaluation/dataset_expanded_locked_no_answer.json `
  --index-path storage/evaluation_splits/locked_test/index/metadata.json `
  --report evaluation/reports/expanded-locked.json `
  --refusal-threshold $threshold --no-enforce
```

锁定集结果中，reranker 将 Recall@1 从 0.7625 提升到 0.85、MRR@5 从 0.8525 提升到 0.9021，但 Recall@5 从 0.975 降到 0.9625；表格题候选召回只有 0.90。开发集阈值在锁定集上的误答率为 5%，但误拒答率达到 35%，因此暂不启用单一分数硬拒答。完整配置、分类指标、未通过项和后续建议见 `evaluation/optimization_results.json`。

扫描提取器默认仍使用 300 DPI。`--render-dpi` 只用于受控实验；本轮由 140-160 DPI 官方原件生成的扫描测试副本经同页对照后使用 160 DPI：

```powershell
uv run python src/extraction/scanned_pdf.py "data/扫描文档.pdf" --render-dpi 160
```

答案级评测读取系统预测、实际引用的 chunk，以及人工复核的正确性、完整性和引用支持字段；`--dataset` 可以重复传入。当前 CLI/MCP 只返回检索证据，没有答案生成组件，因此仓库不声称已有最终答案正确率：

```powershell
uv run python evaluation/evaluate_answers.py `
  --dataset evaluation/dataset_expanded_locked_answerable.json `
  --dataset evaluation/dataset_expanded_locked_no_answer.json `
  --predictions evaluation/predictions.json `
  --report evaluation/reports/answers.json
```

预测文件格式参考 `evaluation/answer_prediction_template.json`。

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
├── pyproject.toml
└── uv.lock
```

`models/`、`storage/` 和 `evaluation/reports/` 是本地生成内容，不提交 Git。`data/` 是受控知识源，也默认不提交。
