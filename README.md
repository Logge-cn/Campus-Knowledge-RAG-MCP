# 校园文档 RAG-MCP

校园文档 RAG-MCP 是一个面向校内 PDF 的本地证据检索服务。它提取原生或扫描 PDF，使用 BM25 与中文 Embedding 混合召回，再由本地 BGE cross-encoder 精排，通过命令行或 MCP 返回带 chunk ID、文件名和页码的证据。

项目定位是 **MCP 检索工具**：仓库负责检索、证据充分性判断和评测接口；最终自然语言答案由 Codex 或其他 MCP 客户端生成。检索 Recall 不等于最终答案正确率。

## 当前进度（2026-08-15）

- 当前位于 `master`，第一优先级“答案级评测”的本地基础设施已完成：固定 12 题、版本化 Prompt/Schema、真实 STDIO MCP 运行器和答案评估器均已实现。
- 真实 12 题答案生成与人工复核尚未执行，因为该步骤会把本地校园 PDF 证据发送给 OpenAI/Codex，仍等待明确的数据外发授权；因此当前没有可声明的答案正确率、引用准确率或拒答遵守率。
- 第二优先级“干净环境复现”已完成：依赖和四个模型输入可固定校验，两份 Git PDF 可从零重建索引，项目级 MCP 配置、工具发现、`knowledge_base_status` 和完整测试均已验收；第三优先级及以后尚未启动。
- 当前运行索引为 schema 4、2 份 PDF、731 个 chunk；完整测试套件实际为 96 项并已通过。
- 当前 `evaluation/release_manifest.json` 已按第二优先级工作区重新生成，覆盖 84 个发布输入并通过校验；这不表示仍待授权的第一优先级真实答案评测已经完成。
- `docs/` 已纳入版本控制；`slides/` 仍被 `.gitignore` 忽略，只作为本地演示保留。

## 安装

需要 Python 3.12 或更高版本以及 [uv](https://docs.astral.sh/uv/)。本轮实际验证环境为 Windows 11、Python 3.12.13、uv 0.11.14；Python 包的精确版本由 `uv.lock` 固定。

```powershell
$env:UV_CACHE_DIR="$PWD\.cache\uv"
uv sync --frozen --all-groups
uv run --frozen python src/prepare_model.py
uv run --frozen python src/prepare_model.py --reranker
uv run --frozen python src/prepare_model.py --ocr
```

模型和生成索引默认放在项目内，均被 `.gitignore` 排除。新 worktree 可以在运行上述模型准备命令前用 `RAG_ASSET_ROOT` 指向独立的本地资产目录；该目录仍须是明确的非根目录：

```powershell
$env:RAG_ASSET_ROOT="$PWD\.local-assets"
```

固定的四个模型输入如下。Embedding 与 reranker 来自 BAAI 的 Hugging Face 仓库，OCR 模型来自 PaddlePaddle；`evaluation/reproduction_plan.json` 记录完整目录 SHA-256，复现检查会同时校验权重、配置和 tokenizer，而不是只检查目录是否存在。

| 用途 | 固定模型 | 目标目录 |
| --- | --- | --- |
| Embedding | [BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5) | `models/bge-base-zh-v1.5/` |
| reranker | [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) | `models/bge-reranker-base/` |
| OCR 检测 | [PaddlePaddle/PP-OCRv6_medium_det](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det) | `models/paddlex/official_models/PP-OCRv6_medium_det/` |
| OCR 识别 | [PaddlePaddle/PP-OCRv6_medium_rec](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec) | `models/paddlex/official_models/PP-OCRv6_medium_rec/` |

## 导入和更新 PDF

仓库的 `data/` 已包含并跟踪两份用于复现的项目 PDF。先执行 dry-run，确认手册被识别为原生 PDF、奖学金细则被识别为扫描 PDF；再从两份输入正式重建提取产物和索引：

```powershell
uv run --frozen python src/ingest.py `
  "data/南京邮电大学学生手册（2023版）+.pdf" `
  "data/奖学金细则.pdf" `
  --dry-run

uv run --frozen python src/ingest.py `
  "data/南京邮电大学学生手册（2023版）+.pdf" `
  "data/奖学金细则.pdf" `
  --render-dpi 160
```

同一制度的多个版本可以共享 `--document-id`。新版本导入后，旧版本保留在清单中但默认不参与检索：

```powershell
uv run python src/ingest.py "data/制度-2026.pdf" `
  --document-id student-policy `
  --version 2026 `
  --effective-date 2026-09-01
```

导入会在临时目录完成全部文档提取、质量检查、版本清单和新索引构建，全部成功后才一起切换；任一步失败都会保留旧产物、旧版本清单和旧索引。真实提取失败或低置信 OCR 默认不能发布；只有经过人工复核并明确接受风险时，才应使用 `--allow-quality-failures`。首次建库失败也不会留下半套索引文件。

## 检索和 MCP

```powershell
uv run python src/cli.py build
uv run python src/cli.py search "本科学生国家奖学金奖励标准是多少"
uv run python src/cli.py retrieve "学校附近明天会不会下雨"
uv run python src/cli.py status
uv run python src/mcp_server.py
```

仓库内的 `.codex/config.toml` 是可移植的最小 Codex 客户端配置；信任项目并重启 Codex/ChatGPT 桌面客户端后即可加载。它使用 `uv` 和相对工作目录，不包含本机 Python 或仓库绝对路径：

```toml
[mcp_servers.njupt-rag]
command = "uv"
args = ["run", "--frozen", "python", "src/mcp_server.py"]
cwd = "."
startup_timeout_sec = 30

[mcp_servers.njupt-rag.env]
PYTHONUTF8 = "1"
PYTHONIOENCODING = "utf-8"
RAG_PREWARM = "0"
```

[Codex 官方 MCP 文档](https://developers.openai.com/codex/mcp/)说明项目级 `.codex/config.toml` 会在受信任项目中加载，并支持 STDIO 服务的 `command`、`args`、`env` 与 `cwd`。不使用 Codex 时，可将同一启动命令换成目标 MCP 客户端对应的 STDIO 配置格式。服务连接后的最小验收命令为：

```powershell
uv run --frozen python evaluation/verify_mcp.py
```

该命令通过真实 STDIO 会话初始化服务、发现 `knowledge_base_status` 和 `search_knowledge_base`，并实际调用 `knowledge_base_status`；不会生成示例对话或调用外部大模型。

`search` 保留兼容的结果列表。`retrieve` 和 MCP 额外返回：

- `evidence_sufficient`：证据是否足以支持作答；
- `confidence` 与 `reason`：可解释的判断结果；
- `assessment.signals`：raw reranker 分数、分差、通道一致性、词面覆盖、来源一致性、OCR 质量及时效性；
- `diagnostics`：缓存命中和本次耗时；
- 每条证据的 `chunk_id`、文件名、页码和正文。

`status` 还会返回文档版本列表和 `freshness.pending_update`，用于判断提取产物是否已领先于当前索引。

如果 `evidence_sufficient` 为 `false`，MCP 客户端不应基于返回候选生成事实答案。时效问题只有在证据带有效日期且处于 active 状态时才允许通过硬规则。

MCP 服务默认启动时预加载索引和两个模型。设置 `RAG_PREWARM=0` 可关闭预热。完全相同的查询使用 128 条进程内缓存，索引元数据变化后缓存键自动失效。

## 测试与评测

运行全部测试：

```powershell
uv run --frozen python -m pytest -q
```

`pytest==9.1.1` 已纳入 dev 依赖和 `uv.lock`；`unittest discover` 不能完整收集当前以 pytest 风格编写的测试。

校准或验证证据充分性时，必须在开发数据校准，再对文档隔离的未见数据使用固定阈值：

```powershell
uv run python evaluation/evaluate_evidence.py `
  --dataset evaluation/dataset.json `
  --report evaluation/reports/evidence-development-fixed.json `
  --threshold 0.60 --enforce `
  --maximum-false-answer-rate 0.05 `
  --maximum-false-refusal-rate 0.35
```

当前开发集固定阈值结果为：90 条可回答问题误拒答 1 条（1.11%），10 条不可回答问题误答 0 条。20 条四类不可回答回归集误答 0 条；该回归集已用于规则开发，不能冒充下一轮独立未见验收集。

固定答案级评测使用 Codex CLI、`gpt-5.6-sol`、`answer-eval-v1` Prompt、`answer-eval-v1` 工具 description 和 Top 5 证据。问题集覆盖普通文本、表格、扫描 PDF、知识库外问题、错误学校和时效性问题。运行真实客户端评测：

```powershell
uv run python evaluation/run_answer_evaluation.py `
  --output-dir evaluation/reports/answer-eval-v1
```

该命令通过真实 STDIO MCP 工具逐题检索，保存客户端原始 JSONL 事件、`evidence_sufficient`、`confidence`、`reason`、完整检索结果、最终答案和引用，并生成自动诊断报告。运行会把问题和检索到的校园 PDF 证据发送给所配置的 OpenAI/Codex 服务，执行前必须确认这些材料允许外发。

在 `predictions.json` 中人工复核 `correct`、`complete`、`citation_supported` 和 `uses_model_memory_or_guess` 后，重新生成最终报告：

```powershell
uv run python evaluation/evaluate_answers.py `
  --dataset evaluation/answer_eval_dataset.json `
  --predictions evaluation/reports/answer-eval-v1/predictions.json `
  --report evaluation/reports/answer-eval-v1/report-reviewed.json
```

报告同时给出正确性、完整性、引用 Precision/Recall、错误回答率、错误拒答率、证据不足拒答遵守率和模型记忆/猜测率，并将失败区分为检索错误、证据判断错误和生成错误。字符匹配只作自动诊断，最终结论以人工复核字段为准。

运行本机冷启动、非缓存和缓存基准：

```powershell
uv run python evaluation/benchmark_runtime.py `
  --query "本科学生国家奖学金奖励标准是多少" `
  --query "休学学生是否需要办理离校手续" `
  --output evaluation/reports/runtime.json
```

本机单并发基准（Windows 11、AMD64 24 逻辑处理器、Python 3.12.13、2 份文档/731 chunks）为：索引约 5.82 MB、进程峰值工作集约 1.86 GB、冷初始化 10.16 秒，非缓存查询 P50 2.53 秒、P95 3.43 秒，缓存查询 P50 1.29 毫秒、P95 1.50 毫秒。分阶段结果表明主要瓶颈是 reranker（平均 2.68 秒），不是 BM25 或向量召回。

## v2 检索基线

当前索引 schema 为 4。v2 策略包括：

- 表格按行切块并重复标题与表头；
- BM25 使用字段—值表格检索文本，Embedding/reranker 使用原始证据；
- rank-normalized cross-encoder 分数加 `0.16 / Hybrid rank`；
- 高置信度与表格类 Hybrid Top 5 保护。

历史 v2 验收集包含 40 条精确 chunk qrel，其中原生/扫描 PDF 各 20 条、表格题 20 条。记录结果为 Recall@1 0.90、Recall@5 1.0、MRR@5 0.9437。旧 `config_freeze.json` 和 `optimization_results.json` 描述的是 v1 历史评测，不应被当成当前生产配置。

`config_freeze_v2.json` 也是历史记录，其部分代码哈希无法由当前仓库内容复核。不要修改历史文件来伪造一致性；新发布周期使用可移植发布清单：

```powershell
uv run --frozen python evaluation/release_manifest.py create `
  --output evaluation/release_manifest.json `
  --tracked

uv run --frozen python evaluation/release_manifest.py verify evaluation/release_manifest.json
```

发布清单应在本轮实现和提交范围确定后创建；清单验证成功是执行完整复现计划的前置条件。

文本文件按 LF 归一化后计算哈希，二进制文件按原始字节计算；校验同时检查规范化大小。任何冻结输入变化都必须开启新的评测周期。

单一复现入口会先校验发布清单、两份 PDF，以及 Embedding、reranker、OCR 检测和 OCR 识别四个完整模型目录的 SHA-256。目录哈希同时覆盖模型权重、配置和 tokenizer；随后从 PDF 原子重建索引，验证 MCP 工具发现与 `knowledge_base_status`，再运行完整 pytest、固定阈值证据评测和性能基准：

```powershell
uv run --frozen python evaluation/reproduce_release.py `
  --asset-root "$PWD"

uv run --frozen python evaluation/reproduce_release.py `
  --asset-root "$PWD" `
  --execute `
  --report evaluation/reports/reproduction.json
```

`--execute` 会更新所指定资产目录下的 `storage/`。需要保留现有运行索引时，可先设置同一个 `RAG_ASSET_ROOT` 并在该目录准备四个模型，再把该目录传给 `--asset-root`；两份输入 PDF 始终从 Git 工作区的 `data/` 读取。
固定复现计划使用 160 DPI 重新 OCR 当前扫描 PDF，并输出原生 PDF 页级进度、OCR 页级进度和每个步骤耗时。日常高质量导入仍默认 300 DPI；如果目标硬件允许，也可以直接用 `src/ingest.py --render-dpi 300` 做更高成本的发布演练。

当前周期的精简复现证据保存在 `evaluation/reproduction_result.json`；详细生成报告继续保持忽略，只在该文件中记录其 SHA-256。2026-08-15 的独立资产目录演练从两份 PDF 重建了 393 页、731 chunks，真实 STDIO MCP 发现 2 个工具并读取到 731 chunks，96 项 pytest 全部通过，两组证据评测和性能基准也通过；总耗时 1519.25 秒，其中索引重建 1119.21 秒。此前一次 300 DPI 演练在 30 分钟内完成 23/30 个 OCR 页面后超时，事务回滚成功且旧索引未被覆盖，因此 300 DPI 仍被视为本机 CPU 上的高成本模式。

## 项目结构

```text
rag/
├── data/                 # 复现所需的两份原始 PDF，纳入 Git
├── docs/                 # 设计、调优路线、当前计划与实施报告
├── evaluation/           # 数据集、冻结、评测和基准工具
├── models/               # 本地模型，不提交 Git
├── slides/               # 本地 HTML 演示；当前被 Git 忽略
├── storage/              # 提取产物和索引，不提交 Git
├── src/
│   ├── extraction/       # 原生/扫描 PDF 提取
│   ├── retrieval/        # 切块、召回、精排、证据判断和缓存
│   ├── ingest.py         # 统一导入和原子更新
│   ├── cli.py
│   └── mcp_server.py
├── tests/                # pytest 测试套件
├── pyproject.toml        # Python 3.12+ 运行依赖
└── uv.lock               # 锁定依赖版本
```
