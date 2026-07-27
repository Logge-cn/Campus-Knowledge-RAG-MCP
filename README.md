# NJUPT RAG 知识库

这是一个按职责划分的 RAG 知识库项目。`crawler` 采集南京邮电大学公开网页、PDF 和 DOCX；后续由 `ingestion` 建库、`retrieval` 检索、`app` 对外提供服务。

## 安装

需要 Python 3.12+ 和 [uv](https://docs.astral.sh/uv/)。在项目根目录执行：

```powershell
uv sync
```

## 运行

```powershell
uv run python -m crawler
```

## 目录结构

```text
.
├── crawler/            # 采集：白名单站点 → 原始文件与 SQLite 文档
│   ├── collector.py
│   └── sources.json
├── ingestion/          # 清洗、切块、向量化、建库
├── retrieval/          # 召回、重排、引用
├── app/                # API、MCP 或 Web 界面
├── shared/             # 跨模块数据模型、配置、工具
├── data/               # 原始文件与本地文档库
├── pyproject.toml      # Python 版本与项目依赖
└── uv.lock             # 锁定的依赖版本
```
