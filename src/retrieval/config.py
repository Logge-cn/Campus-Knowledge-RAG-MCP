"""Shared paths and retrieval settings."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = Path(os.environ.get("RAG_ASSET_ROOT", PROJECT_ROOT / "runtime")).resolve()
if ASSET_ROOT.parent == ASSET_ROOT:
    raise ValueError("RAG_ASSET_ROOT must not be a filesystem root")
DEFAULT_ARTIFACTS_ROOT = ASSET_ROOT / "storage" / "artifacts"
DEFAULT_INDEX_ROOT = ASSET_ROOT / "storage" / "index"
DEFAULT_INDEX_PATH = DEFAULT_INDEX_ROOT / "metadata.json"
DEFAULT_MODEL_PATH = ASSET_ROOT / "models" / "bge-base-zh-v1.5"
DEFAULT_RERANKER_PATH = ASSET_ROOT / "models" / "bge-reranker-base"
CHUNKS_PATH_NAME = "chunks.json"
BM25_PATH_NAME = "bm25.json"
EMBEDDINGS_PATH_NAME = "embeddings.npy"
SCHEMA_VERSION = 4
CHUNK_TARGET_TOKENS = 410
CHUNK_MAX_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 41
# Backwards-compatible names used by callers and index metadata.
CHUNK_SIZE = CHUNK_MAX_TOKENS
CHUNK_OVERLAP = CHUNK_OVERLAP_TOKENS
BM25_K1 = 1.5
BM25_B = 0.75
CANDIDATE_LIMIT = 20
RRF_K = 1
RRF_BM25_WEIGHT = 3.0
RRF_VECTOR_WEIGHT = 1.0
EMBEDDING_MODEL = "BAAI/bge-base-zh-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANK_CANDIDATE_LIMIT = 20
RERANK_BATCH_SIZE = 16
RERANK_SCORE_NORMALIZATION = "rank"
RERANK_RRF_PRIOR_METHOD = "reciprocal_rank"
RERANK_RRF_PRIOR_WEIGHT = 0.16
RERANK_TOP5_PROTECTION_SCORE = 0.97
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
QUERY_EXPANSION_WEIGHT = 2
TOKENIZER_VERSION = "char-bigram-v1"


def inside_project(path: Path) -> Path:
    resolved = path.resolve()
    for root in dict.fromkeys((PROJECT_ROOT, ASSET_ROOT)):
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    raise ValueError(f"Path must be inside the project or configured asset root: {path}")


def relative_asset_path(path: Path) -> Path:
    """Return a stable project-style path for project or asset files."""
    resolved = inside_project(path)
    # Prefer the explicit asset root when it is nested inside the code worktree.
    # This keeps persisted paths stable as ``storage/...`` instead of exposing
    # an environment-specific ``runtime/storage/...`` prefix.
    for root in dict.fromkeys((ASSET_ROOT, PROJECT_ROOT)):
        try:
            return resolved.relative_to(root)
        except ValueError:
            continue
    raise AssertionError("inside_project accepted a path without an allowed root")
