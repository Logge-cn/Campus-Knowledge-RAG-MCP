"""Shared paths and retrieval settings."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / "storage" / "artifacts"
DEFAULT_INDEX_ROOT = PROJECT_ROOT / "storage" / "index"
DEFAULT_INDEX_PATH = DEFAULT_INDEX_ROOT / "metadata.json"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "bge-base-zh-v1.5"
CHUNKS_PATH_NAME = "chunks.json"
BM25_PATH_NAME = "bm25.json"
EMBEDDINGS_PATH_NAME = "embeddings.npy"
SCHEMA_VERSION = 2
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
BM25_K1 = 1.5
BM25_B = 0.75
CANDIDATE_LIMIT = 20
RRF_K = 1
EMBEDDING_MODEL = "BAAI/bge-base-zh-v1.5"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
TOKENIZER_VERSION = "char-bigram-v1"


def inside_project(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(f"Path must be inside the project: {path}") from error
    return resolved
