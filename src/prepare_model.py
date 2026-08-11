"""Download the pinned embedding model and caches into this project only."""

from __future__ import annotations

import os
import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "BAAI/bge-base-zh-v1.5"
MODEL_PATH = PROJECT_ROOT / "models" / "bge-base-zh-v1.5"
RERANKER_NAME = "BAAI/bge-reranker-base"
RERANKER_PATH = PROJECT_ROOT / "models" / "bge-reranker-base"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reranker", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(PROJECT_ROOT / ".cache" / "sentence-transformers"))

    if args.reranker:
        from sentence_transformers import CrossEncoder

        if (RERANKER_PATH / "config.json").exists():
            print(f"Reranker model already exists: {RERANKER_PATH.relative_to(PROJECT_ROOT)}")
            return
        model = CrossEncoder(RERANKER_NAME)
        RERANKER_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(RERANKER_PATH))
        print(f"Reranker model saved: {RERANKER_PATH.relative_to(PROJECT_ROOT)}")
        return

    from sentence_transformers import SentenceTransformer

    if (MODEL_PATH / "model.safetensors").exists():
        print(f"Embedding model already exists: {MODEL_PATH.relative_to(PROJECT_ROOT)}")
        return
    model = SentenceTransformer(MODEL_NAME)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MODEL_PATH))
    print(f"Embedding model saved: {MODEL_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
