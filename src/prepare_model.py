"""Download the pinned local models into the configured asset root."""

from __future__ import annotations

import argparse
import os

from retrieval.config import ASSET_ROOT, DEFAULT_MODEL_PATH, DEFAULT_RERANKER_PATH

MODEL_NAME = "BAAI/bge-base-zh-v1.5"
MODEL_PATH = DEFAULT_MODEL_PATH
RERANKER_NAME = "BAAI/bge-reranker-base"
RERANKER_PATH = DEFAULT_RERANKER_PATH


def main() -> None:
    parser = argparse.ArgumentParser()
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--reranker", action="store_true")
    target.add_argument("--ocr", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(ASSET_ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(ASSET_ROOT / ".cache" / "sentence-transformers"))

    if args.ocr:
        from extraction.scanned_pdf import (
            OCR_DETECTION_MODEL,
            OCR_RECOGNITION_MODEL,
            PADDLEX_CACHE_DIR,
            create_ocr_engine,
        )

        model_root = PADDLEX_CACHE_DIR / "official_models"
        expected = [model_root / OCR_DETECTION_MODEL, model_root / OCR_RECOGNITION_MODEL]
        if all((path / "inference.pdiparams").exists() for path in expected):
            print(f"OCR models already exist: {model_root.relative_to(ASSET_ROOT)}")
            return
        create_ocr_engine()
        missing = [path for path in expected if not (path / "inference.pdiparams").exists()]
        if missing:
            raise RuntimeError(f"OCR model download incomplete: {', '.join(str(path) for path in missing)}")
        print(f"OCR models saved: {model_root.relative_to(ASSET_ROOT)}")
        return

    if args.reranker:
        from sentence_transformers import CrossEncoder

        if (RERANKER_PATH / "config.json").exists():
            print(f"Reranker model already exists: {RERANKER_PATH.relative_to(ASSET_ROOT)}")
            return
        model = CrossEncoder(RERANKER_NAME)
        RERANKER_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(RERANKER_PATH))
        print(f"Reranker model saved: {RERANKER_PATH.relative_to(ASSET_ROOT)}")
        return

    from sentence_transformers import SentenceTransformer

    if (MODEL_PATH / "model.safetensors").exists():
        print(f"Embedding model already exists: {MODEL_PATH.relative_to(ASSET_ROOT)}")
        return
    model = SentenceTransformer(MODEL_NAME)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MODEL_PATH))
    print(f"Embedding model saved: {MODEL_PATH.relative_to(ASSET_ROOT)}")


if __name__ == "__main__":
    main()
