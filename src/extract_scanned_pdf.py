"""Extract scanned PDF text with PaddleOCR into traceable RAG artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import fitz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PADDLEX_CACHE_DIR = PROJECT_ROOT / ".cache" / "paddlex"
os.environ["PADDLE_PDX_CACHE_HOME"] = str(PADDLEX_CACHE_DIR)

from paddleocr import PaddleOCR


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_markdown(path: Path, metadata: dict, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frontmatter = "\n".join(f"{key}: {value}" for key, value in metadata.items())
    path.write_text(f"---\n{frontmatter}\n---\n\n{content}\n", encoding="utf-8")


def process(pdf_path: Path, data_root: Path, output_root: Path, start_page: int = 1, end_page: int | None = None) -> dict:
    relative_path = pdf_path.relative_to(data_root)
    artifact_dir = output_root / relative_path.with_suffix("")
    rendered_dir = artifact_dir / "rendered"
    raw_dir = artifact_dir / "ocr_raw"
    document = fitz.open(pdf_path)
    ocr = PaddleOCR(
        lang="ch",
        device="cpu",
        enable_mkldnn=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    pages = []
    for page_number, page in enumerate(document, 1):
        image_path = rendered_dir / f"page-{page_number:03d}.png"
        raw_path = raw_dir / f"page-{page_number:03d}_res.json"
        markdown_path = artifact_dir / "pages" / f"page-{page_number:03d}.md"
        if not raw_path.exists():
            if page_number < start_page or (end_page is not None and page_number > end_page):
                continue
            image_path.parent.mkdir(parents=True, exist_ok=True)
            page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72), alpha=False).save(image_path)
            raw_dir.mkdir(parents=True, exist_ok=True)
            ocr.predict(str(image_path))[0].save_to_json(save_path=str(raw_dir))
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        ordered = sorted(
            zip(payload["rec_boxes"], payload["rec_texts"], payload["rec_scores"]),
            key=lambda row: (row[0][1], row[0][0]),
        )
        text = "\n\n".join(item[1] for item in ordered)
        confidence = sum(item[2] for item in ordered) / len(ordered) if ordered else 0.0
        note = "自动识别，建议核对原 PDF 第 %d 页" % page_number if confidence < 0.85 else ""
        write_markdown(
            markdown_path,
            {
                "source_file": relative_path.as_posix(),
                "page": page_number,
                "source_type": "ocr",
                "confidence": f"{confidence:.3f}",
                "processing_note": note or "none",
            },
            text,
        )
        pages.append(
            {
                "page": page_number,
                "decision": "ocr",
                "confidence": confidence,
                "processing_note": note,
                "raw_result": raw_path.relative_to(artifact_dir).as_posix(),
            }
        )
    metadata = {
        "source_file": relative_path.as_posix(),
        "file_sha256": sha256(pdf_path),
        "imported_at": datetime.now(UTC).isoformat(),
        "extractor": "extract_scanned_pdf.py",
        "pages": pages,
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int)
    args = parser.parse_args()
    metadata = process(args.pdf, args.data_root, args.output_root, args.start_page, args.end_page)
    print(f"{args.pdf}: ocr_pages={len(metadata['pages'])}")


if __name__ == "__main__":
    main()
