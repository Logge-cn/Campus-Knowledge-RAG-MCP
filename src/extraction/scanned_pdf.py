"""Extract scanned PDF text with PaddleOCR into traceable RAG artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

import fitz

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = Path(os.environ.get("RAG_ASSET_ROOT", PROJECT_ROOT / "runtime")).resolve()
if ASSET_ROOT.parent == ASSET_ROOT:
    raise ValueError("RAG_ASSET_ROOT must not be a filesystem root")
PADDLEX_CACHE_DIR = ASSET_ROOT / "models" / "paddlex"
os.environ["PADDLE_PDX_CACHE_HOME"] = str(PADDLEX_CACHE_DIR)

from paddleocr import PaddleOCR

OCR_DETECTION_MODEL = "PP-OCRv6_medium_det"
OCR_RECOGNITION_MODEL = "PP-OCRv6_medium_rec"


def create_ocr_engine() -> PaddleOCR:
    """Create the OCR engine used by both model preparation and extraction."""
    return PaddleOCR(
        lang="ch",
        device="cpu",
        enable_mkldnn=False,
        text_detection_model_name=OCR_DETECTION_MODEL,
        text_recognition_model_name=OCR_RECOGNITION_MODEL,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_markdown(path: Path, metadata: dict, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frontmatter = "\n".join(f"{key}: {value}" for key, value in metadata.items())
    path.write_text(f"---\n{frontmatter}\n---\n\n{content}\n", encoding="utf-8")


def page_continuation(previous_text: str, text: str) -> str:
    """Preserve a heading split by a scanned PDF page break in the next page's artifact."""
    if not re.match(r"^[\u4e00-\u9fff]{1,4}[：:]", text) or not previous_text:
        return ""
    previous_tail = re.sub(r"\s+", "", previous_text)[-32:]
    return f"上页续文：{previous_tail}"


def process(
    pdf_path: Path,
    data_root: Path,
    output_root: Path,
    start_page: int = 1,
    end_page: int | None = None,
    render_dpi: int = 300,
) -> dict:
    if render_dpi <= 0:
        raise ValueError("render_dpi must be positive")
    relative_path = pdf_path.relative_to(data_root)
    artifact_dir = output_root / relative_path.with_suffix("")
    rendered_dir = artifact_dir / "rendered"
    raw_dir = artifact_dir / "ocr_raw"
    document = fitz.open(pdf_path)
    source_sha256 = sha256(pdf_path)
    imported_at = datetime.now(UTC).isoformat()
    ocr: PaddleOCR | None = None
    pages = []
    previous_text = ""
    for page_number, page in enumerate(document, 1):
        print(f"[ocr] {relative_path.as_posix()}: page {page_number}/{len(document)}", flush=True)
        image_path = rendered_dir / f"page-{page_number:03d}.png"
        raw_path = raw_dir / f"page-{page_number:03d}_res.json"
        markdown_path = artifact_dir / "pages" / f"page-{page_number:03d}.md"
        if not raw_path.exists():
            if page_number < start_page or (end_page is not None and page_number > end_page):
                continue
            image_path.parent.mkdir(parents=True, exist_ok=True)
            page.get_pixmap(matrix=fitz.Matrix(render_dpi / 72, render_dpi / 72), alpha=False).save(image_path)
            raw_dir.mkdir(parents=True, exist_ok=True)
            if ocr is None:
                ocr = create_ocr_engine()
            ocr.predict(str(image_path))[0].save_to_json(save_path=str(raw_dir))
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        ordered = sorted(
            zip(payload["rec_boxes"], payload["rec_texts"], payload["rec_scores"]),
            key=lambda row: (row[0][1], row[0][0]),
        )
        text = "\n\n".join(item[1] for item in ordered)
        continuation = page_continuation(previous_text, text)
        indexed_text = f"{continuation}\n\n{text}" if continuation else text
        confidence = sum(item[2] for item in ordered) / len(ordered) if ordered else 0.0
        note = "自动识别，建议核对原 PDF 第 %d 页" % page_number if confidence < 0.85 else ""
        write_markdown(
            markdown_path,
            {
                "source_file": relative_path.as_posix(),
                "page": page_number,
                "source_type": "ocr",
                "source_sha256": source_sha256,
                "content_sha256": text_sha256(indexed_text),
                "imported_at": imported_at,
                "confidence": f"{confidence:.3f}",
                "low_confidence": str(confidence < 0.85).lower(),
                "processing_note": note or "none",
                "render_dpi": render_dpi,
            },
            indexed_text,
        )
        pages.append(
            {
                "page": page_number,
                "decision": "ocr",
                "confidence": confidence,
                "processing_note": note,
                "continued_from_previous_page": bool(continuation),
                "render_dpi": render_dpi,
                "raw_result": raw_path.relative_to(artifact_dir).as_posix(),
            }
        )
        previous_text = text
    metadata = {
        "source_file": relative_path.as_posix(),
        "file_sha256": source_sha256,
        "imported_at": imported_at,
        "extractor": "extraction/scanned_pdf.py",
        "ocr_models": {
            "text_detection": OCR_DETECTION_MODEL,
            "text_recognition": OCR_RECOGNITION_MODEL,
        },
        "render_dpi": render_dpi,
        "pages": pages,
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ASSET_ROOT / "storage" / "artifacts",
    )
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int)
    parser.add_argument("--render-dpi", type=int, default=300)
    args = parser.parse_args()
    metadata = process(args.pdf, args.data_root, args.output_root, args.start_page, args.end_page, args.render_dpi)
    print(f"{args.pdf}: ocr_pages={len(metadata['pages'])}")


if __name__ == "__main__":
    main()
