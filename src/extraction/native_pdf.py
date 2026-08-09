"""Extract reliable native PDF text and tables into RAG artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import camelot
import fitz


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIN_ALNUM_CHARS = 50
MIN_USABLE_BLOCK_RATIO = 0.01
MAX_REPLACEMENT_RATIO = 0.15
MAX_DUPLICATE_LINE_RATIO = 0.30
MIN_PYMUPDF_TABLE_DENSITY = 0.60


@dataclass
class PageMetrics:
    alnum_chars: int
    replacement_ratio: float
    usable_block_ratio: float
    duplicate_line_ratio: float
    reading_order_anomaly: bool


@dataclass
class TableArtifact:
    rows: list[list[str]]
    bbox: tuple[float, float, float, float]
    method: str
    score: float
    title: str


def clean_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def cell_density(rows: list[list[str]]) -> float:
    cells = [cell for row in rows for cell in row]
    return sum(bool(clean_text(cell)) for cell in cells) / len(cells) if cells else 0.0


def calculate_metrics(page: fitz.Page) -> PageMetrics:
    blocks = page.get_text("blocks", sort=False)
    text_blocks = [block for block in blocks if clean_text(block[4])]
    text = "\n".join(block[4] for block in text_blocks)
    non_whitespace = [char for char in text if not char.isspace()]
    lines = [clean_text(line) for line in text.splitlines() if clean_text(line)]
    usable_area = sum(
        fitz.Rect(block[:4]).get_area()
        for block in text_blocks
        if sum(char.isalnum() for char in block[4]) >= 5
    )
    heights = sorted(block[3] - block[1] for block in text_blocks)
    median_height = heights[len(heights) // 2] if heights else 0.0
    y_reversal = any(
        abs(current[0] - previous[0]) <= page.rect.width * 0.15
        and current[1] + median_height < previous[1]
        for previous, current in zip(
            [(block[0], block[1]) for block in text_blocks],
            [(block[0], block[1]) for block in text_blocks][1:],
        )
    )
    left = [block for block in text_blocks if (block[0] + block[2]) / 2 < page.rect.width / 2]
    right = [block for block in text_blocks if (block[0] + block[2]) / 2 >= page.rect.width / 2]
    columns_overlap = False
    if len(left) >= 3 and len(right) >= 3:
        left_top, left_bottom = min(block[1] for block in left), max(block[3] for block in left)
        right_top, right_bottom = min(block[1] for block in right), max(block[3] for block in right)
        overlap = max(0.0, min(left_bottom, right_bottom) - max(left_top, right_top))
        columns_overlap = overlap >= min(left_bottom - left_top, right_bottom - right_top) * 0.5
    return PageMetrics(
        alnum_chars=sum(char.isalnum() for char in text),
        replacement_ratio=non_whitespace.count("\ufffd") / len(non_whitespace) if non_whitespace else 0.0,
        usable_block_ratio=usable_area / page.rect.get_area(),
        duplicate_line_ratio=(len(lines) - len(set(lines))) / len(lines) if lines else 0.0,
        reading_order_anomaly=y_reversal or columns_overlap,
    )


def decision(metrics: PageMetrics) -> str:
    if (
        metrics.alnum_chars == 0
        or metrics.replacement_ratio > MAX_REPLACEMENT_RATIO
        or metrics.usable_block_ratio < MIN_USABLE_BLOCK_RATIO
    ):
        return "ocr"
    return "native"


def pymupdf_rows(table: fitz.table.Table) -> list[list[str]]:
    return [[clean_text(cell) for cell in row] for row in table.extract()]


def camelot_rows(table: camelot.core.Table) -> list[list[str]]:
    return [[clean_text(cell) for cell in row] for row in table.df.values.tolist()]


def camelot_bbox(table: camelot.core.Table, page_height: float) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = table._bbox
    return (x0, page_height - y1, x1, page_height - y0)


def table_title(page: fitz.Page, bbox: tuple[float, float, float, float], fallback: str) -> str:
    lines: dict[int, list[tuple[float, str]]] = {}
    for x0, y0, _, _, word, *_ in page.get_text("words", sort=True):
        lines.setdefault(round(y0 / 3), []).append((x0, word))
    candidates = []
    for line_y, words in lines.items():
        text = "".join(word for _, word in sorted(words))
        y0 = line_y * 3
        if bbox[1] - 100 <= y0 <= bbox[1] and re.search(r"表\s*\d+", text):
            candidates.append((y0, text))
    return max(candidates, default=(0.0, fallback))[1]


def detect_tables(pdf_path: Path, page: fitz.Page, page_number: int) -> list[TableArtifact]:
    fitz.TOOLS.mupdf_display_errors(False)
    pymupdf_tables = page.find_tables().tables
    pymupdf_candidates = [
        TableArtifact(
            rows=pymupdf_rows(table),
            bbox=tuple(table.bbox),
            method="pymupdf.find_tables",
            score=cell_density(pymupdf_rows(table)),
            title="",
        )
        for table in pymupdf_tables
    ]
    needs_lattice = any(table.score < MIN_PYMUPDF_TABLE_DENSITY for table in pymupdf_candidates)
    lattice_candidates: list[TableArtifact] = []
    if needs_lattice:
        try:
            for table in camelot.read_pdf(str(pdf_path), pages=str(page_number), flavor="lattice"):
                rows = camelot_rows(table)
                report = table.parsing_report
                score = cell_density(rows) * (report["accuracy"] / 100) * (1 - report["whitespace"] / 100)
                lattice_candidates.append(
                    TableArtifact(
                        rows=rows,
                        bbox=camelot_bbox(table, page.rect.height),
                        method="camelot.lattice",
                        score=score,
                        title="",
                    )
                )
        except Exception:
            lattice_candidates = []
    selected: list[TableArtifact] = []
    for index, candidate in enumerate(pymupdf_candidates):
        alternative = lattice_candidates[index] if index < len(lattice_candidates) else None
        chosen = candidate
        if candidate.score < MIN_PYMUPDF_TABLE_DENSITY and alternative and alternative.score > candidate.score:
            chosen = alternative
        chosen.title = table_title(page, chosen.bbox, f"第 {page_number} 页表 {index + 1}")
        selected.append(chosen)
    return selected


def intersects_table(block: tuple, table: TableArtifact) -> bool:
    block_rect, table_rect = fitz.Rect(block[:4]), fitz.Rect(table.bbox)
    intersection = block_rect & table_rect
    return bool(intersection) and intersection.get_area() >= block_rect.get_area() * 0.5


def native_text(page: fitz.Page, tables: list[TableArtifact]) -> str:
    blocks = [
        clean_text(block[4])
        for block in page.get_text("blocks", sort=True)
        if clean_text(block[4]) and not any(intersects_table(block, table) for table in tables)
    ]
    return "\n\n".join(blocks)


def normalize_table(rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    width = max(map(len, rows), default=0)
    rows = [row + [""] * (width - len(row)) for row in rows]
    keep = [column for column in range(width) if any(clean_text(row[column]) for row in rows)]
    rows = [[row[column] for column in keep] for row in rows]
    has_header = sum(bool(cell) for cell in rows[0]) >= 2
    header_rows = 2 if has_header and len(rows) > 1 and sum(bool(cell) for cell in rows[1][:2]) == 0 and any(rows[1][2:]) else 1
    if has_header:
        headers = []
        parent = ""
        for column in range(len(rows[0])):
            if rows[0][column]:
                parent = rows[0][column]
            child = rows[1][column] if header_rows == 2 else ""
            headers.append(" / ".join(filter(None, (parent, child))))
        body = rows[header_rows:]
    else:
        headers = [f"列{column + 1}" for column in range(len(rows[0]))]
        body = rows
    for column in range(len(headers)):
        previous = ""
        for row in body:
            if not row[column] and previous and any(row[column + 1 :]):
                row[column] = previous
            elif row[column]:
                previous = row[column]
    return headers, body


def markdown_table(rows: list[list[str]]) -> str:
    headers, body = normalize_table(rows)
    escape = lambda cell: cell.replace("|", "\\|")
    lines = ["| " + " | ".join(map(escape, headers)) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(map(escape, row)) + " |" for row in body)
    return "\n".join(lines)


def write_markdown(path: Path, metadata: dict, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frontmatter = "\n".join(f"{key}: {value}" for key, value in metadata.items())
    path.write_text(f"---\n{frontmatter}\n---\n\n{content}\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process(pdf_path: Path, data_root: Path, output_root: Path) -> dict:
    fitz.TOOLS.mupdf_display_errors(False)
    relative_path = pdf_path.relative_to(data_root)
    artifact_dir = output_root / relative_path.with_suffix("")
    document = fitz.open(pdf_path)
    pages: list[dict] = []
    for page_number, page in enumerate(document, 1):
        metrics = calculate_metrics(page)
        page_decision = decision(metrics)
        warnings = []
        if 0 < metrics.alnum_chars < MIN_ALNUM_CHARS:
            warnings.append("short_text")
        if metrics.duplicate_line_ratio > MAX_DUPLICATE_LINE_RATIO:
            warnings.append("duplicate_line_ratio")
        if metrics.reading_order_anomaly:
            warnings.append("reading_order_anomaly")
        page_record = {
            "page": page_number,
            "decision": page_decision,
            "metrics": asdict(metrics),
            "quality_warnings": warnings,
            "tables": [],
        }
        if page_decision == "native":
            tables = detect_tables(pdf_path, page, page_number)
            text = native_text(page, tables)
            write_markdown(
                artifact_dir / "pages" / f"page-{page_number:03d}.md",
                {"source_file": relative_path.as_posix(), "page": page_number, "source_type": "pdf"},
                text,
            )
            for table_index, table in enumerate(tables, 1):
                table_path = artifact_dir / "tables" / f"page-{page_number:03d}-table-{table_index:02d}.md"
                write_markdown(
                    table_path,
                    {
                        "source_file": relative_path.as_posix(),
                        "page": page_number,
                        "source_type": "table",
                        "table_index": table_index,
                        "table_title": table.title,
                        "extraction_method": table.method,
                        "extraction_score": f"{table.score:.3f}",
                    },
                    f"# {table.title}\n\n{markdown_table(table.rows)}",
                )
                page_record["tables"].append({"index": table_index, "method": table.method, "score": round(table.score, 3)})
        pages.append(page_record)
    metadata = {
        "source_file": relative_path.as_posix(),
        "file_sha256": sha256(pdf_path),
        "imported_at": datetime.now(UTC).isoformat(),
        "extractor": "extraction/native_pdf.py",
        "pages": pages,
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", nargs="+", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "storage" / "artifacts")
    args = parser.parse_args()
    for pdf_path in args.pdf:
        metadata = process(pdf_path, args.data_root, args.output_root)
        native_pages = sum(page["decision"] == "native" for page in metadata["pages"])
        print(f"{pdf_path}: native_pages={native_pages}/{len(metadata['pages'])}")


if __name__ == "__main__":
    main()
