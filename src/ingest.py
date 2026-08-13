"""Safely extract PDF documents and rebuild the local retrieval index."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

import pymupdf

from retrieval import build_index
from retrieval.config import DEFAULT_ARTIFACTS_ROOT, DEFAULT_INDEX_PATH, PROJECT_ROOT, inside_project


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_pdf_type(pdf_path: Path, *, sample_pages: int = 3, minimum_characters: int = 50) -> str:
    """Classify a PDF as native when its sampled pages contain usable text."""
    document = pymupdf.open(pdf_path)
    try:
        text = "".join(page.get_text("text") for page in document[:sample_pages])
    finally:
        document.close()
    characters = sum(character.isalnum() for character in text)
    return "native" if characters >= minimum_characters else "scanned"


def native_process(*args, **kwargs):
    from extraction.native_pdf import process

    return process(*args, **kwargs)


def scanned_process(*args, **kwargs):
    from extraction.scanned_pdf import process

    return process(*args, **kwargs)


def _load_manifest(artifacts_root: Path) -> dict[str, Any]:
    path = artifacts_root / "documents.json"
    if not path.exists():
        return {"schema_version": 1, "documents": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("documents"), list):
        raise ValueError(f"Invalid document manifest: {path}")
    return payload


def _promote_directories(pairs: list[tuple[Path, Path]], backup_root: Path) -> None:
    promoted: list[tuple[Path, Path]] = []
    backed_up: list[tuple[Path, Path]] = []
    try:
        backup_root.mkdir(parents=True, exist_ok=True)
        for position, (source, target) in enumerate(pairs):
            backup = backup_root / f"target-{position}"
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                target.replace(backup)
                backed_up.append((backup, target))
            source.replace(target)
            promoted.append((target, source))
    except Exception:
        for target, source in reversed(promoted):
            if target.exists():
                target.replace(source)
        for backup, target in reversed(backed_up):
            if backup.exists():
                backup.replace(target)
        raise


def ingest_documents(
    pdf_paths: list[Path],
    *,
    data_root: Path,
    artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT,
    index_path: Path = DEFAULT_INDEX_PATH,
    source_type: str = "auto",
    document_id: str | None = None,
    version: str | None = None,
    effective_date: str | None = None,
    render_dpi: int = 300,
    dry_run: bool = False,
) -> dict[str, Any]:
    if not pdf_paths:
        raise ValueError("At least one PDF is required")
    if document_id and len(pdf_paths) > 1:
        raise ValueError("--document-id can only be used when ingesting one PDF")
    if source_type not in {"auto", "native", "scanned"}:
        raise ValueError("source_type must be auto, native or scanned")
    data_root = inside_project(data_root)
    artifacts_root = inside_project(artifacts_root)
    index_path = inside_project(index_path)
    plans: list[dict[str, Any]] = []
    for pdf_path in pdf_paths:
        resolved = inside_project(pdf_path)
        relative = resolved.relative_to(data_root)
        kind = detect_pdf_type(resolved) if source_type == "auto" else source_type
        plans.append(
            {
                "path": resolved,
                "relative": relative,
                "artifact_relative": relative.with_suffix(""),
                "source_file": relative.as_posix(),
                "source_type": kind,
                "sha256": _sha256(resolved),
                "document_id": document_id or relative.with_suffix("").as_posix(),
                "version": version,
                "effective_date": effective_date,
            }
        )
    public_plans = [{key: value for key, value in plan.items() if key not in {"path", "relative", "artifact_relative"}} for plan in plans]
    if dry_run:
        return {"dry_run": True, "documents": public_plans, "index_rebuilt": False}

    storage_root = artifacts_root.parent
    storage_root.mkdir(parents=True, exist_ok=True)
    transaction_root = Path(tempfile.mkdtemp(prefix=".ingest-staging-", dir=storage_root))
    staging_artifacts = transaction_root / "artifacts"
    staging_index = transaction_root / "index"
    try:
        if artifacts_root.exists():
            shutil.copytree(artifacts_root, staging_artifacts)
        else:
            staging_artifacts.mkdir(parents=True)
        for plan in plans:
            staged_document = staging_artifacts / plan["artifact_relative"]
            if staged_document.exists():
                shutil.rmtree(staged_document)
            if plan["source_type"] == "native":
                native_process(plan["path"], data_root, staging_artifacts)
            else:
                scanned_process(plan["path"], data_root, staging_artifacts, render_dpi=render_dpi)

        manifest = _load_manifest(staging_artifacts)
        incoming_ids = {plan["document_id"] for plan in plans}
        for record in manifest["documents"]:
            if record.get("document_id") in incoming_ids:
                record["active"] = False
        records_by_source = {record["source_file"]: record for record in manifest["documents"]}
        for plan in plans:
            records_by_source[plan["source_file"]] = {
                "document_id": plan["document_id"],
                "source_file": plan["source_file"],
                "source_type": plan["source_type"],
                "sha256": plan["sha256"],
                "version": plan["version"],
                "effective_date": plan["effective_date"],
                "active": True,
            }
        manifest["documents"] = sorted(records_by_source.values(), key=lambda item: item["source_file"])
        (staging_artifacts / "documents.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        staged_index_path = staging_index / index_path.name
        index_result = build_index(staging_artifacts, staged_index_path, force=True)
        _promote_directories(
            [(staging_artifacts, artifacts_root), (staging_index, index_path.parent)],
            transaction_root / "backup",
        )
    finally:
        shutil.rmtree(transaction_root, ignore_errors=True)
    return {"dry_run": False, "documents": public_plans, "index_rebuilt": True, "index": index_result}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", nargs="+", type=Path)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--source-type", choices=("auto", "native", "scanned"), default="auto")
    parser.add_argument("--document-id")
    parser.add_argument("--version")
    parser.add_argument("--effective-date", type=date.fromisoformat)
    parser.add_argument("--render-dpi", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = ingest_documents(
        args.pdf,
        data_root=args.data_root,
        source_type=args.source_type,
        document_id=args.document_id,
        version=args.version,
        effective_date=args.effective_date.isoformat() if args.effective_date else None,
        render_dpi=args.render_dpi,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
