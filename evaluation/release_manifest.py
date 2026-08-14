"""Create or verify a portable release manifest for RAG code and artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".json", ".lock", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
TEXT_FILENAMES = {".editorconfig", ".gitattributes", ".gitignore"}


def _portable_bytes(path: Path) -> tuple[bytes, bool]:
    content = path.read_bytes()
    portable = path.suffix.lower() in TEXT_SUFFIXES or path.name.lower() in TEXT_FILENAMES
    return (content.replace(b"\r\n", b"\n").replace(b"\r", b"\n") if portable else content), portable


def _file_record(root: Path, relative_path: Path) -> dict[str, Any]:
    path = (root / relative_path).resolve()
    path.relative_to(root.resolve())
    content, portable = _portable_bytes(path)
    return {
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "portable_text": portable,
    }


def _git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_tracked_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [Path(value.decode("utf-8")) for value in result.stdout.split(b"\0") if value]


def create_manifest(root: Path, files: Iterable[Path]) -> dict[str, Any]:
    root = root.resolve()
    normalized = sorted({path.as_posix() for path in files})
    missing = [path for path in normalized if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze missing files: {', '.join(missing)}")
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "commit": _git_commit(root),
        "hash_policy": "LF-normalized UTF-8-compatible text; raw bytes for binary files",
        "files": {path: _file_record(root, Path(path)) for path in normalized},
    }


def verify_manifest(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    root = root.resolve()
    errors: list[dict[str, Any]] = []
    files = manifest.get("files")
    if manifest.get("schema_version") != 1 or not isinstance(files, dict):
        return {"valid": False, "checked": 0, "errors": [{"code": "invalid_manifest_schema"}]}
    for relative_path, expected in sorted(files.items()):
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            errors.append({"code": "path_outside_root", "path": relative_path})
            continue
        if not path.is_file():
            errors.append({"code": "missing_file", "path": relative_path})
            continue
        actual = _file_record(root, Path(relative_path))
        if actual["sha256"] != expected.get("sha256"):
            errors.append(
                {
                    "code": "sha256_mismatch",
                    "path": relative_path,
                    "expected": expected.get("sha256"),
                    "actual": actual["sha256"],
                }
            )
        if actual["bytes"] != expected.get("bytes"):
            errors.append(
                {
                    "code": "byte_size_mismatch",
                    "path": relative_path,
                    "expected": expected.get("bytes"),
                    "actual": actual["bytes"],
                }
            )
    return {"valid": not errors, "checked": len(files), "errors": errors}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--output", type=Path, required=True)
    group = create.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=Path, action="append")
    group.add_argument("--tracked", action="store_true", help="Freeze every Git-tracked file except the output")
    verify = subparsers.add_parser("verify")
    verify.add_argument("manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "create":
        output = args.output.resolve()
        output.relative_to(PROJECT_ROOT)
        files = _git_tracked_files(PROJECT_ROOT) if args.tracked else args.file
        output_relative = output.relative_to(PROJECT_ROOT)
        manifest = create_manifest(PROJECT_ROOT, (path for path in files if path != output_relative))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"created": output.relative_to(PROJECT_ROOT).as_posix(), "files": len(manifest["files"])}, ensure_ascii=False))
        return
    manifest_path = args.manifest.resolve()
    manifest_path.relative_to(PROJECT_ROOT)
    report = verify_manifest(PROJECT_ROOT, json.loads(manifest_path.read_text(encoding="utf-8")))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
