"""Check release inputs and optionally execute the full reproducibility plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.release.manifest import verify_manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str:
    """Hash every file name and byte in a directory using a stable order."""
    digest = hashlib.sha256()
    files = sorted(
        (item for item in path.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(path).as_posix(),
    )
    for file_path in files:
        relative = file_path.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(file_path.stat().st_size.to_bytes(8, "big"))
        with file_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def check_prerequisites(
    prerequisites: list[dict[str, Any]],
    project_root: Path,
    asset_root: Path,
) -> list[dict[str, Any]]:
    errors = []
    roots = {"project": project_root.resolve(), "assets": asset_root.resolve()}
    for item in prerequisites:
        root_name = item.get("root")
        relative_path = item.get("path")
        if root_name not in roots or not isinstance(relative_path, str):
            errors.append({"code": "invalid_prerequisite", "item": item})
            continue
        root = roots[root_name]
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            errors.append({"code": "prerequisite_outside_root", "path": relative_path})
            continue
        kind = item.get("kind", "file")
        if kind not in {"file", "tree"}:
            errors.append({"code": "invalid_prerequisite_kind", "path": relative_path, "kind": kind})
            continue
        exists = path.is_file() if kind == "file" else path.is_dir()
        if not exists:
            errors.append({"code": "missing_prerequisite", "root": root_name, "path": relative_path})
            continue
        expected = item.get("sha256")
        actual = _sha256(path) if kind == "file" else _tree_sha256(path)
        if expected and actual != expected:
            errors.append({"code": "prerequisite_sha256_mismatch", "root": root_name, "path": relative_path})
    return errors


def expand_argv(values: list[str], replacements: dict[str, str]) -> list[str]:
    argv = []
    for value in values:
        for placeholder, replacement in replacements.items():
            value = value.replace(placeholder, replacement)
        argv.append(value)
    return argv


def reproduce(
    plan: dict[str, Any],
    project_root: Path,
    asset_root: Path,
    python: Path,
    *,
    execute: bool,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    asset_root = asset_root.resolve()
    errors = check_prerequisites(plan.get("prerequisites", []), project_root, asset_root)
    release_manifest_path = plan.get("release_manifest")
    manifest_report = None
    if release_manifest_path:
        path = (project_root / release_manifest_path).resolve()
        try:
            path.relative_to(project_root)
        except ValueError:
            errors.append({"code": "release_manifest_outside_project"})
        else:
            if not path.is_file():
                errors.append({"code": "release_manifest_missing", "path": release_manifest_path})
            else:
                manifest_report = verify_manifest(project_root, json.loads(path.read_text(encoding="utf-8")))
                if not manifest_report["valid"]:
                    errors.append({"code": "release_manifest_invalid", "details": manifest_report["errors"]})

    steps = []
    run_started = time.perf_counter()
    if execute and not errors:
        replacements = {
            "{python}": str(python.resolve()),
            "{project_root}": str(project_root),
            "{asset_root}": str(asset_root),
        }
        env = os.environ.copy()
        env["RAG_ASSET_ROOT"] = str(asset_root)
        for item in plan.get("steps", []):
            argv = expand_argv(item["argv"], replacements)
            print(f"[reproduce] START {item['id']}", flush=True)
            started = time.perf_counter()
            result = subprocess.run(argv, cwd=project_root, env=env, check=False)
            step = {
                "id": item["id"],
                "returncode": result.returncode,
                "passed": result.returncode == 0,
                "duration_seconds": round(time.perf_counter() - started, 2),
            }
            steps.append(step)
            print(
                f"[reproduce] {'PASS' if step['passed'] else 'FAIL'} {item['id']} "
                f"({step['duration_seconds']:.2f}s)",
                flush=True,
            )
            if result.returncode != 0:
                errors.append({"code": "reproduction_step_failed", **step})
                break
    return {
        "valid": not errors,
        "mode": "execute" if execute else "check",
        "release_manifest": manifest_report,
        "steps": steps,
        "duration_seconds": round(time.perf_counter() - run_started, 2),
        "errors": errors,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PROJECT_ROOT / "evaluation" / "release" / "plan.json")
    parser.add_argument("--asset-root", type=Path, default=PROJECT_ROOT / "runtime")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    plan_path = args.plan.resolve()
    plan_path.relative_to(PROJECT_ROOT)
    report = reproduce(
        json.loads(plan_path.read_text(encoding="utf-8")),
        PROJECT_ROOT,
        args.asset_root,
        args.python,
        execute=args.execute,
    )
    if args.report:
        report_path = args.report.resolve()
        report_path.relative_to(PROJECT_ROOT)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
