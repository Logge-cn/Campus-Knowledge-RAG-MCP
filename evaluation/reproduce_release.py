"""Check release inputs and optionally execute the full reproducibility plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.release_manifest import verify_manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
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
        if not path.is_file():
            errors.append({"code": "missing_prerequisite", "root": root_name, "path": relative_path})
            continue
        expected = item.get("sha256")
        if expected and _sha256(path) != expected:
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
            result = subprocess.run(argv, cwd=project_root, env=env, check=False)
            step = {"id": item["id"], "returncode": result.returncode, "passed": result.returncode == 0}
            steps.append(step)
            if result.returncode != 0:
                errors.append({"code": "reproduction_step_failed", **step})
                break
    return {
        "valid": not errors,
        "mode": "execute" if execute else "check",
        "release_manifest": manifest_report,
        "steps": steps,
        "errors": errors,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PROJECT_ROOT / "evaluation" / "reproduction_plan.json")
    parser.add_argument("--asset-root", type=Path, default=PROJECT_ROOT)
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
