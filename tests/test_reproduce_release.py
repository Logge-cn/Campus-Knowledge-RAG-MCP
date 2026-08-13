import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.reproduce_release import check_prerequisites, expand_argv, reproduce


class ReproduceReleaseTests(unittest.TestCase):
    def test_expands_placeholders_inside_path_arguments(self):
        self.assertEqual(
            expand_argv(["{asset_root}/data/source.pdf"], {"{asset_root}": "C:/assets"}),
            ["C:/assets/data/source.pdf"],
        )

    def test_checks_project_and_asset_prerequisites(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            assets = root / "assets"
            project.mkdir()
            assets.mkdir()
            payload = assets / "model.bin"
            payload.write_bytes(b"model")
            prerequisites = [
                {
                    "root": "assets",
                    "path": "model.bin",
                    "sha256": hashlib.sha256(b"model").hexdigest(),
                }
            ]

            self.assertEqual(check_prerequisites(prerequisites, project, assets), [])

    def test_check_mode_reports_hash_mismatch_without_running_steps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            assets = root / "assets"
            project.mkdir()
            assets.mkdir()
            (assets / "model.bin").write_bytes(b"wrong")
            plan = {
                "prerequisites": [{"root": "assets", "path": "model.bin", "sha256": "0" * 64}],
                "steps": [{"id": "never", "argv": ["{python}", "-c", "raise SystemExit(1)"]}],
            }

            report = reproduce(plan, project, assets, Path(sys.executable), execute=False)

            self.assertFalse(report["valid"])
            self.assertEqual(report["steps"], [])
            self.assertEqual(report["errors"][0]["code"], "prerequisite_sha256_mismatch")


if __name__ == "__main__":
    unittest.main()
