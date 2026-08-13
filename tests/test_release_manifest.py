import json
import tempfile
import unittest
from pathlib import Path


from evaluation.release_manifest import create_manifest, verify_manifest


class ReleaseManifestTests(unittest.TestCase):
    def test_text_hash_is_portable_and_changes_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracked = root / "config.py"
            tracked.write_bytes(b"value = 1\r\n")
            manifest = create_manifest(root, [Path("config.py")])

            tracked.write_bytes(b"value = 1\n")
            self.assertTrue(verify_manifest(root, manifest)["valid"])

            tracked.write_text("value = 2\n", encoding="utf-8")
            report = verify_manifest(root, manifest)
            self.assertFalse(report["valid"])
            self.assertEqual(report["errors"][0]["code"], "sha256_mismatch")

    def test_binary_size_and_missing_files_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "model.safetensors"
            binary.write_bytes(b"model")
            manifest = create_manifest(root, [Path("model.safetensors")])
            binary.unlink()

            report = verify_manifest(root, manifest)

            self.assertFalse(report["valid"])
            self.assertEqual(report["errors"], [{"code": "missing_file", "path": "model.safetensors"}])

    def test_manifest_round_trips_as_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data.json").write_text('{"value": 1}\n', encoding="utf-8")
            manifest = json.loads(json.dumps(create_manifest(root, [Path("data.json")])))
            self.assertTrue(verify_manifest(root, manifest)["valid"])


if __name__ == "__main__":
    unittest.main()
