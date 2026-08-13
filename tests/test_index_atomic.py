import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval.index import _artifacts_digest, _index_paths, _source_records, _write_index_atomically


class AtomicIndexTests(unittest.TestCase):
    def test_staged_artifacts_publish_stable_paths(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            staging = root / "staging"
            published = root / "published"
            page = staging / "document" / "pages" / "page-001.md"
            page.parent.mkdir(parents=True)
            page.write_text(
                "---\nsource_file: document.pdf\npage: 1\nsource_type: pdf\n---\n\nevidence\n",
                encoding="utf-8",
            )

            record = _source_records(staging, published)[0]

            self.assertEqual(record["artifact_path"], f"{published.relative_to(PROJECT_ROOT).as_posix()}/document/pages/page-001.md")

    def test_artifact_digest_changes_with_indexable_content(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            page = root / "document" / "pages" / "page-001.md"
            page.parent.mkdir(parents=True)
            page.write_text("old", encoding="utf-8")
            before = _artifacts_digest(root)
            page.write_text("new", encoding="utf-8")
            self.assertNotEqual(_artifacts_digest(root), before)

    def test_failed_first_index_does_not_leave_partial_files(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            paths = _index_paths(Path(directory) / "index" / "metadata.json")
            real_replace = __import__("os").replace
            calls = 0

            def fail_on_second(source, target):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("simulated promotion failure")
                real_replace(source, target)

            with patch("retrieval.index.os.replace", side_effect=fail_on_second):
                with self.assertRaises(OSError):
                    _write_index_atomically(
                        paths,
                        [{"text": "new"}],
                        {"document_lengths": [1]},
                        np.asarray([[1.0]], dtype=np.float32),
                        {"schema_version": 4},
                    )

            self.assertFalse(any(path.exists() for path in paths.values()))

    def test_failed_promotion_restores_previous_index(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            paths = _index_paths(root / "index" / "metadata.json")
            paths["metadata"].parent.mkdir(parents=True)
            for path in paths.values():
                path.write_bytes(f"old-{path.name}".encode())
            original = {key: path.read_bytes() for key, path in paths.items()}
            real_replace = __import__("os").replace
            calls = 0

            def fail_on_second(source, target):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("simulated promotion failure")
                real_replace(source, target)

            with patch("retrieval.index.os.replace", side_effect=fail_on_second):
                with self.assertRaises(OSError):
                    _write_index_atomically(
                        paths,
                        [{"text": "new"}],
                        {"document_lengths": [1]},
                        np.asarray([[1.0]], dtype=np.float32),
                        {"schema_version": 4},
                    )

            self.assertEqual({key: path.read_bytes() for key, path in paths.items()}, original)

    def test_successful_promotion_writes_metadata_last(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            paths = _index_paths(root / "index" / "metadata.json")
            _write_index_atomically(
                paths,
                [{"text": "new"}],
                {"document_lengths": [1]},
                np.asarray([[1.0]], dtype=np.float32),
                {"schema_version": 4},
            )
            self.assertEqual(json.loads(paths["metadata"].read_text(encoding="utf-8"))["schema_version"], 4)
            self.assertEqual(json.loads(paths["chunks"].read_text(encoding="utf-8"))[0]["text"], "new")


if __name__ == "__main__":
    unittest.main()
