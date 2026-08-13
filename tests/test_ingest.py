import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ingest import detect_pdf_type, ingest_documents


class IngestTests(unittest.TestCase):
    def _pdf(self, path: Path, text: str = "") -> None:
        document = pymupdf.open()
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)
        document.save(path)
        document.close()

    def test_detects_native_and_scanned_pdfs(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            native = root / "native.pdf"
            scanned = root / "scanned.pdf"
            self._pdf(native, "native document text " * 20)
            self._pdf(scanned)

            self.assertEqual(detect_pdf_type(native), "native")
            self.assertEqual(detect_pdf_type(scanned), "scanned")

    def test_dry_run_does_not_create_artifacts(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            data_root = root / "data"
            artifacts_root = root / "storage" / "artifacts"
            data_root.mkdir()
            pdf = data_root / "document.pdf"
            self._pdf(pdf, "document text " * 20)

            result = ingest_documents(
                [pdf],
                data_root=data_root,
                artifacts_root=artifacts_root,
                index_path=root / "storage" / "index" / "metadata.json",
                dry_run=True,
            )

            self.assertEqual(result["documents"][0]["source_type"], "native")
            self.assertEqual(result["changes"]["added"], ["document.pdf"])
            self.assertFalse(artifacts_root.exists())

    def test_promotes_extraction_and_marks_old_version_inactive(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            data_root = root / "data"
            artifacts_root = root / "storage" / "artifacts"
            data_root.mkdir()
            old_pdf = data_root / "rules-2023.pdf"
            new_pdf = data_root / "rules-2024.pdf"
            self._pdf(old_pdf, "old rules " * 20)
            self._pdf(new_pdf, "new rules " * 20)

            def fake_process(pdf_path, active_data_root, output_root):
                relative = pdf_path.relative_to(active_data_root)
                target = output_root / relative.with_suffix("")
                target.mkdir(parents=True)
                (target / "metadata.json").write_text('{"pages": []}', encoding="utf-8")
                return {"source_file": relative.as_posix(), "pages": []}

            def fake_build(active_artifacts_root, active_index_path, force, published_artifacts_root):
                self.assertEqual(published_artifacts_root, artifacts_root)
                active_index_path.parent.mkdir(parents=True)
                active_index_path.write_text('{"schema_version": 4}', encoding="utf-8")
                return {"rebuilt": True}

            with patch("ingest.native_process", side_effect=fake_process), patch(
                "ingest.build_index", side_effect=fake_build
            ) as build:
                ingest_documents(
                    [old_pdf],
                    data_root=data_root,
                    artifacts_root=artifacts_root,
                    index_path=root / "storage" / "index" / "metadata.json",
                    document_id="rules",
                    version="2023",
                )
                result = ingest_documents(
                    [new_pdf],
                    data_root=data_root,
                    artifacts_root=artifacts_root,
                    index_path=root / "storage" / "index" / "metadata.json",
                    document_id="rules",
                    version="2024",
                )

            manifest = json.loads((artifacts_root / "documents.json").read_text(encoding="utf-8"))
            records = {item["source_file"]: item for item in manifest["documents"]}
            self.assertFalse(records["rules-2023.pdf"]["active"])
            self.assertTrue(records["rules-2024.pdf"]["active"])
            self.assertEqual(result["documents"][0]["version"], "2024")
            self.assertEqual(result["quality"]["pages"], 0)
            self.assertTrue(result["quality"]["passed"])
            self.assertEqual(build.call_count, 2)

    def test_failed_index_build_keeps_previous_artifacts_and_index(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            data_root = root / "data"
            artifacts_root = root / "storage" / "artifacts"
            index_path = root / "storage" / "index" / "metadata.json"
            data_root.mkdir()
            artifacts_root.mkdir(parents=True)
            index_path.parent.mkdir(parents=True)
            (artifacts_root / "sentinel.txt").write_text("old-artifacts", encoding="utf-8")
            index_path.write_text("old-index", encoding="utf-8")
            pdf = data_root / "new.pdf"
            self._pdf(pdf, "new rules " * 20)

            def fake_process(pdf_path, active_data_root, output_root):
                target = output_root / pdf_path.relative_to(active_data_root).with_suffix("")
                target.mkdir(parents=True)
                (target / "metadata.json").write_text('{"pages": []}', encoding="utf-8")
                return {"pages": []}

            with patch("ingest.native_process", side_effect=fake_process), patch(
                "ingest.build_index", side_effect=RuntimeError("simulated build failure")
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated build failure"):
                    ingest_documents(
                        [pdf],
                        data_root=data_root,
                        artifacts_root=artifacts_root,
                        index_path=index_path,
                    )

            self.assertEqual((artifacts_root / "sentinel.txt").read_text(encoding="utf-8"), "old-artifacts")
            self.assertEqual(index_path.read_text(encoding="utf-8"), "old-index")
            self.assertFalse((artifacts_root / "new").exists())

    def test_document_id_is_unambiguous_for_batch_ingest(self):
        with self.assertRaisesRegex(ValueError, "one PDF"):
            ingest_documents(
                [PROJECT_ROOT / "one.pdf", PROJECT_ROOT / "two.pdf"],
                data_root=PROJECT_ROOT,
                document_id="shared",
                dry_run=True,
            )


if __name__ == "__main__":
    unittest.main()
