import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch


from evaluation.retrieval.benchmark import hardware_info, index_size_bytes, percentile
from retrieval.runtime import warmup


class RuntimeBenchmarkTests(unittest.TestCase):
    def test_percentile_uses_nearest_rank(self):
        self.assertEqual(percentile([1, 2, 3, 4], 0.5), 2)
        self.assertEqual(percentile([1, 2, 3, 4], 0.95), 4)

    def test_percentile_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            percentile([], 0.95)

    def test_hardware_info_records_reproduction_context(self):
        info = hardware_info()
        self.assertGreaterEqual(info["logical_cpus"], 1)
        self.assertEqual(info["concurrency"], 1)
        self.assertTrue(info["python"])

    def test_index_size_counts_persisted_files(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            (root / "metadata.json").write_bytes(b"12")
            (root / "chunks.json").write_bytes(b"345")
            self.assertEqual(index_size_bytes(root / "metadata.json"), 5)

    def test_warmup_loads_reranker_before_embedding_model(self):
        calls = []
        index = {"metadata": {"documents": 2, "chunks": 731}}
        with (
            patch("retrieval.runtime.load_index", return_value=index),
            patch("retrieval.runtime.load_reranker", side_effect=lambda: calls.append("reranker")),
            patch("retrieval.runtime.load_model", side_effect=lambda: calls.append("embedding")),
        ):
            result = warmup(Path("unused.json"))

        self.assertEqual(calls, ["reranker", "embedding"])
        self.assertEqual(result["documents"], 2)
        self.assertEqual(result["chunks"], 731)


if __name__ == "__main__":
    unittest.main()
