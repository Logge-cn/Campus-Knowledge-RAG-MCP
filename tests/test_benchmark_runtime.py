import unittest
import tempfile
from pathlib import Path


from evaluation.benchmark_runtime import hardware_info, index_size_bytes, percentile


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


if __name__ == "__main__":
    unittest.main()
