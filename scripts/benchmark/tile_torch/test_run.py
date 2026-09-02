import importlib.util
from pathlib import Path
import sys
import unittest


SPEC = importlib.util.spec_from_file_location("tile_torch_benchmark", Path(__file__).with_name("run.py"))
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BenchmarkContractTests(unittest.TestCase):
    def test_shape_matrix_is_not_only_squares(self):
        cases = MODULE.make_cases(["gemm"])
        self.assertEqual(len(cases), 8)
        self.assertTrue(any(c.m == c.n == c.k == 1024 for c in cases))
        self.assertTrue(any(c.m > c.n for c in cases))
        self.assertTrue(any(c.m < c.n for c in cases))
        self.assertTrue(any(c.m % 8 and c.n % 8 and c.k % 16 for c in cases))

    def test_reduction_sizes_include_wide_and_single_row(self):
        cases = MODULE.make_cases(["sum", "softmax"])
        self.assertTrue(any(c.m == 1 for c in cases))
        self.assertTrue(any(c.n == 4096 for c in cases))

    def test_percentiles(self):
        self.assertEqual(MODULE.percentile([9, 1, 5], 0.5), 5)
        self.assertAlmostEqual(MODULE.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9], 0.9), 8.2)
        with self.assertRaises(ValueError):
            MODULE.percentile([float("nan")], 0.5)

    def test_tolerances_are_shared_and_add_is_exact(self):
        self.assertEqual(MODULE.tolerance("add"), (0.0, 0.0))
        self.assertEqual(MODULE.tolerance("gemm"), (1e-4, 1e-4))

    def test_unknown_operation_is_not_silently_skipped(self):
        with self.assertRaises(ValueError):
            MODULE.make_cases(["gemm", "unknown"])


if __name__ == "__main__":
    unittest.main()
