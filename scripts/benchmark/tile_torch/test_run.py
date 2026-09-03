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

    def test_native_mapping_metadata_matches_the_request(self):
        case = MODULE.Case("gemm", 7, 17, 37)
        native = dict(backend="metal", operation="gemm", execution_scope="group", pipeline_window=2, mma_operations=1,
                      cooperative_matrix=False, matrix_intrinsics=0)
        MODULE.validate_native_metadata(native, case, "metal", "group")
        for key, value in (("backend", "cpu"), ("operation", "add"), ("execution_scope", "worker"), ("mma_operations", 0)):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                MODULE.validate_native_metadata(native | {key: value}, case, "metal", "group")
        del native["execution_scope"]
        with self.assertRaisesRegex(RuntimeError, "execution-scope"):
            MODULE.validate_native_metadata(native, case, "metal", "group")

    def test_native_pipeline_window_matches_the_request(self):
        case = MODULE.Case("gemm", 7, 17, 37)
        native = dict(backend="metal", operation="gemm", execution_scope="worker", mma_operations=1,
                      cooperative_matrix=False, matrix_intrinsics=0)
        for window in (1, 2):
            MODULE.validate_native_metadata(native | {"pipeline_window": window}, case, "metal", "worker", window)
            with self.assertRaisesRegex(RuntimeError, "pipeline-window"):
                MODULE.validate_native_metadata(native | {"pipeline_window": 3 - window}, case, "metal", "worker", window)
        with self.assertRaisesRegex(RuntimeError, "pipeline-window"):
            MODULE.validate_native_metadata(native, case, "metal", "worker")

    def test_matrix_capability_is_not_confused_with_emitted_instructions(self):
        case = MODULE.Case("gemm", 7, 17, 37)
        for enabled in (False, True):
            for backend, scope, block in (("cpu", "worker", (8, 8, 16)),
                                          ("metal", "worker", (8, 8, 16)),
                                          ("metal", "group", (3, 5, 7)),
                                          ("metal", "group", (8, 8, 16))):
                calls = int(enabled and backend == "metal" and scope == "group" and block == (8, 8, 16))
                native = dict(backend=backend, operation="gemm", execution_scope=scope, pipeline_window=2,
                              mma_operations=1, cooperative_matrix=enabled, matrix_intrinsics=calls)
                arguments = (case, backend, scope, 2, enabled, block)
                with self.subTest(enabled=enabled, backend=backend, scope=scope, block=block):
                    MODULE.validate_native_metadata(native, *arguments)
                    with self.assertRaisesRegex(RuntimeError, "cooperative-matrix"):
                        MODULE.validate_native_metadata(native | {"cooperative_matrix": not enabled}, *arguments)
                    with self.assertRaisesRegex(RuntimeError, "matrix-intrinsic"):
                        MODULE.validate_native_metadata(native | {"matrix_intrinsics": 1 - calls}, *arguments)
                    for invalid in (None, -1, 1.5, True):
                        with self.assertRaisesRegex(RuntimeError, "matrix-intrinsic"):
                            MODULE.validate_native_metadata(native | {"matrix_intrinsics": invalid}, *arguments)


if __name__ == "__main__":
    unittest.main()
