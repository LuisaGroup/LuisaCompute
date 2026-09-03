import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("tile_torch_benchmark", Path(__file__).with_name("run.py"))
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

REPEAT_SPEC = importlib.util.spec_from_file_location("tile_torch_repeat", Path(__file__).with_name("repeat.py"))
REPEAT = importlib.util.module_from_spec(REPEAT_SPEC)
with patch.dict(sys.modules, {"run": MODULE}):
    REPEAT_SPEC.loader.exec_module(REPEAT)


class RepeatContractTests(unittest.TestCase):
    def test_variant_and_framework_orders_are_both_balanced(self):
        keys = [("metal", str(i)) for i in range(8)]
        rounds = [list(REPEAT.order_for_round(keys, i)) for i in range(4)]
        for key in keys:
            first_variants = [next(v for k, v, _ in r if k == key) for r in rounds]
            self.assertEqual(first_variants.count("candidate"), 2)
            for variant in ("reference", "candidate"):
                orders = [order for r in rounds for k, v, order in r if (k, v) == (key, variant)]
                self.assertEqual(sorted(orders), [0, 0, 1, 1])
        self.assertEqual([r[0][0] for r in rounds], keys[:4])

    @staticmethod
    def row():
        return {"case": {"operation": "gemm", "m": 17, "n": 19, "k": 13}, "backend": "metal",
                "valid": True, "block": [16, 32, 32], "native": {
                    "execution_scope": "group", "pipeline_window": 1, "cooperative_matrix": True,
                    "vectorize": True, "auto_vectorize": False, "throughput_us_p50": 1.0}}

    def test_plan_uses_recorded_configuration_but_not_recorded_score(self):
        row = self.row()
        with patch.object(Path, "read_text", return_value=json.dumps({"results": [row]})):
            plan = REPEAT.load_plan(Path("unused.json"), {"gemm"})
        config = plan["metal", "gemm_17x19x13"]
        self.assertEqual(config["gemm_block"], (16, 32, 32))
        self.assertEqual(config["pipeline_window"], 1)
        self.assertNotIn("throughput_us_p50", config)

    def test_plan_does_not_guess_historical_vectorization_semantics(self):
        row = self.row()
        del row["native"]["auto_vectorize"]
        with patch.object(Path, "read_text", return_value=json.dumps({"results": [row]})):
            with self.assertRaisesRegex(ValueError, "auto_vectorize"):
                REPEAT.load_plan(Path("unused.json"), {"gemm"})

    def test_plan_rejects_invalid_duplicate_and_empty_selections(self):
        for rows in ([self.row(), self.row()], [self.row() | {"valid": False}], []):
            with self.subTest(rows=rows), patch.object(Path, "read_text", return_value=json.dumps({"results": rows})):
                with self.assertRaises(ValueError):
                    REPEAT.load_plan(Path("unused.json"), {"gemm"})


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
                      cooperative_matrix=False, matrix_intrinsics=0, vectorize=True, auto_vectorize=False)
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
                      cooperative_matrix=False, matrix_intrinsics=0, vectorize=True, auto_vectorize=False)
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
                              mma_operations=1, cooperative_matrix=enabled, matrix_intrinsics=calls, vectorize=True, auto_vectorize=False)
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

    def test_native_vectorization_metadata_matches_the_request(self):
        case = MODULE.Case("gemm", 7, 17, 37)
        native = dict(backend="cpu", operation="gemm", execution_scope="worker", pipeline_window=2,
                      cooperative_matrix=False, matrix_intrinsics=0, mma_operations=1, auto_vectorize=False)
        for enabled in (False, True):
            MODULE.validate_native_metadata(native | {"vectorize": enabled}, case, "cpu", "worker", vectorize=enabled)
            for invalid in (None, int(enabled), not enabled):
                with self.subTest(enabled=enabled, invalid=invalid), self.assertRaisesRegex(RuntimeError, "vectorization"):
                    MODULE.validate_native_metadata(native | {"vectorize": invalid}, case, "cpu", "worker", vectorize=enabled)

    def test_automatic_vectorization_is_a_separate_explicit_option(self):
        case = MODULE.Case("gemm", 7, 17, 37)
        native = dict(backend="cpu", operation="gemm", execution_scope="worker", pipeline_window=2,
                      cooperative_matrix=False, matrix_intrinsics=0, mma_operations=1, vectorize=True)
        for enabled in (False, True):
            MODULE.validate_native_metadata(native | {"auto_vectorize": enabled}, case, "cpu", "worker", auto_vectorize=enabled)
            for invalid in (None, int(enabled), not enabled):
                with self.subTest(enabled=enabled, invalid=invalid), self.assertRaisesRegex(RuntimeError, "automatic-vectorization"):
                    MODULE.validate_native_metadata(native | {"auto_vectorize": invalid}, case, "cpu", "worker", auto_vectorize=enabled)

    def test_jit_candidates_are_explicit_and_deduplicated(self):
        self.assertEqual(MODULE.tuning_candidates((8, 8, 16), 2, None, None), [])
        self.assertEqual(MODULE.tuning_candidates((8, 8, 16), 2, "8,8,16;16,32,32;8,8,16", "1,2,1"),
                         [((8, 8, 16), 1), ((8, 8, 16), 2), ((16, 32, 32), 1), ((16, 32, 32), 2)])
        self.assertEqual(MODULE.tuning_candidates((8, 8, 16), 2, None, "1"), [((8, 8, 16), 1)])
        for block in ("", "8,8", "0,8,16", "8,8,16,32", "8,a,16"):
            with self.subTest(block=block), self.assertRaises(ValueError):
                MODULE.parse_gemm_block(block)
        with self.assertRaises(ValueError):
            MODULE.tuning_candidates((8, 8, 16), 2, None, "0,2")

    def test_jit_search_revalidates_winner_instead_of_publishing_minimum(self):
        args = argparse.Namespace(tuning_candidates=[((8, 8, 16), 2), ((16, 32, 32), 1), ((64, 64, 64), 2)],
                                  gemm_block=(8, 8, 16), pipeline_window=2)
        calls = []
        ordinals = []

        def measure(torch, np, candidate, case, backend, ordinal):
            calls.append((candidate.gemm_block, candidate.pipeline_window))
            ordinals.append(ordinal)
            if candidate.gemm_block == (64, 64, 64):
                raise RuntimeError("capacity")
            score = 100.0 if len(calls) == 4 else (2.0 if candidate.gemm_block == (8, 8, 16) else 1.0)
            return {"valid": True, "native": {"throughput_us_p50": score}}

        with patch.object(MODULE, "run_case", side_effect=measure), contextlib.redirect_stdout(io.StringIO()):
            result = MODULE.run_tuned_case(None, None, args, MODULE.Case("gemm", 32, 32, 32), "cpu", 1)
        self.assertEqual(calls[0], ((16, 32, 32), 1))  # Rotated trial order.
        self.assertEqual(calls[-1], ((16, 32, 32), 1))
        self.assertEqual(len(calls), 4)  # Includes a fresh recapture/JIT after selection.
        self.assertEqual(ordinals[-1], 1)  # Fresh comparison preserves alternating framework order.
        self.assertEqual(result["native"]["throughput_us_p50"], 100.0)
        self.assertEqual(sum(trial["valid"] for trial in result["tuning"]["trials"]), 2)
        self.assertEqual(args.gemm_block, (8, 8, 16))  # No mutation of caller configuration.

    def test_jit_search_cannot_hide_failed_candidates_or_revalidation(self):
        args = argparse.Namespace(tuning_candidates=[((8, 8, 16), 2)], gemm_block=(8, 8, 16), pipeline_window=2)
        case = MODULE.Case("gemm", 32, 32, 32)
        for failure in (RuntimeError("oracle"), {"valid": True, "native": {"throughput_us_p50": float("nan")}}):
            effect = [failure] if isinstance(failure, Exception) else lambda *args: failure
            with patch.object(MODULE, "run_case", side_effect=effect), contextlib.redirect_stdout(io.StringIO()):
                result = MODULE.run_tuned_case(None, None, args, case, "cpu", 0)
            self.assertFalse(result["valid"])
            self.assertFalse(result["tuning"]["trials"][0]["valid"])
        with patch.object(MODULE, "run_case", side_effect=[{"valid": True, "native": {"throughput_us_p50": 1.0}},
                                                          RuntimeError("fresh oracle")]), contextlib.redirect_stdout(io.StringIO()):
            result = MODULE.run_tuned_case(None, None, args, case, "cpu", 0)
        self.assertFalse(result["valid"])
        self.assertIn("revalidation", result["error"])
        self.assertTrue(result["tuning"]["trials"][0]["valid"])


if __name__ == "__main__":
    unittest.main()
