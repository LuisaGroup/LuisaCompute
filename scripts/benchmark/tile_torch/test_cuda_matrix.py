"""Host-only tests for workload identity, receipts and failure classification."""
import copy
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest

from cuda_matrix import native_result, tensor_receipts, validate_case


class CudaMatrixTests(unittest.TestCase):
    @staticmethod
    def case(**updates):
        row = dict(id="topk-fixture", operation="topk", dimensions=[1, 4, 2],
                   tile=[1, 4, 1], precision="fp32", seed=19, pattern="adversarial")
        row.update(updates)
        return row

    def read_native(self, case, status="passed", returncode=0, reason="", **updates):
        packet = dict(case, schema=1, backend="cuda", lowering="native", status=status,
                      reason=reason, samples=3, host_wall_us=[2.0, 2.1, 2.2],
                      cuda_event_stream_span_us=[1.0, 1.1, 1.2], graph_batch=4,
                      graph_event_stream_span_us_per_op=[0.5, 0.6, 0.7],
                      correctness=dict(errors=0, inputs_unchanged=True,
                                       guards_unchanged=True, all_outputs_finite=True))
        packet.update(updates)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.json"
            path.write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")
            original = path.read_bytes()
            result = native_result(dict(status="exited", returncode=returncode), path, case,
                                   SimpleNamespace(samples=3, graph_batch=4), "native")
            self.assertEqual(path.read_bytes(), original)
        self.assertEqual(result["result"], packet)
        return result

    @staticmethod
    def write_fixture(directory, case):
        # Tied top values require stable indices [0, 1] in the native contract.
        (directory / "input0.f32").write_bytes(struct.pack("<4f", 3.0, 3.0, -1.0, 2.0))
        (directory / "expected.f64").write_bytes(struct.pack("<2d", 3.0, 3.0))
        (directory / "bound.f64").write_bytes(struct.pack("<2d", 0.0, 0.0))
        (directory / "expected_indices.i64").write_bytes(struct.pack("<2q", 0, 1))
        ranking = case.get("ranking_algorithm", "full_sort_prefix")
        manifest = dict(case, schema=1, endianness="little", ranking_algorithm=ranking,
                        algorithm="stable_repeated_extrema" if ranking == "repeated_extrema" else "padded_bitonic_full_sort_prefix",
                        inputs=[dict(name="input0", path="input0.f32", shape=[1, 4], storage_dtype="float32")],
                        output=dict(path="output.f32", shape=[1, 2], storage_dtype="float32"),
                        expected=dict(path="expected.f64", bound_path="bound.f64", storage_dtype="float64"),
                        indices=dict(path="output_indices.i64", expected_path="expected_indices.i64", shape=[1, 2], storage_dtype="int64"))
        path = directory / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path, manifest

    def test_fast_math_requires_a_boolean_and_keeps_legacy_default(self):
        legacy = validate_case(self.case(), 0)
        self.assertFalse(legacy.get("fast_math", False))
        self.assertEqual(legacy["ranking_algorithm"], "full_sort_prefix")
        for flag in (True, False):
            self.assertIs(validate_case(self.case(fast_math=flag), 0)["fast_math"], flag)
        for invalid in (None, 0, 1, "false", "true"):
            with self.subTest(flag=invalid), self.assertRaisesRegex(ValueError, "fast_math must be boolean"):
                validate_case(self.case(fast_math=invalid), 0)

    def test_repeated_extrema_requires_topk_and_known_algorithm(self):
        for algorithm in ("full_sort_prefix", "repeated_extrema"):
            self.assertEqual(validate_case(self.case(ranking_algorithm=algorithm), 0)["ranking_algorithm"], algorithm)
        for updates, message in ((dict(ranking_algorithm="heap"), "unknown ranking algorithm"),
                                 (dict(operation="sort", ranking_algorithm="repeated_extrema"), "requires topk"),
                                 (dict(operation="rmsnorm", ranking_algorithm="full_sort_prefix"), "only applicable to ranking")):
            with self.subTest(updates=updates), self.assertRaisesRegex(ValueError, message):
                validate_case(self.case(**updates), 0)

    def test_native_requested_policy_must_match_result(self):
        case = self.case(fast_math=True, ranking_algorithm="repeated_extrema")
        self.assertEqual(self.read_native(case)["status"], "passed")
        for changes, message in ((dict(fast_math=False), "fast_math mismatch"),
                                 (dict(fast_math=1), "fast_math mismatch"),
                                 (dict(ranking_algorithm="full_sort_prefix"), "ranking algorithm mismatch")):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                self.read_native(case, **changes)

    def test_native_historical_missing_policy_fields_remain_strict(self):
        self.assertEqual(self.read_native(self.case())["status"], "passed")

    def test_compiler_failure_retains_original_packet(self):
        result = self.read_native(self.case(), status="compiler_failure", returncode=1,
                                  reason="CUDA Tile IR NVRTC failed (exit 0xc00000fd): stack overflow")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_kind"], "compiler_failure")
        self.assertEqual(result["result"]["status"], "compiler_failure")

    def test_historical_process_failures_are_not_shape_rejections(self):
        for reason in ("CUDA Tile IR NVRTC failed (exit 0xc00000fd):",
                       "CUDA Tile IR tileiras failed (exit 0x00000001):",
                       "CUDA Tile IR NVRTC could not start: missing executable",
                       "CUDA Tile IR tileiras could not start: access denied"):
            with self.subTest(reason=reason):
                result = self.read_native(self.case(), status="unsupported", returncode=3, reason=reason)
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["failure_kind"], "compiler_failure")
                self.assertEqual(result["result"]["status"], "unsupported")

    def test_real_capability_rejection_remains_unsupported(self):
        result = self.read_native(self.case(), status="unsupported", returncode=3,
                                  reason="CUDA Tile IR: unsupported gather")
        self.assertEqual(result["status"], "unsupported")
        self.assertNotIn("failure_kind", result)

    def test_failure_status_requires_failure_exit_and_reason(self):
        for status, code, reason in (("compiler_failure", 0, "compiler failed"),
                                     ("compiler_failure", 1, ""),
                                     ("unsupported", 0, "unsupported operation")):
            with self.subTest(status=status, code=code, reason=reason), self.assertRaises(ValueError):
                self.read_native(self.case(), status=status, returncode=code, reason=reason)

    def test_manifest_requested_and_realized_ranking_are_checked(self):
        case = self.case(fast_math=True, ranking_algorithm="repeated_extrema")
        with tempfile.TemporaryDirectory() as temporary:
            path, original = self.write_fixture(Path(temporary), case)
            self.assertEqual(tensor_receipts(path, case)[0]["algorithm"], "stable_repeated_extrema")
            for updates, message in ((dict(fast_math=False), "fast_math mismatch"),
                                     (dict(ranking_algorithm="full_sort_prefix"), "ranking algorithm mismatch"),
                                     (dict(algorithm="padded_bitonic_full_sort_prefix"), "realized ranking algorithm mismatch")):
                changed = copy.deepcopy(original)
                changed.update(updates)
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.subTest(updates=updates), self.assertRaisesRegex(ValueError, message):
                    tensor_receipts(path, case)

    def test_policy_changes_do_not_change_fixture_or_oracle_receipts(self):
        strict = self.case(fast_math=False, ranking_algorithm="full_sort_prefix")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path, original = self.write_fixture(directory, strict)
            strict_hashes = tensor_receipts(path, strict)[1]
            files_before = {name: (directory / name).read_bytes() for name in strict_hashes}
            for updates in (dict(fast_math=True), dict(ranking_algorithm="repeated_extrema")):
                case = {**strict, **updates}
                changed = {**original, **updates}
                if case["ranking_algorithm"] == "repeated_extrema":
                    changed["algorithm"] = "stable_repeated_extrema"
                path.write_text(json.dumps(changed), encoding="utf-8")
                self.assertEqual(tensor_receipts(path, case)[1], strict_hashes)
                self.assertEqual({name: (directory / name).read_bytes() for name in strict_hashes}, files_before)


if __name__ == "__main__":
    unittest.main()
