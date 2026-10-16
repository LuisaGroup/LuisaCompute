"""Pure ranking oracle, typed exports, timing scopes, and failed-route tests.

No Torch import, native executable, device, or performance measurement is used.
"""
import argparse
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import ranking as rank


def expected(case, inputs):
    values, indices = [], []
    for row in range(case.rows):
        source = inputs[row * case.columns:(row + 1) * case.columns]
        order = sorted(range(case.columns), key=lambda index: (
            -source[index] if case.direction == "descending" else source[index], index))[:case.k]
        indices.extend(order)
        values.extend(source[index] for index in order)
    return values, indices


def measurement(case, prefix=None):
    result = dict(status="passed", operation=case.operation, dimensions=case.dimensions,
                  backend="simd", implementation="tile_xir_simd",
                  direction=case.direction, precision="fp32", index_dtype="int64", stable_ties=True,
                  algorithm="quadratic_rank_reference", input_shape=[case.rows, case.columns],
                  output_shape=[case.rows, case.k], timing="synchronized_host_wall", repetitions=2,
                  correctness=dict(checks=2, values_per_check=case.rows * case.k, indices_per_check=case.rows * case.k,
                                   input_elements_per_check=case.rows * case.columns, guard_elements_per_check=102,
                                   values_bitwise_equal=True, indices_exact=True, input_immutable=True, all_guards_intact=True),
                  throughput_us=[2., 4.], latency_us=[5., 7.])
    if prefix is not None:
        result.update(input_path=str(prefix) + ".input.f32", values_path=str(prefix) + ".values.f32",
                      indices_path=str(prefix) + ".indices.i64")
    return result


def arguments():
    return argparse.Namespace(samples=2, sample_ms=3., warmup_ms=4., threads=1,
                              timeout=5., metal_device_timing=None, metal4_device_timing=False)


def metal4_sample(sample_id, count, counters):
    return dict(sample_id=sample_id, error="", overflow=False, dispatch_timestamps_enabled=counters,
                timestamp_frequency_hz=1000000000 if counters else 0,
                command_buffers=[dict(ordinal=0, dispatch_count=count, contains_non_dispatch_work=False,
                                      valid=True, gpu_begin_seconds=1., gpu_end_seconds=1.000004,
                                      host_commit_begin_ns=1, host_commit_return_ns=2,
                                      host_feedback_begin_ns=3, host_callbacks_end_ns=4,
                                      host_completion_publish_ns=5)],
                dispatches=[dict(ordinal=index, command_buffer_ordinal=0, dispatch_size=[32, 1, 1],
                                 block_size=[32, 1, 1], shader_checksum="1234", valid=counters,
                                 begin_ticks=100 if counters else 0, end_ticks=1100 if counters else 0,
                                 elapsed_ns=1000. if counters else 0.) for index in range(count)])


def metal4_device():
    device = dict(method="metal4_precise_dispatch_timestamps_v1", scope="instrumented_dispatch_intervals",
                  host_samples_instrumented=False, zero_overhead_kernel_time=False, repetitions=2,
                  capabilities=dict(timestamp_heap=True, timestamp_frequency_hz=1000000000),
                  control=dict(method="metal4_commit_feedback_v1", scope="command_buffer_gpu_intervals",
                               encoder_instrumentation=False, repetitions=2))
    sample_id = 1
    for phase, count in (("throughput", 2), ("latency", 1)):
        for target, counters in ((device, True), (device["control"], False)):
            target[phase] = [metal4_sample(sample_id + index, count, counters) for index in range(2)]
            sample_id += 2
    return device


def metal_device():
    record = dict(compute_ns=1000., compute_span_ns=1100., command_buffer_ns=4000.,
                  calibration_cpu_ns=100., calibration_gpu_ticks=100., compute_passes=1, command_buffers=1)
    return dict(method="metal_compute_pass_timestamps_v1", scope="sum_of_compute_encoder_gpu_intervals",
                host_samples_instrumented=False, repetitions=2,
                throughput=[deepcopy(record) for _ in range(2)], latency=[deepcopy(record) for _ in range(2)],
                control=dict(method="metal_command_buffer_timestamps_v1", scope="sum_of_command_buffer_gpu_intervals",
                             encoder_instrumentation=False, repetitions=2,
                             throughput=[dict(command_buffer_ns=3000., command_buffers=1) for _ in range(2)],
                             latency=[dict(command_buffer_ns=3000., command_buffers=1) for _ in range(2)]))


class RankingOracleTests(unittest.TestCase):
    def test_case_bounds(self):
        self.assertEqual(rank.parse_case("topk:17,65,8"), rank.Case("topk", 17, 65, 8))
        self.assertEqual(rank.parse_case("sort:1,31,31").key, "sort-1x31x31-descending")
        for text in ("x:1,2,1", "sort:1,2,1", "topk:0,1,1", "topk:1,2,0", "topk:1,2,3",
                     "topk:1,65537,1", "topk:65536,65536,1", "topk:1,2", "topk:1,2,1:extra"):
            with self.subTest(text=text), self.assertRaises(argparse.ArgumentTypeError):
                rank.parse_case(text)
        for case in (("topk", True, 2, 1), ("sort", 1, 2, 1), ("topk", 1, 2, 1, "bad")):
            with self.subTest(case=case), self.assertRaises(ValueError):
                rank.Case(*case)

    def test_native_routes(self):
        for route in rank.NATIVE_ROUTES:
            name, binary = rank.parse_native(route + "=some-binary")
            self.assertEqual(name, route)
            self.assertTrue(Path(binary).is_absolute())
        for text in ("cuda=a", "xir-simd=", "simd", "torch-cpu=a"):
            with self.subTest(text=text), self.assertRaises(argparse.ArgumentTypeError):
                rank.parse_native(text)

    def test_fixture_and_all_default_cases(self):
        self.assertEqual(rank.fixture(rank.Case("topk", 2, 4, 1)), [-3.75, -2.25, -.75, .75, .5, 2., 3.5, -2.75])
        for text in rank.DEFAULT_CASES:
            initial = rank.parse_case(text)
            for direction in ("ascending", "descending"):
                case = rank.Case(initial.operation, initial.rows, initial.columns, initial.k, direction)
                inputs = rank.fixture(case)
                values, indices = expected(case, inputs)
                with self.subTest(case=case):
                    check = rank.validate_output(case, inputs, values, indices, stable=True)
                    self.assertEqual(check["elements"], case.rows * case.k)
                    self.assertTrue(check["stable_ties"])

    def test_arbitrary_topk_ties_but_stable_tile(self):
        case = rank.Case("topk", 1, 5, 3)
        inputs, values, indices = [9., 9., 9., 3., -1.], [9., 9., 9.], [2, 0, 1]
        self.assertFalse(rank.validate_output(case, inputs, values, indices, stable=False)["stable_ties"])
        with self.assertRaisesRegex(ValueError, "stable tie"):
            rank.validate_output(case, inputs, values, indices, stable=True)

    def test_threshold_tie_subset_is_legal(self):
        case = rank.Case("topk", 1, 5, 2)
        self.assertTrue(rank.validate_output(case, [5., 4., 4., 4., 1.], [5., 4.], [0, 3], stable=False)["topk_threshold"])

    def test_corrupt_indices_values_and_shapes_fail(self):
        case = rank.Case("topk", 1, 5, 3)
        inputs = [9., 9., 9., 3., -1.]
        bad = [([9., 9.], [0, 1]), ([9., 9., 9.], [0, 1, 1]),
               ([9., 9., 9.], [0, 1, -1]), ([9., 9., 9.], [0, 1, 5]),
               ([9., 9., 9.], [0, 1, 2.]), ([9., 9., 9.], [False, 1, 2]),
               ([9., 9., 3.], [0, 1, 2]), ([9., 9., 3.], [0, 1, 3]),
               ([float("nan"), 9., 9.], [0, 1, 2]), ([float("inf"), 9., 9.], [0, 1, 2]),
               ([True, 9., 9.], [0, 1, 2])]
        for values, indices in bad:
            with self.subTest(values=values, indices=indices), self.assertRaises(ValueError):
                rank.validate_output(case, inputs, values, indices, stable=False)
        for malformed in (inputs[:-1], [float("nan")] + inputs[1:]):
            with self.assertRaises(ValueError):
                rank.validate_output(case, malformed, [9.] * 3, [0, 1, 2], stable=False)
        with self.assertRaises(ValueError):
            rank.validate_output(case, inputs, [9.] * 3, [0, 1, 2], stable=1)

    def test_sorted_direction_not_just_value_membership(self):
        for direction in ("ascending", "descending"):
            case = rank.Case("sort", 1, 3, 3, direction)
            with self.subTest(direction=direction), self.assertRaisesRegex(ValueError, "incorrect order"):
                rank.validate_output(case, [3., 1., 2.], [2., 3., 1.], [2, 0, 1], stable=True)

    def test_typed_exports_and_lengths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "typed"
            for code, values in (("f", [1.25, -3.5]), ("q", [0, 2**40])):
                encoded = rank.typed_bytes(values, code)
                path.write_bytes(encoded)
                self.assertEqual(rank.read_typed(path, code, 2), values)
                for data in (encoded[:-1], encoded + b"\x00"):
                    path.write_bytes(data)
                    with self.assertRaisesRegex(ValueError, "byte count"):
                        rank.read_typed(path, code, 2)
            for code in ("d", "i", "x"):
                with self.subTest(code=code), self.assertRaises(ValueError):
                    rank.typed_bytes([1], code)
            for count in (-1, True, 1.5):
                with self.subTest(count=count), self.assertRaises(ValueError):
                    rank.read_typed(path, "q", count)


class RankingMetadataTimingTests(unittest.TestCase):
    def setUp(self):
        self.case = rank.Case("topk", 1, 31, 1)

    def test_metadata_and_strict_types(self):
        rank.check_native_metadata(measurement(self.case), self.case)
        mutations = dict(status="error", operation="sort", dimensions=[True, 31, 1], direction="ascending",
                         precision="fp16", index_dtype="int32", stable_ties=1, input_shape=[1, 30],
                         output_shape=[1, 2], algorithm="")
        for key, value in mutations.items():
            changed = measurement(self.case)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                rank.check_native_metadata(changed, self.case)
        for value in ([], None, 1):
            with self.assertRaises(ValueError):
                rank.check_native_metadata(value, self.case)

    def test_native_correctness_metadata_missing_corrupt_and_bool_types(self):
        base = measurement(self.case)
        for key, expected_value in base["correctness"].items():
            for value in (None, False, 0, 1, 1.0, "true", expected_value + 1):
                if type(value) is type(expected_value) and value == expected_value:
                    continue
                changed = deepcopy(base)
                if value is None:
                    del changed["correctness"][key]
                else:
                    changed["correctness"][key] = value
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "correctness metadata"):
                    rank.check_native_metadata(changed, self.case)
        del base["correctness"]
        with self.assertRaisesRegex(ValueError, "missing native correctness"):
            rank.check_native_metadata(base, self.case)

    def test_cpp_producer_source_contract_not_only_mock_schema(self):
        # This is a static cross-language contract check, not a native launch.
        # It deliberately reads the real producer, so a cpu/simd alias change
        # cannot leave mutually self-consistent Python mocks passing silently.
        root = Path(__file__).resolve().parents[3]
        source = (root / "src/tests/common/tile_rank_benchmark.h").read_text()
        fixture_source = (root / "src/tests/common/tile_rank_test_utils.h").read_text()
        self.assertRegex(source, r'<<\s*",\\"backend\\":"\s*<<\s*std::quoted\(backend\)')
        implementation = re.search(
            r'std::quoted\(backend == "metal" \? "([^"]+)" : backend == "metal4" \? "([^"]+)" :\s*"([^"]+)"\)', source)
        self.assertIsNotNone(implementation)
        actual = dict(zip(("metal", "metal4", "simd"), implementation.groups()))
        for route, backend in rank.NATIVE_ROUTES.items():
            self.assertEqual(actual[backend], "tile_" + route.replace("-", "_"))
        self.assertIn(r'\"correctness\":{\"checks\":2', source)
        self.assertIn(r'\"guard_elements_per_check\":', source)
        self.assertIn("6u * GuardedData::pad", source)
        self.assertRegex(fixture_source, r'pad\s*=\s*17u?')
        for key in ("values_bitwise_equal", "indices_exact", "input_immutable", "all_guards_intact"):
            self.assertIn('\\"' + key + '\\":true', source)

    def test_host_scope_and_samples(self):
        self.assertEqual(rank.timing_metrics(measurement(self.case), 2), dict(e2e_batch_us=3., e2e_single_us=6.))
        for key, value in (("timing", "pure_kernel"), ("repetitions", True), ("repetitions", 0),
                           ("repetitions", 100001), ("latency_us", [1.]), ("throughput_us", [1., float("nan")]),
                           ("throughput_us", [1., 0.]), ("throughput_us", [True, 2.]), ("device_timing", None)):
            changed = measurement(self.case)
            changed[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                rank.timing_metrics(changed, 2)

    def test_legacy_metal_control_is_not_instrumented_compute(self):
        changed = measurement(self.case)
        changed["device_timing"] = metal_device()
        metrics = rank.timing_metrics(changed, 2)
        self.assertEqual(metrics["gpu_cb_batch_us"], 1.5)
        self.assertEqual(metrics["gpu_instrumented_compute_batch_us"], .5)
        self.assertEqual(metrics["e2e_batch_us"], 3.)
        self.assertFalse(any("pure" in key for key in metrics))
        changed["device_timing"]["host_samples_instrumented"] = True
        with self.assertRaises(ValueError):
            rank.timing_metrics(changed, 2)

    def test_metal4_scopes_and_accounting(self):
        changed = measurement(self.case)
        changed["device_timing"] = metal4_device()
        metrics = rank.timing_metrics(changed, 2)
        self.assertEqual(metrics["gpu_instrumented_dispatch_batch_us"], 1.)
        self.assertAlmostEqual(metrics["gpu_cb_batch_us"], 2.)
        self.assertEqual(metrics["e2e_batch_us"], 3.)
        bad = []
        for key, value in (("scope", "pure_kernel"), ("host_samples_instrumented", True),
                           ("zero_overhead_kernel_time", True), ("repetitions", True)):
            mutated = deepcopy(changed)
            mutated["device_timing"][key] = value
            bad.append(mutated)
        mutated = deepcopy(changed)
        mutated["device_timing"]["throughput"][0]["dispatches"].pop()
        bad.append(mutated)
        mutated = deepcopy(changed)
        mutated["device_timing"]["throughput"][1]["sample_id"] = 1
        bad.append(mutated)
        mutated = deepcopy(changed)
        mutated["device_timing"]["latency"][0]["dispatches"][0]["shader_checksum"] = "5555"
        bad.append(mutated)
        mutated = deepcopy(changed)
        mutated["device_timing"]["control"]["throughput"][0]["dispatches"][0]["begin_ticks"] = 1
        bad.append(mutated)
        for index, mutated in enumerate(bad):
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                rank.timing_metrics(mutated, 2)


class RankingExecutionTests(unittest.TestCase):
    def test_native_cli(self):
        self.assertEqual(rank.native_command("binary", rank.Case("sort", 17, 65, 65, "ascending"), arguments(), Path("out")),
                         ["binary", "rank", "sort", "17", "65", "65", "ascending", "2", "3.0", "4.0", "out"])

    def test_plan_is_predeclared_and_balanced(self):
        cases = [rank.Case("topk", 1, 31, 1), rank.Case("sort", 17, 65, 65)]
        routes = ["xir-simd", "tirx-metal", "torch-cpu"]
        rows = rank.make_plan(cases, routes, 6)
        self.assertEqual(len(rows), 36)
        self.assertEqual({row["status"] for row in rows}, {"NotRun"})
        orders = [next(row["order"] for row in rows if row["round"] == iteration) for iteration in range(6)]
        for route in routes:
            self.assertEqual(Counter(order.index(route) for order in orders), {0: 2, 1: 2, 2: 2})
        for first in routes:
            for second in routes:
                if first != second:
                    self.assertEqual(sum(order.index(first) < order.index(second) for order in orders), 3)
        for bad_cases, bad_routes in ((cases + cases, routes), (cases, routes + routes), ([], routes), (cases, [])):
            with self.assertRaises(ValueError):
                rank.make_plan(bad_cases, bad_routes, 2)

    def test_errors_unavailable_and_interrupt_preserve_rows(self):
        rows = rank.make_plan([rank.Case("topk", 1, 31, 1)], ["native", "missing", "torch"], 1)
        visited, persisted = [], []

        def invoke(row):
            visited.append(row["route"])
            if row["route"] == "native":
                raise ValueError("compile failure")
            if row["route"] == "missing":
                raise rank.NotAvailable("no MPS")
            return dict(metrics=dict(e2e_batch_us=1.))

        self.assertTrue(rank.execute_rows(rows, invoke, lambda: persisted.append(1)))
        self.assertEqual(visited, ["native", "missing", "torch"])
        self.assertEqual([row["status"] for row in rows], ["Error", "NotRun", "OK"])
        self.assertEqual(len(persisted), 3)
        fresh = rank.make_plan([rank.Case("topk", 1, 31, 1)], ["native", "torch"], 1)
        with patch("ranking.native_visit", side_effect=KeyboardInterrupt):
            self.assertFalse(rank.execute_rows(fresh, lambda row: rank.native_visit(), lambda: None))
        self.assertEqual([row["status"] for row in fresh], ["Error", "NotRun"])

    def test_incomplete_summary_does_not_select_survivors(self):
        rows = rank.make_plan([rank.Case("topk", 1, 31, 1)], ["native", "torch"], 2)
        for row in rows:
            row.update(status="OK", metrics=dict(e2e_batch_us=1. + row["round"]))
        rows[0]["status"] = "Error"
        summary = {item["route"]: item for item in rank.make_summary(rows, 2)}
        self.assertFalse(summary["native"]["complete"])
        self.assertNotIn("median_us", summary["native"])
        self.assertEqual(summary["torch"]["median_us"], dict(e2e_batch_us=1.5))

    def native_fixture(self, directory, mutate=None, status="OK"):
        case = rank.Case("topk", 2, 31, 8)
        inputs = rank.fixture(case)
        prefix = directory / "output"
        values, indices = expected(case, inputs)
        result = measurement(case, prefix)
        Path(result["input_path"]).write_bytes(rank.typed_bytes(inputs, "f"))
        Path(result["values_path"]).write_bytes(rank.typed_bytes(values, "f"))
        Path(result["indices_path"]).write_bytes(rank.typed_bytes(indices, "q"))
        if mutate:
            mutate(result)

        def capture(command, environment, timeout, stdout, stderr):
            stdout.write_text(json.dumps(result))
            stderr.write_text("fixture stderr")
            return dict(status=status, exit_code=0 if status == "OK" else 1)

        return case, inputs, capture

    def test_native_complete_exports_and_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            case, inputs, capture = self.native_fixture(directory)
            with patch("metal4_timing.capture", side_effect=capture):
                result = rank.native_visit("binary", "xir-simd", case, inputs, arguments(), directory, {})
            self.assertTrue(result["correctness"]["stable_ties"])
            self.assertEqual(result["device_interval"]["status"], "NotRun")
            command = json.loads((directory / "command.json").read_text())
            self.assertEqual(command["environment"]["LUISA_TILE_BENCH_XIR_BACKEND"], "simd")

    def test_native_corrupt_artifacts_and_requested_missing_gpu_fail(self):
        mutations = [lambda result: result.update(operation="sort"),
                     lambda result: result.update(backend="metal"),
                     lambda result: result.update(implementation="tile_tirx_metal"),
                     lambda result: result.update(values_path="wrong-output"),
                     lambda result: Path(result["input_path"]).write_bytes(b"wrong input"),
                     lambda result: Path(result["indices_path"]).write_bytes(b"truncated"),
                     lambda result: Path(result["values_path"]).write_bytes(b""),
                     lambda result: Path(result["indices_path"]).write_bytes(rank.typed_bytes([0] * 16, "q"))]
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                case, inputs, capture = self.native_fixture(directory, mutation)
                with patch("metal4_timing.capture", side_effect=capture), self.assertRaises(ValueError):
                    rank.native_visit("binary", "xir-simd", case, inputs, arguments(), directory, {})
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            case, inputs, capture = self.native_fixture(directory, lambda result: result.update(backend="metal4", implementation="tile_xir_metal4"))
            args = arguments()
            args.metal4_device_timing = True
            with patch("metal4_timing.capture", side_effect=capture), self.assertRaisesRegex(ValueError, "not reported"):
                rank.native_visit("binary", "xir-metal4", case, inputs, args, directory, {})

    def test_native_failure_and_timeout_keep_logs(self):
        for status in ("Error", "Timeout", "Interrupted"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                case, inputs, capture = self.native_fixture(directory, status=status)
                with patch("metal4_timing.capture", side_effect=capture), self.assertRaises(KeyboardInterrupt if status == "Interrupted" else ValueError):
                    rank.native_visit("binary", "xir-simd", case, inputs, arguments(), directory, {})
                self.assertTrue((directory / "stdout.log").is_file())
                self.assertEqual((directory / "stderr.log").read_text(), "fixture stderr")
                self.assertEqual(json.loads((directory / "process.json").read_text())["status"], status)

    def test_native_oserror_is_retained_in_reason_and_process_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            case, inputs, capture_fixture = self.native_fixture(directory, status="Error")

            def capture(*args):
                result = capture_fixture(*args)
                result.update(exit_code=None, error="[Errno 2] No such file or directory: missing-binary")
                return result

            with patch("metal4_timing.capture", side_effect=capture), self.assertRaisesRegex(ValueError, "No such file or directory: missing-binary"):
                rank.native_visit("missing-binary", "xir-simd", case, inputs, arguments(), directory, {})
            result = json.loads((directory / "process.json").read_text())
            self.assertIsNone(result["exit_code"])
            self.assertIn("missing-binary", result["error"])

    def test_main_native_failure_cannot_skip_torch_and_keeps_matrix(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ):
            output = Path(temporary) / "matrix"
            seen = []

            def torch_visit(device, case, inputs, args, directory):
                seen.append((device, case.key, len(inputs)))
                return dict(metrics=dict(e2e_batch_us=2., e2e_single_us=3.))

            with patch("ranking.native_visit", side_effect=ValueError("native failed")), patch("ranking.torch_visit", side_effect=torch_visit), redirect_stdout(io.StringIO()):
                code = rank.main(["--native", "xir-simd=/missing", "--torch", "cpu", "--case", "topk:1,31,1",
                                  "--rounds", "1", "--direction", "descending", "--output", str(output)])
            self.assertEqual(code, 1)
            result = json.loads((output / "results.json").read_text())
            self.assertEqual(result["counts"], {"Error": 1, "OK": 1})
            self.assertEqual(len(seen), 1)
            self.assertTrue(result["source_unchanged"])
            self.assertEqual(result["fixed_environment"]["PYTORCH_ENABLE_MPS_FALLBACK"], "0")
            self.assertIn("not pure kernel", result["timing_scope"])
            self.assertIn("no full dynamic-library/build closure attestation", result["identity_coverage"])
            self.assertEqual((output / "topk-1x31x1-descending.input.f32").stat().st_size, 31 * 4)

    def test_cli_rejects_duplicates_invalid_bounds_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "not-created"
            for extras in (("--torch", "cpu", "--torch", "cpu"), ("--torch", "cpu", "--samples", "0"),
                           ("--torch", "cpu", "--samples", "102"), ("--torch", "cpu", "--sample-ms", "10001"),
                           ("--torch", "cpu", "--warmup-ms", "60001"),
                           ("--torch", "cpu", "--sample-ms", "nan"), ("--torch", "cpu", "--rounds", "-1"), ()):
                with self.subTest(extras=extras), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    rank.main(["--output", str(output), *extras])
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
