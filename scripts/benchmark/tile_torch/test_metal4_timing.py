"""CPU-only protocol tests: no Metal device, native binary, or build required."""

import argparse
import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import compare_llm
import metal4_timing as timing


def sample_record(sample_id, count, counters):
    return dict(
        error="", overflow=False, dispatch_timestamps_enabled=counters,
        timestamp_frequency_hz=10**9 if counters else 0, sample_id=sample_id,
        command_buffers=[dict(ordinal=0, dispatch_count=count, contains_non_dispatch_work=False,
                             valid=True, gpu_begin_seconds=1., gpu_end_seconds=1. + count * 1e-6,
                             host_commit_begin_ns=10, host_commit_return_ns=11,
                             host_feedback_begin_ns=20, host_callbacks_end_ns=30, host_completion_publish_ns=40)],
        dispatches=[dict(ordinal=i, command_buffer_ordinal=0, dispatch_size=[32, 1, 1], block_size=[32, 1, 1],
                         shader_checksum="123", valid=counters,
                         begin_ticks=100 + 200 * i if counters else 0,
                         end_ticks=200 + 200 * i if counters else 0,
                         elapsed_ns=100. if counters else 0.) for i in range(count)])


def payload_for(op="rope", dims=(2, 4), block=(1, 1), samples=2, repetitions=8):
    inputs, output = timing.shapes_for(op, dims)
    count = min(repetitions, 64)
    payload = dict(
        implementation="tile_xir_metal4", backend="metal4", operation=op, dimensions=list(dims),
        precision="fp32", fast_math=False, relaxed_precision=False, runtime="luisa", repetitions=repetitions,
        repetition_policy="fixed", timing="synchronized_host_wall", batch_policy="one_runtime_command_list_per_batch",
        attention_block=list(block), attention_qk="mma" if op == "attention" else "not_applicable",
        attention_pv="mma" if op == "attention" else "not_applicable",
        input_shapes=[list(s) for s in inputs], output_shape=list(output),
        realization="test-only; local_lanes=1", dispatch=[32, 1, 1],
        correctness=dict(checks=2, elements_per_check=timing.math.prod(output), guard_elements_per_check=34,
                         max_abs_error=0., atol=5e-5, rtol=5e-5),
        throughput_us=[2.] * samples, latency_us=[3.] * samples)
    payload["device_timing"] = dict(
        method="metal4_precise_dispatch_timestamps_v1", scope="instrumented_dispatch_intervals",
        host_samples_instrumented=False, zero_overhead_kernel_time=False, repetitions=count,
        capabilities=dict(timestamp_frequency_hz=10**9, timestamp_heap=True),
        throughput=[sample_record(1 + i, count, True) for i in range(samples)],
        latency=[sample_record(1 + samples + i, 1, True) for i in range(samples)],
        control=dict(method="metal4_commit_feedback_v1", scope="command_buffer_gpu_intervals",
                     encoder_instrumentation=False, repetitions=count,
                     throughput=[sample_record(1 + 2 * samples + i, count, False) for i in range(samples)],
                     latency=[sample_record(1 + 3 * samples + i, 1, False) for i in range(samples)]))
    return payload


def write_exports(path, op, dims, arrays=None):
    import numpy as np
    inputs, _ = timing.shapes_for(op, dims)
    if arrays is None:
        arrays = [np.full(s, .25, np.float32) for s in inputs]
    for index, value in enumerate(arrays):
        value.astype(np.float32).tofile(str(path) + f".input{index}.f32")
    expected = timing.reference(op, dims, arrays)
    expected.astype(np.float32).tofile(path)
    return expected


class Metal4AttentionProtocolTests(unittest.TestCase):
    def test_reuses_shared_contract_without_loading_torch(self):
        self.assertIs(timing.parse_case, compare_llm.parse_case)
        self.assertIs(timing.shapes_for, compare_llm.shapes_for)
        self.assertIs(timing.reference, compare_llm.reference)
        self.assertIs(timing.gpu_failure_diagnostics, compare_llm.gpu_failure_diagnostics)

    def test_attention_dimensions_and_row_regression(self):
        for text in ("attention:1,8,8,1,2048,64,64", "attention:2,6,2,3,5,7,9",
                     "attention:1,4,1,5,5,3,7", "rope:17,258", "rmsnorm:128,1024"):
            with self.subTest(text=text):
                op, dims = timing.parse_case(text)
                self.assertEqual(len(dims), 7 if op == "attention" else 2)
        for text in ("attention:1,2,1,0,3,4,5", "attention:1,3,2,1,3,4,5",
                     "attention:1,2,1,4,3,4,5", "attention:1,2,1,1,3,4",
                     "attention:1,2,1,1,3,4,5,6", "attention:1,2,1,1,3,4,-5",
                     "attention:65536,2,1,1,65536,4,5", "attention:1,1,1,65536,65536,1,1",
                     "rope:1,3", "swiglu:65536,65536", "unknown:1,2"):
            with self.subTest(text=text), self.assertRaises(argparse.ArgumentTypeError):
                timing.parse_case(text)
        self.assertEqual(timing.shapes_for("attention", (2, 6, 2, 3, 5, 7, 9)),
                         ([(2, 6, 3, 7), (2, 2, 5, 7), (2, 2, 5, 9)], (2, 6, 3, 9)))
        self.assertEqual(timing.shapes_for("layernorm", (3, 9)), ([(3, 9), (1, 9), (1, 9)], (3, 9)))

    def test_parser_defaults_and_attention_block_limits(self):
        common = ["--binary", "unused", "--output", "unused"]
        with patch("metal4_timing.subprocess.Popen") as spawn:
            args = timing.parse_arguments(common)
            self.assertEqual(args.case, [(op, (128, 1024)) for op in ("rmsnorm", "masked_softmax", "swiglu")])
            self.assertEqual(args.local_lanes, [1, 32, 0])
            self.assertEqual(args.attention_block, (16, 32))
            args = timing.parse_arguments(common + ["--case", "attention:1,2,1,3,5,7,9", "--attention-block", "3", "7"])
            self.assertEqual(args.attention_block, [3, 7])
            spawn.assert_not_called()
        for block in ((0, 1), (129, 1), (1, 0), (1, 257)):
            with self.subTest(block=block), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                timing.parse_arguments(common + ["--attention-block", *map(str, block)])

    def test_metadata_requires_actual_attention_shapes_block_and_modes(self):
        dims, block = (2, 6, 2, 3, 5, 7, 9), (3, 7)
        payload = payload_for("attention", dims, block)
        timing.validate(payload, "attention", dims, 2, 8, block)
        mutations = [("attention_block", [1, 1]), ("attention_qk", "reduce"), ("attention_pv", "reduce"),
                     ("attention_qk", None), ("output_shape", list(dims)), ("input_shapes", [[1], [1], [1]]),
                     ("dimensions", [True, *dims[1:]]), ("fast_math", 0), ("repetitions", 8.),
                     ("backend", "metal"), ("implementation", "tile_xir_simd")]
        for key, value in mutations:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                timing.validate(payload | {key: value}, "attention", dims, 2, 8, block)
        for key in ("attention_qk", "attention_pv"):
            bad = copy.deepcopy(payload)
            del bad[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                timing.validate(bad, "attention", dims, 2, 8, block)
        for key, value in (("elements_per_check", timing.math.prod(dims)), ("guard_elements_per_check", 0),
                           ("max_abs_error", float("nan")), ("max_abs_error", -1.), ("atol", .001)):
            bad = copy.deepcopy(payload)
            bad["correctness"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                timing.validate(bad, "attention", dims, 2, 8, block)
        legacy_row = payload_for()
        del legacy_row["attention_qk"], legacy_row["attention_pv"]
        timing.validate(legacy_row, "rope", (2, 4), 2, 8)

    def test_bottom_right_mask_triangular_decode_ragged_and_gqa(self):
        import numpy as np
        for queries in (1, 2, 3):
            dims = (1, 4, 2, queries, 3, 1, 1)
            arrays = [np.zeros((1, 4, queries, 1), np.float32), np.zeros((1, 2, 3, 1), np.float32),
                      np.array([[[[1], [3], [8]], [[2], [6], [10]]]], np.float32)]
            expected = timing.reference("attention", dims, arrays)
            means = ([1, 2, 4], [2, 4, 6])
            for head in range(4):
                np.testing.assert_allclose(expected[0, head, :, 0], means[head // 2][3 - queries:])
            with self.subTest(queries=queries), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "result.f32"
                write_exports(path, "attention", dims, arrays)
                self.assertEqual(timing.validate_exports(path, "attention", dims)["elements"], 4 * queries)

    def test_complete_exports_reject_tail_corruption_nonfinite_and_wrong_size(self):
        import numpy as np
        dims = (2, 6, 2, 3, 5, 7, 9)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.f32"
            write_exports(path, "attention", dims)
            self.assertEqual(path.stat().st_size, 2 * 6 * 3 * 9 * 4)
            timing.validate_exports(path, "attention", dims)
            # Corrupt the last element, not a sampled prefix.
            for index in range(4):
                for mode in ("nan", "extra", "truncated", "finite_wrong"):
                    if mode == "finite_wrong" and index != 3:
                        continue
                    write_exports(path, "attention", dims)
                    target = path if index == 3 else Path(str(path) + f".input{index}.f32")
                    data = np.fromfile(target, np.float32)
                    if mode == "nan":
                        data[-1] = np.nan
                    elif mode == "extra":
                        data = np.append(data, np.float32(0))
                    elif mode == "truncated":
                        data = data[:-1]
                    else:
                        data[-1] += 1
                    data.tofile(target)
                    with self.subTest(index=index, mode=mode), self.assertRaises(ValueError):
                        timing.validate_exports(path, "attention", dims)
            for op, row_dims in (("rope", (2, 4)), ("rmsnorm", (3, 5)), ("layernorm", (3, 5)),
                                 ("masked_softmax", (7, 5)), ("swiglu", (3, 5)), ("gelu_residual", (3, 5))):
                write_exports(path, op, row_dims)
                self.assertEqual(timing.validate_exports(path, op, row_dims)["elements"], timing.math.prod(row_dims))


class Metal4TimingAccountingTests(unittest.TestCase):
    def test_local_lanes_are_unique_and_acknowledge_explicit_requests(self):
        for lanes in (1, 32, 64, 128, 65536):
            for request in (0, lanes):
                with self.subTest(lanes=lanes, request=request):
                    self.assertEqual(timing.validate_local_lanes(f"backend; local_lanes={lanes}; suffix", request), lanes)
        for realization, request in (("backend", 0), ("local_lanes=1; local_lanes=1", 1),
                                     ("local_lanes=1; local_lanes=32", 0), ("local_lanes=0", 0),
                                     ("local_lanes=-1", 0), ("local_lanes=1.0", 0), ("local_lanes", 0),
                                     ("local_lanes=auto", 0), ("local_lanes=32", 64),
                                     ("local_lanes=4294967296", 0), ("local_lanes=1", True)):
            with self.subTest(realization=realization, request=request), self.assertRaises(ValueError):
                timing.validate_local_lanes(realization, request)

    def test_consistent_sample_geometry_must_match_benchmark_dispatch(self):
        payload = payload_for()
        timing.validate(payload, "rope", (2, 4), 2, 8)
        for dispatch in ([64, 1, 1], [32, 1], [32, 1, 0], [32, True, 1], [32., 1, 1]):
            with self.subTest(dispatch=dispatch), self.assertRaises(ValueError):
                timing.validate(payload | {"dispatch": dispatch}, "rope", (2, 4), 2, 8)
        # Both counter and feedback arms independently bind to the top-level
        # dispatch; matching each other alone is insufficient.
        for control in (False, True):
            bad = copy.deepcopy(payload)
            parent = bad["device_timing"]["control"] if control else bad["device_timing"]
            parent["throughput"][0]["dispatches"][0]["dispatch_size"] = [64, 1, 1]
            with self.subTest(control=control), self.assertRaisesRegex(ValueError, "benchmark dispatch"):
                timing.validate(bad, "rope", (2, 4), 2, 8)

    def test_three_metrics_keep_their_own_denominators(self):
        payload = payload_for(repetitions=100)
        metrics = timing.validate(payload, "rope", (2, 4), 2, 100)
        self.assertEqual(metrics["throughput_instrumented_dispatch_ns"]["median"], 100.)
        self.assertAlmostEqual(metrics["throughput_feedback_only_command_buffer_ns_per_dispatch"]["median"], 1000., places=5)
        self.assertEqual(metrics["throughput_host_wall_us_per_dispatch"]["median"], 2.)
        self.assertAlmostEqual(metrics["latency_feedback_only_command_buffer_ns_per_dispatch"]["median"], 1000., places=5)
        self.assertEqual(metrics["latency_host_wall_us_per_dispatch"]["median"], 3.)
        self.assertEqual(len(payload["device_timing"]["throughput"][0]["dispatches"]), 64)

    def test_multiple_command_buffers_sum_then_divide_by_total_dispatches(self):
        sample = sample_record(1, 4, False)
        first = sample["command_buffers"][0]
        first.update(dispatch_count=1, gpu_end_seconds=1. + 1e-6)
        second = copy.deepcopy(first)
        second.update(ordinal=5, dispatch_count=3, gpu_end_seconds=1. + 9e-6)
        sample["command_buffers"].append(second)
        for dispatch in sample["dispatches"][1:]:
            dispatch["command_buffer_ordinal"] = 5
        self.assertAlmostEqual(timing.validate_sample(sample, 4, False, 10**9), 2500., places=5)
        with self.assertRaises(ValueError):
            timing.validate_sample(sample, 0, False, 10**9)

    def test_counter_clock_count_mode_and_identity_denials(self):
        good = payload_for()
        mutations = [lambda p: p["device_timing"].update(repetitions=7),
                     lambda p: p["device_timing"]["control"].update(repetitions=7),
                     lambda p: p["device_timing"]["capabilities"].update(timestamp_frequency_hz=0),
                     lambda p: p["device_timing"]["throughput"][0].update(overflow=True),
                     lambda p: p["device_timing"]["throughput"][0]["dispatches"][0].update(elapsed_ns=99.),
                     lambda p: p["device_timing"]["control"]["throughput"][0].update(dispatch_timestamps_enabled=True),
                     lambda p: p["device_timing"]["latency"][0].update(sample_id=1),
                     lambda p: p["device_timing"]["latency"][0]["dispatches"][0].update(shader_checksum="456"),
                     lambda p: p.update(throughput_us=[float("nan"), 1.])]
        for index, mutate in enumerate(mutations):
            bad = copy.deepcopy(good)
            mutate(bad)
            with self.subTest(index=index), self.assertRaises(ValueError):
                timing.validate(bad, "rope", (2, 4), 2, 8)


class Metal4FailureStopTests(unittest.TestCase):
    def test_known_driver_diagnostics_are_shared_and_not_normal_json(self):
        for diagnostic in ("GPUHangError", "GPU_Hang_Error", "GPU-Hang-Error", "MTLCommandBufferErrorDomain Code=2",
                           "execution of the command buffer was aborted", "GPU Address Fault Error"):
            self.assertTrue(timing.gpu_failure_diagnostics("", diagnostic))
        self.assertEqual(timing.gpu_failure_diagnostics('{"backend":"metal4","max_abs_error":0,"error":""}', ""), [])

    def test_failure_stops_later_visits_and_invalidates_prior_success(self):
        for failure in ("GPUHangError", "Timeout"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                binary = root / "benchmark_tile_xir"
                binary.write_bytes(b"mocked")
                binary.chmod(0o700)
                (root / "libluisa-backend-metal4.dylib").write_bytes(b"mocked")
                tensors = root / "tensors"
                tensors.mkdir()
                args = timing.parse_arguments(["--binary", str(binary), "--output", str(root / "report"),
                                               "--case", "attention:1,2,1,2,3,1,1", "--attention-block", "2", "3",
                                               "--samples", "1", "--repetitions", "2", "--rounds", "1"])
                calls = []

                def capture(command, environment, timeout, stdout, stderr):
                    calls.append(command)
                    self.assertEqual(command[4:6], ["2", "3"])
                    payload = payload_for("attention", (1, 2, 1, 2, 3, 1, 1), (2, 3), 1, 2)
                    write_exports(Path(command[-1]), "attention", (1, 2, 1, 2, 3, 1, 1))
                    stdout.write_text(json.dumps(payload))
                    stderr.write_text("GPUHangError (0x3)" if len(calls) == 2 and failure == "GPUHangError" else "")
                    return dict(status="Timeout", exit_code=-9, error="TimeoutExpired") if len(calls) == 2 and failure == "Timeout" else dict(status="OK", exit_code=0)

                with patch("metal4_timing.parse_arguments", return_value=args), \
                        patch("metal4_timing.tempfile.mkdtemp", return_value=str(tensors)), \
                        patch("metal4_timing.capture", side_effect=capture), redirect_stdout(io.StringIO()):
                    self.assertEqual(timing.main(), 1)
                report = json.loads((root / "report" / "results.json").read_text())
                self.assertEqual(len(calls), 2)
                self.assertEqual([r["status"] for r in report["results"]], ["OK", "Timeout" if failure == "Timeout" else "Error", "NotRun"])
                self.assertFalse(report["cohort_valid"])
                self.assertFalse(report["gpu_diagnostics_valid"])
                self.assertTrue(all(not r["valid"] for r in report["results"]))
                self.assertIn("not launched:", report["results"][-1]["error"])
                self.assertIn("metrics", report["results"][0])  # Retained but not accepted as a cohort.
                self.assertEqual(report["results"][0]["actual_local_lanes"], 1)
                self.assertNotIn("metrics", report["results"][1])
                self.assertEqual(len(list((root / "report").glob("*.stderr.log"))), 2)


if __name__ == "__main__":
    unittest.main()
