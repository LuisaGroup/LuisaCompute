"""Host-only tests for workload identity, receipts and failure classification."""
import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import contextlib
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import cuda_matrix

from cuda_matrix import main, native_result, route_environment, tensor_receipts, validate_case, worker_warps_receipts, structural_receipts


class CudaMatrixTests(unittest.TestCase):
    @staticmethod
    def case(**updates):
        row = dict(id="topk-fixture", operation="topk", dimensions=[1, 4, 2],
                   tile=[1, 4, 1], precision="fp32", seed=19, pattern="adversarial")
        row.update(updates)
        return row

    def read_native(self, case, status="passed", returncode=0, reason="", requested_worker_warps=0, source_text=None,
                    requested_scan_chunk=0, requested_independent_axis=0, **updates):
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
            if source_text is not None:
                (path.parent / "source.txt").write_text(source_text, encoding="utf-8")
            original = path.read_bytes()
            result = native_result(dict(status="exited", returncode=returncode), path, case,
                                   SimpleNamespace(samples=3, graph_batch=4, native_worker_warps=requested_worker_warps,
                                                   native_scan_chunk=requested_scan_chunk, native_independent_axis=requested_independent_axis), "native")
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

    def test_native_single_source_receipt_freezes_actual_exported_bytes(self):
        first = self.read_native(self.case(), source_text="original kernel")
        changed = self.read_native(self.case(), source_text="changed kernel")
        self.assertEqual(set(first["generated_sources"]), {"source.txt"})
        self.assertNotEqual(first["generated_sources"], changed["generated_sources"])
        self.assertEqual(self.read_native(self.case())["generated_sources"], {})

    def test_worker_hint_environment_is_explicit_and_native_only(self):
        inherited = {"LUISA_CUDA_TILE_WORKER_WARPS": "8", "LUISA_CUDA_TILE_IR": "1", "PATH": "preserve"}
        original = dict(inherited)
        for route in ("native", "tirx", "simd", "torch"):
            for request in (0, 4, 8):
                with self.subTest(route=route, request=request):
                    env = route_environment(inherited, route, native_worker_warps=request)
                    self.assertEqual(env.get("LUISA_CUDA_TILE_WORKER_WARPS"), str(request) if route == "native" and request else None)
                    self.assertEqual(env["PATH"], "preserve")
        self.assertEqual(inherited, original)
        for invalid in (True, -1, 1, 2, 3, 16, "4"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "invalid native worker"):
                route_environment(inherited, "native", native_worker_warps=invalid)

    def test_worker_hint_cli_rejects_invalid_choices_without_execution(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--list-cases", "--native-worker-warps", "4"]), 0)
            for value in ("-1", "1", "2", "3", "16", "true"):
                with self.subTest(value=value), self.assertRaises(SystemExit) as error:
                    main(["--list-cases", "--native-worker-warps", value])
                self.assertEqual(error.exception.code, 2)

    def test_worker_hint_native_packet_requires_exact_requested_marker(self):
        case = self.case()
        for request in (4, 8):
            result = self.read_native(case, requested_worker_warps=request, realization=f"native Tile; worker-warps-hint={request};")
            self.assertEqual(result["native_worker_warps"], [dict(stage=0, requested=request, reported_hint=request)])
        for marker, request in (("", 4), ("worker-warps-hint=8", 4), ("worker-warps-hint=4", 0),
                                ("worker-warps-hint=4; worker-warps-hint=4", 4), ("worker-warps-hint=4junk", 4),
                                ("worker-warps-hint=0", 0)):
            with self.subTest(marker=marker, request=request), self.assertRaisesRegex(ValueError, "worker-warps request/realization"):
                self.read_native(case, requested_worker_warps=request, realization=marker)

    def test_worker_hint_each_pipeline_stage_must_agree(self):
        result = dict(pipeline_stages=[dict(realization="Tile; worker-warps-hint=4;") for _ in range(3)])
        self.assertEqual([r["reported_hint"] for r in worker_warps_receipts(result, 4)], [4, 4, 4])
        result["pipeline_stages"][1]["realization"] = "Tile default"
        with self.assertRaisesRegex(ValueError, "worker-warps request/realization"):
            worker_warps_receipts(result, 4)
        self.assertEqual(worker_warps_receipts(dict(realization="historical native Tile"), 0),
                         [dict(stage=0, requested=0, reported_hint=None)])

    def test_native_only_is_an_explicit_native_calibration_route(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--list-cases", "--routes", "native", "--native-only", "--native-worker-warps", "8"]), 0)
            for routes in ("native,tirx", "simd", "tirx"):
                with self.subTest(routes=routes), self.assertRaisesRegex(ValueError, "native-only requires"):
                    main(["--list-cases", "--routes", routes, "--native-only"])

    def test_native_only_never_creates_a_torch_child_or_comparison(self):
        # Exercise main's actual child dispatch and completion path. Only process
        # creation, machine inventory and packet validation are replaced by mocks.
        class Telemetry:
            pid = 7
            _handle = 7
            stopped = False
            def poll(self):
                return 0 if self.stopped else None
            def kill(self):
                self.stopped = True
            def wait(self, timeout=None):
                return 0

        cpu = SimpleNamespace(CREATE_SUSPENDED=4,
            topology=lambda: dict(caller_affinity=dict(system_mask="0xf"), records=[dict(group=0, logical_processor=i, core_index=i) for i in range(4)]),
            OwnedJob=lambda: SimpleNamespace(attach=lambda p: None, close=lambda: None),
            set_owned_affinity=lambda handle, mask: {}, resume_owned_primary_thread=lambda p: 1)
        launches = []
        def child(cpu, command, work, environment, mask, timeout):
            launches.append(dict(command=command, environment=environment))
            (work / "artifacts").mkdir(parents=True)
            (work / "artifacts/manifest.json").write_text("{}", encoding="utf-8")
            return dict(status="exited", returncode=0)
        native = dict(status="passed", generated_sources={"source.txt": "mock-sha"}, result=dict(compile_ms=1, cold_ms=1, host_wall_p50_us=1,
                      realization="Tile; worker-warps-hint=4", cuda_event_stream_span_us=[], graph_event_stream_span_us_per_op=[]))
        with tempfile.TemporaryDirectory() as temporary, contextlib.ExitStack() as stack:
            root = Path(temporary)
            (root / "bin").mkdir()
            (root / "bin/benchmark_tile_workloads.exe").write_bytes(b"test")
            (root / "CMakeCache.txt").write_text("test", encoding="utf-8")
            marker = root / "marker.json"
            marker.write_text("{}", encoding="utf-8")
            stack.enter_context(mock.patch.dict("sys.modules", {"windows_affinity": cpu}))
            for name, replacement in (("selected_cases", lambda args: [self.case()]), ("child", child),
                                      ("native_result", lambda *args: copy.deepcopy(native)),
                                      ("tensor_receipts", lambda *args: ({}, {"input": "same"})),
                                      ("digest", lambda path: "mock-sha")):
                stack.enter_context(mock.patch.object(cuda_matrix, name, replacement))
            stack.enter_context(mock.patch.object(cuda_matrix.shutil, "which", return_value="mock-nvidia-smi"))
            stack.enter_context(mock.patch.object(cuda_matrix.subprocess, "Popen", return_value=Telemetry()))
            stack.enter_context(redirect_stdout(io.StringIO()))
            code = main(["--routes", "native", "--native-only", "--native-worker-warps", "4", "--native-scan-chunk", "1024",
                         "--build-dir", str(root), "--build-marker", str(marker),
                         "--output", str(root / "results"), "--affinity-mask", "0xf"])
            result = json.loads((root / "results/results.json").read_text())
        self.assertEqual(code, 0)
        self.assertEqual(len(launches), 1)
        self.assertEqual(launches[0]["environment"]["LUISA_CUDA_TILE_WORKER_WARPS"], "4")
        self.assertEqual(launches[0]["environment"]["LUISA_CUDA_TILE_SCAN_CHUNK"], "1024")
        self.assertNotIn("LUISA_CUDA_TILE_INDEPENDENT_AXIS", launches[0]["environment"])
        self.assertEqual(result["options"]["native_scan_chunk"], 1024)
        self.assertFalse(result["torch_requested"])
        self.assertEqual(result["status"], "passed")
        item = result["cases"][0]
        self.assertEqual(item["runs"]["native"]["native_scan_chunk_requested"], 1024)
        self.assertEqual(item["runs"]["native"]["native_independent_axis_requested"], 0)
        self.assertEqual(item["runs"]["native"]["environment_overrides"]["LUISA_CUDA_TILE_SCAN_CHUNK"], "1024")
        self.assertEqual(item["runs"]["torch"]["native_scan_chunk_requested"], 0)
        self.assertEqual(item["runs"]["torch"]["native_independent_axis_requested"], 0)
        self.assertEqual(item["runs"]["torch"]["status"], "not_requested")
        self.assertNotIn("process", item["runs"]["torch"])
        self.assertEqual(item["comparison_status"], "absent_native_only_calibration")

    def test_structural_environment_is_native_only_and_does_not_mutate_parent(self):
        inherited = {"LUISA_CUDA_TILE_SCAN_CHUNK": "2048", "LUISA_CUDA_TILE_INDEPENDENT_AXIS": "4", "PATH": "keep"}
        original = dict(inherited)
        for route in ("native", "tirx", "simd", "torch"):
            for scan, axis in ((0, 0), (1024, 0), (2048, 0), (0, 1), (0, 2), (0, 4)):
                with self.subTest(route=route, scan=scan, axis=axis):
                    env = route_environment(inherited, route, native_scan_chunk=scan, native_independent_axis=axis)
                    self.assertEqual(env.get("LUISA_CUDA_TILE_SCAN_CHUNK"), str(scan) if route == "native" and scan else None)
                    self.assertEqual(env.get("LUISA_CUDA_TILE_INDEPENDENT_AXIS"), str(axis) if route == "native" and axis else None)
                    self.assertEqual(env["PATH"], "keep")
        self.assertEqual(inherited, original)
        for scan, axis, message in ((True, 0, "invalid native scan"), (512, 0, "invalid native scan"),
                                   ("1024", 0, "invalid native scan"), (0, True, "invalid native independent"),
                                   (0, 3, "invalid native independent"), (0, 8, "invalid native independent"),
                                   (1024, 2, "mutually exclusive")):
            with self.subTest(scan=scan, axis=axis), self.assertRaisesRegex(ValueError, message):
                route_environment(inherited, "native", native_scan_chunk=scan, native_independent_axis=axis)

    def test_structural_cli_validates_choices_and_exclusion_before_execution(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            for scan, axis in ((0, 0), (1024, 0), (2048, 0), (0, 1), (0, 2), (0, 4)):
                self.assertEqual(main(["--list-cases", "--native-scan-chunk", str(scan), "--native-independent-axis", str(axis)]), 0)
            for option, value in (("--native-scan-chunk", "512"), ("--native-scan-chunk", "-1"),
                                  ("--native-independent-axis", "3"), ("--native-independent-axis", "8")):
                with self.subTest(option=option, value=value), self.assertRaises(SystemExit) as error:
                    main(["--list-cases", option, value])
                self.assertEqual(error.exception.code, 2)
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                main(["--list-cases", "--native-scan-chunk", "1024", "--native-independent-axis", "1"])

    def test_structural_packet_requires_actual_rewrites(self):
        scan = self.read_native(self.case(), requested_scan_chunk=1024,
                                realization="Tile; scan-chunk=1024; chunked-scans=2;")["native_structure"][0]
        self.assertEqual((scan["scan_chunk"], scan["chunked_scans"], scan["independent_axis_extent"], scan["partitioned_collectives"]),
                         (1024, 2, None, 0))
        axis = self.read_native(self.case(), requested_independent_axis=1,
                                realization="Tile; independent-axis-extent=1; partitioned-collectives=3;")["native_structure"][0]
        self.assertEqual((axis["independent_axis_extent"], axis["partitioned_collectives"], axis["scan_chunk"], axis["chunked_scans"]),
                         (1, 3, None, 0))
        legacy = self.read_native(self.case())["native_structure"][0]
        self.assertEqual((legacy["scan_chunk_requested"], legacy["independent_axis_requested"],
                          legacy["chunked_scans"], legacy["partitioned_collectives"]), (0, 0, 0, 0))

    def test_structural_packet_rejects_noop_wrong_duplicate_and_stale_markers(self):
        bad = [
            (1024, 0, ""), (1024, 0, "scan-chunk=2048; chunked-scans=1"),
            (1024, 0, "scan-chunk=1024; chunked-scans=0"), (1024, 0, "scan-chunk=1024; chunked-scans=-1"),
            (1024, 0, "scan-chunk=1024; chunked-scans=1junk"),
            (1024, 0, "scan-chunk=1024; chunked-scans=1; chunked-scans=2"),
            (1024, 0, "scan-chunk=1024; chunked-scans=1; independent-axis-extent=1; partitioned-collectives=1"),
            (0, 2, "independent-axis-extent=1; partitioned-collectives=1"),
            (0, 2, "independent-axis-extent=2; partitioned-collectives=0"),
            (0, 2, "independent-axis-extent=2"), (0, 2, "partitioned-collectives=2"),
            (0, 0, "scan-chunk=0"), (0, 0, "chunked-scans=0"),
            (0, 0, "independent-axis-extent=0"), (0, 0, "partitioned-collectives=0"),
        ]
        for scan, axis, marker in bad:
            with self.subTest(scan=scan, axis=axis, marker=marker), self.assertRaisesRegex(ValueError, "native structural"):
                self.read_native(self.case(), requested_scan_chunk=scan, requested_independent_axis=axis, realization=marker)

    def test_structural_every_stage_must_match_requested_transform(self):
        packet = dict(pipeline_stages=[dict(realization=f"scan-chunk=2048; chunked-scans={count}") for count in (1, 3)])
        self.assertEqual([row["chunked_scans"] for row in structural_receipts(packet, 2048)], [1, 3])
        packet["pipeline_stages"][1]["realization"] = "historical Tile"
        with self.assertRaisesRegex(ValueError, "native structural"):
            structural_receipts(packet, 2048)
        self.assertEqual(structural_receipts(dict(realization="historical Tile"))[0]["chunked_scans"], 0)


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


class CudaMatrixAtomicSaveTests(unittest.TestCase):
    def test_transient_replace_permission_error_preserves_old_json_until_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.json"
            old = b'{"status":"old"}\n'
            path.write_bytes(old)
            pending = path.with_suffix(".json.writing")
            actual_replace = Path.replace
            attempts = []

            def replace(source, destination):
                attempts.append(source.read_bytes())
                self.assertEqual(path.read_bytes(), old)
                if len(attempts) < 3:
                    raise PermissionError(13, "temporary sharing violation")
                return actual_replace(source, destination)

            with mock.patch.object(Path, "replace", autospec=True, side_effect=replace) as replacement, \
                 mock.patch.object(cuda_matrix.time, "monotonic", side_effect=[100.0, 100.2, 100.4]), \
                 mock.patch.object(cuda_matrix.time, "sleep") as sleep:
                cuda_matrix.save(path, {"status": "passed", "samples": [1.0, 2.0]})
            self.assertEqual(replacement.call_count, 3)
            self.assertEqual(sleep.call_args_list, [mock.call(0.05), mock.call(0.05)])
            self.assertEqual(len(set(attempts)), 1)
            self.assertEqual(json.loads(path.read_text()), {"status": "passed", "samples": [1.0, 2.0]})
            self.assertFalse(pending.exists())

    def test_permanent_replace_permission_error_keeps_failure_and_pending_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.json"
            original = b'{"status":"running"}\n'
            path.write_bytes(original)
            failure = PermissionError(13, "permanent sharing violation")
            with mock.patch.object(Path, "replace", side_effect=failure) as replacement, \
                 mock.patch.object(cuda_matrix.time, "monotonic", side_effect=[10.0, 10.0, 11.99, 12.01]), \
                 mock.patch.object(cuda_matrix.time, "sleep") as sleep:
                with self.assertRaises(PermissionError) as caught:
                    cuda_matrix.save(path, {"status": "passed"})
            self.assertIs(caught.exception, failure)
            self.assertEqual(replacement.call_count, 3)
            self.assertEqual(sleep.call_count, 2)
            self.assertEqual(sleep.call_args_list[0], mock.call(0.05))
            self.assertAlmostEqual(sleep.call_args_list[1].args[0], 0.01)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(json.loads(path.with_suffix(".json.writing").read_text()), {"status": "passed"})

    def test_other_replace_errors_are_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.json"
            failure = OSError(28, "filesystem error")
            with mock.patch.object(Path, "replace", side_effect=failure) as replacement, \
                 mock.patch.object(cuda_matrix.time, "sleep") as sleep:
                with self.assertRaises(OSError) as caught:
                    cuda_matrix.save(path, {"status": "passed"})
            self.assertIs(caught.exception, failure)
            replacement.assert_called_once()
            sleep.assert_not_called()

    def test_write_permission_errors_are_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.json"
            failure = PermissionError(13, "cannot write temporary file")
            with mock.patch.object(Path, "write_text", side_effect=failure) as write, \
                 mock.patch.object(Path, "replace") as replacement, \
                 mock.patch.object(cuda_matrix.time, "sleep") as sleep:
                with self.assertRaises(PermissionError) as caught:
                    cuda_matrix.save(path, {"status": "passed"})
            self.assertIs(caught.exception, failure)
            write.assert_called_once()
            replacement.assert_not_called()
            sleep.assert_not_called()


class StreamingReceiptTests(unittest.TestCase):
    @staticmethod
    def packet():
        return dict(realization="native; streaming-scan-chunk=2048; streaming-scan-available; "
                    "streaming-scan-input-slot=0; streaming-scan-output-slot=1; "
                    "streaming-scan-input-bytes=32768; streaming-scan-output-bytes=32768",
                    native_streaming=[dict(stage=0, chunk_requested=2048, available=True, input_slot=0, output_slot=1,
                                           input_bytes=32768, output_bytes=32768, static_ranges_disjoint=True,
                                           expected_selected_entry="luisa_tile_stream_scan")])

    def test_streaming_receipt_and_structural_marker_do_not_conflict(self):
        packet = self.packet()
        self.assertEqual(len(cuda_matrix.streaming_scan_receipts(packet, 2048)), 1)
        self.assertEqual(cuda_matrix.structural_receipts(packet)[0]["chunked_scans"], 0)
        packet["realization"] += ";"
        self.assertEqual(len(cuda_matrix.streaming_scan_receipts(packet, 2048)), 1)

    def test_calibration_rejects_fallback_and_changed_range_facts(self):
        for updates in (dict(available=False), dict(static_ranges_disjoint=False), dict(input_bytes=65536),
                        dict(input_slot=1), dict(chunk_requested=1024), dict(expected_selected_entry="luisa_tile_main")):
            packet = self.packet()
            packet["native_streaming"][0].update(updates)
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                cuda_matrix.streaming_scan_receipts(packet, 2048)

    def test_streaming_is_removed_from_controls_and_other_routes(self):
        environment = {"LUISA_CUDA_TILE_STREAMING_SCAN": "2048"}
        for route in ("native", "tirx", "simd"):
            self.assertNotIn("LUISA_CUDA_TILE_STREAMING_SCAN", route_environment(environment, route))
        self.assertEqual(route_environment(environment, "native", native_streaming_scan=1024)["LUISA_CUDA_TILE_STREAMING_SCAN"], "1024")
        self.assertNotIn("LUISA_CUDA_TILE_STREAMING_SCAN", route_environment(environment, "tirx", native_streaming_scan=1024))
        for kwargs in (dict(native_scan_chunk=1024), dict(native_independent_axis=1)):
            with self.assertRaises(ValueError):
                route_environment(environment, "native", native_streaming_scan=2048, **kwargs)

    def test_default_historical_receipt_remains_valid(self):
        self.assertEqual(cuda_matrix.streaming_scan_receipts(dict(realization="native")), [])
        with self.assertRaises(ValueError):
            cuda_matrix.streaming_scan_receipts(self.packet(), 0)


if __name__ == "__main__":
    unittest.main()
