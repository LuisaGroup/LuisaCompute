import argparse
import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from compare_llm import capture_benchmark, check_metadata, configure_probe_environment, gpu_failure_diagnostics, main, make_summary, make_visit_plan, mark_remaining_gpu_visits_not_run, native_operation_environment, parse_arguments, parse_case, reference, run_torch_worker, shapes_for, validate_output


class LlmBenchmarkTests(unittest.TestCase):
    def test_shapes_and_invalid_cases(self):
        self.assertEqual(shapes_for("rope", (17, 258)), ([(17, 258), (17, 129), (17, 129)], (17, 258)))
        self.assertEqual(parse_case("attention:1,4,2,1,128,64,64")[0], "attention")
        for text in ("rope:1,7", "swiglu:0,4", "attention:1,3,2,1,4,4,4", "attention:1,2,1,5,4,4,4", "swiglu:65536,65536", "unknown:1,2"):
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_case(text)

    def test_rope_oracle_and_complete_check(self):
        import numpy as np
        x = np.array([[1, 2, 3, 4]], np.float32)
        c, s = np.array([[1, 0]], np.float32), np.array([[0, 1]], np.float32)
        expected = reference("rope", (1, 4), [x, c, s])
        np.testing.assert_array_equal(expected, [[1, -4, 3, 2]])
        self.assertEqual(validate_output(expected, expected)["elements"], 4)
        for actual in (expected[:, :3], expected + 1, np.full_like(expected, np.nan)):
            with self.assertRaises(ValueError):
                validate_output(actual, expected)

    def test_attention_bottom_right_mask_and_gqa(self):
        import numpy as np
        dims = (1, 4, 2, 2, 3, 1, 1)
        q = np.zeros((1, 4, 2, 1), np.float32)
        k = np.zeros((1, 2, 3, 1), np.float32)
        v = np.array([[[[1], [3], [8]], [[2], [6], [10]]]], np.float32)
        expected = reference("attention", dims, [q, k, v])
        np.testing.assert_allclose(expected[0, :2, :, 0], [[2, 4], [2, 4]])
        np.testing.assert_allclose(expected[0, 2:, :, 0], [[4, 6], [4, 6]])

    def test_metadata_fail_closed(self):
        row = dict(implementation="tile_xir_simd", backend="cpu", precision="fp32", fast_math=False,
                   relaxed_precision=False, runtime="luisa", timing="synchronized_host_wall",
                   batch_policy="one_runtime_command_list_per_batch", operation="rope", dimensions=[1, 4],
                   attention_block=[1, 1], input_shapes=[[1, 4], [1, 2], [1, 2]], output_shape=[1, 4],
                   correctness=dict(checks=2, elements_per_check=4, guard_elements_per_check=34, atol=5e-5, rtol=5e-5),
                   repetitions=10, throughput_us=[1., 2.], latency_us=[3., 4.])
        check_metadata(row, "cpu", "rope", (1, 4), (1, 1), 2)
        current = row | dict(source_reduction_policy="unordered_tree", reduction_candidate_setting="not_applicable")
        check_metadata(current, "cpu", "rope", (1, 4), (1, 1), 2)
        for key, value in (("source_reduction_policy", "fold_left"), ("reduction_candidate_setting", "enabled")):
            with self.assertRaises(ValueError):
                check_metadata(current | {key: value}, "cpu", "rope", (1, 4), (1, 1), 2)
        for tree, threads in ((True, 0), (False, 128)):
            with self.assertRaises(ValueError):
                check_metadata(row, "cpu", "rope", (1, 4), (1, 1), 2, tree, threads)
        explicit = row | dict(reduction_tree=True, requested_group_threads=128)
        check_metadata(explicit, "cpu", "rope", (1, 4), (1, 1), 2, True, 128)
        for key, value in (("reduction_tree", 1), ("requested_group_threads", 128.0)):
            with self.assertRaises(ValueError):
                check_metadata(explicit | {key: value}, "cpu", "rope", (1, 4), (1, 1), 2, True, 128)
        with self.assertRaises(ValueError):
            check_metadata(row, "cpu", "rope", (1, 4), (1, 1), 2, forward_input_views=True)
        check_metadata(row | dict(requested_input_views=True), "cpu", "rope", (1, 4), (1, 1), 2, forward_input_views=True)
        for value in (1, False, "true"):
            with self.assertRaises(ValueError):
                check_metadata(row | dict(requested_input_views=value), "cpu", "rope", (1, 4), (1, 1), 2, forward_input_views=True)
        for key, value in (("fast_math", 0), ("operation", "swiglu"), ("repetitions", 0), ("throughput_us", [float("nan"), 2])):
            bad = copy.deepcopy(row)
            bad[key] = value
            with self.assertRaises(ValueError):
                check_metadata(bad, "cpu", "rope", (1, 4), (1, 1), 2)

    def test_paired_ratios_and_missing_case_not_omitted(self):
        rows = [dict(operation="rope", dimensions=[1, 4], round=r, path=p, valid=True,
                     measurement=dict(throughput_us_p50=t, latency_us_p50=2 * t))
                for r, a, b in ((0, 2, 4), (1, 9, 3)) for p, t in (("native", a), ("torch", b))]
        result = make_summary(rows, 2)[0]
        self.assertEqual(result["throughput_us_p50"]["paired_native_over_torch_median"], 1.75)
        self.assertEqual(result["throughput_us_p50"]["slower_rounds"], 1)
        rows[0]["valid"] = False
        self.assertEqual(make_summary(rows, 2), [dict(operation="rope", dimensions=[1, 4], complete=False)])

    def test_attention_decomposition_acknowledged(self):
        dims = (1, 2, 1, 1, 3, 4, 5)
        inputs, output = shapes_for("attention", dims)
        row = dict(implementation="tile_tirx_metal", backend="metal", precision="fp32", fast_math=False,
                   relaxed_precision=False, runtime="luisa", timing="synchronized_host_wall",
                   batch_policy="one_runtime_command_list_per_batch", operation="attention", dimensions=list(dims),
                   attention_block=[1, 3], input_shapes=[list(s) for s in inputs], output_shape=list(output),
                   correctness=dict(checks=2, elements_per_check=10, guard_elements_per_check=34, atol=5e-5, rtol=5e-5),
                   repetitions=10, throughput_us=[1., 2.], latency_us=[3., 4.])
        check = lambda result, mode: check_metadata(result, "metal", "attention", dims, (1, 3), 2, attention_qk=mode)
        check(row, "mma")  # Legacy default is still accepted.
        check(row | dict(attention_qk="reduce"), "reduce")
        for bad in (row, row | dict(attention_qk="mma"), row | dict(attention_qk=True)):
            with self.assertRaises(ValueError):
                check(bad, "reduce")
        with self.assertRaises(ValueError):
            check(row | dict(attention_qk="reduce"), "mma")

    def test_attention_pv_default_backcompat_and_explicit_acknowledgment(self):
        dims = (1, 2, 1, 1, 3, 4, 5)
        inputs, output = shapes_for("attention", dims)
        row = dict(implementation="tile_tirx_metal", backend="metal", precision="fp32", fast_math=False,
                   relaxed_precision=False, runtime="luisa", timing="synchronized_host_wall",
                   batch_policy="one_runtime_command_list_per_batch", operation="attention", dimensions=list(dims),
                   attention_block=[1, 3], input_shapes=[list(s) for s in inputs], output_shape=list(output),
                   correctness=dict(checks=2, elements_per_check=10, guard_elements_per_check=34, atol=5e-5, rtol=5e-5),
                   repetitions=10, throughput_us=[1., 2.], latency_us=[3., 4.])
        check = lambda result, **modes: check_metadata(result, "metal", "attention", dims, (1, 3), 2, **modes)
        check(copy.deepcopy(row))  # Old binaries/artifacts need not report the default.
        check(row | dict(attention_qk="mma", attention_pv="mma"))
        check(row | dict(attention_pv="reduce"), attention_pv="reduce")
        check(row | dict(attention_qk="reduce", attention_pv="reduce"), attention_qk="reduce", attention_pv="reduce")
        check(row | dict(attention_qk="reduce", attention_pv="mma"), attention_qk="reduce")
        for policy in (None, "mma", True, 1, "not_applicable", "invalid"):
            candidate = row if policy is None else row | dict(attention_pv=policy)
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                check(candidate, attention_pv="reduce")
        with self.assertRaises(ValueError):
            check(row | dict(attention_pv="reduce"))
        # Neither an acknowledged PV probe nor its math-equivalent source
        # decomposition relaxes the complete oracle and timing contract.
        for key, value in (("repetitions", 0), ("throughput_us", [1.]), ("latency_us", [float("nan"), 1.]),
                           ("correctness", row["correctness"] | dict(elements_per_check=9)),
                           ("correctness", row["correctness"] | dict(guard_elements_per_check=0)),
                           ("correctness", row["correctness"] | dict(atol=1e-2))):
            candidate = row | dict(attention_pv="reduce") | {key: value}
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                check(candidate, attention_pv="reduce")

    def test_attention_pv_non_attention_is_not_applicable(self):
        row = dict(implementation="tile_tirx_metal", backend="metal", precision="fp32", fast_math=False,
                   relaxed_precision=False, runtime="luisa", timing="synchronized_host_wall",
                   batch_policy="one_runtime_command_list_per_batch", operation="rope", dimensions=[1, 4],
                   attention_block=[1, 1], input_shapes=[[1, 4], [1, 2], [1, 2]], output_shape=[1, 4],
                   correctness=dict(checks=2, elements_per_check=4, guard_elements_per_check=34, atol=5e-5, rtol=5e-5),
                   repetitions=10, throughput_us=[1., 2.], latency_us=[3., 4.])
        check = lambda result, **modes: check_metadata(result, "metal", "rope", (1, 4), (1, 1), 2, **modes)
        check(copy.deepcopy(row))
        check(row | dict(attention_pv="not_applicable"))
        check(row | dict(attention_pv="not_applicable"), attention_pv="reduce")
        for policy in (None, "mma", "reduce", True):
            candidate = row if policy is None else row | dict(attention_pv=policy)
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                check(candidate, attention_pv="reduce")

    def test_attention_probe_parser_guards(self):
        common = ["--native", "not-resolved-native", "--build-dir", "not-resolved-build", "--output", "not-created"]
        attention = "attention:1,2,1,1,3,4,5"
        with patch("compare_llm.subprocess.run") as process:
            default = parse_arguments(common + ["--backend", "metal", "--case", attention])
            self.assertEqual((default.attention_qk, default.attention_pv), ("mma", "mma"))
            for qk in ("mma", "reduce"):
                for pv in ("mma", "reduce"):
                    selected = parse_arguments(common + ["--backend", "metal", "--case", attention,
                                                         "--case", "rope:1,4", "--attention-qk", qk, "--attention-pv", pv])
                    self.assertEqual((selected.attention_qk, selected.attention_pv), (qk, pv))
            process.assert_not_called()  # Parsing never builds or launches.
        for backend in ("cpu", "metal4"):
            selected = parse_arguments(common + ["--backend", backend, "--case", attention,
                                                 "--attention-qk", "reduce", "--attention-pv", "reduce"])
            self.assertEqual((selected.attention_qk, selected.attention_pv), ("reduce", "reduce"))
        invalid = [(["--backend", "metal", "--case", "rope:1,4", "--attention-pv", "reduce"], "PV decomposition"),
                   (["--backend", "metal", "--case", attention, "--attention-pv", "invalid"], "invalid choice"),
                   (["--backend", "cpu", "--case", "rope:1,4", "--attention-qk", "reduce"], "QK decomposition")]
        for options, message in invalid:
            output = io.StringIO()
            with self.subTest(options=options), redirect_stderr(output), self.assertRaises(SystemExit):
                parse_arguments(common + options)
            self.assertIn(message, output.getvalue())

    def test_attention_probe_environment_is_sanitized_and_independent(self):
        common = ["--native", "unused", "--build-dir", "unused", "--output", "unused", "--backend", "metal",
                  "--case", "attention:1,2,1,1,3,4,5"]
        for qk in ("mma", "reduce"):
            for pv in ("mma", "reduce"):
                args = parse_arguments(common + ["--attention-qk", qk, "--attention-pv", pv])
                inherited = {"LUISA_TILE_BENCH_ATTENTION_QK": "stale-qk", "LUISA_TILE_BENCH_ATTENTION_PV": "stale-pv",
                             "LUISA_TILE_BENCH_UNRELATED": "stale", "LUISA_SIMD_WARP_WIDTH": "64"}
                with self.subTest(qk=qk, pv=pv), patch.dict(os.environ, inherited, clear=True):
                    removed = configure_probe_environment(args)
                    self.assertEqual(removed, inherited)
                    for key, mode in (("LUISA_TILE_BENCH_ATTENTION_QK", qk), ("LUISA_TILE_BENCH_ATTENTION_PV", pv)):
                        if mode == "mma":
                            self.assertNotIn(key, os.environ)
                        else:
                            self.assertEqual(os.environ[key], mode)
                    self.assertNotIn("LUISA_TILE_BENCH_UNRELATED", os.environ)
                    self.assertEqual(os.environ["LUISA_SIMD_WARP_WIDTH"], "8")

    def test_xir_attention_probe_backend_and_metadata(self):
        common = ["--native", "unused", "--build-dir", "unused", "--output", "unused",
                  "--case", "attention:1,2,1,1,3,4,5", "--attention-qk", "reduce", "--attention-pv", "reduce"]
        dims = (1, 2, 1, 1, 3, 4, 5)
        inputs, output = shapes_for("attention", dims)
        for backend, runtime in (("cpu", "simd"), ("metal4", "metal4")):
            args = parse_arguments(common + ["--backend", backend])
            with patch.dict(os.environ, clear=True):
                configure_probe_environment(args)
                self.assertEqual(os.environ["LUISA_TILE_BENCH_XIR_BACKEND"], runtime)
                self.assertEqual(os.environ["LUISA_TILE_BENCH_ATTENTION_QK"], "reduce")
                self.assertEqual(os.environ["LUISA_TILE_BENCH_ATTENTION_PV"], "reduce")
            row = dict(implementation="tile_xir_" + runtime, backend=backend, precision="fp32", fast_math=False,
                       relaxed_precision=False, runtime="luisa", timing="synchronized_host_wall",
                       batch_policy="one_runtime_command_list_per_batch", operation="attention", dimensions=list(dims),
                       attention_qk="reduce", attention_pv="reduce", attention_block=[1, 3],
                       input_shapes=[list(s) for s in inputs], output_shape=list(output),
                       correctness=dict(checks=2, elements_per_check=10, guard_elements_per_check=34, atol=5e-5, rtol=5e-5),
                       repetitions=10, throughput_us=[1., 2.], latency_us=[3., 4.])
            check_metadata(row, backend, "attention", dims, (1, 3), 2, attention_qk="reduce", attention_pv="reduce")
            del row["attention_pv"]
            with self.assertRaises(ValueError):
                check_metadata(row, backend, "attention", dims, (1, 3), 2, attention_qk="reduce", attention_pv="reduce")

    def test_mixed_matrix_only_passes_decomposition_environment_to_attention(self):
        environment = dict(LUISA_TILE_BENCH_ATTENTION_QK="reduce", LUISA_TILE_BENCH_ATTENTION_PV="reduce",
                           LUISA_TILE_BENCH_XIR_BACKEND="metal4", LUISA_TILE_BENCH_GROUP_THREADS="64")
        self.assertEqual(native_operation_environment(environment, "attention"), environment)
        for operation in ("rmsnorm", "rope", "layernorm"):
            filtered = native_operation_environment(environment, operation)
            self.assertNotIn("LUISA_TILE_BENCH_ATTENTION_QK", filtered)
            self.assertNotIn("LUISA_TILE_BENCH_ATTENTION_PV", filtered)
            self.assertEqual(filtered["LUISA_TILE_BENCH_XIR_BACKEND"], "metal4")
            self.assertEqual(filtered["LUISA_TILE_BENCH_GROUP_THREADS"], "64")
        self.assertEqual(environment["LUISA_TILE_BENCH_ATTENTION_PV"], "reduce")


class LlmProcessDiagnosticTests(unittest.TestCase):
    def test_visit_plan_and_gpu_failure_stop_policy(self):
        cases = [("rope", (1, 4)), ("rmsnorm", (17, 65))]
        rows = make_visit_plan(cases, 2)
        self.assertEqual(len(rows), 8)
        self.assertEqual([row["path"] for row in rows[:4]], ["native", "torch", "torch", "native"])
        self.assertTrue(all(row["status"] == "NotRun" and not row["valid"] for row in rows))
        rows[0].update(status="Error", gpu_failure_diagnostics=[dict(excerpt="GPU Hang Error")])
        self.assertFalse(mark_remaining_gpu_visits_not_run(rows, "cpu"))
        self.assertTrue(mark_remaining_gpu_visits_not_run(rows, "metal4"))
        self.assertTrue(all(row["status"] == "NotRun" and row["error"].startswith("not launched:") for row in rows[1:]))
        self.assertEqual(len(make_visit_plan(cases[:1], 6, baseline=True)), 18)

    def test_main_never_launches_remaining_gpu_arms_after_diagnostic(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ):
            directory = Path(temporary)
            binary = directory / "native"
            binary.write_bytes(b"not an executable; subprocess mocked")
            output = directory / "report"
            args = parse_arguments(["--native", str(binary), "--build-dir", str(directory), "--output", str(output),
                                    "--backend", "metal", "--case", "attention:1,2,1,1,3,4,5",
                                    "--case", "rmsnorm:1,4", "--rounds", "2"])

            def capture(command, environment, timeout, destination, stem, row):
                row["gpu_failure_diagnostics"] = [dict(channel="stderr", excerpt="GPU Hang Error")]
                raise RuntimeError("GPU failure diagnostic; entire cohort invalid")

            with patch("compare_llm.parse_arguments", return_value=args), patch("compare_llm.artifact_hashes", return_value={}), \
                    patch("compare_llm.platform.platform", return_value="test-platform"), \
                    patch("compare_llm.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "", "")), \
                    patch("compare_llm.subprocess.check_output", return_value="test\n"), \
                    patch("compare_llm.capture_benchmark", side_effect=capture) as native, \
                    patch("compare_llm.run_torch_worker") as torch, redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 1)
            native.assert_called_once()
            torch.assert_not_called()
            result = json.loads((output / "results.json").read_text())
            self.assertEqual(len(result["results"]), 8)
            self.assertEqual(result["results"][0]["status"], "Error")
            self.assertTrue(all(row["status"] == "NotRun" for row in result["results"][1:]))
            self.assertFalse(result["metadata"]["gpu_diagnostics_valid"])
            self.assertFalse(result["metadata"]["cohort_valid"])
            self.assertTrue(all(not row["complete"] for row in result["summary"]))

    def test_known_diagnostics_and_normal_output(self):
        failures = ["Caused GPU Hang Error (0x00000003)", "MTLCommandBufferErrorDomain Code=2",
                    "MTLCommandBufferStatusError", "Execution of the command buffer was aborted",
                    "Metal command buffer failed with error", "MPS backend out of memory",
                    "Error Domain=AGXMetalG16X Code=3", "GPU Address Fault Error (0x1)",
                    "Error: command buffer completion failed"]
        for failure in failures:
            for channel in ("stdout", "stderr"):
                streams = {"stdout": b'{"valid":true}', "stderr": b""}
                streams[channel] = failure.encode()
                with self.subTest(failure=failure, channel=channel):
                    found = gpu_failure_diagnostics(**streams)
                    self.assertTrue(found)
                    self.assertEqual(found[0]["channel"], channel)
        normal = ["", "MPS allocated 128 MB; command buffer completed successfully",
                  '{"backend":"metal","max_abs_error":0,"error":""}',
                  "GPU validation passed; kernel max_abs_error=0; PASS"]
        for text in normal:
            with self.subTest(text=text):
                self.assertEqual(gpu_failure_diagnostics(text, "ordinary warning: unused argument"), [])

    def test_gpu_hang_spelling_separators_and_normal_json(self):
        for first in ("", " ", "_", "-", "\t", " _- "):
            for second in ("", " ", "_", "-", "\t", " _- "):
                diagnostic = f"Caused gPu{first}hAnG{second}eRrOr (0x3)"
                for channel in ("stdout", "stderr"):
                    streams = {"stdout": "", "stderr": ""}
                    streams[channel] = diagnostic
                    with self.subTest(first=first, second=second, channel=channel):
                        found = gpu_failure_diagnostics(**streams)
                        self.assertTrue(found)
                        self.assertEqual(found[0]["channel"], channel)
        for text in ('{"backend":"metal","max_abs_error":0,"error":""}',
                     '{"backend":"mps","max_abs_error":1e-7,"error":null}',
                     '{"backend":"metal","error":false}',
                     '{"backend":"mps","error":0}',
                     'GPUHangError_count=0; GPU validation passed',
                     'prefixGPUHangErrorSuffix is an identifier'):
            with self.subTest(normal=text):
                self.assertEqual(gpu_failure_diagnostics(text, text), [])

    def test_diagnostic_evidence_is_bounded(self):
        found = gpu_failure_diagnostics(b"", (("prefix " * 100) + "GPU Hang Error\n").encode() * 100)
        self.assertEqual(len(found), 16)
        self.assertTrue(all(len(item["excerpt"]) <= 360 for item in found))
        self.assertEqual(found[0]["line"], 1)

    def test_zero_exit_fd_stderr_is_rejected_and_preserved(self):
        # Real lightweight Python subprocess, no Torch import or device. An
        # os.write(2) reproduces the C/C++ fd channel missed by redirect_stderr.
        with tempfile.TemporaryDirectory() as temporary:
            directory, row = Path(temporary), dict(valid=False)
            command = [sys.executable, "-c", "import os; os.write(1,b'{\"valid\":true}'); os.write(2,b'Caused GPU Hang Error (0x3)\\n')"]
            with self.assertRaisesRegex(RuntimeError, "entire cohort invalid"):
                capture_benchmark(command, dict(os.environ), 5, directory, "torch", row)
            self.assertEqual(row["process"]["exit_code"], 0)
            self.assertFalse(row["valid"])
            self.assertEqual((directory / "torch.stdout.log").read_bytes(), b'{"valid":true}')
            self.assertEqual((directory / "torch.stderr.log").read_bytes(), b"Caused GPU Hang Error (0x3)\n")
            self.assertTrue(json.loads((directory / "torch.process.json").read_text())["gpu_failure_diagnostics"])

    def test_success_and_nonzero_failures_preserve_raw_output(self):
        for code in (0, 1):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as temporary:
                directory, row = Path(temporary), {}
                process = Mock(returncode=code)
                process.communicate.return_value = (b'{"valid":true}', b"ordinary informational message")
                with patch("compare_llm.subprocess.Popen", return_value=process):
                    if code:
                        with self.assertRaises(subprocess.CalledProcessError):
                            capture_benchmark(["fake"], {}, 5, directory, "visit", row)
                    else:
                        result = capture_benchmark(["fake"], {}, 5, directory, "visit", row)
                        self.assertEqual(result.stdout, b'{"valid":true}')
                self.assertEqual(row["gpu_failure_diagnostics"], [])
                self.assertEqual((directory / "visit.stderr.log").read_bytes(), b"ordinary informational message")

    def test_compact_hang_zero_exit_fd_output_is_rejected(self):
        # Exercise both OS fd channels, not only a mocked process or the regex.
        for diagnostic in (b"GPUHangError (0x3)\n", b"GPU_Hang_Error (0x3)\n",
                           b"GPU-Hang-Error (0x3)\n", b"GPUHang_Error (0x3)\n"):
            for channel in ("stdout", "stderr"):
                with self.subTest(diagnostic=diagnostic, channel=channel), tempfile.TemporaryDirectory() as temporary:
                    directory, row = Path(temporary), dict(valid=False)
                    streams = {"stdout": b'{"valid":true}', "stderr": b""}
                    streams[channel] += diagnostic
                    command = [sys.executable, "-c", f"import os; os.write(1,{streams['stdout']!r}); os.write(2,{streams['stderr']!r})"]
                    with self.assertRaisesRegex(RuntimeError, "entire cohort invalid"):
                        capture_benchmark(command, dict(os.environ), 5, directory, "visit", row)
                    self.assertEqual(row["process"]["exit_code"], 0)
                    self.assertFalse(row["valid"])
                    for name, payload in streams.items():
                        self.assertEqual((directory / f"visit.{name}.log").read_bytes(), payload)
                    receipt = json.loads((directory / "visit.process.json").read_text())
                    self.assertEqual(receipt["gpu_failure_diagnostics"][0]["channel"], channel)

    def test_zero_exit_normal_json_fd_output_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, row = Path(temporary), {}
            stdout = b'{"backend":"metal","max_abs_error":0,"error":""}'
            stderr = b'{"backend":"mps","max_abs_error":1e-7,"error":null}\n'
            command = [sys.executable, "-c", f"import os; os.write(1,{stdout!r}); os.write(2,{stderr!r})"]
            result = capture_benchmark(command, dict(os.environ), 5, directory, "visit", row)
            self.assertEqual(result.returncode, 0)
            self.assertEqual((result.stdout, result.stderr), (stdout, stderr))
            self.assertEqual(row["gpu_failure_diagnostics"], [])

    def test_timeout_keeps_full_output_diagnostic_and_process_group_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, row = Path(temporary), {}
            process = Mock(returncode=-9, pid=12345)
            process.communicate.side_effect = [subprocess.TimeoutExpired(["fake"], 1, output=b"partial"),
                                               (b"complete stdout", b"GPU Hang Error before timeout")]
            with patch("compare_llm.subprocess.Popen", return_value=process), patch("compare_llm.os.killpg") as kill:
                with self.assertRaises(subprocess.TimeoutExpired) as caught:
                    capture_benchmark(["fake"], {}, 1, directory, "timeout", row)
            if os.name == "posix":
                kill.assert_called_once()
                self.assertEqual(kill.call_args.args[0], 12345)
            self.assertEqual(caught.exception.output, b"complete stdout")
            self.assertTrue(row["process"]["timed_out"])
            self.assertTrue(row["gpu_failure_diagnostics"])
            self.assertEqual((directory / "timeout.stdout.log").read_bytes(), b"complete stdout")

    def test_one_gpu_diagnostic_invalidates_other_case_summaries(self):
        rows = [dict(operation=op, dimensions=[1, 4], round=r, path=p, valid=True,
                     measurement=dict(throughput_us_p50=1., latency_us_p50=2.))
                for op in ("rope", "swiglu") for r in range(2) for p in ("native", "torch")]
        rows[0].update(valid=False, gpu_failure_diagnostics=[dict(excerpt="GPU Hang Error")])
        summaries = make_summary(rows, 2)
        self.assertEqual(len(summaries), 2)
        self.assertTrue(all(not item["complete"] and "invalid_reason" in item for item in summaries))
        self.assertTrue(all("throughput_us_p50" not in item for item in summaries))

    def test_torch_worker_is_a_captured_process_with_explicit_inputs(self):
        import hashlib
        import numpy as np
        with tempfile.TemporaryDirectory() as temporary:
            directory, row = Path(temporary), {}
            args = argparse.Namespace(backend="metal", samples=2, sample_ms=10, warmup_ms=20, threads=1,
                                      timeout=15, metal_device_timing=None)
            arrays = [np.ones((1, 4), np.float32), np.ones((1, 2), np.float32), np.ones((1, 2), np.float32)]
            output = directory / "out.f32"

            def capture(command, environment, timeout, destination, stem, record):
                self.assertEqual(command[:3], [sys.executable, str(Path(__file__).resolve().with_name("compare_llm.py")), "--torch-worker"])
                request = json.loads(Path(command[3]).read_text())
                self.assertEqual(request["backend"], "metal")
                self.assertEqual(request["timing"]["samples"], 2)
                for path, array in zip(request["input_paths"], arrays):
                    self.assertEqual(Path(path).read_bytes(), array.tobytes())
                actual = np.array([[0, 0, 2, 2]], np.float32)
                actual.tofile(output)
                result = {key: request[key] for key in ("format", "operation", "dimensions", "backend", "input_sha256", "output_path")}
                check = dict(elements=4, max_abs_error=0., atol=5e-5, rtol=5e-5)
                result.update(output_shape=[1, 4], output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                              torch_info=dict(version="test", git_version="test", config="test", mps_cpu_fallback=False),
                              correctness=check,
                              measurement=dict(repetitions=2, throughput_us=[1., 2.], latency_us=[3., 4.],
                                               precision="fp32", expression="test", pre_timing_correctness=check))
                return subprocess.CompletedProcess(command, 0, json.dumps(result).encode(), b"")

            with patch("compare_llm.capture_benchmark", side_effect=capture) as process:
                result = run_torch_worker("rope", (1, 4), arrays, args, directory, "torch", output, row)
            process.assert_called_once()
            self.assertEqual(result["measurement"]["throughput_us_p50"], 1.5)


if __name__ == "__main__":
    unittest.main()
