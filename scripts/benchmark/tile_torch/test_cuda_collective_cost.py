"""Host-only checks for isolated collective-cost requests and actual receipts."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import cuda_matrix as matrix


def realization(status="selected", workers=8, reason="predicted-saving", score="-0.3", hint=8):
    value = ("native Tile; collective-cost-profile=sm89-24-cuda134-v3; "
             f"collective-cost-workers={workers}; collective-cost-status={status}; collective-cost-reason={reason};")
    if score is not None:
        value += f" collective-cost-log-score={score};"
    if hint:
        value += f" worker-warps-hint={hint};"
    return value


class CollectiveCostTests(unittest.TestCase):
    @staticmethod
    def packet(marker="native Tile", *, requested=True, fast_math=False, route="native", stages=None):
        case = dict(id="cost-fixture", operation="reduce_sum", precision="fp32", dimensions=[1, 4],
                    tile=[1, 4, 1], seed=19, pattern="random", fast_math=fast_math)
        result = dict(case, schema=1, backend="cuda", lowering=route, status="passed", realization=marker,
                      samples=3, host_wall_us=[2.0, 2.1, 2.2], cuda_event_stream_span_us=[1.0, 1.1, 1.2],
                      graph_batch=4, graph_event_stream_span_us_per_op=[0.5, 0.6, 0.7],
                      correctness=dict(errors=0, inputs_unchanged=True, guards_unchanged=True, all_outputs_finite=True))
        if stages is not None:
            result["pipeline_stages"] = [dict(realization=stage) for stage in stages]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.json"
            original = (json.dumps(result) + "\n").encode("utf-8")
            path.write_bytes(original)
            checked = matrix.native_result(dict(status="exited", returncode=0), path, case,
                                           SimpleNamespace(samples=3, graph_batch=4, native_collective_cost=requested), route)
            assert path.read_bytes() == original
        return checked

    def test_environment_is_native_only_explicit_and_parent_unchanged(self):
        env = {"LUISA_CUDA_TILE_COLLECTIVE_COST": "1", "LUISA_CUDA_TILE_WORKER_WARPS": "8",
               "LUISA_CUDA_TILE_SCAN_CHUNK": "1024", "LUISA_CUDA_TILE_INDEPENDENT_AXIS": "2",
               "LUISA_CUDA_TILE_STREAMING_SCAN": "2048", "LUISA_CUDA_TILE_IR_ALIGNED16": "1", "PATH": "preserve"}
        original = dict(env)
        for route in ("native", "tirx", "simd", "torch"):
            for requested in (False, True):
                with self.subTest(route=route, requested=requested):
                    actual = matrix.route_environment(env, route, native_collective_cost=requested)
                    self.assertEqual(actual.get("LUISA_CUDA_TILE_COLLECTIVE_COST"), "1" if requested and route == "native" else None)
                    for key in original.keys() - {"LUISA_CUDA_TILE_COLLECTIVE_COST", "PATH"}:
                        self.assertNotIn(key, actual)
                    self.assertEqual(actual["PATH"], "preserve")
        self.assertEqual(env, original)

    def test_cli_and_environment_reject_every_other_experiment(self):
        conflicts = [("--native-aligned16", [], "native_aligned16", True),
                     ("--native-worker-warps", ["4"], "native_worker_warps", 4),
                     ("--native-worker-warps", ["8"], "native_worker_warps", 8),
                     ("--native-scan-chunk", ["1024"], "native_scan_chunk", 1024),
                     ("--native-independent-axis", ["1"], "native_independent_axis", 1),
                     ("--native-streaming-scan", ["2048"], "native_streaming_scan", 2048)]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(matrix.main(["--list-cases", "--native-collective-cost"]), 0)
            for flag, values, key, value in conflicts:
                with self.subTest(flag=flag, values=values), self.assertRaisesRegex(ValueError, "collective-cost is mutually exclusive"):
                    matrix.main(["--list-cases", "--native-collective-cost", flag] + values)
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "collective-cost is mutually exclusive"):
                    matrix.route_environment({}, "native", native_collective_cost=True, **{key: value})
        for value in (None, 0, 1, "true"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "must be boolean"):
                matrix.route_environment({}, "native", native_collective_cost=value)

    def test_selected_and_fast_math_fallback_validate_actual_native_packet(self):
        chosen = self.packet(realization())
        self.assertEqual(chosen["status"], "passed")
        self.assertTrue(chosen["native_collective_cost"][0]["nondefault_hint_selected"])
        self.assertEqual(chosen["native_worker_warps"][0]["reported_hint"], 8)
        self.assertEqual(chosen["native_worker_warps"][0]["request_source"], "collective_cost_profile")
        fallback = self.packet(realization("ineligible", 0, "fast-math", None, 0), fast_math=True)
        self.assertEqual(fallback["status"], "passed")
        self.assertEqual(fallback["native_collective_cost"][0]["reason"], "fast-math")
        self.assertFalse(fallback["native_collective_cost"][0]["nondefault_hint_selected"])
        self.assertIsNone(fallback["native_worker_warps"][0]["reported_hint"])

    def test_default_prefix_and_model_default_are_not_labeled_selected(self):
        for marker in (realization("default", 0, "prefix", None, 0),
                       realization("default", 0, "predicted-default", "0.14", 0),
                       realization("ineligible", 0, "target-profile", None, 0)):
            row = self.packet(marker)["native_collective_cost"][0]
            self.assertFalse(row["nondefault_hint_selected"])
            self.assertEqual(row["workers"], 0)
        historical = self.packet(requested=False)
        self.assertEqual(historical["native_collective_cost"], [])
        self.assertEqual(historical["native_worker_warps"], [dict(stage=0, requested=0, reported_hint=None)])

    def test_every_pipeline_stage_must_match_its_selected_worker(self):
        stages = [realization(), realization("default", 0, "prefix", None, 0)]
        checked = self.packet(stages=stages)
        self.assertEqual([row["workers"] for row in checked["native_collective_cost"]], [8, 0])
        self.assertEqual([row["reported_hint"] for row in checked["native_worker_warps"]], [8, None])
        for index, changed in ((0, realization(hint=0)), (0, realization(hint=4)),
                               (1, realization("default", 0, "prefix", None, 8))):
            modified = list(stages)
            modified[index] = changed
            with self.subTest(index=index, changed=changed), self.assertRaisesRegex(ValueError, "worker-warps"):
                self.packet(stages=modified)

    def test_malformed_duplicate_missing_and_inconsistent_decisions_fail_closed(self):
        selected = realization()
        bad = ["native Tile", selected.replace("sm89-24-cuda134-v3", "other-profile"),
               selected.replace("workers=8", "workers=4"), selected.replace("workers=8", "workers=08"),
               selected.replace("status=selected", "status=default"), realization("selected", 0, hint=0),
               selected.replace("status=selected", "status=unknown"),
               selected + " collective-cost-workers=8;", selected + " collective-cost-unknown=1;",
               selected + " collective-cost-reason;", selected.replace("reason=predicted-saving", "reason=bad/reason"),
               realization(score=None), realization(score="NaN"), realization(score="inf"), realization(score="1e999"),
               realization("ineligible", 0, "analysis", "0.1", 0),
               realization("default", 0, "predicted-default", None, 0),
               realization("default", 0, "prefix", "0.1", 0),
               realization(score="0.1"), realization(reason="whatever", score="-0.3"),
               realization("default", 0, "predicted-default", "-0.3", 0),
               realization("ineligible", 0, "predicted-saving", None, 0),
               realization(score="-0.05129329438755058")]
        for marker in bad:
            with self.subTest(marker=marker), self.assertRaisesRegex(ValueError, "collective-cost"):
                self.packet(marker)
        boundary = self.packet(realization("default", 0, "predicted-default", "-0.05129329438755058", 0))
        self.assertFalse(boundary["native_collective_cost"][0]["nondefault_hint_selected"])

    def test_default_and_other_routes_reject_cost_metadata(self):
        with self.assertRaisesRegex(ValueError, "unrequested collective-cost"):
            self.packet(realization(), requested=False)
        with self.assertRaisesRegex(ValueError, "unrequested collective-cost"):
            self.packet(realization(), requested=True, route="tirx")

    def test_main_records_cost_request_and_clears_it_from_native_only_torch_record(self):
        class Telemetry:
            pid, _handle, stopped = 7, 7, False
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
        case = dict(id="cost-fixture", operation="reduce_sum", dimensions=[1, 4], tile=[1, 4, 1],
                    precision="fp32", seed=19, pattern="random")
        native = dict(status="passed", generated_sources={"source.txt": "mock-sha"}, result=dict(compile_ms=1, cold_ms=1,
                      host_wall_p50_us=1, realization=realization(), cuda_event_stream_span_us=[], graph_event_stream_span_us_per_op=[]))
        with tempfile.TemporaryDirectory() as temporary, contextlib.ExitStack() as stack:
            root = Path(temporary)
            (root / "bin").mkdir()
            (root / "bin/benchmark_tile_workloads.exe").write_bytes(b"test")
            (root / "CMakeCache.txt").write_text("test", encoding="utf-8")
            marker = root / "marker.json"
            marker.write_text("{}", encoding="utf-8")
            stack.enter_context(mock.patch.dict("sys.modules", {"windows_affinity": cpu}))
            stack.enter_context(mock.patch.dict(matrix.os.environ, {"LUISA_CUDA_TILE_COLLECTIVE_COST": "8", "LUISA_CUDA_TILE_STREAMING_SCAN": "2048"}))
            for name, replacement in (("selected_cases", lambda args: [case]), ("child", child),
                                      ("native_result", lambda *args: copy.deepcopy(native)),
                                      ("tensor_receipts", lambda *args: ({}, {"input": "same"})), ("digest", lambda path: "mock-sha")):
                stack.enter_context(mock.patch.object(matrix, name, replacement))
            stack.enter_context(mock.patch.object(matrix.shutil, "which", return_value="mock-nvidia-smi"))
            stack.enter_context(mock.patch.object(matrix.subprocess, "Popen", return_value=Telemetry()))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            code = matrix.main(["--routes", "native", "--native-only", "--native-collective-cost",
                                "--build-dir", str(root), "--build-marker", str(marker),
                                "--output", str(root / "results"), "--affinity-mask", "0xf"])
            result = json.loads((root / "results/results.json").read_text())
        self.assertEqual(code, 0)
        self.assertEqual(len(launches), 1)
        self.assertEqual(launches[0]["environment"]["LUISA_CUDA_TILE_COLLECTIVE_COST"], "1")
        self.assertNotIn("LUISA_CUDA_TILE_STREAMING_SCAN", launches[0]["environment"])
        self.assertEqual(result["environment_removed"]["LUISA_CUDA_TILE_COLLECTIVE_COST"], "8")
        self.assertTrue(result["options"]["native_collective_cost"])
        self.assertTrue(result["cases"][0]["runs"]["native"]["native_collective_cost_requested"])
        self.assertFalse(result["cases"][0]["runs"]["torch"]["native_collective_cost_requested"])
        self.assertEqual(result["cases"][0]["comparison_status"], "absent_native_only_calibration")


if __name__ == "__main__":
    unittest.main()
