"""CPU-only frozen scan-cost provenance; never imports Torch or CUDA."""
import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import cuda_matrix as matrix


def packet(*, ineligible=False, unknown=False):
    # Independent fixed geometry, capacity and expected feature vectors.
    rows, width = 128, 8192
    original = [1.0, 1.0 / 6.0, 49152.0]
    feature_rows = ([1.0, 1.0 / 6.0, 64.0, 32.0], [1.0, 1.0 / 6.0, 32.0, 32.0],
                    [1.0, 1.0 / 6.0, 32.0, 64.0], [1.0, 1.0 / 6.0, 48.0, 192.0])
    capacities = (12, 6, 3, 1)
    original_score = sum(a * b for a, b in zip((0.7284119403590213, 0, 0.00026641302734655293), original))
    scores = [sum(a * b for a, b in zip((0.6027021529393383, 12.227593513162688, 0.027854484240835357, 0.022923736019806327), f)) for f in feature_rows]
    selected = 0 if ineligible or unknown else 256
    facts = dict(requested="1", profile=matrix.CUB_SCAN_COST_PROFILE, fit=matrix.CUB_SCAN_COST_FIT,
                 status="ineligible" if ineligible else "retained" if unknown else "selected",
                 reason="fast-math" if ineligible else "predicted-original" if unknown else "predicted-saving-installed",
                 sm=89, processors=24, warp=32, toolkit=13040, nvrtc=130400)
    facts.update({"profile-sha256": matrix.CUB_SCAN_COST_PROFILE_SHA256, "selected-threads": selected,
                  "declared-count": 4, "search-count": 0 if ineligible else 4, "compiler-call-count": 0 if ineligible else 4,
                  "search-ms": 3.5, "device-query-ok": "true", "resident-threads": 1536, "driver-api": 13040,
                  "original-entry": "luisa_tile_main", "original-source-key": "0123456789abcdef",
                  "original-registers": 22, "original-static-shared-bytes": 64, "original-local-bytes": 0,
                  "original-max-threads": 128, "original-resource-status": "ok"})
    if not ineligible:
        facts.update({"original-score": original_score, "selected-score": original_score if unknown else scores[1],
                      "original-features": ",".join(map(str, original))})
    for index, threads in enumerate((128, 256, 512, 1024)):
        prefix = f"t{threads}-"
        installed = threads == selected
        item = {"compile": "not-attempted" if ineligible else "ok", "load": "not-attempted" if ineligible else "ok",
                "entry": "not-attempted" if ineligible else "ok", "disposition": "profile-ineligible" if ineligible else "installed" if installed else "cost-ineligible" if unknown else "not-best",
                "cleanup": "not-needed" if ineligible else "shader-owned" if installed else "ok",
                "compile-key-known": "false" if ineligible else "true", "compile-key": f"{0 if ineligible else threads:016x}",
                "source-key": f"{0 if ineligible else threads + 1:016x}", "compile-ms": 0 if ineligible else 0.5,
                "query-scope": "not-queried" if ineligible else "loaded-candidate", "resource-entry": "luisa_tile_cub_scan",
                "reason": "candidate-unavailable" if ineligible else "resident-capacity-unknown" if unknown else "scored", "diagnostic": "",
                "registers": "unknown" if ineligible else 24, "static-shared-bytes": "unknown" if ineligible else 1024,
                "local-bytes": "unknown" if ineligible else 0, "max-threads": "unknown" if ineligible else 1024,
                "resource-status": "not-queried" if ineligible else "ok", "resident-cta-capacity": "unknown" if ineligible or unknown else capacities[index],
                "capacity-status": "not-queried" if ineligible else "cuda-1" if unknown else "ok",
                "capacity-threads": threads, "capacity-dynamic-shared-bytes": 0}
        if not ineligible:
            item["source-file"] = f"cache/candidate {threads}.cu"
        if not ineligible and not unknown:
            item.update(score=scores[index], features=",".join(map(str, feature_rows[index])))
        facts.update({prefix + key: value for key, value in item.items()})
    realization = "native; " + "; ".join(f"cub-scan-cost-{key}={value}" for key, value in facts.items())
    realization += f"; cub-scan-requested; cub-scan-threads={selected}; cub-scan-chunk={selected * 8}; cub-scan-{'available' if selected else 'unavailable'}; cub-scan-compile-key={selected:016x}"
    if selected:
        realization += ("; cub-scan-source-file=cache/candidate 256.cu; cub-scan-resource-scope=installed-entry; cub-scan-input-slot=0; cub-scan-output-slot=3; "
                        "cub-scan-input-bytes=2097152; cub-scan-output-bytes=2097152; cub-scan-alignment-mask=9; cub-scan-grid-x=128; cub-scan-block-x=256;")
    else:
        realization += "; cub-scan-resource-scope=not-queried;"
    entry = "luisa_tile_cub_scan" if selected else "luisa_tile_main"
    return dict(realization=realization, operation="scan", dimensions=[rows, width], tile=[1, width, 1], precision="bf16", fast_math=ineligible,
        native_alignment=[dict(stage=0, requested=False, eligible_buffer_mask=0, final_argument_mod16=[0, 0, 0, 0], expected_selected_entry=entry)],
        native_cub_scan=[dict(stage=0, threads_requested=selected, available=bool(selected), input_slot=0, output_slot=3 if selected else 0,
            input_bytes=2097152 if selected else 0, output_bytes=2097152 if selected else 0,
            static_ranges_disjoint=bool(selected), final_pointers_aligned16=bool(selected), expected_selected_entry=entry,
            expected_selected_grid=[rows, 1, 1], expected_selected_block=[selected or 1, 1, 1])])


class CubScanCostTests(unittest.TestCase):
    def test_environment_cli_and_all_experiment_conflicts(self):
        inherited = {"LUISA_CUDA_TILE_CUB_SCAN_COST": "1", "LUISA_CUDA_TILE_CUB_SCAN": "128"}
        for route in ("native", "tirx", "simd", "torch"):
            self.assertNotIn("LUISA_CUDA_TILE_CUB_SCAN_COST", matrix.route_environment(inherited, route))
            env = matrix.route_environment(inherited, route, native_cub_scan_cost=True)
            self.assertEqual(env.get("LUISA_CUDA_TILE_CUB_SCAN_COST"), "1" if route == "native" else None)
            self.assertNotIn("LUISA_CUDA_TILE_CUB_SCAN", env)
        for name, value in dict(native_aligned16=True, native_worker_warps=8, native_scan_chunk=1024,
                native_independent_axis=1, native_streaming_scan=2048, native_collective_cost=True, native_program_rows=1,
                native_partition_cost=True, native_cub_scan_threads=128).items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                matrix.route_environment({}, "native", native_cub_scan_cost=True, **{name: value})
        with self.assertRaises(ValueError):
            matrix.validate_native_cub_scan_cost(1)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(matrix.main(["--list-cases", "--native-cub-scan-cost"]), 0)

    def test_default_has_no_cost_and_fixed_recipe_unchanged(self):
        self.assertEqual(matrix.cub_scan_cost_receipts(dict(realization="native")), [])
        env = matrix.route_environment({}, "native", native_cub_scan_threads=256)
        self.assertEqual(env["LUISA_CUDA_TILE_CUB_SCAN"], "256")
        self.assertNotIn("LUISA_CUDA_TILE_CUB_SCAN_COST", env)
        with self.assertRaises(ValueError):
            matrix.cub_scan_cost_receipts(packet(), False)

    def test_frozen_model_prediction_and_actual_guard_fallback_separate(self):
        p = packet()
        cost = matrix.cub_scan_cost_receipts(p, True)
        self.assertEqual(cost[0]["selected_threads"], 256)
        self.assertEqual(cost[0]["candidates"][1]["features"], [1, 1/6, 32, 32])
        self.assertTrue(matrix.cub_scan_receipts(p, 0, cost)[0]["selected"])
        for aligned, disjoint in ((False, True), (True, False)):
            p = packet()
            p["native_cub_scan"][0].update(final_pointers_aligned16=aligned, static_ranges_disjoint=disjoint,
                expected_selected_entry="luisa_tile_main", expected_selected_block=[1, 1, 1])
            p["native_alignment"][0]["expected_selected_entry"] = "luisa_tile_main"
            if not aligned:
                p["native_alignment"][0]["final_argument_mod16"][0] = 2
            cost = matrix.cub_scan_cost_receipts(p, True)
            self.assertEqual(cost[0]["selected_threads"], 256)  # installed, not invocation selection
            self.assertFalse(matrix.cub_scan_receipts(p, 0, cost)[0]["selected"])
            matrix.alignment_receipts(p, False)

    def test_unknown_resources_and_fast_math_keep_explicit_zero_final_receipt(self):
        for p in (packet(unknown=True), packet(ineligible=True)):
            cost = matrix.cub_scan_cost_receipts(p, True)
            final = matrix.cub_scan_receipts(p, 0, cost)
            self.assertEqual(cost[0]["selected_threads"], 0)
            self.assertFalse(final[0]["selected"])
            self.assertEqual(final[0]["compile_key"], "0" * 16)
            self.assertTrue(all(c["score"] is None for c in cost[0]["candidates"]))

    def test_model_resource_identity_selection_and_cleanup_mutants_rejected(self):
        replacements = [
            (matrix.CUB_SCAN_COST_FIT, "0" * 64), ("cub-scan-cost-nvrtc=130400", "cub-scan-cost-nvrtc=13040"),
            ("original-features=1.0,0.16666666666666666,49152.0", "original-features=1,0.16666666666666666,8192"),
            ("t256-features=1.0,0.16666666666666666,32.0,32.0", "t256-features=1,0.16666666666666666,31,32"),
            ("t256-resident-cta-capacity=6", "t256-resident-cta-capacity=unknown"),
            ("t256-local-bytes=0", "t256-local-bytes=8"), ("t256-query-scope=loaded-candidate", "t256-query-scope=installed-entry"),
            ("t256-capacity-threads=256", "t256-capacity-threads=128"), ("t256-capacity-dynamic-shared-bytes=0", "t256-capacity-dynamic-shared-bytes=16"),
            ("t128-cleanup=ok", "t128-cleanup=cuda-1"), ("t256-cleanup=shader-owned", "t256-cleanup=ok"),
            ("selected-threads=256", "selected-threads=128"), ("compiler-call-count=4", "compiler-call-count=3"),
            ("reason=predicted-saving-installed", "reason=predicted-original"),
            ("t256-compile-key-known=true", "t256-compile-key-known=false"),
        ]
        for before, after in replacements:
            p = packet()
            self.assertIn(before, p["realization"])
            p["realization"] = p["realization"].replace(before, after, 1)
            with self.subTest(before=before), self.assertRaises(ValueError):
                matrix.cub_scan_cost_receipts(p, True)
        p = packet()
        p["realization"] += "; cub-scan-cost-selected-threads=256"
        with self.assertRaises(ValueError):
            matrix.cub_scan_cost_receipts(p, True)

    def test_final_winner_identity_cannot_borrow_losing_candidate(self):
        p = packet()
        cost = matrix.cub_scan_cost_receipts(p, True)
        p["realization"] = p["realization"].replace("cub-scan-compile-key=0000000000000100", "cub-scan-compile-key=0000000000000080")
        with self.assertRaises(ValueError):
            matrix.cub_scan_receipts(p, 0, cost)

    def test_all_candidate_sources_independent_hashes_and_full_oracle_remain_required(self):
        p = packet()
        row = dict(id="probe", operation="scan", precision="bf16", dimensions=[128, 8192], tile=[1, 8192, 1], seed=19, pattern="random")
        p.update(schema=1, backend="cuda", lowering="native", status="passed", seed=19, pattern="random", samples=3,
                 host_wall_us=[1, 2, 3], cuda_event_stream_span_us=[1, 2, 3], graph_batch=4,
                 graph_event_stream_span_us_per_op=[1, 2, 3],
                 correctness=dict(errors=0, inputs_unchanged=True, guards_unchanged=True, all_outputs_finite=True))
        args = SimpleNamespace(samples=3, graph_batch=4, native_cub_scan_cost=True)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = directory / "results.json"
            path.write_text(json.dumps(p), encoding="utf-8")
            original = directory / "source.txt"
            original.write_text("original Tile unchanged", encoding="utf-8")
            (directory / "cub-source-stage0.cu").write_text("source256", encoding="utf-8")
            for threads in (128, 256, 512):
                (directory / f"cub-cost-source-stage0-t{threads}.cu").write_text(f"source{threads}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "CUB cost candidate source"):
                matrix.native_result(dict(status="exited", returncode=0), path, row, args, "native")
            (directory / "cub-cost-source-stage0-t1024.cu").write_text("source1024", encoding="utf-8")
            checked = matrix.native_result(dict(status="exited", returncode=0), path, row, args, "native")
            self.assertEqual(checked["generated_sources"], {"source.txt": matrix.digest(original)})
            self.assertEqual(len(checked["cub_cost_generated_sources"]), 4)
            self.assertEqual(checked["cub_generated_sources"]["cub-source-stage0.cu"]["sha256"],
                             checked["cub_cost_generated_sources"]["cub-cost-source-stage0-t256.cu"]["sha256"])
            (directory / "cub-source-stage0.cu").write_text("wrong winner", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "winner source/key"):
                matrix.native_result(dict(status="exited", returncode=0), path, row, args, "native")
            p["correctness"]["errors"] = 1
            path.write_text(json.dumps(p), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "correctness gate"):
                matrix.native_result(dict(status="exited", returncode=0), path, row, args, "native")


if __name__ == "__main__":
    unittest.main()
