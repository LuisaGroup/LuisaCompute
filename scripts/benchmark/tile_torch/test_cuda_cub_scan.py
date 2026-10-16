"""CPU-only CUB realization provenance and isolation checks."""
import copy
import io
import json
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import cuda_matrix as matrix


class CubScanTests(unittest.TestCase):
    @staticmethod
    def packet():
        realization = (
            "native; cub-scan-requested; cub-scan-threads=256; cub-scan-chunk=2048; cub-scan-available; "
            "cub-scan-compile-key=123456789abcdef0; cub-scan-input-slot=0; cub-scan-output-slot=3; "
            "cub-scan-input-bytes=65536; cub-scan-output-bytes=65536; cub-scan-alignment-mask=9; "
            "cub-scan-grid-x=16; cub-scan-block-x=256; host-selected-disjoint-aligned16-cub-scan-v1")
        return dict(realization=realization, operation="scan", precision="bf16", dimensions=[16, 2048], tile=[1, 2048, 1],
                    native_alignment=[dict(stage=0, requested=False, eligible_buffer_mask=0,
                                           final_argument_mod16=[0, 0, 0, 0], expected_selected_entry="luisa_tile_cub_scan")],
                    native_cub_scan=[dict(stage=0, threads_requested=256, available=True, input_slot=0, output_slot=3,
                        input_bytes=65536, output_bytes=65536, static_ranges_disjoint=True, final_pointers_aligned16=True,
                        expected_selected_entry="luisa_tile_cub_scan", expected_selected_grid=[16, 1, 1], expected_selected_block=[256, 1, 1])])

    def test_native_only_explicit_environment_and_all_conflicts(self):
        inherited = {"LUISA_CUDA_TILE_CUB_SCAN": "1024", "KEEP": "x"}
        for route in ("native", "tirx", "simd", "torch"):
            self.assertNotIn("LUISA_CUDA_TILE_CUB_SCAN", matrix.route_environment(inherited, route))
            env = matrix.route_environment(inherited, route, native_cub_scan_threads=256)
            self.assertEqual(env.get("LUISA_CUDA_TILE_CUB_SCAN"), "256" if route == "native" else None)
            self.assertEqual(env["KEEP"], "x")
        for option, value in dict(native_aligned16=True, native_worker_warps=4, native_scan_chunk=1024,
                                 native_independent_axis=1, native_streaming_scan=2048, native_collective_cost=True,
                                 native_program_rows=1, native_partition_cost=True).items():
            with self.subTest(option=option), self.assertRaises(ValueError):
                matrix.route_environment({}, "native", native_cub_scan_threads=256, **{option: value})
        for invalid in (True, 1, 2, 64, 2048, "256"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                matrix.validate_native_cub_scan(invalid)

    def test_selected_and_runtime_fallback_are_distinct(self):
        p = self.packet()
        self.assertTrue(matrix.cub_scan_receipts(p, 256)[0]["selected"])
        matrix.alignment_receipts(p, False)
        for aligned, disjoint in ((False, True), (True, False)):
            p = self.packet()
            p["native_cub_scan"][0].update(final_pointers_aligned16=aligned, static_ranges_disjoint=disjoint,
                expected_selected_entry="luisa_tile_main", expected_selected_block=[1, 1, 1])
            p["native_alignment"][0]["expected_selected_entry"] = "luisa_tile_main"
            if not aligned:
                p["native_alignment"][0]["final_argument_mod16"][0] = 2
            self.assertFalse(matrix.cub_scan_receipts(p, 256)[0]["selected"])
            matrix.alignment_receipts(p, False)

    def test_unavailable_and_historical_default_keep_their_meaning(self):
        self.assertEqual(matrix.cub_scan_receipts(dict(realization="native")), [])
        p = self.packet()
        p["realization"] = "native; cub-scan-requested; cub-scan-threads=256; cub-scan-chunk=2048; cub-scan-unavailable; cub-scan-compile-key=0000000000000000; cub-scan-diagnostic=shape"
        p["native_cub_scan"][0].update(available=False, input_slot=0, output_slot=0, input_bytes=0, output_bytes=0,
            static_ranges_disjoint=False, final_pointers_aligned16=False, expected_selected_entry="luisa_tile_main", expected_selected_block=[1, 1, 1])
        self.assertFalse(matrix.cub_scan_receipts(p, 256)[0]["selected"])
        with self.assertRaises(ValueError):
            matrix.cub_scan_receipts(p, 0)

    def test_changed_recipe_geometry_and_guard_evidence_are_rejected(self):
        mutations = [
            lambda p: p.update(realization=p["realization"].replace("chunk=2048", "chunk=1024")),
            lambda p: p.update(realization=p["realization"] + "; cub-scan-threads=256"),
            lambda p: p.update(realization=p["realization"].replace("123456789abcdef0", "not-a-hash")),
            lambda p: p["native_cub_scan"][0].update(input_bytes=2),
            lambda p: p["native_cub_scan"][0].update(input_slot=1),
            lambda p: p["native_cub_scan"][0].update(expected_selected_block=[1, 1, 1]),
            lambda p: p["native_cub_scan"][0].update(expected_selected_grid=[1, 1, 1]),
            lambda p: p["native_alignment"][0].update(final_argument_mod16=[2, 0, 0, 0]),
            lambda p: p.update(precision="fp32"),
            lambda p: p.update(dimensions=[16, 2047]),
        ]
        for mutate in mutations:
            p = self.packet()
            mutate(p)
            with self.subTest(packet=p), self.assertRaises(ValueError):
                matrix.cub_scan_receipts(p, 256)

    def test_native_packet_preserves_two_sources_and_full_oracle(self):
        row = dict(id="probe", operation="scan", precision="bf16", dimensions=[16, 2048], tile=[1, 2048, 1], seed=19, pattern="random")
        p = dict(self.packet(), **{k: v for k, v in row.items() if k not in self.packet()})
        p.update(schema=1, backend="cuda", lowering="native", status="passed", samples=3, host_wall_us=[1, 2, 3],
                 cuda_event_stream_span_us=[1, 2, 3], graph_batch=4, graph_event_stream_span_us_per_op=[1, 2, 3],
                 correctness=dict(errors=0, inputs_unchanged=True, guards_unchanged=True, all_outputs_finite=True))
        args = SimpleNamespace(samples=3, graph_batch=4, native_cub_scan_threads=256)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = directory / "results.json"
            path.write_text(json.dumps(p), encoding="utf-8")
            (directory / "source.txt").write_text("original Tile", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "CUB candidate source"):
                matrix.native_result(dict(status="exited", returncode=0), path, row, args, "native")
            source = directory / "cub-source-stage0.cu"
            source.write_text("separate CUDA candidate", encoding="utf-8")
            result = matrix.native_result(dict(status="exited", returncode=0), path, row, args, "native")
            self.assertEqual(set(result["generated_sources"]), {"source.txt"})
            self.assertEqual(result["cub_generated_sources"][source.name]["sha256"], matrix.digest(source))
            p["correctness"]["errors"] = 1
            path.write_text(json.dumps(p), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "correctness gate"):
                matrix.native_result(dict(status="exited", returncode=0), path, row, args, "native")

    def test_cli_list_mode_is_host_only_and_rejects_unsupported_threads(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(matrix.main(["--list-cases", "--native-cub-scan-threads", "256"]), 0)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            matrix.main(["--list-cases", "--native-cub-scan-threads", "16"])


if __name__ == "__main__":
    unittest.main()
