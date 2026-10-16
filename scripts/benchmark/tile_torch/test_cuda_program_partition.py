"""Program repartition must prove both the expected entry and launch grid."""
import contextlib
import copy
import io
import unittest

import cuda_matrix as matrix


def packet(request=1, available=True, disjoint=True):
    fields = dict(input_slot=0, output_slot=3, input_bytes=2051 * 7 * 2,
                  output_bytes=7 * 2, original_rows=4, grid_x=7, original_grid_x=2)
    realization = "native Tile"
    if request:
        realization += f"; program-partition-rows={request}"
    if available:
        realization += "; program-partition-available; host-selected-disjoint-program-rows-v1"
        realization += "".join(f"; program-partition-{key.replace('_', '-')}={value}" for key, value in fields.items())
    else:
        fields = {key: 0 for key in fields}
    selected = available and disjoint
    receipt = dict(stage=0, rows_requested=request, available=available,
                   static_ranges_disjoint=disjoint, original_grid=[2, 1, 1],
                   expected_selected_grid=[7, 1, 1] if selected else [2, 1, 1],
                   expected_selected_entry="luisa_tile_partition" if selected else "luisa_tile_main", **fields)
    return dict(realization=realization, native_program_partition=[receipt])


class ProgramPartitionTests(unittest.TestCase):
    def test_available_entry_and_changed_grid(self):
        result = packet()
        self.assertEqual(matrix.program_partition_receipts(result, 1)[0]["expected_selected_grid"], [7, 1, 1])
        result["native_program_partition"][0]["expected_selected_grid"] = [2, 1, 1]
        with self.assertRaisesRegex(ValueError, "selected grid mismatch"):
            matrix.program_partition_receipts(result, 1)

    def test_runtime_fallback_is_not_a_candidate_measurement(self):
        for available, disjoint in ((False, False), (True, False)):
            with self.subTest(available=available), self.assertRaisesRegex(ValueError, "calibration did not select"):
                matrix.program_partition_receipts(packet(available=available, disjoint=disjoint), 1)

    def test_default_and_historical_records(self):
        self.assertEqual(matrix.program_partition_receipts(dict(realization="native")), [])
        self.assertFalse(matrix.program_partition_receipts(packet(0, False, False))[0]["available"])
        with self.assertRaisesRegex(ValueError, "missing program partition"):
            matrix.program_partition_receipts(dict(realization="native; program-partition-rows=1"))

    def test_wrong_metadata_or_grid_fails(self):
        for field, value in (("rows_requested", True), ("input_slot", 3), ("input_bytes", 0),
                             ("original_rows", 1), ("grid_x", 2), ("original_grid_x", 7),
                             ("expected_selected_entry", "luisa_tile_main"),
                             ("original_grid", [7, 1, 1]), ("expected_selected_grid", [7, True, 1])):
            result = packet()
            result["native_program_partition"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                matrix.program_partition_receipts(result, 1)
        result = packet()
        result["realization"] += "; program-partition-grid-x=7"
        with self.assertRaises(ValueError):
            matrix.program_partition_receipts(result, 1)

    def test_multiple_stage_identity(self):
        result = packet()
        result["pipeline_stages"] = [dict(realization=result["realization"])] * 2
        result["native_program_partition"].append(copy.deepcopy(result["native_program_partition"][0]))
        with self.assertRaisesRegex(ValueError, "request/stage"):
            matrix.program_partition_receipts(result, 1)
        result["native_program_partition"][1]["stage"] = 1
        self.assertEqual(len(matrix.program_partition_receipts(result, 1)), 2)

    def test_parent_environment_never_leaks(self):
        env = {"LUISA_CUDA_TILE_PROGRAM_ROWS": "4", "PATH": "original"}
        for route in ("native", "tirx", "simd", "torch"):
            for request in (0, 1, 2, 4):
                result = matrix.route_environment(env, route, native_program_rows=request)
                self.assertEqual(result.get("LUISA_CUDA_TILE_PROGRAM_ROWS"), str(request) if request and route == "native" else None)
        self.assertEqual(env, {"LUISA_CUDA_TILE_PROGRAM_ROWS": "4", "PATH": "original"})

    def test_conflicting_schedules(self):
        conflicts = dict(native_aligned16=True, native_worker_warps=8, native_scan_chunk=1024,
                         native_independent_axis=1, native_streaming_scan=2048, native_collective_cost=True)
        for name, value in conflicts.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "program-rows is mutually exclusive"):
                matrix.route_environment({}, "native", native_program_rows=1, **{name: value})
        for value in (True, None, "1", 3, 8):
            with self.subTest(value=value), self.assertRaises(ValueError):
                matrix.route_environment({}, "native", native_program_rows=value)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(matrix.main(["--list-cases", "--native-program-rows", "1"]), 0)
            with self.assertRaises(ValueError):
                matrix.main(["--list-cases", "--native-program-rows", "1", "--native-collective-cost"])


if __name__ == "__main__":
    unittest.main()
