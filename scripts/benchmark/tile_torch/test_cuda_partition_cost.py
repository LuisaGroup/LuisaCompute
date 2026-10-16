"""Automatic row selection must agree with its frozen profile and actual launch receipts."""
import contextlib
import copy
import io
import unittest

import cuda_matrix as matrix
from test_cuda_program_partition import packet


def cost_packet(status="selected", reason=None):
    selected = status == "selected"
    result = packet(1 if selected else 0, selected, selected)
    original_rows = 0 if status == "ineligible" else 4
    fields = {
        "requested": 1, "profile": "sm89-24-cuda134-partition-linear-v1",
        "fit": "63e0677c8554b46b707aa9f1fca3f29f223c653b1ec35fd57b5534b79595b6c2",
        "status": status, "reason": reason or {"selected": "predicted-saving", "retained": "predicted-original",
                                                "ineligible": "target"}[status],
        "original-rows": original_rows, "selected-rows": 1 if selected else original_rows,
    }
    if status != "ineligible":
        fields.update({"original-score": 2, "selected-score": 1 if selected else 2})
    result["realization"] += "".join(f"; partition-cost-{key}={value}" for key, value in fields.items())
    return result


class PartitionCostTests(unittest.TestCase):
    def test_actual_candidate_and_grid(self):
        result = cost_packet()
        decisions = matrix.partition_cost_receipts(result, True)
        self.assertEqual(decisions[0]["selected_rows"], 1)
        self.assertEqual(matrix.program_partition_receipts(result, cost_decisions=decisions)[0]["expected_selected_grid"], [7, 1, 1])
        for field, value in (("expected_selected_grid", [2, 1, 1]), ("static_ranges_disjoint", False), ("available", False)):
            bad = copy.deepcopy(result)
            bad["native_program_partition"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                matrix.program_partition_receipts(bad, cost_decisions=decisions)

    def test_retained_unavailable_and_ineligible(self):
        for status, reason in (("retained", "predicted-original"), ("retained", "candidate-unavailable"), ("ineligible", "target")):
            result = cost_packet(status, reason)
            decisions = matrix.partition_cost_receipts(result, True)
            self.assertFalse(matrix.program_partition_receipts(result, cost_decisions=decisions)[0]["available"])

    def test_marker_corruption_and_threshold(self):
        for old, new in (("requested=1", "requested=0"), ("original-score=2", "original-score=NaN"),
                         ("selected-score=1", "selected-score=1.9"), ("selected-score=1", "selected-score=inf"),
                         ("selected-rows=1", "selected-rows=4"), ("reason=predicted-saving", "reason=target"),
                         ("linear-v1", "linear-v2"), ("original-rows=4", "original-rows=true")):
            result = cost_packet()
            result["realization"] = result["realization"].replace("partition-cost-" + old, "partition-cost-" + new) if "=" in old else result["realization"].replace(old, new)
            with self.subTest(new=new), self.assertRaises(ValueError):
                matrix.partition_cost_receipts(result, True)
        for extra in ("; partition-cost-fit=duplicate", "; partition-cost-unknown=1", "; partition-cost-broken"):
            result = cost_packet()
            result["realization"] += extra
            with self.assertRaises(ValueError):
                matrix.partition_cost_receipts(result, True)

    def test_unrequested_or_missing_model_rejected(self):
        self.assertEqual(matrix.partition_cost_receipts(packet(0, False, False)), [])
        with self.assertRaises(ValueError):
            matrix.partition_cost_receipts(cost_packet())
        with self.assertRaises(ValueError):
            matrix.partition_cost_receipts(packet(), True)
        result = cost_packet()
        with self.assertRaises(ValueError):
            matrix.program_partition_receipts(result, 1, matrix.partition_cost_receipts(result, True))

    def test_multiple_stages_keep_decisions_separate(self):
        chosen, retained = cost_packet(), cost_packet("retained")
        result = dict(pipeline_stages=[dict(realization=x["realization"]) for x in (chosen, retained)],
                      native_program_partition=[chosen["native_program_partition"][0], retained["native_program_partition"][0]])
        result["native_program_partition"][1]["stage"] = 1
        decisions = matrix.partition_cost_receipts(result, True)
        self.assertEqual(len(matrix.program_partition_receipts(result, cost_decisions=decisions)), 2)
        with self.assertRaises(ValueError):
            matrix.program_partition_receipts(result, cost_decisions=list(reversed(decisions)))

    def test_environment_isolation_and_conflicts(self):
        environment = {"LUISA_CUDA_TILE_PARTITION_COST": "1", "PATH": "control"}
        for route in ("native", "tirx", "simd", "torch"):
            for request in (False, True):
                actual = matrix.route_environment(environment, route, native_partition_cost=request)
                self.assertEqual(actual.get("LUISA_CUDA_TILE_PARTITION_COST"), "1" if route == "native" and request else None)
        self.assertEqual(environment["LUISA_CUDA_TILE_PARTITION_COST"], "1")
        for key, value in dict(native_aligned16=True, native_worker_warps=8, native_scan_chunk=1024,
                               native_independent_axis=1, native_streaming_scan=2048,
                               native_collective_cost=True, native_program_rows=1).items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                matrix.route_environment({}, "native", native_partition_cost=True, **{key: value})
        with self.assertRaises(ValueError):
            matrix.route_environment({}, "native", native_partition_cost=1)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(matrix.main(["--list-cases", "--native-partition-cost"]), 0)
            with self.assertRaises(ValueError):
                matrix.main(["--list-cases", "--native-partition-cost", "--native-worker-warps", "8"])


if __name__ == "__main__":
    unittest.main()
