"""Synthetic CPU-only contract tests; none of these numbers are GPU evidence."""
import copy
import math
from pathlib import Path
import tempfile
import unittest
import json
import io
from contextlib import redirect_stdout
import calibrate as c

def device():
    return dict(schema=1, compute_capability=89, sm_count=24, warp_size=32, max_resident_warps_per_sm=48,
                identity={"fixture": "synthetic device facts, not measurement"},
                evidence=[dict(path="synthetic", sha256="0" * 64)])

def realization(width=128, independent=2, programs=10, kind=0):
    return (f"unrelated emitter; collective-work-v1: programs={programs}, elementwork={width * independent}, "
            f"read-bytes={width * independent * 2}, write-bytes={independent * 2}, tile-live-bytes={width * independent * 6}, "
            f"largest-tile={width * independent}; collective=kind{kind}:width{width}:independent{independent}")

def record(text, median):
    return dict(status="validated", cohort_completion=dict(comparison_eligible=True),
                routes=dict(native=dict(realization=text, event_us=dict(p50=median, samples=[median * (1 + x * 0.001) for x in (-3, -2, -1, 0, 1, 2, 3)]))),
                source_files={"source.txt": dict(normalized_source_sha256="a" * 64)},
                fixture_sha256={"input": "b" * 64}, manifest={"synthetic": True})

def summary(groups=6):
    result = dict(schema=1, status="completed_validated", issues=[], cohorts=[], cases=[])
    for label, worker in (("baseline", 0), ("hint4", 4), ("hint8", 8), ("recheck", 0)):
        result["cohorts"].append(dict(label=label, worker=worker, status="passed", finished="synthetic complete",
                                      comparison_eligible=True, implementation_receipt_set_sha256="c" * 64, configuration={}))
    for i in range(groups):
        text = realization(64 << i, 1 << (i % 3), 7 + i)
        ratio4 = [1.02, 0.95, 1.10, 0.92, 1.0, 1.3][i % 6]
        ratio8 = [1.3, 1.2, 0.99, 0.91, 1.1, 0.85][i % 6]
        records = {label: record(text, 10.0 * ratio) for label, ratio in
                   (("baseline", 1), ("hint4", ratio4), ("hint8", ratio8), ("recheck", 1.002))}
        result["cases"].append(dict(case={"id": f"synthetic-{i}", "fast_math": False}, cohorts=records,
            comparisons_to_native0={label: dict(status="valid_matched_worker_pair", nativeN_over_native0=ratio)
                                    for label, ratio in (("hint4", ratio4), ("hint8", ratio8), ("recheck", 1.002))}))
    return result

class CalibrationTests(unittest.TestCase):
    def test_actual_device_facts_interface_and_expected_feature_equations(self):
        d = c.device_facts(device())
        work = c.parse_work(realization())
        x = c.features(work, d)
        self.assertEqual(x[0], 1)
        self.assertAlmostEqual(x[1], math.log2(1 + 10 / (24 * 48 / 4)))
        self.assertAlmostEqual(x[2], math.log2(1 + 128 / 32))
        self.assertEqual(x[6], 0)
        self.assertEqual(c.features(c.parse_work(realization(kind=3)), d)[6], 1)

    def test_parser_fails_closed(self):
        for text in ("missing", realization() + "; " + realization(), realization().replace("kind0", "kind4"),
                     realization().replace("width128", "width0"), realization().replace("programs=10", "programs=-1"),
                     realization() + "; scan-chunk=1024", realization() + "; collective=invalid"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                c.parse_work(text)
        work = c.parse_work(realization())
        work["collectives"][0]["width"] = c.MAX_U64
        with self.assertRaises(ValueError):
            c.features(work, c.device_facts(device()))

    def test_geometry_groups_ignore_names_and_storage_work(self):
        a = c.parse_work(realization())
        b = copy.deepcopy(a)
        b["read_bytes"] *= 2
        b["elementwork"] *= 4
        b["collectives"][0]["kind"] = 3
        self.assertEqual(c.geometry_group(a), c.geometry_group(b))
        b["collectives"][0]["width"] *= 2
        self.assertNotEqual(c.geometry_group(a), c.geometry_group(b))

    def test_complete_gate_and_no_silent_dropping(self):
        original = summary()
        for mutation in (lambda x: x.update(status="partial"),
                         lambda x: x["cohorts"][0].update(status="failed"),
                         lambda x: x["cases"][1]["cohorts"]["hint4"].update(status="failed"),
                         lambda x: x["cases"][0]["cohorts"]["baseline"]["routes"]["native"].update(realization="old packet"),
                         lambda x: x["cases"].append(copy.deepcopy(x["cases"][0]))):
            bad = copy.deepcopy(original)
            mutation(bad)
            with self.assertRaises(ValueError):
                c.calibrate(bad, device())

    def test_all_negative_and_tie_rows_retained(self):
        report = c.calibrate(summary(), device())
        self.assertEqual(len(report["training_observations"]), 6)
        self.assertFalse(report["production_autoselection_enabled"])
        self.assertEqual(report["production_hint"], 0)
        self.assertEqual(report["leave_one_geometry_group_out"]["summary"]["measured_regressions"]["4"], 3)
        self.assertEqual(report["leave_one_geometry_group_out"]["summary"]["measured_worst_ratio"]["4"], 1.3)
        for fold in report["leave_one_geometry_group_out"]["folds"]:
            for p in fold["predictions"]:
                self.assertEqual(p["decision"]["production_hint"], 0)

    def test_held_group_outcomes_cannot_change_its_fit_or_decision(self):
        rows, _, _ = c.dataset(summary(), device())
        original = c.diagnostics(rows)
        changed = copy.deepcopy(rows)
        held_group = rows[2]["group"]
        changed[2]["relative_log_scores"] = {"4": 4.0, "8": -4.0}
        changed[2]["relative_ratios"] = {"4": math.exp(4), "8": math.exp(-4)}
        changed[2]["recheck"]["ratio"] = 100
        changed[2]["samples"]["4"] = [0.0001] * 7
        after = c.diagnostics(changed)
        a = next(f for f in original["folds"] if f["held_group"] == held_group)
        b = next(f for f in after["folds"] if f["held_group"] == held_group)
        self.assertEqual(a["profile"], b["profile"])
        self.assertEqual(a["predictions"][0]["decision"], b["predictions"][0]["decision"])
        self.assertNotEqual(a["predictions"][0]["signed_log_errors"], b["predictions"][0]["signed_log_errors"])

    def test_out_of_support_and_missing_recheck_remain_zero(self):
        rows, _, _ = c.dataset(summary(), device())
        profile = c.train_profile(rows)
        vector = rows[0]["features"].copy()
        vector[2] += 100
        self.assertEqual(c.decide(profile, vector)["diagnostic_candidate"], 0)
        for row in rows:
            row["recheck"] = None
        profile = c.train_profile(rows)
        self.assertIn("missing_default_recheck", profile["reasons"])
        self.assertEqual(c.decide(profile, rows[0]["features"])["diagnostic_candidate"], 0)

    def test_ridge_coordinates_and_empirical_candidate_contract(self):
        rows, _, _ = c.dataset(summary(), device())
        for row in rows:
            row["relative_log_scores"] = {"4": math.log(0.7), "8": 0.0}
            row["samples"] = {"0": [10.0] * 7, "4": [7.0] * 7, "8": [10.0] * 7}
            row["medians_us"] = {"0": 10.0, "4": 7.0, "8": 10.0}
            row["recheck"] = dict(ratio=1.0)
        profile = c.train_profile(rows)
        decision = c.decide(profile, rows[3]["features"])
        self.assertEqual(decision["diagnostic_candidate"], 4)
        self.assertEqual(decision["production_hint"], 0)
        self.assertAlmostEqual(decision["relative_log_scores"]["4"], math.log(0.7))
        for row in rows:
            row["relative_log_scores"]["8"] = math.log(0.7)
        tied = c.decide(c.train_profile(rows), rows[3]["features"])
        self.assertEqual(tied["diagnostic_candidate"], 0)

    def test_hardware_and_sample_validation(self):
        for field, value in (("sm_count", 0), ("warp_size", True), ("max_resident_warps_per_sm", None), ("compute_capability", 90)):
            invalid = device()
            invalid[field] = value
            with self.assertRaises(ValueError):
                c.device_facts(invalid)
        packet = summary()
        packet["cases"][0]["cohorts"]["hint4"]["routes"]["native"]["event_us"]["samples"][0] = float("nan")
        with self.assertRaises(ValueError):
            c.calibrate(packet, device())

    def test_cli_snapshots_and_rejected_fit_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path, facts_path = root / "summary.json", root / "device.json"
            summary_path.write_text(json.dumps(summary()))
            facts_path.write_text(json.dumps(device()))
            with redirect_stdout(io.StringIO()):
                self.assertEqual(c.main(["--summary", str(summary_path), "--device-facts", str(facts_path), "--output", str(root / "valid")]), 0)
            report = c.read(root / "valid/report.json")
            self.assertEqual(report["inputs"]["summary"]["sha256"], c.sha((root / "valid/input-summary.json").read_bytes()))
            self.assertEqual(report["inputs"]["device_facts"]["sha256"], c.sha((root / "valid/input-device-facts.json").read_bytes()))
            with self.assertRaises(ValueError):
                c.main(["--summary", str(summary_path), "--device-facts", str(facts_path), "--output", str(root / "valid")])
            invalid = summary()
            invalid["status"] = "partial"
            summary_path.write_text(json.dumps(invalid))
            with redirect_stdout(io.StringIO()):
                self.assertEqual(c.main(["--summary", str(summary_path), "--device-facts", str(facts_path), "--output", str(root / "invalid")]), 2)
            rejected = c.read(root / "invalid/report.json")
            self.assertEqual(rejected["status"], "rejected_no_fit")
            self.assertEqual(rejected["production_hint"], 0)
            self.assertNotIn("full_training_profile", rejected)
            self.assertEqual((root / "invalid/input-summary.json").read_bytes(), summary_path.read_bytes())

if __name__ == "__main__":
    unittest.main(verbosity=2)
