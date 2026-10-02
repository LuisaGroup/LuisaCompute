"""Synthetic contract tests only; no real measurement or heldout outcome is read."""
import copy
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
import calibrate_v2 as c
sys.path.insert(0, str(c.V1_PATH.parent))
from test_calibrate import summary, device

def rows(count=8, ratio=0.7):
    packet = summary(count)
    for item in packet["cases"]:
        item["comparisons_to_native0"]["hint8"]["nativeN_over_native0"] = ratio
        native = item["cohorts"]["hint8"]["routes"]["native"]
        native["event_us"] = dict(p50=10 * ratio, samples=[10 * ratio] * 7)
        item["cohorts"]["baseline"]["routes"]["native"]["event_us"] = dict(p50=10, samples=[10] * 7)
    return c.load_dataset(packet, "a" * 64, c.v1.device_facts(device()))[0]

class ConditionalModelTests(unittest.TestCase):
    def test_fixed_features_and_no_name_lookup(self):
        data = rows()
        work = data[0]["logical_work"]
        self.assertAlmostEqual(data[0]["features"][0], math.log2(1 + work["programs"] / 24))
        changed = copy.deepcopy(data)
        for row in changed:
            row["case"] = dict(id="any name", precision="unused label", operation="unused operation")
        self.assertEqual(c.train_profile(data), c.train_profile(changed))

    def test_native_only_dataset_needs_no_torch_or_worker4(self):
        packet = summary()
        packet["cohorts"] = [x for x in packet["cohorts"] if x["worker"] != 4]
        for item in packet["cases"]:
            del item["cohorts"]["hint4"]
            del item["comparisons_to_native0"]["hint4"]
        report = c.calibrate([("b" * 64, packet)], device())
        self.assertEqual(len(report["training_observations"]), 6)
        self.assertFalse(report["production_autoselection_enabled"])

    def test_prefix_default_retained_but_not_fitted(self):
        data = rows()
        data[2]["prefix"] = True
        data[2]["target_log_score"] = -1000
        profile = c.train_profile(data)
        self.assertEqual(profile["retained_rows"], 8)
        self.assertEqual(profile["fitted_nonprefix_rows"], 7)
        self.assertEqual(c.decide(profile, data[2])["reason"], "prefix_semantic_default")
        self.assertEqual(c.decide(profile, data[2])["diagnostic_candidate"], 0)

    def test_candidate_requires_support_and_calibration(self):
        data = rows()
        profile = c.train_profile(data)
        decision = c.decide(profile, data[3])
        self.assertEqual(decision["diagnostic_candidate"], 8)
        self.assertEqual(decision["production_hint"], 0)
        out = copy.deepcopy(data[3])
        out["features"][0] = 100
        self.assertEqual(c.decide(profile, out)["reason"], "outside_leaf_feature_support")
        profile = c.train_profile(data[:2])
        self.assertEqual(c.decide(profile, data[0])["diagnostic_candidate"], 0)

    def test_all_residuals_including_unsupported_are_retained(self):
        data = rows()
        data[0]["target_log_score"] = 1.0
        profile = c.train_profile(data)
        self.assertEqual(len(profile["inner_predictions"]), len(data))
        self.assertAlmostEqual(profile["global_worst_inner_absolute_error"], max(p["absolute_error"] for p in profile["inner_predictions"]))
        self.assertTrue(any(not p["inner_supported"] for p in profile["inner_predictions"]))
        for leaf in c.leaves(profile["tree"]):
            self.assertGreaterEqual(leaf["calibration"]["worst_all_inner_error"], leaf["calibration"]["worst_supported_inner_error"] or 0)

    def test_depth_leaf_group_and_equal_geometry_weight(self):
        data = rows()
        for i, row in enumerate(data):
            row["target_log_score"] = -0.3 if i < 4 else 0.2
        tree = c.fit_tree(data)
        self.assertTrue(all(len(leaf["path"]) - len("root") <= 2 for leaf in c.leaves(tree)))
        self.assertTrue(all(len(leaf["groups"]) >= 2 for leaf in c.leaves(tree)))
        duplicate = copy.deepcopy(data[2])
        duplicate["record_id"] += "-another-run"
        repeated = c.fit_tree(data + [duplicate])
        for row in data:
            self.assertAlmostEqual(c.locate(tree, row["features"])["mean_log_score"], c.locate(repeated, row["features"])["mean_log_score"])

    def test_outer_held_outcomes_do_not_change_profile_or_decision(self):
        data = rows()
        original = c.outer_diagnostics(data)
        altered = copy.deepcopy(data)
        altered[3]["target_log_score"] = 100
        altered[3]["observed_ratio"] = 100
        altered[3]["samples"]["8"] = [1e-10] * 7
        altered[3]["recheck_ratio"] = 100
        modified = c.outer_diagnostics(altered)
        before = next(f for f in original["folds"] if f["held_group"] == data[3]["group"])
        after = next(f for f in modified["folds"] if f["held_group"] == data[3]["group"])
        self.assertEqual(before["profile"], after["profile"])
        self.assertEqual(before["predictions"][0]["decision"], after["predictions"][0]["decision"])
        self.assertNotEqual(before["predictions"][0]["observed_ratio"], after["predictions"][0]["observed_ratio"])

    def test_drift_and_regressions_remain_default(self):
        data = rows(ratio=1.2)
        profile = c.train_profile(data)
        self.assertTrue(all(c.decide(profile, r)["diagnostic_candidate"] == 0 for r in data))
        self.assertEqual(c.outer_diagnostics(data)["summary"]["all_observed_worker8_regressions"], len(data))
        data = rows()
        data[3]["recheck_ratio"] = 1.1
        profile = c.train_profile(data)
        self.assertIn("leaf_training_drift_exceeds_limit", c.locate(profile["tree"], data[3]["features"])["calibration"]["reasons"])
        self.assertEqual(c.decide(profile, data[3])["diagnostic_candidate"], 0)

    def test_identity_merge_and_incomplete_fail_closed(self):
        old, new = summary(), summary()
        for cohort in new["cohorts"]:
            cohort["implementation_receipt_set_sha256"] = "d" * 64
        with self.assertRaises(ValueError):
            c.calibrate([("a" * 64, old), ("b" * 64, new)], device())
        with self.assertRaises(ValueError):
            c.calibrate([("a" * 64, old), ("a" * 64, old)], device())
        old["status"] = "partial"
        with self.assertRaises(ValueError):
            c.calibrate([("a" * 64, old)], device())

    def test_cli_exact_provenance_and_new_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "summary.json").write_text(json.dumps(summary()))
            (root / "device.json").write_text(json.dumps(device()))
            args = ["--summary", str(root / "summary.json"), "--device-facts", str(root / "device.json"), "--output", str(root / "out")]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(c.main(args), 0)
            report = json.loads((root / "out/report.json").read_text())
            self.assertEqual(report["inputs"]["summaries"][0]["sha256"], c.sha((root / "out/input-summary-0.json").read_bytes()))
            with self.assertRaises(ValueError):
                c.main(args)

if __name__ == "__main__":
    unittest.main(verbosity=2)
