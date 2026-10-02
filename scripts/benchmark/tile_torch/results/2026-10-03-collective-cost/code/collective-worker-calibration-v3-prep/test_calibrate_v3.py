"""Synthetic-only host contracts; no actual heldout data is read."""
import copy
import math
import unittest
import sys
import calibrate_v3 as c
sys.path.insert(0, str(c.v2.V1_PATH.parent))
from test_calibrate import summary, device

def facts():
    result = device()
    result["identity"].update(cuda_driver_api_version=13040, toolkit="synthetic/v13.4")
    return result

def rows():
    return c.load_dataset(summary(8), "a" * 64, c.v1.device_facts(facts()))[0]

class ModelTests(unittest.TestCase):
    def test_features_use_actual_algebra_geometry_and_storage(self):
        r = rows()[0]
        work = r["logical_work"]
        a = work["collectives"][0]
        total = a["width"] * a["independent"]
        expected = [math.log2(1 + work["programs"] / 24), math.log2(1 + work["largest_tile"]),
                    math.log2(1 + work["tile_live_bytes"] / 128), math.log2(1 + work["elementwork"] / total),
                    math.log2(1 + a["width"] / 32), math.log2(1 + a["independent"]), 1, 0,
                    math.log2(1 + (work["read_bytes"] + work["write_bytes"]) / (4 * total))]
        self.assertEqual(r["features"], expected)
        changed = copy.deepcopy(work)
        changed["collectives"][0]["kind"] = 2
        f = c.feature_vector(changed, c.v1.device_facts(facts()))
        self.assertEqual(f[6:8], [0, 1])
        self.assertEqual(f[:6] + f[8:], expected[:6] + expected[8:])

    def test_tree_contract_names_irrelevant_and_default_disabled(self):
        data = rows()
        report = c.calibrate([("a" * 64, summary(8))], facts())
        self.assertFalse(report["production_autoselection_enabled"])
        self.assertTrue(all(len(n["path"]) - 4 <= 2 for n in c.leaves(report["full_training_profile"]["tree"])))
        self.assertTrue(all(len(n["groups"]) >= 2 for n in c.leaves(report["full_training_profile"]["tree"])))
        before = c.train_profile(data)
        for row in data:
            row["case"] = {"id": "unused", "operation": "unused", "precision": "unused"}
        self.assertEqual(before, c.train_profile(data))

    def test_prefix_minimum_remain_default_and_all_negatives_retained(self):
        data = rows()
        data[0]["prefix"] = True
        data[1]["unseen_minimum"] = True
        profile = c.train_profile(data)
        self.assertEqual(profile["retained_rows"], 8)
        for row in data[:2]:
            self.assertEqual(c.decide(profile, row)["experimental_candidate"], 0)
        self.assertEqual(c.diagnostics(data)["summary"]["all_worker8_regressions"], sum(r["observed_ratio"] > 1 for r in data))

    def test_point_gate_is_strict_and_does_not_claim_support(self):
        data = rows()
        for row in data:
            row["target_log_score"] = math.log(0.8)
        profile = c.train_profile(data)
        row = copy.deepcopy(data[0])
        row["features"][0] = 100.0
        decision = c.decide(profile, row)
        self.assertEqual(decision["experimental_candidate"], 8)
        self.assertFalse(decision["in_leaf_feature_box_diagnostic"])
        self.assertEqual(decision["production_hint"], 0)
        profile["tree"]["mean_log_score"] = math.log(0.95)
        self.assertEqual(c.decide(profile, row)["experimental_candidate"], 0)

    def test_held_group_outcome_cannot_change_training_or_decision(self):
        data = rows()
        before = c.diagnostics(data)
        altered = copy.deepcopy(data)
        altered[3].update(target_log_score=100, observed_ratio=100, recheck_ratio=100)
        after = c.diagnostics(altered)
        a = next(f for f in before["folds"] if f["held_group"] == data[3]["group"])
        b = next(f for f in after["folds"] if f["held_group"] == data[3]["group"])
        self.assertEqual(a["profile"], b["profile"])
        self.assertEqual(a["predictions"][0]["decision"], b["predictions"][0]["decision"])
        self.assertNotEqual(a["predictions"][0]["observed_ratio"], b["predictions"][0]["observed_ratio"])

    def test_flat_tree_evaluation_parity(self):
        data = rows()
        tree = c.train_profile(data)["tree"]
        nodes = c.flat_nodes(tree)
        for row in data:
            i = 0
            while nodes[i]["feature"] >= 0:
                n = nodes[i]
                i = n["left"] if row["features"][n["feature"]] <= n["threshold"] else n["right"]
            self.assertEqual(nodes[i]["score"], c.locate(tree, row["features"])["mean_log_score"])

    def test_incomplete_mismatch_and_overflow_rejected(self):
        bad = summary()
        bad["status"] = "partial"
        with self.assertRaises(ValueError): c.calibrate([("a", bad)], facts())
        d = facts()
        d["sm_count"] = 25
        with self.assertRaises(ValueError): c.calibrate([("a", summary())], d)
        d = facts()
        d["identity"]["cuda_driver_api_version"] = 13030
        with self.assertRaises(ValueError): c.calibrate([("a", summary())], d)
        w = rows()[0]["logical_work"]
        w["collectives"][0]["width"] = c.v1.MAX_U64
        w["collectives"][0]["independent"] = 2
        with self.assertRaises(ValueError): c.feature_vector(w, c.v1.device_facts(facts()))

if __name__ == "__main__":
    unittest.main(verbosity=2)
