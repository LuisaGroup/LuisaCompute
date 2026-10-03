"""Small synthetic CPU checks; never read measurement/heldout files."""
import copy
import math
import unittest
import calibrate as c


def observations():
    result = []
    truth = [0.6, 0.002, 0.00003]
    for rows, width in ((7, 64), (33, 512), (97, 1024), (257, 2048)):
        group = str([rows, width - 1, 4, width])
        # Algebra/storage labels are intentionally never numeric features.
        for variant in ("sum_half", "max_float"):
            case_id = group + variant
            for schedule, extent in (("default", 4), ("rows1", 1), ("rows2", 2)):
                facts = c.candidate_facts(rows, width - 1, width, extent, 2)
                result.append(dict(record_id=case_id + schedule, case_id=case_id, group=group,
                    schedule=schedule, requested_extent=0 if schedule == "default" else extent,
                    features=facts["features"], measured_us=sum(a*x for a, x in zip(truth, facts["features"]))))
    return result, truth


class CalibrationTests(unittest.TestCase):
    def test_exact_three_parameter_fit(self):
        rows, truth = observations()
        model = c.fit(rows)
        for actual, expected in zip(model["coefficients"], truth):
            self.assertAlmostEqual(actual, expected, places=11)
        self.assertLess(model["weighted_sse"], 1e-20)
        self.assertEqual(model["full_scaled_design_rank"], 3)

    def test_group_weight_not_number_of_variants(self):
        rows, _ = observations()
        first = rows[0]["group"]
        duplicated = rows + [dict(row, record_id=row["record_id"] + "extra", case_id=row["case_id"] + "extra")
                             for row in rows if row["group"] == first]
        original, more = c.fit(rows), c.fit(duplicated)
        for actual, expected in zip(more["coefficients"], original["coefficients"]):
            self.assertAlmostEqual(actual, expected, places=12)

    def test_nnls_rejects_negative_coefficient(self):
        rows = [dict(record_id=str(p), group=str(p), features=[1., float(p), 4.], measured_us=5.-p) for p in (1, 2, 3)]
        model = c.fit(rows)
        self.assertEqual(model["coefficients"][1], 0.)
        self.assertTrue(all(x >= 0 for x in model["coefficients"]))
        self.assertTrue(any(item["status"] == "rank_deficient" for item in model["active_subsets"]))

    def test_logo_held_times_never_fit_or_choose(self):
        rows, _ = observations()
        original = c.logo(rows)
        held = original["folds"][0]["held_group"]
        changed = copy.deepcopy(rows)
        for row in changed:
            if row["group"] == held:
                row["measured_us"] *= 100 if row["schedule"] == "rows1" else 0.1
        after = next(fold for fold in c.logo(changed)["folds"] if fold["held_group"] == held)
        before = original["folds"][0]
        self.assertEqual(before["model"], after["model"])
        self.assertEqual([x["decision"] for x in before["decisions"]], [x["decision"] for x in after["decisions"]])
        self.assertNotEqual(before["residuals"], after["residuals"])
        self.assertFalse(any(row["record_id"] in before["model"]["training_record_ids"] for row in rows if row["group"] == held))

    def test_threshold_equality_and_tie_keep_default(self):
        rows = [dict(schedule=name, requested_extent=extent, features=[1., 1., demand])
                for name, extent, demand in (("default", 0, 100.), ("rows1", 1, 95.), ("rows2", 2, 96.))]
        model = dict(coefficients=[0., 0., .01])
        self.assertEqual(c.decide(model, rows)["selected_schedule"], "default")
        rows[1]["features"][2] = 94.
        self.assertEqual(c.decide(model, rows)["selected_schedule"], "rows1")
        for row in rows:
            row["features"][2] = 100.
        self.assertEqual(c.decide(model, rows)["selected_schedule"], "default")

    def test_group_split_holds_all_schedules_and_variants(self):
        rows, _ = observations()
        report = c.logo(rows)
        for fold in report["folds"]:
            self.assertEqual(len(fold["residuals"]), 6)
            self.assertEqual(len(fold["decisions"]), 2)
            self.assertNotIn(fold["held_group"], fold["model"]["training_groups"])

    def test_facts_unknown_peak_and_overflow(self):
        facts = c.candidate_facts(17, 61, 64, 2, 2)
        self.assertEqual(facts["features"], [1., 9., 128.])
        self.assertEqual(facts["tail_valid_extent"], 1)
        self.assertFalse(facts["candidate_peak_state"]["known"])
        self.assertNotIn("value", facts["candidate_peak_state"])
        for args in ((1 << 63, 2, 2, 1, 4), (17, 65, 64, 1, 2), (17, 61, 64, 0, 2), (1, 1, 1 << 63, 4, 2)):
            with self.assertRaises(ValueError):
                c.candidate_facts(*args)

    def test_actual_ir_parser_only_sum_max(self):
        text = "native; collective-work-v1: programs=5, elementwork=1, read-bytes=8, write-bytes=4, tile-live-bytes=256, largest-tile=64; collective=kind0:width64:independent4;"
        self.assertEqual(c.work_facts(text)["independent"], 4)
        for invalid in (text + text, text.replace("kind0", "kind3"), text.replace("width64", "width0")):
            with self.assertRaises(ValueError):
                c.work_facts(invalid)

    def test_incomplete_or_failed_summary_never_consumed(self):
        for status in ("running", "pending_queue", "queue_failed", "completed_with_failures"):
            with self.assertRaises(ValueError):
                c.load_dataset(dict(status=status))
        with self.assertRaisesRegex(ValueError, "provenance"):
            c.load_dataset(dict(status="completed_validated_native"))

    def test_regressions_remain_in_metrics(self):
        rows, truth = observations()
        model = dict(coefficients=[0., 0., 1.])
        chosen = c.decide(model, rows[:3])["selected_schedule"]
        self.assertNotEqual(chosen, "default")
        changed = copy.deepcopy(rows[:3])
        next(row for row in changed if row["schedule"] == chosen)["measured_us"] = 100.
        evaluated = c.evaluate(model, changed)
        self.assertEqual(len(evaluated["residuals"]), 3)
        self.assertEqual(c.metric(evaluated["decisions"])["selected_regressions_over_five_percent"], 1)


if __name__ == "__main__":
    unittest.main()
