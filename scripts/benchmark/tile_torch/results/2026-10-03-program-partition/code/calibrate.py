"""CPU-only diagnostic partition NNLS. Never opens raw cohorts, compiles or executes a kernel."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import numpy as np

SCHEMA = "independent-partition-nnls-v1"
MAX_U64 = (1 << 64) - 1
SCHEDULES = ("default", "rows1", "rows2")
FEATURES = ("constant", "programs", "logical_batches_times_collective_volume")
POLICY = dict(processor_reference=24, improvement_fraction=0.05,
              candidates=[0, 1, 2], default_enabled=False,
              confidence="uncalibrated_point_prediction", training_statistic="seven_sample_median_us",
              recheck_usage="diagnostic_only_not_a_training_observation",
              group_weight="equal_total_weight_per_R_N_B_W_geometry",
              solver="enumerate_eight_nonnegative_active_subsets_binary64_lstsq",
              feature_scaling="training_only_weighted_RMS",
              positive_coefficient_tolerance=1e-12, objective_tie_tolerance=1e-12)
WORK = re.compile(r"(?:^|;\s*)collective-work-v1: programs=(\d+), elementwork=(\d+), read-bytes=(\d+), write-bytes=(\d+), tile-live-bytes=(\d+), largest-tile=(\d+)(?=;|$)")
COLLECTIVE = re.compile(r"(?:^|;\s*)collective=kind([0-3]):width(\d+):independent(\d+)(?=;|$)")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def receipt(path):
    data = Path(path).read_bytes()
    return dict(path=str(Path(path).resolve()), sha256=sha(data), bytes=len(data))


def integer(value, name, positive=False):
    require(type(value) is int and (1 if positive else 0) <= value <= MAX_U64,
            "invalid/overflowed uint64 fact: " + name)
    return value


def positive(value, name):
    require(type(value) in (float, int) and math.isfinite(value) and value > 0, "invalid positive " + name)
    return float(value)


def work_facts(realization):
    require(isinstance(realization, str) and realization.count("collective-work-v1") == 1,
            "missing/duplicate actual IR work facts")
    work, collective = WORK.findall(realization), COLLECTIVE.findall(realization)
    require(len(work) == 1 and len(collective) == realization.count("collective=") == 1,
            "requires one actual collective record")
    fields = ("programs", "elementwork", "read_bytes", "write_bytes", "original_live_bytes", "largest_tile")
    facts = {key: integer(int(value), key) for key, value in zip(fields, work[0])}
    kind, width, independent = map(int, collective[0])
    require(kind in (0, 2), "partition model requires SUM/MAXIMUM IR algebra")
    facts.update(kind=kind, width=integer(width, "width", True), independent=integer(independent, "independent", True))
    return facts


def candidate_facts(rows, columns, width, extent, storage_bytes, processors=24):
    for name, value in (("rows", rows), ("columns", columns), ("width", width),
                        ("extent", extent), ("storage", storage_bytes), ("processors", processors)):
        integer(value, name, True)
    require(columns <= width, "valid contribution width exceeds Tile width")
    programs = rows // extent + int(rows % extent != 0)
    batches = programs // processors + int(programs % processors != 0)
    volume = integer(extent * width, "per-program collective volume", True)
    facts = dict(programs=programs, independent_extent=extent, full_programs=rows // extent,
        tail_valid_extent=rows % extent, contribution_extent=width, logical_contribution_extent=columns,
        padded_independent=integer(programs * extent, "padded independent extent", True),
        collective_input_per_program=volume,
        collective_input_total=integer(programs * volume, "whole-launch collective volume", True),
        valid_input_bytes=integer(rows * columns * storage_bytes, "valid input bytes", True),
        valid_output_bytes=integer(rows * storage_bytes, "valid output bytes", True),
        input_snapshot_bytes=integer(volume * storage_bytes, "input snapshot bytes", True),
        fp32_source_bytes=integer(volume * 4, "FP32 source bytes", True),
        fp32_result_bytes=integer(extent * 4, "FP32 result bytes", True),
        output_value_bytes=integer(extent * storage_bytes, "output value bytes", True),
        independent_bounds_elidable=rows % extent == 0, contribution_bounds_elidable=columns == width,
        candidate_peak_state=dict(known=False, reason="no candidate IR/recipe SSA liveness analysis"),
        logical_batches=batches, demand=integer(batches * volume, "logical batch demand", True))
    facts["features"] = [1.0, float(programs), float(facts["demand"])]
    return facts


def samples(cohort):
    require(cohort.get("status") == "validated", "native cohort lacks explicit revalidation")
    timing = cohort["routes"]["native"]["event_us"]
    values = [positive(x, "timing sample") for x in timing["samples"]]
    require(len(values) == 7, "all seven samples required")
    middle = statistics.median(values)
    require(middle == positive(timing["p50"], "median"), "reported median differs from raw samples")
    return values, middle


def queue_identity(summary):
    if summary.get("status") == "completed_validated_native":
        revalidation = summary.get("native_revalidation")
        require(isinstance(revalidation, dict) and revalidation.get("status") == "passed" and
                revalidation.get("original_cohort_status") == "failed" and
                revalidation.get("cases") == 16 and revalidation.get("native_executions") == 64,
                "native-only derived validation must retain explicit passed revalidation provenance")
        require(summary.get("rescue_queue_receipt") and summary.get("original_failed_queue"), "missing rescue/original queue identity")
        return dict(rescue=summary["rescue_queue_receipt"], original_failed=summary["original_failed_queue"])
    require(summary.get("queue_receipt"), "missing original queue identity")
    return dict(original=summary["queue_receipt"])


def load_dataset(summary):
    status = summary.get("status")
    require(status in ("completed_validated", "completed_validated_native"), "only completed validated summaries can train")
    queue_identity(summary)
    require(summary.get("kind") == "partition" and summary.get("build_snapshot"),
            "missing partition/build/queue identity")
    cases = summary.get("cases")
    require(isinstance(cases, list) and len(cases) == 16, "training inventory must contain the frozen sixteen cases")
    require(len({item["case"]["id"] for item in cases}) == len(cases), "duplicate training case")
    observations, diagnostics = [], []
    for item in cases:
        case, cohorts = item["case"], item["cohorts"]
        require(case.get("fast_math", False) is False, "strict math required")
        require(len(case["dimensions"]) == 2 and len(case["tile"]) >= 2, "unsupported fixture geometry")
        rows, columns = case["dimensions"]
        original_extent, width = case["tile"][:2]
        storage_bytes = {"fp32": 4, "fp16": 2, "bf16": 2}.get(case["precision"])
        require(storage_bytes is not None, "unsupported storage")
        original = work_facts(cohorts["default"]["routes"]["native"]["realization"])
        require(original["independent"] == original_extent and original["width"] == width,
                "fixture Tile geometry differs from actual original IR")
        group = canonical([rows, columns, original_extent, width]).decode()
        all_times = {}
        for schedule in (*SCHEDULES, "recheck"):
            cohort = cohorts[schedule]
            values, measured = samples(cohort)
            work = work_facts(cohort["routes"]["native"]["realization"])
            require(work == original, "original IR facts changed between realizations")
            requested = {"default": 0, "rows1": 1, "rows2": 2, "recheck": 0}[schedule]
            extent = requested or original_extent
            require(not requested or (0 < extent < original_extent and original_extent % extent == 0), "illegal partition factor")
            facts = candidate_facts(rows, columns, width, extent, storage_bytes)
            require(original["programs"] == rows // original_extent + int(rows % original_extent != 0), "original IR program count mismatch")
            receipts = cohort["selection"]["program_partition"]
            require(len(receipts) == 1, "requires one selected launch receipt")
            launch = receipts[0]
            require(launch["rows_requested"] == requested and
                    launch["expected_selected_grid"] == [facts["programs"], 1, 1], "candidate selected grid/request mismatch")
            expected_entry = "luisa_tile_partition" if requested else "luisa_tile_main"
            require(launch["expected_selected_entry"] == expected_entry, "candidate did not select expected entry")
            if requested:
                require(launch["available"] is True and launch["static_ranges_disjoint"] is True and
                        launch["input_bytes"] == facts["valid_input_bytes"] and launch["output_bytes"] == facts["valid_output_bytes"],
                        "missing disjoint complete typed interval receipt")
            if schedule != "default":
                require(item["comparisons"][schedule]["status"] == "valid_matched_pair", "source/fixture matched-pair validation missing")
            all_times[schedule] = dict(samples=values, p50=measured)
            if schedule in SCHEDULES:
                observations.append(dict(record_id=case["id"] + ":" + schedule, case_id=case["id"], group=group,
                    schedule=schedule, requested_extent=requested, actual_ir=work, facts=facts,
                    features=facts["features"], measured_us=measured, samples=values,
                    case_provenance=case, source_receipts=cohort["source_files"], launch_receipt=launch))
        baseline, recheck = all_times["default"]["p50"], all_times["recheck"]["p50"]
        diagnostics.append(dict(case_id=case["id"], group=group, timings=all_times,
            default_recheck_ratio=recheck / baseline, default_recheck_drift=recheck / baseline - 1.0,
            candidates={name: dict(over_default=all_times[name]["p50"] / baseline,
                                   over_recheck=all_times[name]["p50"] / recheck,
                                   regression=all_times[name]["p50"] > baseline) for name in SCHEDULES[1:]}))
    return observations, diagnostics


def fit(observations):
    require(observations, "no training observations")
    groups = Counter(row["group"] for row in observations)
    weights = np.array([1.0 / groups[row["group"]] for row in observations], dtype=np.float64)
    x = np.array([row["features"] for row in observations], dtype=np.float64)
    y = np.array([positive(row["measured_us"], "training timing") for row in observations], dtype=np.float64)
    require(x.shape == (len(observations), 3) and np.isfinite(x).all() and (x >= 0).all(), "invalid three-feature matrix")
    scales = np.sqrt(np.sum(weights[:, None] * x * x, axis=0) / weights.sum())
    require(np.isfinite(scales).all() and (scales > 0).all(), "invalid training-only feature scale")
    a = x / scales * np.sqrt(weights)[:, None]
    b = y * np.sqrt(weights)
    best = None
    candidates = []
    for mask in range(8):
        columns = [i for i in range(3) if mask & (1 << i)]
        scaled = np.zeros(3, dtype=np.float64)
        rank, singular = 0, np.array([], dtype=np.float64)
        if columns:
            solution, _, rank, singular = np.linalg.lstsq(a[:, columns], b, rcond=None)
            if rank != len(columns):
                candidates.append(dict(mask=mask, status="rank_deficient", rank=int(rank)))
                continue
            tolerance = POLICY["positive_coefficient_tolerance"] * max(1.0, float(np.max(np.abs(solution))))
            if float(solution.min()) < -tolerance:
                candidates.append(dict(mask=mask, status="negative_active_coefficient"))
                continue
            scaled[columns] = np.maximum(solution, 0.0)
        coefficients = scaled / scales
        residual = x @ coefficients - y
        objective = float(np.dot(weights, residual * residual))
        candidate = dict(mask=mask, status="feasible", coefficients=coefficients.tolist(), weighted_sse=objective,
            rank=int(rank), singular_values=singular.tolist(),
            scaled_condition=float(singular[0] / singular[-1]) if len(singular) else None)
        candidates.append(candidate)
        tie = POLICY["objective_tie_tolerance"] * max(1.0, objective, best["weighted_sse"] if best else 0.0)
        if best is None or objective < best["weighted_sse"] - tie or (
                abs(objective - best["weighted_sse"]) <= tie and (mask.bit_count(), mask) < (best["mask"].bit_count(), best["mask"])):
            best = candidate
    require(best is not None and all(math.isfinite(value) and value >= 0 for value in best["coefficients"]), "NNLS has no finite solution")
    return dict(coefficients=best["coefficients"], active_mask=best["mask"], weighted_sse=best["weighted_sse"],
        training_scales=scales.tolist(), training_groups=sorted(groups),
        training_record_ids=[row["record_id"] for row in observations], active_subsets=candidates,
        full_scaled_design_rank=int(np.linalg.matrix_rank(a)),
        full_scaled_design_singular_values=np.linalg.svd(a, compute_uv=False).tolist(),
        sample_weights=weights.tolist())


def predict(model, row):
    score = sum(float(a) * float(x) for a, x in zip(model["coefficients"], row["features"]))
    require(math.isfinite(score) and score > 0, "nonfinite/nonpositive predicted score")
    return score


def decide(model, case_rows):
    require({row["schedule"] for row in case_rows} == set(SCHEDULES) and len(case_rows) == 3,
            "decision requires exactly original/rows1/rows2")
    by_schedule = {row["schedule"]: row for row in case_rows}
    scores = {name: predict(model, by_schedule[name]) for name in SCHEDULES}
    winner = min(SCHEDULES, key=lambda name: (scores[name], SCHEDULES.index(name)))
    selected = winner if scores[winner] / scores["default"] < 1.0 - POLICY["improvement_fraction"] else "default"
    return dict(selected_schedule=selected, experimental_extent=by_schedule[selected]["requested_extent"],
        scores_us_proxy=scores, predicted_ratio=scores[selected] / scores["default"],
        reason="point_prediction_exceeds_five_percent" if selected != "default" else "retain_original",
        confidence="uncalibrated", production_enabled=False)


def evaluate(model, observations):
    residuals = [dict(record_id=row["record_id"], predicted_us=predict(model, row), observed_us=row["measured_us"],
        signed_error_us=predict(model, row) - row["measured_us"],
        relative_error=predict(model, row) / row["measured_us"] - 1.0) for row in observations]
    decisions = []
    for case_id in sorted({row["case_id"] for row in observations}):
        rows = [row for row in observations if row["case_id"] == case_id]
        decision = decide(model, rows)  # Only geometry/features enter selection.
        original = next(row for row in rows if row["schedule"] == "default")
        selected = next(row for row in rows if row["schedule"] == decision["selected_schedule"])
        decisions.append(dict(case_id=case_id, group=original["group"], decision=decision,
            observed_policy_ratio=selected["measured_us"] / original["measured_us"],
            all_candidate_ratios={row["schedule"]: row["measured_us"] / original["measured_us"] for row in rows}))
    return dict(residuals=residuals, decisions=decisions)


def metric(decisions):
    groups = sorted({item["group"] for item in decisions})
    group_logs = [statistics.mean(math.log(item["observed_policy_ratio"]) for item in decisions if item["group"] == group) for group in groups]
    return dict(cases=len(decisions), groups=len(groups), selected=sum(item["decision"]["experimental_extent"] != 0 for item in decisions),
        selected_regressions=sum(item["decision"]["experimental_extent"] != 0 and item["observed_policy_ratio"] > 1 for item in decisions),
        selected_regressions_over_five_percent=sum(item["decision"]["experimental_extent"] != 0 and item["observed_policy_ratio"] > 1.05 for item in decisions),
        worst_policy_ratio=max(item["observed_policy_ratio"] for item in decisions),
        geometry_equal_weight_policy_geomean=math.exp(statistics.mean(group_logs)))


def logo(observations):
    folds = []
    for group in sorted({row["group"] for row in observations}):
        training = [row for row in observations if row["group"] != group]
        held = [row for row in observations if row["group"] == group]
        model = fit(training)
        evaluated = evaluate(model, held)
        folds.append(dict(held_group=group, model=model, **evaluated))
    return dict(folds=folds, summary=metric([decision for fold in folds for decision in fold["decisions"]]))


def calibrate(summary, device):
    require(device.get("sm_count") == 24 and device.get("compute_capability") == 89 and device.get("identity"),
            "the initial model uses the declared SM89/24-processor reference")
    observations, drift = load_dataset(summary)
    require(len({row["group"] for row in observations}) >= 3, "LOGO needs multiple training geometry groups")
    model = fit(observations)
    evaluated = evaluate(model, observations)
    profile = dict(schema=SCHEMA, formula="q = a0 + aP*P + aD*ceil(P/24)*b*W", fixed_policy=POLICY,
        feature_names=FEATURES, coefficients=model["coefficients"],
        device_declared_identity=device, summary_provenance=dict(status=summary["status"], build_snapshot=summary["build_snapshot"],
            queue_identity=queue_identity(summary), native_revalidation=summary.get("native_revalidation")),
        confidence="uncalibrated", default_enabled=False)
    return dict(schema=SCHEMA, status="fitted_diagnostic_not_deployed", profile_id=sha(canonical(profile)), profile=profile,
        full_training_model=model, training_observations=observations, full_training_evaluation=evaluated,
        full_training_metrics=metric(evaluated["decisions"]), outer_leave_geometry_out=logo(observations),
        baseline_recheck_and_all_candidates=drift,
        limitations=["Logical batches are not occupancy or measured residency.",
                    "No candidate peak-state/register/shared-memory features are manufactured.",
                    "Mask lowering, storage/algebra effects and compiler implementation changes are not separate model terms.",
                    "Five percent is a point-score decision threshold, not a confidence guarantee.",
                    "External held-out timings are neither accepted nor read by this fitter."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--device", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "preserve earlier reports: choose a fresh output directory")
    report = calibrate(read(args.summary), read(args.device))
    report["input_receipts"] = dict(summary=receipt(args.summary), device=receipt(args.device), script=receipt(__file__))
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(status=report["status"], profile_id=report["profile_id"], coefficients=report["profile"]["coefficients"],
                         logo=report["outer_leave_geometry_out"]["summary"])))


if __name__ == "__main__":
    main()
