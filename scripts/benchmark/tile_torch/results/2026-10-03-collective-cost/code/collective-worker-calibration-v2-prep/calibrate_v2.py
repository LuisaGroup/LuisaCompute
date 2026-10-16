"""Conditional numeric worker0/8 diagnostic; no subprocess, compiler or GPU."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
V1_PATH = HERE.parent / "collective-worker-calibration-prep/calibrate.py"
V1_SHA256 = "48505217a47d257645f324287ad063c0d4b185e897665293c708f9bc12f40b16"
if hashlib.sha256(V1_PATH.read_bytes()).hexdigest() != V1_SHA256:
    raise ValueError("frozen v1 parser dependency changed")
sys.path.insert(0, str(V1_PATH.parent))
import calibrate as v1

FEATURES = ["log2_programs_per_sm", "log2_largest_logical_tile_elements",
            "log2_explicit_live_bytes_per_warp_fp32_reference", "log2_elementwork_per_collective_input"]
POLICY = dict(candidates=[0, 8], prefix_default=0, maximum_depth=2, minimum_leaf_geometry_groups=2,
              minimum_calibrated_geometry_groups=2, improvement_fraction=0.05, maximum_leaf_drift=0.05)
NATIVE_KEYS = ["routes", "samples", "sample_ms", "warmup_ms", "graph_batch", "threads", "affinity_mask",
               "native_aligned16", "native_timeout", "telemetry_ms", "path_prefix", "declared_environment_overrides"]
require, canonical, sha = v1.require, v1.canonical, v1.sha

def feature_vector(work, device):
    total = 0
    for item in work["collectives"]:
        total = v1.checked(total + v1.checked(item["width"] * item["independent"], "collective input", True), "total collective input", True)
    return [math.log2(1 + work["programs"] / device["sm_count"]),
            math.log2(1 + work["largest_tile"]),
            math.log2(1 + work["tile_live_bytes"] / (device["warp_size"] * 4)),
            math.log2(1 + work["elementwork"] / total)]

def load_dataset(summary, summary_id, device):
    require(summary.get("schema") == 1 and summary.get("status") == "completed_validated" and not summary.get("issues"), "requires completed validated summary")
    cohorts = summary["cohorts"]
    require(all(c.get("status") == "passed" and c.get("finished") and c.get("comparison_eligible") is True for c in cohorts), "cohort final gate failed")
    labels = {w: [c["label"] for c in cohorts if c["worker"] == w] for w in (0, 4, 8)}
    require(set(labels[0]) == {"baseline", "recheck"} and len(labels[0]) == 2 and len(labels[8]) == 1 and len(labels[4]) <= 1 and
            len(cohorts) == sum(map(len, labels.values())), "requires baseline0, worker8, recheck0 and optional worker4")
    implementations = {c["implementation_receipt_set_sha256"] for c in cohorts}
    require(len(implementations) == 1, "implementation receipts differ within summary")
    configurations = []
    for cohort in cohorts:
        config = {k: cohort["configuration"].get(k) for k in NATIVE_KEYS}
        for name in ("native_scan_chunk", "native_independent_axis"):
            value = cohort["configuration"].get(name, 0)
            require(type(value) is int and value == 0, "worker calibration requires structural flags zero")
            config[name] = value
        configurations.append(config)
    require(all(c == configurations[0] for c in configurations), "native measurement configurations differ")
    seen, rows = set(), []
    for item in summary["cases"]:
        case = item["case"]
        require(case.get("fast_math", False) is False, "strict math model only")
        identity = sha(canonical(case))
        require(identity not in seen, "duplicate case within dataset")
        seen.add(identity)
        base = item["cohorts"]["baseline"]
        samples0, median0 = v1.event_samples(base)
        work = v1.parse_work(base["routes"]["native"]["realization"])
        group, geometry = v1.geometry_group(work)
        sources = {k: v["normalized_source_sha256"] for k, v in base["source_files"].items()}
        row = dict(record_id=summary_id + ":" + identity, dataset_id=summary_id, case_identity=identity, case=case,
                   group=group, geometry=geometry, logical_work=work, features=feature_vector(work, device),
                   prefix=any(c["kind"] == 3 for c in work["collectives"]),
                   source_identity=sources, fixture_sha256=base["fixture_sha256"], samples={"0": samples0}, medians_us={"0": median0})
        for key, label in (("8", labels[8][0]), ("recheck", "recheck")):
            other = item["cohorts"][label]
            comparison = item["comparisons_to_native0"][label]
            require(comparison["status"] == "valid_matched_worker_pair", "invalid paired comparison")
            require(other["fixture_sha256"] == base["fixture_sha256"] and other["manifest"] == base["manifest"], "paired fixture/manifest mismatch")
            require({k: v["normalized_source_sha256"] for k, v in other["source_files"].items()} == sources, "paired source changes beyond worker hint")
            require(v1.parse_work(other["routes"]["native"]["realization"]) == work, "paired logical facts differ")
            samples, median = v1.event_samples(other)
            ratio = median / median0
            require(math.isfinite(ratio) and ratio > 0 and math.isclose(comparison["nativeN_over_native0"], ratio, rel_tol=1e-12), "paired ratio mismatch")
            row["samples"][key], row["medians_us"][key] = samples, median
            if key == "8":
                row["observed_ratio"] = ratio
                row["target_log_score"] = math.log(ratio)
            else:
                row["recheck_ratio"] = ratio
        rows.append(row)
    require(rows, "empty dataset")
    return rows, dict(dataset_id=summary_id, implementation_identity=next(iter(implementations)), native_configuration=configurations[0], retained_rows=len(rows))

def fit_tree(rows):
    require(rows and not any(r["prefix"] for r in rows), "tree requires non-prefix training observations")
    counts = Counter(r["group"] for r in rows)
    weights = {r["record_id"]: 1.0 / counts[r["group"]] for r in rows}
    def stats(items):
        weight = sum(weights[r["record_id"]] for r in items)
        mean = sum(weights[r["record_id"]] * r["target_log_score"] for r in items) / weight
        loss = sum(weights[r["record_id"]] * (r["target_log_score"] - mean) ** 2 for r in items)
        return mean, loss
    def build(items, depth, path):
        mean, loss = stats(items)
        node = dict(path=path, type="leaf", mean_log_score=mean, weighted_squared_error=loss,
                    rows=len(items), groups=sorted({r["group"] for r in items}), record_ids=[r["record_id"] for r in items],
                    feature_min=[min(r["features"][i] for r in items) for i in range(len(FEATURES))],
                    feature_max=[max(r["features"][i] for r in items) for i in range(len(FEATURES))])
        best = None
        if depth < POLICY["maximum_depth"]:
            for feature in range(len(FEATURES)):
                values = sorted({r["features"][feature] for r in items})
                for lo, hi in zip(values, values[1:]):
                    threshold = lo + (hi - lo) / 2
                    left = [r for r in items if r["features"][feature] <= threshold]
                    right = [r for r in items if r["features"][feature] > threshold]
                    if min(len({r["group"] for r in left}), len({r["group"] for r in right})) < POLICY["minimum_leaf_geometry_groups"]:
                        continue
                    split_loss = stats(left)[1] + stats(right)[1]
                    # Deterministic ties: feature order, then lower threshold.
                    if split_loss < loss - 1e-12 and (best is None or split_loss < best[0] - 1e-12):
                        best = (split_loss, feature, threshold, left, right)
        if best is not None:
            _, feature, threshold, left, right = best
            node.update(type="split", feature=feature, threshold=threshold,
                        left=build(left, depth + 1, path + "L"), right=build(right, depth + 1, path + "R"))
        return node
    return build(rows, 0, "root")

def locate(tree, vector):
    while tree["type"] == "split":
        tree = tree["left"] if vector[tree["feature"]] <= tree["threshold"] else tree["right"]
    return tree

def supported(leaf, vector):
    return all(lo - 1e-12 <= value <= hi + 1e-12 for lo, hi, value in zip(leaf["feature_min"], leaf["feature_max"], vector))

def leaves(tree):
    return [tree] if tree["type"] == "leaf" else leaves(tree["left"]) + leaves(tree["right"])

def train_profile(all_rows):
    rows = [r for r in all_rows if not r["prefix"]]
    result = dict(status="no_nonprefix_training_data", tree=None, inner_predictions=[],
                  retained_rows=len(all_rows), fitted_nonprefix_rows=len(rows), prefix_default_rows=len(all_rows) - len(rows))
    if not rows:
        return result
    tree = fit_tree(rows)
    predictions = []
    for group in sorted({r["group"] for r in rows}):
        train = [r for r in rows if r["group"] != group]
        held = [r for r in rows if r["group"] == group]
        if not train:
            continue
        inner = fit_tree(train)
        for row in held:
            leaf = locate(inner, row["features"])
            predictions.append(dict(record_id=row["record_id"], held_group=group,
                predicted_log_score=leaf["mean_log_score"], observed_log_score=row["target_log_score"],
                absolute_error=abs(leaf["mean_log_score"] - row["target_log_score"]),
                inner_supported=supported(leaf, row["features"]) and len(leaf["groups"]) >= POLICY["minimum_leaf_geometry_groups"],
                inner_leaf_path=leaf["path"], full_training_leaf_path=locate(tree, row["features"])["path"]))
    for leaf in leaves(tree):
        items = [r for r in rows if r["record_id"] in leaf["record_ids"]]
        residuals = [p for p in predictions if p["inner_supported"] and p["full_training_leaf_path"] == leaf["path"]]
        groups = sorted({p["held_group"] for p in residuals})
        all_residuals = [p for p in predictions if p["full_training_leaf_path"] == leaf["path"]]
        spread = max(abs(math.log(sample / r["medians_us"][worker])) for r in items for worker in ("0", "8") for sample in r["samples"][worker])
        drift = max(abs(math.log(r["recheck_ratio"])) for r in items)
        reasons = []
        if len(leaf["groups"]) < POLICY["minimum_leaf_geometry_groups"]:
            reasons.append("insufficient_leaf_training_groups")
        if len(groups) < POLICY["minimum_calibrated_geometry_groups"]:
            reasons.append("insufficient_in_support_calibration_groups")
        if any(abs(r["recheck_ratio"] - 1) > POLICY["maximum_leaf_drift"] for r in items):
            reasons.append("leaf_training_drift_exceeds_limit")
        worst = max((p["absolute_error"] for p in residuals), default=None)
        leaf["calibration"] = dict(eligible_groups=groups, eligible_record_ids=[p["record_id"] for p in residuals],
            unsupported_record_ids=[p["record_id"] for p in all_residuals if not p["inner_supported"]],
            worst_all_inner_error=max((p["absolute_error"] for p in all_residuals), default=None),
            worst_supported_inner_error=worst, timing_log_deviation=spread, default_drift_log_deviation=drift,
            empirical_margin=None if worst is None else worst + 2 * spread + drift, reasons=reasons)
    result.update(status="fitted_diagnostic_only", tree=tree, inner_predictions=predictions,
                  global_worst_inner_absolute_error=max((p["absolute_error"] for p in predictions), default=None))
    return result

def decide(profile, row):
    decision = dict(production_hint=0, diagnostic_candidate=0, reason="default")
    if row["prefix"]:
        decision["reason"] = "prefix_semantic_default"
        return decision
    if profile["tree"] is None:
        decision["reason"] = "no_nonprefix_training_data"
        return decision
    leaf = locate(profile["tree"], row["features"])
    decision.update(leaf_path=leaf["path"], relative_log_score=leaf["mean_log_score"])
    if not supported(leaf, row["features"]):
        decision["reason"] = "outside_leaf_feature_support"
        return decision
    calibration = leaf["calibration"]
    if calibration["reasons"]:
        decision["reason"] = ";".join(calibration["reasons"])
        return decision
    upper = leaf["mean_log_score"] + calibration["empirical_margin"]
    decision["empirical_upper_log_score"] = upper
    if upper < math.log1p(-POLICY["improvement_fraction"]):
        decision.update(diagnostic_candidate=8, reason="supported_conditional_candidate_requires_heldout_validation")
    else:
        decision["reason"] = "no_conservative_improvement"
    return decision

def outer_diagnostics(rows):
    folds = []
    for group in sorted({r["group"] for r in rows}):
        train, held = [r for r in rows if r["group"] != group], [r for r in rows if r["group"] == group]
        profile = train_profile(train)
        predictions = []
        for row in held:
            decision = decide(profile, row)
            predictions.append(dict(record_id=row["record_id"], decision=decision, observed_ratio=row["observed_ratio"],
                observed_policy_ratio=row["observed_ratio"] if decision["diagnostic_candidate"] else 1.0,
                signed_log_error=decision.get("relative_log_score", row["target_log_score"]) - row["target_log_score"] if not row["prefix"] else None))
        folds.append(dict(held_group=group, training_record_ids=[r["record_id"] for r in train], profile=profile, predictions=predictions))
    predictions = [p for fold in folds for p in fold["predictions"]]
    return dict(folds=folds, summary=dict(retained_rows=len(rows), geometry_groups=len({r["group"] for r in rows}),
        reasons=dict(Counter(p["decision"]["reason"] for p in predictions)),
        diagnostic_nondefault_rows=sum(p["decision"]["diagnostic_candidate"] != 0 for p in predictions),
        observed_worst_policy_ratio=max(p["observed_policy_ratio"] for p in predictions),
        observed_selected_regressions=sum(p["decision"]["diagnostic_candidate"] != 0 and p["observed_ratio"] > 1 for p in predictions),
        all_observed_worker8_regressions=sum(r["observed_ratio"] > 1 for r in rows),
        all_observed_worker8_worst_ratio=max(r["observed_ratio"] for r in rows)))

def calibrate(summaries, facts):
    device = v1.device_facts(facts)
    rows, datasets = [], []
    for summary_id, summary in summaries:
        items, identity = load_dataset(summary, summary_id, device)
        rows.extend(items)
        datasets.append(identity)
    require(len({d["dataset_id"] for d in datasets}) == len(datasets), "same dataset supplied twice")
    require(len({d["implementation_identity"] for d in datasets}) == 1, "cannot merge different implementation receipts")
    require(all(d["native_configuration"] == datasets[0]["native_configuration"] for d in datasets), "cannot merge different native timing configurations")
    profile = train_profile(rows)
    return dict(schema="collective-worker-conditional-v2", status="diagnostic_fit_only", production_autoselection_enabled=False,
        production_hint=0, fixed_policy=POLICY, feature_names=FEATURES, device_normalization=device,
        device_declared_identity=facts["identity"], datasets=datasets, training_observations=rows,
        full_training_profile=profile, full_training_predictions=[dict(record_id=r["record_id"], decision=decide(profile, r)) for r in rows],
        outer_leave_one_geometry_group_out=outer_diagnostics(rows),
        limitations=["All predictions are dimensionless relative log scores, not latency or physical occupancy.",
                     "Leaf support boxes and empirical envelopes do not imply joint coverage or statistical confidence.",
                     "Local envelope excludes unsupported inner predictions but retains every residual and global maximum.",
                     "Full-training decisions are not validation. Deployment requires subsequent replicated heldout validation.",
                     "Device receipts are declared provenance; completed summaries own the actual raw-file/oracle validation."])

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--device-facts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "new output directory required")
    packets = [(path, path.read_bytes()) for path in args.summary]
    device_bytes = args.device_facts.read_bytes()
    args.output.mkdir(parents=True)
    inputs = dict(summaries=[], device_facts=dict(path=str(args.device_facts.resolve()), sha256=sha(device_bytes)),
                  script=v1.receipt(__file__), parser_dependency=v1.receipt(V1_PATH), design=v1.receipt(HERE / "DESIGN.md"))
    for index, (path, data) in enumerate(packets):
        (args.output / f"input-summary-{index}.json").write_bytes(data)
        inputs["summaries"].append(dict(path=str(path.resolve()), sha256=sha(data), bytes=len(data)))
    (args.output / "input-device-facts.json").write_bytes(device_bytes)
    try:
        report = calibrate([(sha(data), json.loads(data.decode("utf-8-sig"))) for _, data in packets], json.loads(device_bytes.decode("utf-8-sig")))
        code = 0
    except (ValueError, KeyError, TypeError, OverflowError) as error:
        report = dict(schema="collective-worker-conditional-v2", status="rejected_no_fit", error=str(error),
                      production_hint=0, production_autoselection_enabled=False, fixed_policy=POLICY)
        code = 2
    report["inputs"] = inputs
    report["profile_id"] = sha(canonical(report))
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(status=report["status"], profile_id=report["profile_id"], output=str(args.output / "report.json"))))
    return code

if __name__ == "__main__":
    sys.exit(main())
