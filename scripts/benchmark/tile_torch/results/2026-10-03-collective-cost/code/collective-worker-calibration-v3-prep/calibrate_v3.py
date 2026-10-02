"""Frozen experimental worker0/8 point-score profile; no GPU, confidence claim or heldout input."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
V2_PATH = HERE.parent / "collective-worker-calibration-v2-prep/calibrate_v2.py"
V2_SHA256 = "b7915f90993ffe5afbe00a7f4c98f4cccf92e48e71109809b5afa2472c09b4fe"
if hashlib.sha256(V2_PATH.read_bytes()).hexdigest() != V2_SHA256:
    raise ValueError("frozen v2 parser dependency changed")
sys.path.insert(0, str(V2_PATH.parent))
import calibrate_v2 as v2
v1 = v2.v1
require, canonical, sha = v1.require, v1.canonical, v1.sha

FEATURES = ["log2_programs_per_sm", "log2_largest_logical_tile_elements",
            "log2_explicit_live_bytes_per_warp_fp32_reference", "log2_elementwork_per_collective_input",
            "log2_maximum_contribution_width_per_warp", "log2_total_independent_elements",
            "sum_input_fraction", "maximum_input_fraction", "log2_nominal_access_bytes_per_fp32_collective_input"]
POLICY = dict(candidates=[0, 8], prefix_default=0, unseen_minimum_default=0, maximum_depth=2,
              minimum_leaf_geometry_groups=2, improvement_fraction=0.05,
              confidence="uncalibrated_point_prediction", enabled_by_default=False)

def feature_vector(work, device):
    """Binary64 formulas; integer products/sums are checked uint64 before conversion."""
    total = independent = summed = maximum = 0
    require(work["collectives"], "no collective facts")
    for item in work["collectives"]:
        volume = v1.checked(item["width"] * item["independent"], "collective input", True)
        total = v1.checked(total + volume, "total collective input", True)
        independent = v1.checked(independent + item["independent"], "total independent elements", True)
        if item["kind"] == 0:
            summed = v1.checked(summed + volume, "SUM input", True)
        elif item["kind"] == 2:
            maximum = v1.checked(maximum + volume, "MAXIMUM input", True)
    nominal = v1.checked(work["read_bytes"] + work["write_bytes"], "nominal access bytes")
    width = max(item["width"] for item in work["collectives"])
    values = [math.log2(1.0 + float(work["programs"]) / float(device["sm_count"])),
              math.log2(1.0 + float(work["largest_tile"])),
              math.log2(1.0 + float(work["tile_live_bytes"]) / (float(device["warp_size"]) * 4.0)),
              math.log2(1.0 + float(work["elementwork"]) / float(total)),
              math.log2(1.0 + float(width) / float(device["warp_size"])),
              math.log2(1.0 + float(independent)),
              float(summed) / float(total), float(maximum) / float(total),
              math.log2(1.0 + float(nominal) / (float(total) * 4.0))]
    require(all(math.isfinite(value) for value in values), "nonfinite features")
    return values

def load_dataset(summary, identity, device):
    rows, receipt = v2.load_dataset(summary, identity, device)
    for row in rows:
        row["features"] = feature_vector(row["logical_work"], device)
        row["unseen_minimum"] = any(c["kind"] == 1 for c in row["logical_work"]["collectives"])
    return rows, receipt

def fit_tree(rows):
    require(rows, "tree requires training observations")
    counts = Counter(r["group"] for r in rows)
    weights = {r["record_id"]: 1.0 / counts[r["group"]] for r in rows}
    def stats(items):
        weight = sum(weights[r["record_id"]] for r in items)
        mean = sum(weights[r["record_id"]] * r["target_log_score"] for r in items) / weight
        return mean, sum(weights[r["record_id"]] * (r["target_log_score"] - mean) ** 2 for r in items)
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
                    threshold = lo + (hi - lo) / 2.0
                    left = [r for r in items if r["features"][feature] <= threshold]
                    right = [r for r in items if r["features"][feature] > threshold]
                    if min(len({r["group"] for r in left}), len({r["group"] for r in right})) < POLICY["minimum_leaf_geometry_groups"]:
                        continue
                    split_loss = stats(left)[1] + stats(right)[1]
                    if split_loss < loss - 1e-12 and (best is None or split_loss < best[0] - 1e-12):
                        best = (split_loss, feature, threshold, left, right)
        if best is not None:
            _, feature, threshold, left, right = best
            node.update(type="split", feature=feature, threshold=threshold,
                        left=build(left, depth + 1, path + "L"), right=build(right, depth + 1, path + "R"))
        return node
    return build(rows, 0, "root")

locate, leaves = v2.locate, v2.leaves

def train_profile(rows):
    fitted = [r for r in rows if not r["prefix"] and not r["unseen_minimum"]]
    return dict(tree=fit_tree(fitted) if fitted else None, retained_rows=len(rows), fitted_rows=len(fitted),
                excluded_prefix_rows=sum(r["prefix"] for r in rows),
                excluded_unseen_minimum_rows=sum(r["unseen_minimum"] for r in rows))

def decide(profile, row):
    result = dict(production_hint=0, experimental_candidate=0, reason="default", confidence="uncalibrated")
    if row["prefix"] or row["unseen_minimum"]:
        result["reason"] = "prefix_semantic_default" if row["prefix"] else "unmeasured_minimum_default"
        return result
    if profile["tree"] is None:
        result["reason"] = "no_training_data"
        return result
    leaf = locate(profile["tree"], row["features"])
    result.update(leaf_path=leaf["path"], relative_log_score=leaf["mean_log_score"],
                  in_leaf_feature_box_diagnostic=v2.supported(leaf, row["features"]))
    if len(leaf["groups"]) < POLICY["minimum_leaf_geometry_groups"]:
        result["reason"] = "insufficient_training_groups"
    elif leaf["mean_log_score"] < math.log(1.0 - POLICY["improvement_fraction"]):
        result.update(experimental_candidate=8, reason="point_prediction_exceeds_five_percent_requires_heldout")
    else:
        result["reason"] = "point_prediction_no_five_percent_improvement"
    return result

def diagnostics(rows):
    folds = []
    for group in sorted({r["group"] for r in rows}):
        train, held = [r for r in rows if r["group"] != group], [r for r in rows if r["group"] == group]
        profile = train_profile(train)
        predictions = []
        for row in held:
            decision = decide(profile, row)
            predictions.append(dict(record_id=row["record_id"], decision=decision, observed_ratio=row["observed_ratio"],
                observed_policy_ratio=row["observed_ratio"] if decision["experimental_candidate"] else 1.0,
                signed_log_error=decision["relative_log_score"] - row["target_log_score"] if "relative_log_score" in decision else None))
        folds.append(dict(held_group=group, training_record_ids=[r["record_id"] for r in train], profile=profile, predictions=predictions))
    flat = [p for f in folds for p in f["predictions"]]
    return dict(folds=folds, summary=dict(retained_rows=len(rows), geometry_groups=len({r["group"] for r in rows}),
        reasons=dict(Counter(p["decision"]["reason"] for p in flat)),
        experimental_nondefault_rows=sum(p["decision"]["experimental_candidate"] != 0 for p in flat),
        selected_regressions=sum(p["decision"]["experimental_candidate"] != 0 and p["observed_ratio"] > 1 for p in flat),
        selected_regressions_over_five_percent=sum(p["decision"]["experimental_candidate"] != 0 and p["observed_ratio"] > 1.05 for p in flat),
        observed_worst_policy_ratio=max(p["observed_policy_ratio"] for p in flat),
        observed_policy_geomean=math.exp(sum(math.log(p["observed_policy_ratio"]) for p in flat) / len(flat)),
        worst_absolute_prediction_error=max((abs(p["signed_log_error"]) for p in flat if p["signed_log_error"] is not None), default=None),
        all_worker8_regressions=sum(r["observed_ratio"] > 1 for r in rows),
        all_worker8_worst_ratio=max(r["observed_ratio"] for r in rows)))

def flat_nodes(tree):
    nodes = []
    def visit(node):
        index = len(nodes)
        nodes.append({})
        if node["type"] == "leaf":
            nodes[index] = dict(feature=-1, threshold=0.0, left=-1, right=-1, score=node["mean_log_score"], training_groups=len(node["groups"]))
        else:
            left, right = visit(node["left"]), visit(node["right"])
            nodes[index] = dict(feature=node["feature"], threshold=node["threshold"], left=left, right=right, score=node["mean_log_score"], training_groups=len(node["groups"]))
        return index
    if tree is not None:
        visit(tree)
    return nodes

def calibrate(summaries, facts):
    device = v1.device_facts(facts)
    require(device == dict(compute_capability=89, sm_count=24, warp_size=32, max_resident_warps_per_sm=48), "device profile mismatch")
    require(facts["identity"].get("cuda_driver_api_version") == 13040 and facts["identity"].get("toolkit", "").replace("\\", "/").endswith("/v13.4"), "CUDA13.4 declared identity required")
    rows, datasets = [], []
    for identity, summary in summaries:
        loaded, receipt = load_dataset(summary, identity, device)
        rows.extend(loaded)
        datasets.append(receipt)
    require(len({d["dataset_id"] for d in datasets}) == len(datasets), "duplicate dataset")
    require(len({d["implementation_identity"] for d in datasets}) == 1, "implementation identities differ")
    require(all(d["native_configuration"] == datasets[0]["native_configuration"] for d in datasets), "native configurations differ")
    profile = train_profile(rows)
    report = dict(schema="collective-worker-experimental-v3", status="fitted_experimental_uncalibrated", production_hint=0,
        production_autoselection_enabled=False, fixed_policy=POLICY, feature_names=FEATURES,
        device_normalization=device, device_declared_identity=facts["identity"], datasets=datasets,
        training_observations=rows, full_training_profile=profile,
        full_training_predictions=[dict(record_id=r["record_id"], decision=decide(profile, r)) for r in rows],
        outer_leave_one_geometry_group_out=diagnostics(rows),
        deployment=dict(default_enabled=False, confidence="uncalibrated", strict_math=True, structural_rewrites=False,
            allowed_collective_kinds=[0, 2], score_threshold=math.log(0.95), comparison="strict_less_than",
            nodes=flat_nodes(profile["tree"]), required_device=device, declared_toolchain=facts["identity"]),
        timing_diagnostics=dict(max_default_recheck_deviation=max(abs(r["recheck_ratio"] - 1) for r in rows),
            max_sample_log_deviation=max(abs(math.log(sample / r["medians_us"][key])) for r in rows for key in ("0", "8") for sample in r["samples"][key])),
        limitations=["Optional point prediction; no empirical confidence interval or statistical guarantee is claimed.",
            "All LOGO residuals and negative observations are retained; feature boxes are diagnostics and do not gate decisions.",
            "Algebra and storage features are predictors, not proof of causality or actual register occupancy.",
            "Full-training decisions are not validation. Freeze this profile before independent heldout experiments.",
            "Device/toolchain receipts are declared provenance; runtime must verify its supported boundaries."])
    return report

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--device-facts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "new output directory required")
    packets = [(path, path.read_bytes()) for path in args.summary]
    facts = args.device_facts.read_bytes()
    args.output.mkdir(parents=True)
    inputs = dict(summaries=[], device_facts=v1.receipt(args.device_facts), script=v1.receipt(__file__),
                  v2_dependency=v1.receipt(V2_PATH), v1_dependency=v1.receipt(v2.V1_PATH), design=v1.receipt(HERE / "DESIGN.md"))
    for i, (path, data) in enumerate(packets):
        (args.output / f"input-summary-{i}.json").write_bytes(data)
        inputs["summaries"].append(dict(path=str(path.resolve()), sha256=sha(data), bytes=len(data)))
    (args.output / "input-device-facts.json").write_bytes(facts)
    report = calibrate([(sha(data), json.loads(data.decode("utf-8-sig"))) for _, data in packets], json.loads(facts.decode("utf-8-sig")))
    report["inputs"] = inputs
    report["profile_id"] = sha(canonical(report))
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    parity = dict(schema=1, profile_id=report["profile_id"], feature_names=FEATURES, deployment=report["deployment"],
        cases=[dict(record_id=r["record_id"], logical_work=r["logical_work"], features=r["features"], decision=decide(report["full_training_profile"], r)) for r in report["training_observations"]])
    (args.output / "cpp-parity.json").write_text(json.dumps(parity, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(status=report["status"], profile_id=report["profile_id"], output=str(args.output / "report.json"))))
    return 0

if __name__ == "__main__":
    sys.exit(main())
