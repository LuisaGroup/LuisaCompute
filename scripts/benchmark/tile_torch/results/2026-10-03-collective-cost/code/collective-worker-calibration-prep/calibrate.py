"""Offline diagnostic fitting only. No subprocess, compiler, GPU or auto-selection."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import numpy as np

SCHEMA = "collective-worker-relative-v1"
FEATURES = ["bias", "log2_program_demand_w4_reference", "log2_width_per_warp",
            "log2_independent_elements", "log2_explicit_tile_bytes_per_w4_reference",
            "log2_elementwork_per_collective_input", "prefix_input_fraction",
            "log2_nominal_bytes_per_fp32_collective_input"]
# Fixed before the measurements; no search or held-out tuning is performed.
POLICY = dict(ridge_alpha=1.0, improvement_fraction=0.05, maximum_training_drift=0.05,
              minimum_training_groups=4, minimum_inner_training_groups=3)
MAX_U64 = (1 << 64) - 1
FACT_PATTERN = re.compile(r"(?:^|;\s*)collective-work-v1: programs=(\d+), elementwork=(\d+), read-bytes=(\d+), write-bytes=(\d+), tile-live-bytes=(\d+), largest-tile=(\d+)(?=;|$)")
COLLECTIVE_PATTERN = re.compile(r"(?:^|;\s*)collective=kind([0-3]):width(\d+):independent(\d+)(?=;|$)")

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

def checked(value, name, positive=False):
    require(type(value) is int and (1 if positive else 0) <= value <= MAX_U64, "invalid integer fact: " + name)
    return value

def parse_work(realization):
    require(isinstance(realization, str), "missing native realization")
    require(realization.count("collective-work-v1") == 1, "missing/duplicate collective-work-v1")
    facts = FACT_PATTERN.findall(realization)
    require(len(facts) == 1, "malformed collective-work-v1 facts")
    keys = ["programs", "elementwork", "read_bytes", "write_bytes", "tile_live_bytes", "largest_tile"]
    result = {key: checked(int(value), key, key in {"programs", "tile_live_bytes", "largest_tile"}) for key, value in zip(keys, facts[0])}
    collectives = COLLECTIVE_PATTERN.findall(realization)
    require(collectives and len(collectives) == realization.count("collective="), "missing/malformed collective enum records")
    require(not any(token in realization for token in ("scan-chunk=", "chunked-scans=", "independent-axis-extent=", "partitioned-collectives=")), "structural candidate cannot train worker-only model")
    result["collectives"] = [dict(kind=int(k), width=checked(int(w), "width", True), independent=checked(int(i), "independent", True)) for k, w, i in collectives]
    return result

def device_facts(raw):
    require(raw.get("schema") == 1, "unsupported device-facts schema")
    result = {name: checked(raw.get(name), name, True) for name in
              ("compute_capability", "sm_count", "warp_size", "max_resident_warps_per_sm")}
    require(result["compute_capability"] == 89, "worker-hint experiment is currently admitted only for SM89")
    require(isinstance(raw.get("identity"), dict) and raw["identity"], "missing declared device/toolchain identity")
    require(isinstance(raw.get("evidence"), list) and raw["evidence"], "missing device-fact evidence receipts")
    for entry in raw["evidence"]:
        require(isinstance(entry, dict) and isinstance(entry.get("path"), str) and
                re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")), "invalid declared device-fact receipt")
    return result

def features(work, device):
    total = independent = prefix = 0
    for collective in work["collectives"]:
        volume = checked(collective["width"] * collective["independent"], "collective volume", True)
        total = checked(total + volume, "collective input total", True)
        independent = checked(independent + collective["independent"], "independent total", True)
        if collective["kind"] == 3:  # INCLUSIVE_SUM in the recorded IR enum schema.
            prefix = checked(prefix + volume, "prefix total", True)
    width = max(c["width"] for c in work["collectives"])
    nominal_bytes = checked(work["read_bytes"] + work["write_bytes"], "nominal access total")
    reference_slots = device["sm_count"] * device["max_resident_warps_per_sm"] / 4
    vector = [1.0, math.log2(1 + work["programs"] / max(1.0, reference_slots)),
              math.log2(1 + width / device["warp_size"]), math.log2(1 + independent),
              math.log2(1 + work["tile_live_bytes"] / (4 * device["warp_size"] * 4)),
              math.log2(1 + work["elementwork"] / total), prefix / total,
              math.log2(1 + nominal_bytes / (total * 4))]
    require(all(math.isfinite(v) for v in vector), "nonfinite feature vector")
    return vector

def geometry_group(work):
    # Group storage/arithmetic variants with the same logical input geometry.
    # No benchmark ID, operation name, external shape or dtype enters a feature.
    geometry = dict(programs=work["programs"], collective_geometry=sorted({(c["width"], c["independent"]) for c in work["collectives"]}))
    return sha(canonical(geometry)), geometry

def event_samples(cohort):
    require(cohort.get("status") == "validated" and cohort.get("cohort_completion", {}).get("comparison_eligible") is True,
            "unvalidated/incomplete case cannot train")
    record = cohort["routes"]["native"]["event_us"]
    samples = record["samples"]
    require(isinstance(samples, list) and len(samples) == 7 and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in samples), "requires seven positive finite native event samples")
    median = statistics.median(samples)
    require(math.isclose(median, record["p50"], rel_tol=1e-12), "stored event median mismatch")
    return samples, median

def dataset(summary, facts):
    device = device_facts(facts)
    require(summary.get("schema") == 1 and summary.get("status") == "completed_validated" and not summary.get("issues"), "requires completed_validated worker summary without issues")
    cohorts = summary["cohorts"]
    require(all(c.get("status") == "passed" and c.get("finished") and c.get("comparison_eligible") is True for c in cohorts), "all cohort final gates must pass")
    labels = {worker: [c["label"] for c in cohorts if c["worker"] == worker] for worker in (0, 4, 8)}
    require(labels[0] in (["baseline"], ["baseline", "recheck"]) and len(labels[4]) == len(labels[8]) == 1 and len(cohorts) == sum(map(len, labels.values())), "expected baseline, one worker4/8 and optional final recheck")
    identities = {c["implementation_receipt_set_sha256"] for c in cohorts}
    require(len(identities) == 1, "cohort implementation identities differ")
    require(all(isinstance(identity, str) and re.fullmatch(r"[0-9a-f]{64}", identity) for identity in identities), "invalid implementation receipt identity")
    configs = [c["configuration"] for c in cohorts]
    require(all(c == configs[0] for c in configs), "cohort non-worker configuration differs")
    require(all(c.get("native_scan_chunk", 0) == c.get("native_independent_axis", 0) == 0 for c in configs), "structural settings cannot train worker-only model")
    rows = []
    seen = set()
    for ordinal, row in enumerate(summary["cases"]):
        case = row["case"]
        require(case.get("fast_math", False) is False, "strict math profile only")
        case_identity = sha(canonical(case))
        require(case_identity not in seen, "duplicate case identity")
        seen.add(case_identity)
        base = row["cohorts"]["baseline"]
        base_samples, base_median = event_samples(base)
        work = parse_work(base["routes"]["native"]["realization"])
        vector = features(work, device)
        group, geometry = geometry_group(work)
        observation = dict(ordinal=ordinal, case_identity=case_identity, case=case, group=group,
                           logical_geometry=geometry, logical_work=work, features=vector,
                           source_identity={k: v["normalized_source_sha256"] for k, v in base["source_files"].items()},
                           fixture_sha256=base["fixture_sha256"], samples={"0": base_samples}, medians_us={"0": base_median},
                           relative_log_scores={}, relative_ratios={}, recheck=None)
        for worker in (4, 8):
            label = labels[worker][0]
            candidate = row["cohorts"][label]
            require(row["comparisons_to_native0"][label]["status"] == "valid_matched_worker_pair", "candidate lacks a valid matched comparison")
            require(candidate["fixture_sha256"] == base["fixture_sha256"] and candidate["manifest"] == base["manifest"], "candidate fixture/manifest differs")
            require({k: v["normalized_source_sha256"] for k, v in candidate["source_files"].items()} == observation["source_identity"], "candidate source differs beyond worker hint")
            require(parse_work(candidate["routes"]["native"]["realization"]) == work, "worker candidate logical facts differ")
            samples, median = event_samples(candidate)
            ratio = median / base_median
            require(math.isfinite(ratio) and ratio > 0, "candidate ratio is not finite positive")
            require(math.isclose(row["comparisons_to_native0"][label]["nativeN_over_native0"], ratio, rel_tol=1e-12), "candidate relative ratio mismatch")
            observation["samples"][str(worker)] = samples
            observation["medians_us"][str(worker)] = median
            observation["relative_ratios"][str(worker)] = ratio
            observation["relative_log_scores"][str(worker)] = math.log(ratio)
        if "recheck" in labels[0]:
            recheck = row["cohorts"]["recheck"]
            require(row["comparisons_to_native0"]["recheck"]["status"] == "valid_matched_worker_pair", "default recheck lacks matched comparison")
            require(parse_work(recheck["routes"]["native"]["realization"]) == work, "recheck logical facts differ")
            samples, median = event_samples(recheck)
            require(math.isfinite(median / base_median) and median / base_median > 0, "recheck ratio is not finite positive")
            observation["recheck"] = dict(samples=samples, p50_us=median, ratio=median / base_median)
        rows.append(observation)
    require(rows, "empty training data")
    return rows, device, next(iter(identities))

def fit(rows, worker):
    x = np.asarray([r["features"][1:] for r in rows], dtype=np.float64)
    y = np.asarray([r["relative_log_scores"][str(worker)] for r in rows], dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    penalty = np.diag([0.0] + [POLICY["ridge_alpha"]] * x.shape[1])
    beta = np.linalg.solve(z.T @ z / len(x) + penalty, z.T @ y / len(x))
    require(np.isfinite(beta).all(), "nonfinite fitted coefficients")
    original = np.r_[beta[0] - np.dot(beta[1:], mean / scale), beta[1:] / scale]
    return dict(worker=worker, standardized_coefficients=beta.tolist(), original_feature_coefficients=original.tolist(),
                center=mean.tolist(), scale=scale.tolist(), training_rows=len(rows),
                feature_min=x.min(axis=0).tolist(), feature_max=x.max(axis=0).tolist())

def predict(model, vector):
    return float(np.dot(np.asarray(model["original_feature_coefficients"]), np.asarray(vector)))

def in_support(model, vector):
    return all(lo - 1e-12 <= v <= hi + 1e-12 for lo, hi, v in zip(model["feature_min"], model["feature_max"], vector[1:]))

def train_profile(rows):
    groups = sorted({r["group"] for r in rows})
    profile = dict(models={str(w): fit(rows, w) for w in (4, 8)}, groups=groups, uncertainty={}, reasons=[])
    if len(groups) < POLICY["minimum_training_groups"]:
        profile["reasons"].append("insufficient_training_groups")
    if any(r["recheck"] is None for r in rows):
        profile["reasons"].append("missing_default_recheck")
    drift = max((abs(math.log(r["recheck"]["ratio"])) for r in rows if r["recheck"] is not None), default=0.0)
    if any(abs(r["recheck"]["ratio"] - 1) > POLICY["maximum_training_drift"] for r in rows if r["recheck"] is not None):
        profile["reasons"].append("training_drift_exceeds_predeclared_limit")
    for worker in (4, 8):
        residuals = []
        for group in groups:
            train = [r for r in rows if r["group"] != group]
            test = [r for r in rows if r["group"] == group]
            if len({r["group"] for r in train}) < POLICY["minimum_inner_training_groups"]:
                continue
            model = fit(train, worker)
            residuals.extend(abs(r["relative_log_scores"][str(worker)] - predict(model, r["features"])) for r in test)
        # Empirical envelopes, not confidence intervals: seven timing samples
        # are correlated and do not represent seven independent experiments.
        spread = max(max(abs(math.log(x / r["medians_us"][str(w)])) for x in r["samples"][str(w)])
                     for r in rows for w in (0, worker))
        enough = len(residuals) == len(rows)
        margin = max(residuals, default=0.0) + 2 * spread + drift if enough else None
        profile["uncertainty"][str(worker)] = dict(inner_logo_absolute_residuals=residuals,
            worst_inner_logo_error=max(residuals, default=None), timing_log_deviation=spread,
            default_drift_log_deviation=drift, empirical_margin=margin,
            interpretation="training-only conservative empirical envelope; not calibrated coverage probability")
        if not enough:
            profile["reasons"].append("insufficient_inner_group_residuals")
    profile["reasons"] = sorted(set(profile["reasons"]))
    return profile

def decide(profile, vector):
    scores = {str(w): predict(profile["models"][str(w)], vector) for w in (4, 8)}
    result = dict(production_hint=0, diagnostic_candidate=0, reason="diagnostic_only_default", relative_log_scores=scores)
    if profile["reasons"]:
        result["reason"] = ";".join(profile["reasons"])
        return result
    if not all(in_support(profile["models"][str(w)], vector) for w in (4, 8)):
        result["reason"] = "outside_training_feature_bounds"
        return result
    intervals = {str(w): [scores[str(w)] - profile["uncertainty"][str(w)]["empirical_margin"],
                          scores[str(w)] + profile["uncertainty"][str(w)]["empirical_margin"]] for w in (4, 8)}
    result["empirical_log_envelopes"] = intervals
    winners = [w for w in (4, 8) if intervals[str(w)][1] < math.log1p(-POLICY["improvement_fraction"]) and
               intervals[str(w)][1] < intervals[str(12 - w)][0]]
    if len(winners) == 1:
        result.update(diagnostic_candidate=winners[0], reason="in_support_separated_empirical_candidate_not_deployable")
    else:
        result["reason"] = "ambiguous_or_no_conservative_improvement"
    return result

def diagnostics(rows):
    folds = []
    for group in sorted({r["group"] for r in rows}):
        train = [r for r in rows if r["group"] != group]
        held = [r for r in rows if r["group"] == group]
        if not train:
            folds.append(dict(held_group=group, status="insufficient_training_groups", held_case_identities=[r["case_identity"] for r in held]))
            continue
        profile = train_profile(train)
        predictions = []
        for row in held:
            decision = decide(profile, row["features"])
            selected = decision["diagnostic_candidate"]
            predictions.append(dict(case_identity=row["case_identity"], decision=decision,
                observed_relative_ratios=row["relative_ratios"],
                signed_log_errors={str(w): decision["relative_log_scores"][str(w)] - row["relative_log_scores"][str(w)] for w in (4, 8)},
                observed_diagnostic_policy_ratio=1.0 if selected == 0 else row["relative_ratios"][str(selected)]))
        folds.append(dict(held_group=group, status="diagnostic_only", training_case_identities=[r["case_identity"] for r in train],
                          profile=profile, predictions=predictions))
    predicted = [p for f in folds for p in f.get("predictions", [])]
    summary = dict(predicted_rows=len(predicted), retained_rows=len(rows),
        measured_regressions={str(w): sum(r["relative_ratios"][str(w)] > 1 for r in rows) for w in (4, 8)},
        measured_worst_ratio={str(w): max(r["relative_ratios"][str(w)] for r in rows) for w in (4, 8)},
        defaulted_rows=sum(p["decision"]["diagnostic_candidate"] == 0 for p in predicted),
        nondefault_diagnostic_rows=sum(p["decision"]["diagnostic_candidate"] != 0 for p in predicted),
        worst_held_group_diagnostic_policy_ratio=max((p["observed_diagnostic_policy_ratio"] for p in predicted), default=None),
        mean_absolute_log_error={str(w): statistics.mean(abs(p["signed_log_errors"][str(w)]) for p in predicted) if predicted else None for w in (4, 8)})
    return dict(folds=folds, summary=summary)

def calibrate(summary, facts):
    rows, device, implementation = dataset(summary, facts)
    return dict(schema=SCHEMA, status="diagnostic_fit_only", production_autoselection_enabled=False, production_hint=0,
        feature_names=FEATURES, fixed_policy=POLICY, device_normalization=device, device_declared_identity=facts["identity"],
        implementation_receipt_set_sha256=implementation, training_observations=rows,
        full_training_profile=train_profile(rows), leave_one_geometry_group_out=diagnostics(rows),
        limitations=["Relative log scores are dimensionless, not predicted latency or occupancy.",
                     "Support is the training coordinate-wise feature box; it does not prove full joint coverage.",
                     "Empirical residual/spread/drift envelopes are not statistical confidence intervals.",
                     "All source/data gates rely on the separately completed validated summary; raw files are not revalidated here.",
                     "Device evidence receipts are declared provenance, not proof that a device query matched the timed execution.",
                     "External heldout measurements are not read, fitted, tuned or scored by this tool.",
                     "Production selection remains zero regardless of diagnostic candidates; heldout proof is a later gate."])

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--device-facts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "output must be a new directory")
    summary_bytes = args.summary.read_bytes()
    device_bytes = args.device_facts.read_bytes()
    args.output.mkdir(parents=True)
    inputs = dict(summary=dict(path=str(args.summary.resolve()), sha256=sha(summary_bytes), bytes=len(summary_bytes)),
                  device_facts=dict(path=str(args.device_facts.resolve()), sha256=sha(device_bytes), bytes=len(device_bytes)),
                  script=receipt(__file__), numpy_version=np.__version__)
    # Preserve exact evidence even when preflight rejects the requested fit.
    (args.output / "input-summary.json").write_bytes(summary_bytes)
    (args.output / "input-device-facts.json").write_bytes(device_bytes)
    try:
        report = calibrate(json.loads(summary_bytes.decode("utf-8-sig")), json.loads(device_bytes.decode("utf-8-sig")))
        code = 0
    except (ValueError, KeyError, TypeError, OverflowError, np.linalg.LinAlgError) as error:
        report = dict(schema=SCHEMA, status="rejected_no_fit", error=str(error), production_hint=0,
                      production_autoselection_enabled=False, fixed_policy=POLICY)
        code = 2
    report["inputs"] = inputs
    report["profile_id"] = sha(canonical(report))
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(status=report["status"], profile_id=report["profile_id"], output=str(args.output / "report.json"))))
    return code

if __name__ == "__main__":
    sys.exit(main())
