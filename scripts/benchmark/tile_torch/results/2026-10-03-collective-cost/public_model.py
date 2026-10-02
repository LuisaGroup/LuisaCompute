"""Stable numerical projections used to verify the published calibration inputs."""
import hashlib
import json
import math


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_ids(value, mapping):
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [replace_ids(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: replace_ids(item, mapping) for key, item in value.items()}
    return value


def numerical_report(report, version):
    rows = report["training_observations"]
    # The public input has new byte hashes, hence new opaque dataset IDs. Row
    # order is the fixed original16 followed by extra16 order, never sorted by
    # observed timing or selected decision.
    mapping = {}
    for index, row in enumerate(rows):
        reference = f"row{index:02d}:{row['case']['id']}"
        if version == 1:
            mapping[row["case_identity"]] = reference
        else:
            mapping[row["record_id"]] = reference
    training = []
    for index, row in enumerate(rows):
        fields = ("case", "group", "logical_work", "features", "samples", "medians_us", "fixture_sha256", "source_identity")
        retained = {key: row[key] for key in fields}
        retained["reference"] = f"row{index:02d}:{row['case']['id']}"
        for key in ("relative_log_scores", "relative_ratios", "recheck", "prefix", "unseen_minimum",
                    "observed_ratio", "target_log_score", "recheck_ratio"):
            if key in row:
                retained[key] = row[key]
        training.append(retained)
    result = dict(version=version, status=report["status"], feature_names=report["feature_names"],
                  fixed_policy=report["fixed_policy"], device_normalization=report["device_normalization"],
                  training_observations=training)
    diagnostics = report["leave_one_geometry_group_out" if version == 1 else "outer_leave_one_geometry_group_out"]
    folds = []
    for fold in diagnostics["folds"]:
        retained = {key: value for key, value in fold.items() if key != "profile"}
        if "profile" in fold:
            # v1/v2 conservative profiles contain repeated nested calibration
            # residuals; retain their actual fitted models/tree and abstention
            # reasons once, with all fold predictions below. v3 retains its
            # full small tree including every training group/row receipt.
            profile = fold["profile"]
            if version == 1:
                retained["model"] = {key: profile[key] for key in ("models", "groups", "reasons")}
            elif version == 2:
                retained["model"] = compact_tree_profile(profile)
            else:
                retained["model"] = profile
        folds.append(retained)
    result["leave_one_geometry_group_out"] = dict(summary=diagnostics["summary"], folds=folds)
    if version == 1:
        result["full_training_profile"] = report["full_training_profile"]
    elif version == 2:
        result["full_training_profile"] = compact_tree_profile(report["full_training_profile"])
    else:
        result["full_training_profile"] = report["full_training_profile"]
        result["deployment"] = {key: value for key, value in report["deployment"].items() if key != "declared_toolchain"}
        result["timing_diagnostics"] = report["timing_diagnostics"]
        per_group = [math.exp(sum(math.log(p["observed_policy_ratio"]) for p in fold["predictions"]) / len(fold["predictions"]))
                     for fold in diagnostics["folds"]]
        result["geometry_equal_weight_logo"] = dict(groups=len(per_group),
            per_group_policy_geomean=per_group,
            policy_geomean=math.exp(sum(math.log(value) for value in per_group) / len(per_group)),
            interpretation="Each geometry group receives equal weight; variants/repeated rows within it share that weight.")
    if "full_training_predictions" in report:
        result["full_training_predictions"] = report["full_training_predictions"]
    result["limitations"] = report["limitations"]
    return replace_ids(result, mapping)


def compact_tree_profile(profile):
    def tree(node):
        if node is None:
            return None
        value = {key: val for key, val in node.items() if key not in ("left", "right", "calibration")}
        if "calibration" in node:
            value["calibration_summary"] = {key: val for key, val in node["calibration"].items()
                                             if key not in ("inner_residuals", "residuals")}
        for key in ("left", "right"):
            if key in node:
                value[key] = tree(node[key])
        return value
    return {key: tree(value) if key == "tree" else value for key, value in profile.items() if key != "inner_predictions"}
