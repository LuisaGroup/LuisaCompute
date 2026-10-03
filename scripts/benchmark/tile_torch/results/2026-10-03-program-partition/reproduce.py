"""CPU-only verification of public bytes, recorded evidence and the NNLS fit."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
import calibrate


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def equivalent(left, right):
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(equivalent(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return isinstance(right, list) and len(left) == len(right) and all(equivalent(a, b) for a, b in zip(left, right))
    if isinstance(left, float):
        return type(right) in (float, int) and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
    return type(left) is type(right) and left == right


for name, recorded in read("receipts.json")["files"].items():
    path = (ROOT / name).resolve()
    assert path.is_relative_to(ROOT) and path.is_file(), name
    data = path.read_bytes()
    assert len(data) == recorded["bytes"] and hashlib.sha256(data).hexdigest() == recorded["sha256"], name

summary, recorded = read("validation.json"), read("model.json")
assert summary["status"] == "completed_validated_native"
assert summary["native_revalidation"]["status"] == "passed"
assert summary["native_revalidation"]["original_cohort_status"] == "failed"
assert len(summary["cases"]) == 16
native_count = torch_count = sample_count = 0
for row in summary["cases"]:
    default = row["cohorts"]["default"]
    assert default["original_cohort_status"] == "failed"
    for name, item in row["cohorts"].items():
        assert item["status"] == "validated" and item["native_comparison_eligible"]
        assert item["fixture_sha256"] == default["fixture_sha256"] and item["manifest"] == default["manifest"]
        for source, proof in item["source_files"].items():
            assert proof["original_source_sha256"] == default["source_files"][source]["normalized_source_sha256"]
    groups = list(row["cohorts"].values())
    if "independent_torch_retest" in row:
        assert default["routes"]["torch"]["status"] == "failed"
        assert default["routes"]["torch"]["comparison"] == "missing_initial_torch_baseline"
        groups.append(row["independent_torch_retest"]["evidence"])
    for item in groups:
        for route, result in item["routes"].items():
            if "event_us" not in result:
                continue
            native_count += route == "native"
            torch_count += route == "torch"
            times = result["event_us"]["samples"]
            assert len(times) == 7 and all(math.isfinite(x) and x > 0 for x in times)
            assert statistics.median(times) == result["event_us"]["p50"]
            assert result["saved_output_recheck"]["failed_elements"] == 0
            sample_count += len(times)
assert (native_count, torch_count, sample_count) == (65, 16, 567)
assert calibrate.sha(calibrate.canonical(recorded["profile"])) == recorded["profile_id"]
replayed = json.loads(json.dumps(calibrate.calibrate(summary, read("device.json"))))
keys = ("profile", "full_training_model", "training_observations", "full_training_evaluation", "full_training_metrics",
        "outer_leave_geometry_out", "baseline_recheck_and_all_candidates", "limitations")
for key in keys:
    assert equivalent(replayed[key], recorded[key]), key
print(json.dumps(dict(status="passed", native_executions=native_count, torch_executions=torch_count, samples=sample_count,
    bit_exact_numerical_projection=all(replayed[k] == recorded[k] for k in keys),
    public_profile_id=recorded["profile_id"], original_profile_id=recorded["original_profile_id"],
    logo=replayed["outer_leave_geometry_out"]["summary"])))
