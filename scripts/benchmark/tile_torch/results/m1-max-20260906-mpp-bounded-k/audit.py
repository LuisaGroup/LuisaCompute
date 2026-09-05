#!/usr/bin/env python3
"""Independent evidence audit; imports no benchmark/statistics implementation.

Checks recorded complete-output receipts, not discarded output arrays. Never
turns admission failures or the scope search minimum into an accepted speedup.
"""

import argparse
from collections import Counter
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import median

if not __debug__:
    raise RuntimeError("Run this audit without -O; assertions are required")

HERE = Path(__file__).resolve().parent
SHAPES = [(128, 128, 61), (1024, 1024, 1537), (4096, 4096, 11008), (8192, 8192, 8192)]
PATHS = ("tile_native_mpp", "tile_tirx", "handwritten_mpp", "mps", "torch", "tile_tirx_mpp", "tile_tirx_mpp_views")
VIEW = PATHS[-1]
METRICS = ("e2e_throughput", "e2e_latency", "gpu_throughput", "gpu_latency")
ALIGNED_SOURCE = "b232075c58949157966874ef4a229d124b47e0df1c983804d22adb740c440ff5"
SCOPE_SHAPES = [(1024, 1024, 1024), (4096, 4096, 4096), (8192, 8192, 8192),
                (256, 11008, 4096), (4096, 4096, 11008), (2049, 4097, 1025)]
CONFIGS = [(32, 32, 1, 1, 0, 1, 4, 4), (128, 32, 4, 1, 0, 1, 4, 1),
           (64, 64, 4, 1, 0, 1, 4, 1), (32, 128, 4, 1, 0, 1, 4, 1), (64, 32, 2, 1, 0, 1, 2, 1)]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_artifacts(hashes):
    for name, sha in hashes.items():
        assert digest(Path(name)) == sha, "Artifact changed: " + name


def samples(values):
    assert isinstance(values, list) and len(values) == 5
    assert all(type(v) in (float, int) and math.isfinite(v) and v > 0 for v in values)
    return values


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8), (a, b)


def times(measurement, path):
    values = {"e2e_" + phase: samples(measurement[phase + "_us"])
              for phase in ("throughput", "latency")}
    for phase in ("throughput", "latency"):
        close(median(values["e2e_" + phase]), measurement[phase + "_us_p50"])
    if path == "handwritten_mpp":
        values.update({"gpu_" + phase: samples(measurement["gpu_" + phase + "_us"])
                       for phase in ("throughput", "latency")})
    else:
        device = measurement["device_timing"]
        assert device["method"] == "metal_compute_pass_timestamps_v1"
        assert device["scope"] == "sum_of_compute_encoder_gpu_intervals"
        assert device["host_samples_instrumented"] is False
        control = device["control"]
        assert control["method"] == "metal_command_buffer_timestamps_v1"
        assert control["scope"] == "sum_of_command_buffer_gpu_intervals"
        assert control["encoder_instrumentation"] is False
        assert type(control["repetitions"]) is int and 1 <= control["repetitions"] <= 64
        assert device["repetitions"] == control["repetitions"]
        for phase, repeats in (("throughput", control["repetitions"]), ("latency", 1)):
            entries = control[phase]
            assert all(type(e["command_buffers"]) is int and e["command_buffers"] > 0 for e in entries)
            values["gpu_" + phase] = samples([e["command_buffer_ns"] / (1000 * repeats) for e in entries])
            assert len(control["command_buffer_" + phase + "_us"]) == 5
            for a, b in zip(values["gpu_" + phase], control["command_buffer_" + phase + "_us"]):
                close(a, b)
            samples([e["compute_ns"] for e in device[phase]])
    for phase in ("throughput", "latency"):
        close(median(values["gpu_" + phase]), measurement["gpu_control_" + phase + "_us_p50"])
    return {key: median(v) for key, v in values.items()}


def describe(values):
    return dict(median=median(values), minimum=min(values), maximum=max(values), rounds=values)


def comparison(a, b):
    assert len(a) == len(b) and a
    ratios = [x / y for x, y in zip(a, b)]
    return describe(ratios) | dict(faster_rounds=sum(r < 1 for r in ratios), total_rounds=len(ratios))


def correctness(receipt, elements):
    assert receipt["checked_elements"] == elements
    assert receipt["atol"] == receipt["rtol"] == 1e-4
    assert type(receipt["max_abs_error"]) in (float, int)
    assert math.isfinite(receipt["max_abs_error"]) and receipt["max_abs_error"] >= 0


def replay_audit(report, current=False):
    meta, rows = report["metadata"], report["results"]
    assert meta["rounds"] == 14 and meta["balanced"] is True and meta["samples"] == 5
    assert meta["sample_ms"] == 20 and meta["warmup_ms"] == 100 and meta["threads"] == 8
    assert meta["shapes"] == [list(s) for s in SHAPES] and tuple(meta["paths"]) == PATHS
    assert meta["tirx_view_subgroup_fences"] == "reported"
    assert meta["artifacts_unchanged"] is True
    assert meta["loader_environment"]["DYLD_LIBRARY_PATH"] == "/tmp/luisa-tvm-mpp.VaKmzx/build/lib"
    if current:
        check_artifacts(meta["artifacts_sha256"])
    expected = [(r, s) for r in range(14) for s in SHAPES[r % 4:] + SHAPES[:r % 4]]
    assert [(r["round"], tuple(r["shape"])) for r in rows] == expected
    assert len(rows) == 56 and all(r["valid"] is True for r in rows)
    summaries, outputs, elements, errors = [], 0, 0, []
    for shape in SHAPES:
        cohort = [r for r in rows if tuple(r["shape"]) == shape]
        orders = [r["order"] for r in cohort]
        assert all(sorted(order) == sorted(PATHS) for order in orders)
        for path in PATHS:
            assert Counter(o.index(path) for o in orders) == {i: 2 for i in range(7)}
        for a, b in itertools.combinations(PATHS, 2):
            assert sum(o.index(a) < o.index(b) for o in orders) == 7
        measurements, sources, plans = {p: [] for p in PATHS}, {p: set() for p in PATHS}, set()
        for row in cohort:
            assert row["config"] == list(CONFIGS[0])
            for key, block in (("tirx_schedule", [32, 32, 32]), ("tirx_view_schedule", [128, 32, 1024])):
                schedule = row[key]
                assert schedule["gemm_block"] == block
                assert schedule["group_threads"] == 128 and schedule["copy_batch"] == 1
                assert schedule["pipeline_window"] == 1 and schedule["execution_scope"] == "group"
                assert schedule["cooperative_matrix"] is True
                assert schedule["no_vectorize"] is False and schedule["auto_vectorize"] is False
            for path in PATHS:
                item = row[path]
                assert item["valid"] is True and "error" not in item, (shape, path, item.get("error"))
                correctness(item["correctness"], shape[0] * shape[1])
                errors.append(item["correctness"]["max_abs_error"])
                outputs += 1
                elements += shape[0] * shape[1]
                measurement = item["measurement"]
                if path in ("tile_native_mpp", "handwritten_mpp"):
                    assert measurement["fast_math"] is False and measurement["relaxed_precision"] is False
                if path in ("tile_tirx_mpp", VIEW):
                    assert measurement["mpp_intrinsics"] > 0 and measurement["simdgroup_intrinsics"] == 0
                    assert measurement["forward_readonly_tile_loads"] is (path == VIEW)
                if path == "tile_tirx":
                    assert measurement["simdgroup_intrinsics"] > 0 and measurement["mpp_intrinsics"] == 0
                if path.startswith("tile_tirx"):
                    sha = item["source_sha256"]
                    source_file = HERE / "replay/sources" / (sha + ".metal")
                    assert digest(source_file) == sha
                    sources[path].add(sha)
                    if path == VIEW:
                        source = source_file.read_text()
                        assert ("dynamic_extent" in source) is (shape in SHAPES[:3])
                        assert "threadgroup float" not in source
                        assert len(measurement["execution_plans"]) == 1
                        plan = measurement["execution_plans"][0]
                        assert plan["threads"] == 128 and plan["shared_memory_bytes"] == 0
                        assert plan["cost_basis"] == "metal_mpp_memory_v2"
                        assert plan["independent_subgroups"] is True
                        assert len(plan["matrices"]) == 1
                        matrix = plan["matrices"][0]
                        assert [matrix[k] for k in ("subgroups_m", "subgroups_n", "atom_rows", "atom_columns")] == [4, 1, 4, 4]
                        assert matrix["persistent_accumulator"] and matrix["direct_accumulator_store"]
                        plans.add(json.dumps(plan, sort_keys=True))
                measurements[path].append(times(measurement, path))
        assert all(len(sources[p]) == 1 for p in PATHS if p.startswith("tile_tirx"))
        assert len(plans) == 1
        if shape == SHAPES[-1]:
            assert sources[VIEW] == {ALIGNED_SOURCE}, "Aligned control source changed"
        summary = dict(shape=list(shape), view_plan=json.loads(next(iter(plans))), paths={})
        for path in PATHS:
            summary["paths"][path] = dict(sources=sorted(sources[path]), metrics={})
            for metric in METRICS:
                values = [r[metric] for r in measurements[path]]
                summary["paths"][path]["metrics"][metric] = dict(
                    time_us=describe(values), **{
                        "versus_" + reference: comparison(values, [r[metric] for r in measurements[reference]])
                        for reference in ("mps", "torch", "tile_tirx_mpp")})
        summaries.append(summary)
    assert outputs == 392 and elements == 8325201920
    return dict(full_output_validations=outputs, checked_output_elements=elements, max_abs_error=max(errors),
                artifacts=len(meta["artifacts_sha256"]), all_paths_pass=True, results=summaries)


def admission_audit(current):
    report = read(HERE / "admission.json")
    assert report["artifacts_unchanged"] is True
    if current:
        check_artifacts(report["artifacts_sha256"])
    assert [r["shape"] for r in report["results"]] == [list(s) for s in SHAPES[:3]]
    for row in report["results"]:
        assert row["returncode"] == 2 and row["admitted"] is False and row["output_validated"] is False
        assert "no legal Metal MPP group plan" in row["stderr"]
        assert row["command"][6:9] == ["128", "32", "1024"]
    return dict(admission_rejections=3, validated_executions=0, report_sha256=digest(HERE / "admission.json"))


def scope_audit(current):
    file = HERE.parent / "m1-max-20260906-mpp-scope/search/results.json"
    report = read(file)
    meta, rows = report["metadata"], report["results"]
    assert meta["mode"] == "search" and meta["rounds"] == 1
    assert meta["samples"] == 5 and meta["sample_ms"] == 20 and meta["warmup_ms"] == 100
    assert meta["artifacts_unchanged"] is True
    if current:
        check_artifacts(meta["artifacts_sha256"])
    assert [(r["shape"], r["config"]) for r in rows] == [
        (list(shape), None if config is None else list(config))
        for index, shape in enumerate(SCOPE_SHAPES)
        for config in (None, *(CONFIGS[index % 5:] + CONFIGS[:index % 5]))]
    summaries = []
    for row in rows:
        assert row["valid"] is True
        correctness(row["correctness"], row["shape"][0] * row["shape"][1])
        measurement = row["measurement"]
        if row["config"] is not None:
            assert measurement["fast_math"] is False and measurement["relaxed_precision"] is False
        values = {}
        for metric, field in zip(METRICS, ("throughput_us", "latency_us", "gpu_throughput_us", "gpu_latency_us")):
            values[metric] = median(samples(measurement[field]))
            close(values[metric], measurement[field + "_p50"])
        summaries.append(dict(shape=row["shape"], config=row["config"], time_us=values))
    return dict(full_output_validations=len(rows), report_sha256=digest(file),
                acceptance=False, reason="Single-order screening; no independently accepted selection", results=summaries)


def self_test(report):
    probes = {
        "missing_row": lambda r: r["results"].pop(),
        "wrong_order": lambda r: r["results"][0]["order"].reverse(),
        "invalid_output": lambda r: r["results"][0][VIEW].update(valid=False),
        "nan_time": lambda r: r["results"][0][VIEW]["measurement"]["throughput_us"].__setitem__(0, float("nan")),
        "missing_source": lambda r: r["results"][0][VIEW].update(source_sha256="0" * 64),
        "wrong_element_count": lambda r: r["results"][0][VIEW]["correctness"].update(checked_elements=1),
        "wrong_scope": lambda r: r["results"][0][VIEW]["measurement"]["device_timing"]["control"].update(encoder_instrumentation=True),
        "changed_artifacts": lambda r: r["metadata"].update(artifacts_unchanged=False),
    }
    for name, mutate in probes.items():
        modified = copy.deepcopy(report)
        mutate(modified)
        try:
            replay_audit(modified)
        except (AssertionError, FileNotFoundError):
            continue
        raise AssertionError("Negative audit probe was accepted: " + name)
    return list(probes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-artifacts", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "audit.json")
    args = parser.parse_args()
    report = read(HERE / "replay/results.json")
    result = dict(replay=replay_audit(report, args.current_artifacts),
                  baseline=admission_audit(args.current_artifacts), scope=scope_audit(args.current_artifacts),
                  current_artifacts_checked=args.current_artifacts,
                  replay_report_sha256=digest(HERE / "replay/results.json"))
    if args.self_test:
        result["negative_probes_rejected"] = self_test(report)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"replay_outputs": result["replay"]["full_output_validations"],
                      "scope_outputs": result["scope"]["full_output_validations"],
                      "baseline_rejections": result["baseline"]["admission_rejections"],
                      "negative_probes": result.get("negative_probes_rejected", [])}))


if __name__ == "__main__":
    main()
