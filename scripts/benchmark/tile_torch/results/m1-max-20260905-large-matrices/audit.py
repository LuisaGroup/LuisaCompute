#!/usr/bin/env python3
"""Independently audit scale evidence; no benchmark statistics helpers imported.

Expected MPP-view admission failures remain failures, not successful samples.
The audit validates recorded full-output checks; output arrays are not saved.
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import median

if not __debug__:
    raise RuntimeError("This audit requires assertions; do not run with -O")

HERE = Path(__file__).resolve().parent
GEMM = [(2048, 2048, 2048), (4096, 4096, 4096), (8192, 8192, 8192),
        (256, 11008, 4096), (4096, 4096, 11008), (2049, 4097, 1025)]
ROWS = [(37, 8191), (1024, 8192), (1024, 16384), (4096, 8192), (8192, 4096), (16384, 4096)]
OPS = ("softmax", "rmsnorm", "layernorm")
PATHS = ("tile_native_mpp", "tile_tirx", "handwritten_mpp", "mps", "torch", "tile_tirx_mpp", "tile_tirx_mpp_views")
VIEW = PATHS[-1]
METRICS = ("e2e_throughput", "e2e_latency", "gpu_throughput", "gpu_latency")


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def samples(values, count=5):
    assert isinstance(values, list) and len(values) == count
    assert all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values)
    return values


def times(measurement, path):
    answer = {"e2e_" + phase: samples(measurement[phase + "_us"])
              for phase in ("throughput", "latency")}
    if path == "handwritten_mpp":
        answer.update({"gpu_" + phase: samples(measurement["gpu_" + phase + "_us"])
                       for phase in ("throughput", "latency")})
        return answer
    device = measurement["device_timing"]
    assert device["method"] == "metal_compute_pass_timestamps_v1"
    assert device["scope"] == "sum_of_compute_encoder_gpu_intervals"
    assert device["host_samples_instrumented"] is False
    control = device["control"]
    assert control["method"] == "metal_command_buffer_timestamps_v1"
    assert control["scope"] == "sum_of_command_buffer_gpu_intervals"
    assert control["encoder_instrumentation"] is False
    assert type(control["repetitions"]) is int and 1 <= control["repetitions"] <= 64
    assert control["repetitions"] == device["repetitions"]
    for phase, repetitions in (("throughput", control["repetitions"]), ("latency", 1)):
        entries = control[phase]
        assert all(type(v["command_buffers"]) is int and v["command_buffers"] > 0 for v in entries)
        answer["gpu_" + phase] = samples([v["command_buffer_ns"] / (1000 * repetitions) for v in entries])
        probe = device[phase]
        samples([v["compute_ns"] for v in probe])
        samples([v["command_buffer_ns"] for v in probe])
    return answer


def describe(values):
    return dict(median=median(values), minimum=min(values), maximum=max(values), rounds=values)


def comparison(a, b):
    ratios = [x / y for x, y in zip(a, b)]
    assert len(a) == len(b) and ratios
    return describe(ratios) | dict(faster_rounds=sum(r < 1 for r in ratios), total_rounds=len(ratios))


def check_correctness(record, count, operation="gemm"):
    assert record["atol"] == (1e-4 if operation == "gemm" else 1e-5 if operation == "layernorm" else 2e-6)
    assert record["rtol"] == (1e-4 if operation == "gemm" else 2e-5)
    assert math.isfinite(record["max_abs_error"]) and record["max_abs_error"] >= 0
    if operation == "gemm":
        assert record["checked_elements"] == count


def check_artifacts(hashes):
    for path, expected in hashes.items():
        assert digest(Path(path)) == expected, "Artifact changed: " + path


def pilot_audit():
    failed = read(HERE / "view-pilot/results.json")
    pilot = read(HERE / "view-pilot-patched/results.json")
    assert len(failed["results"]) == len(pilot["results"]) == 6
    assert all(r["valid"] is False and "lacks Metal MPP memory contract v2" in r["error"] for r in failed["results"])
    sources = {}
    for shape, row in zip(GEMM, pilot["results"]):
        assert [row["case"][k] for k in ("m", "n", "k")] == list(shape)
        if shape in GEMM[-2:]:
            assert row["valid"] is False and "no legal Metal MPP group plan" in row["error"]
            continue
        assert row["valid"] is True and row["block"] == [128, 32, 1024]
        for path in ("native", "torch"):
            record = row[path]["correctness"]
            assert record["atol"] == record["rtol"] == 1e-4
            assert math.isfinite(record["max_abs_error"]) and record["max_abs_error"] >= 0
        sha = row["native_source_sha256"]
        assert digest(HERE / "view-pilot-patched/sources" / (sha + ".metal")) == sha
        sources[shape] = sha
    return sources


def gemm_audit(current):
    directory = HERE / "gemm-replay"
    report = read(directory / "results.json")
    meta, rows = report["metadata"], report["results"]
    assert meta["rounds"] == 14 and meta["balanced"] is True and meta["samples"] == 5
    assert meta["sample_ms"] == 20 and meta["warmup_ms"] == 100
    assert meta["shapes"] == [list(s) for s in GEMM] and tuple(meta["paths"]) == PATHS
    assert meta["loader_environment"]["DYLD_LIBRARY_PATH"] == "/tmp/luisa-tvm-mpp.VaKmzx/build/lib"
    assert meta["artifacts_unchanged"] is True
    if current:
        check_artifacts(meta["artifacts_sha256"])
    assert len(rows) == 84
    expected_order = [(r, s) for r in range(14) for s in GEMM[r % 6:] + GEMM[:r % 6]]
    assert [(r["round"], tuple(r["shape"])) for r in rows] == expected_order
    pilot_sources = pilot_audit()
    valid_count, rejected_count, checked_count, results = 0, 0, 0, []
    for shape in GEMM:
        cohort = [r for r in rows if tuple(r["shape"]) == shape]
        orders = [r["order"] for r in cohort]
        assert all(sorted(o) == sorted(PATHS) for o in orders)
        for path in PATHS:
            assert Counter(o.index(path) for o in orders) == {i: 2 for i in range(7)}
        for a, b in itertools.combinations(PATHS, 2):
            assert sum(o.index(a) < o.index(b) for o in orders) == 7
        measurements, source_sets = {}, {}
        for path in PATHS:
            measurements[path] = []
            source_sets[path] = set()
            rejected = path == VIEW and shape in GEMM[-2:]
            for row in cohort:
                assert row["config"] == [32, 32, 1, 1, 0, 1, 4, 4]
                for key, block in (("tirx_schedule", [32, 32, 32]), ("tirx_view_schedule", [128, 32, 1024])):
                    schedule = row[key]
                    assert schedule["gemm_block"] == block
                    assert schedule["group_threads"] == 128 and schedule["copy_batch"] == 1
                    assert schedule["pipeline_window"] == 1 and schedule["execution_scope"] == "group"
                    assert schedule["cooperative_matrix"] is True
                item = row[path]
                if rejected:
                    assert item["valid"] is False and "no legal Metal MPP group plan" in item["error"]
                    assert "measurement" not in item
                    rejected_count += 1
                    continue
                assert item["valid"] is True, (shape, path, item.get("error"))
                check_correctness(item["correctness"], shape[0] * shape[1])
                valid_count += 1
                checked_count += shape[0] * shape[1]
                measurement = item["measurement"]
                if path in ("tile_native_mpp", "handwritten_mpp"):
                    assert measurement["fast_math"] is False and measurement["relaxed_precision"] is False
                if path in ("tile_tirx_mpp", VIEW):
                    assert measurement["mpp_intrinsics"] > 0 and measurement["simdgroup_intrinsics"] == 0
                    assert measurement["forward_readonly_tile_loads"] is (path == VIEW)
                if path.startswith("tile_tirx"):
                    sha = item["source_sha256"]
                    assert digest(directory / "sources" / (sha + ".metal")) == sha
                    source_sets[path].add(sha)
                measurements[path].append({metric: median(values) for metric, values in times(measurement, path).items()})
            if path.startswith("tile_tirx") and not rejected:
                assert len(source_sets[path]) == 1
                if path == VIEW:
                    assert source_sets[path] == {pilot_sources[shape]}
        result = dict(shape=list(shape), paths={})
        for path in PATHS:
            if not measurements[path]:
                result["paths"][path] = dict(status="rejected", rejected_rounds=14)
                continue
            values = {metric: [r[metric] for r in measurements[path]] for metric in METRICS}
            result["paths"][path] = dict(status="validated", sources=sorted(source_sets[path]), metrics={
                metric: dict(time_us=describe(v),
                             versus_mps=comparison(v, [r[metric] for r in measurements["mps"]]),
                             versus_torch=comparison(v, [r[metric] for r in measurements["torch"]]))
                for metric, v in values.items()})
        assert all(r["valid"] is (shape not in GEMM[-2:]) for r in cohort)
        results.append(result)
    assert valid_count == 560 and rejected_count == 28
    return dict(full_output_validations=valid_count, checked_output_elements=checked_count,
                pilot_output_validations=8, pilot_admission_rejections=2, initial_capability_rejections=6,
                admission_rejections=rejected_count, all_paths_pass=False,
                report_sha256=digest(directory / "results.json"), artifacts=len(meta["artifacts_sha256"]),
                environment={k: meta[k] for k in ("platform", "torch_version", "torch_git_version", "loader_environment")}, results=results)


def reduction_facts(operation, m, n, subgroups, packing):
    width = subgroups * 32
    full, tail = divmod(n, width * 4)
    slots = full * 4 + min(4, tail)
    reductions = 1 if operation == "rmsnorm" else 2
    storage = slots * (1 if operation == "rmsnorm" else 2)
    rounds = slots * (3 if operation == "rmsnorm" else 5)
    score = rounds + reductions * subgroups * 2 + 16 / packing
    return dict(subgroups=subgroups, packing=packing, slots=slots, storage=storage,
                rounds=rounds, score=score, kernel_score=score * max(1, (m + 63) // 64))


def reduction_audit(current):
    before = read(HERE / "reduction-artifacts-before.json")
    assert before == read(HERE / "reduction-artifacts-after.json")
    if current:
        check_artifacts(before)
    reports = [read(HERE / name / "results.json") for name in ("reduction-forward", "reduction-reverse")]
    results = []
    for report, shapes in zip(reports, (ROWS, ROWS[::-1])):
        meta = report["metadata"]
        assert meta["samples"] == 5 and meta["sample_ms"] == 30 and meta["warmup_ms"] == 100
        assert meta["metal_subgroup_reductions"] and meta["input_views"] and meta["cache_reduction_inputs"]
        assert meta["reduction_lane_elements"] == 4 and meta["reduction_unroll"] == 1
        assert meta["group_threads"] == 0 and meta["reduction_programs_per_group"] == 0
        assert meta["reduction_cost_profile"] == "analytic" and meta["shared_tile_materialization"] == "preserve"
        assert meta["loader_environment"]["DYLD_LIBRARY_PATH"] is None
        assert [(r["case"]["operation"], r["case"]["m"], r["case"]["n"]) for r in report["results"]] == [
            (op, m, n) for op in OPS for m, n in shapes]
    for op in OPS:
        for m, n in ROWS:
            cohort = [next(r for r in report["results"] if r["name"] == f"{op}_{m}x{n}") for report in reports]
            assert all(r["valid"] for r in cohort)
            assert cohort[0]["timing_order"] != cohort[1]["timing_order"]
            assert cohort[0]["native_source_sha256"] == cohort[1]["native_source_sha256"]
            assert cohort[0]["native"]["execution_plans"] == cohort[1]["native"]["execution_plans"]
            candidates = [reduction_facts(op, m, n, s, p) for s in range(1, 33)
                          for p in (range(1, 9) if s == 1 else [1])]
            legal = [c for c in candidates if c["storage"] <= 64]
            assert legal
            plan = cohort[0]["native"]["execution_plans"][0]
            selected = reduction_facts(op, m, n, plan["reduction_subgroups_per_program"], plan["reduction_programs_per_group"])
            assert selected["score"] == min(c["score"] for c in legal)
            assert plan["normalized_cost"] == selected["score"]
            assert plan["striped_storage_scalars_per_worker"] == selected["storage"] <= 64
            assert plan["reduction_scalar_rounds"] == selected["rounds"]
            measures = []
            for directory, row in zip(("reduction-forward", "reduction-reverse"), cohort):
                sha = row["native_source_sha256"]
                source_path = HERE / directory / "sources" / (sha + ".metal")
                assert digest(source_path) == sha
                source = source_path.read_text()
                expected_intrinsics = {"simd_sum(": 4 if op == "layernorm" else 2,
                                       "simd_max(": 2 if op == "softmax" else 0,
                                       "simd_min(": 0,
                                       "threadgroup_barrier(": 1 if op == "rmsnorm" else 2}
                assert all(source.count(name) == count for name, count in expected_intrinsics.items())
                for path in ("native", "torch"):
                    check_correctness(row[path]["correctness"], m * n, op)
                measures.append({path: {metric: median(v) for metric, v in times(row[path], path).items()}
                                 for path in ("native", "torch")})
            results.append(dict(name=f"{op}_{m}x{n}", selected=selected, candidates_legal=len(legal),
                                generated_intrinsics=expected_intrinsics, metrics={
                metric: dict(native_us=describe([r["native"][metric] for r in measures]),
                             torch_us=describe([r["torch"][metric] for r in measures]),
                             native_torch=comparison([r["native"][metric] for r in measures], [r["torch"][metric] for r in measures]))
                for metric in METRICS}))
    return dict(full_output_validations=72, artifacts=len(before), results=results,
                reports_sha256={name: digest(HERE / name / "results.json") for name in ("reduction-forward", "reduction-reverse")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-reductions", choices=("before", "after"))
    parser.add_argument("--gemm-only", action="store_true")
    parser.add_argument("--check-current-artifacts", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.snapshot_reductions:
        previous = read(HERE.parent / "m1-max-20260905-reduction-fixed-group/fixed-replay/results.json")
        paths = set(previous["metadata"]["artifacts_sha256"])
        paths.add("/tmp/luisa-tvmx-venv/lib/python3.14/site-packages/tvm/lib/libtvm_runtime_extra.dylib")
        answer = {p: digest(Path(p)) for p in sorted(paths)}
        with (HERE / f"reduction-artifacts-{args.snapshot_reductions}.json").open("x") as output:
            json.dump(answer, output, indent=2)
            output.write("\n")
        print("Recorded", len(answer), "reduction artifacts", args.snapshot_reductions, "timing")
    else:
        answer = dict(status="evidence_consistent_with_reported_rejections", audit_sha256=digest(Path(__file__)),
                      gemm=gemm_audit(args.check_current_artifacts))
        if not args.gemm_only:
            answer["reductions"] = reduction_audit(args.check_current_artifacts)
        if args.receipt:
            with args.receipt.open("x") as output:
                json.dump(answer, output, indent=2, allow_nan=False)
                output.write("\n")
        print(json.dumps({k: {n: v for n, v in value.items() if n != "results"} if isinstance(value, dict) else value
                          for k, value in answer.items()}, indent=2))
