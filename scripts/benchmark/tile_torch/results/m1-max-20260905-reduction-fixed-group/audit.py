"""Independent audit of the two predeclared fixed-group reduction comparisons.

No production scoring, statistics or selection helper is imported. Raw output
arrays are not archived; this checks full-output validation records, not a
post-hoc rerun of the FP64 oracle. --pilots-only never accepts replay timing.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median


HERE = Path(__file__).resolve().parent
SHAPES = ((37, 769), (1024, 1024), (16384, 257), (4096, 1024))
OPS = ("softmax", "rmsnorm", "layernorm")
CASES = tuple(f"{op}_{m}x{n}" for op in OPS for m, n in SHAPES)
VARIANTS = ("reference", "candidate", "automatic")
REPLAYS = {"fixed-replay": "reference", "automatic-replay": "automatic"}
METRICS = ("gpu_throughput", "e2e_throughput", "gpu_latency", "e2e_latency")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def facts(m, n, op, subgroups, packing):
    """Enumerate worker ownership; do not reuse stripe_slots or planner helpers."""
    width = 32 * subgroups
    counts = [0] * width
    for element in range(n):
        counts[(element // 4) % width] += 1
    slots = max(counts)
    reductions = 1 if op == "rmsnorm" else 2
    rounds = slots * (3 if op == "rmsnorm" else 5)
    storage = slots * (1 if op == "rmsnorm" else 2)
    score = rounds + reductions * subgroups * 2 + 16 / packing
    waves = max(1, (m + 63) // 64)
    return dict(subgroups=subgroups, packing=packing, width=width,
                threads=width * packing, slots=slots, rounds=rounds,
                storage=storage, reductions=reductions, score=score,
                waves=waves, kernel_score=score * waves)


def automatic_candidates(m, n, op):
    result = []
    for subgroups in range(1, 33):
        for packing in (range(1, 9) if subgroups == 1 else (1,)):
            candidate = facts(m, n, op, subgroups, packing)
            assert candidate["threads"] <= 1024 and candidate["storage"] <= 64
            result.append(candidate)
    assert len(result) == 39
    return result


def gpu(item, phase):
    control = item["device_timing"]["control"]
    repetitions = control["repetitions"] if phase == "throughput" else 1
    return median(s["command_buffer_ns"] for s in control[phase]) / repetitions / 1000


def validate(row, directory, samples, variant, require_gpu):
    assert row["valid"] is True and row["backend"] == "metal" and row["name"] in CASES
    case, native = row["case"], row["native"]
    m, n, op = case["m"], case["n"], case["operation"]
    assert (m, n) in SHAPES and op in OPS and case["k"] == 1
    assert row["name"] == f"{op}_{m}x{n}"
    if variant == "automatic":
        candidates = automatic_candidates(m, n, op)
        expected = min(candidates, key=lambda f: f["kernel_score"])
        assert native["planner_threads"] == native["reduction_programs_per_group"] == 0
    else:
        expected = facts(m, n, op, 1 if variant == "reference" else 2,
                         8 if variant == "reference" else 4)
        candidates = [expected]
        assert native["planner_threads"] == 256
        assert native["reduction_programs_per_group"] == expected["packing"]
    assert native["reduction_unroll_factor"] == 1 and native["reduction_lane_elements"] == 4
    assert native["cache_reduction_inputs"] is True
    assert native["forward_readonly_tile_loads"] is True
    assert native["shared_tile_materialization"] == "preserve"
    assert native["reduction_cost_profile"] == "analytic"
    assert native["metal_subgroup_reductions"] is True and native["metal_max_threads"] == 1024
    assert native["max_reduction_striped_scalars_per_worker"] == 64
    assert native["output_elements"] == m * n
    assert len(native["execution_plans"]) == 1
    plan = native["execution_plans"][0]
    assert plan["optimized"] is True and plan["cost_basis"] == "metal_subgroup_reduction_v1"
    assert plan["candidates_considered"] == len(candidates) and plan["candidates_rejected"] == 0
    assert plan["programs"] == m and plan["threads"] == expected["threads"]
    subgroups, packing = expected["subgroups"], expected["packing"]
    reductions, slots = expected["reductions"], expected["slots"]
    assert plan["reduction_subgroups_per_program"] == subgroups
    assert plan["reduction_programs_per_group"] == packing
    assert plan["reduction_threadgroups"] == (m + packing - 1) // packing
    assert plan["reduction_unroll_factor"] == 1 and plan["reduction_lane_elements"] == 4
    assert plan["independent_subgroups"] is (subgroups == 1)
    assert plan["reduction_scalar_rounds"] == expected["rounds"]
    assert plan["striped_storage_scalars_per_worker"] == expected["storage"] <= 64
    assert plan["reduction_operations"] == reductions
    partial_bytes = 0 if subgroups == 1 else reductions * subgroups * packing * 4
    barriers = 0 if subgroups == 1 else reductions
    assert plan["shared_memory_bytes"] == partial_bytes
    assert plan["group_barrier_sites_before"] == plan["group_barrier_sites_after"] == barriers
    assert plan["reduction_payload_accesses_known"] is True
    accesses = dict(global_read_bytes={"softmax": 4, "rmsnorm": 8, "layernorm": 12}[op],
                    global_write_bytes=4, private_read_bytes=12 if op == "rmsnorm" else 24,
                    private_write_bytes=4 if op == "rmsnorm" else 8)
    for field, scale in (("reduction_payload_accesses_per_program", n),
                         ("reduction_payload_accesses_per_worker", slots)):
        assert plan[field] == {k: v * scale for k, v in accesses.items()}
    assert plan["normalized_cost"] == expected["score"]
    assert plan["concurrent_waves"] == expected["waves"]
    assert plan["normalized_kernel_cost"] == expected["kernel_score"]
    assert math.isclose(plan["reduction_lane_utilization"], n / (expected["width"] * slots), rel_tol=1e-12)

    source = directory / "sources" / (row["native_source_sha256"] + ".metal")
    assert digest(source) == row["native_source_sha256"]
    emitted = source.read_text()
    assert "simd_sum(" in emitted
    assert ("simd_max(" in emitted) == (op == "softmax")
    assert emitted.count("threadgroup_barrier(") == barriers
    assert row["torch"]["output_policy"] == ("preallocated_out" if op == "softmax" else "framework_return_value")
    for provider in ("native", "torch"):
        item = row[provider]
        correctness = item["correctness"]
        assert math.isfinite(correctness["max_abs_error"]) and correctness["max_abs_error"] >= 0
        assert correctness["atol"] == (1e-5 if op == "layernorm" else 2e-6)
        assert correctness["rtol"] == 2e-5
        for phase in ("throughput", "latency"):
            assert len(item[phase + "_us"]) == samples
            assert all(math.isfinite(s) and s > 0 for s in item[phase + "_us"])
            assert math.isclose(median(item[phase + "_us"]), item[phase + "_us_p50"], rel_tol=1e-9)
        if require_gpu:
            timing, control = item["device_timing"], item["device_timing"]["control"]
            assert timing["host_samples_instrumented"] is False
            assert control["encoder_instrumentation"] is False
            assert control["method"] == "metal_command_buffer_timestamps_v1"
            assert control["repetitions"] > 0
            for phase in ("throughput", "latency"):
                assert len(control[phase]) == samples
                assert all(math.isfinite(s["command_buffer_ns"]) and s["command_buffer_ns"] > 0
                           and s["command_buffers"] > 0 for s in control[phase])
                assert math.isclose(gpu(item, phase), control[f"command_buffer_{phase}_us_p50"], rel_tol=1e-12)
        else:
            assert item.get("device_timing") is None


def audit(pilots_only, check_current_artifacts):
    if not __debug__:
        raise RuntimeError("Do not disable this audit's assertions with -O")
    frozen = {}
    environments = []
    environment_keys = ("platform", "torch_version", "torch_config", "threads", "thread_environment")
    for variant in VARIANTS:
        directory = HERE / variant
        report = json.loads((directory / "results.json").read_text())
        environments.append({k: report["metadata"][k] for k in environment_keys})
        assert report["metadata"]["samples"] == 3
        assert report["metadata"]["sample_ms"] == 10 and report["metadata"]["warmup_ms"] == 100
        assert len(report["results"]) == len(CASES)
        assert {row["name"] for row in report["results"]} == set(CASES)
        frozen[variant] = {row["name"]: row for row in report["results"]}
        for row in report["results"]:
            validate(row, directory, 3, variant, require_gpu=False)
    automatic = {name: automatic_candidates(row["case"]["m"], row["case"]["n"], row["case"]["operation"])
                 for name, row in frozen["automatic"].items()}
    result = dict(status="pilots_only" if pilots_only else "pass", audit_sha256=digest(Path(__file__)),
                  source_reports_sha256={v: digest(HERE / v / "results.json") for v in VARIANTS},
                  executed_output_validations=2 * len(CASES) * len(VARIANTS),
                  automatic_candidates=automatic, replays={})
    if pilots_only:
        assert not check_current_artifacts, "Pilot-only mode has no replay artifact manifest"
        return result
    artifact_sets = []
    for directory_name, baseline in REPLAYS.items():
        directory = HERE / directory_name
        report = json.loads((directory / "results.json").read_text())
        meta = report["metadata"]
        environments.append({k: meta[k] for k in environment_keys})
        assert meta["rounds"] == 4 and meta["samples"] == 9
        assert meta["sample_ms"] == 30 and meta["warmup_ms"] == 200
        assert meta["artifacts_unchanged"] is True
        assert meta["native_variants"]["reference"] == meta["native_variants"]["candidate"]
        artifact_sets.append(meta["artifacts_sha256"])
        assert {"benchmark_tile_tirx", "libluisa-tile-bridge-tirx.dylib", "libluisa-benchmark-metal-timing.dylib",
                "libtvm_compiler.dylib", "libtvm_runtime.dylib", "libtvm_runtime_metal.dylib", "libtvm_ffi.dylib"} <= {
                    Path(p).name for p in meta["artifacts_sha256"]}
        if check_current_artifacts:
            for path, expected in meta["artifacts_sha256"].items():
                assert digest(Path(path)) == expected, path
        sources = {"reference": baseline, "candidate": "candidate"}
        for variant, source in sources.items():
            assert result["source_reports_sha256"][source] == meta["source_reports"][variant]["sha256"]
        assert len(report["results"]) == 8 * len(CASES)
        perturbations = []
        for row in report["results"]:
            variant = sources[row["variant"]]
            validate(row, directory, 9, variant, require_gpu=True)
            original = frozen[variant][row["name"]]
            assert row["native_source_sha256"] == original["native_source_sha256"]
            assert row["native"]["execution_plans"] == original["native"]["execution_plans"]
            timing = row["torch"]["device_timing"]
            perturbations.append(timing["command_buffer_throughput_us_p50"] / gpu(row["torch"], "throughput"))
        records = []
        for name in CASES:
            rows = [row for row in report["results"] if row["name"] == name]
            assert len(rows) == 8
            first = [next(row["variant"] for row in rows if row["round"] == r) for r in range(4)]
            assert first.count("reference") == first.count("candidate") == 2
            for variant in sources:
                orders = [row["implementation_order"] for row in rows if row["variant"] == variant]
                assert orders.count(["native", "torch"]) == orders.count(["torch", "native"]) == 2
            pairs = [{row["variant"]: row for row in rows if row["round"] == r} for r in range(4)]
            assert all(set(pair) == set(sources) for pair in pairs)
            score = lambda v: frozen[v][name]["native"]["execution_plans"][0]["normalized_kernel_cost"]
            record = dict(name=name, baseline_kernel_score=score(baseline), candidate_kernel_score=score("candidate"))
            for metric in METRICS:
                kind, phase = metric.split("_")

                def read(row, provider):
                    item = row[provider]
                    return gpu(item, phase) if kind == "gpu" else median(item[phase + "_us"])

                a = [read(pair["reference"], "native") for pair in pairs]
                b = [read(pair["candidate"], "native") for pair in pairs]
                t = [read(pair["candidate"], "torch") for pair in pairs]
                reference_torch = [read(pair["reference"], "torch") for pair in pairs]
                reference_ratios = [x / y for x, y in zip(a, reference_torch)]
                gains, ratios = [x / y for x, y in zip(a, b)], [x / y for x, y in zip(b, t)]
                record[metric] = dict(reference_us=median(a), candidate_us=median(b), torch_us=median(t),
                                      reference_rounds_us=a, candidate_rounds_us=b, torch_rounds_us=t,
                                      reference_torch_us=median(reference_torch), reference_torch_rounds_us=reference_torch,
                                      reference_native_torch_ratio=median(reference_ratios),
                                      reference_ratio_min=min(reference_ratios), reference_ratio_max=max(reference_ratios),
                                      gain=median(gains), gain_min=min(gains), gain_max=max(gains),
                                      native_torch_ratio=median(ratios), ratio_min=min(ratios), ratio_max=max(ratios))
            records.append(record)
        result["executed_output_validations"] += 2 * len(report["results"])
        result["replays"][directory_name] = dict(
            baseline=baseline, replay_sha256=digest(directory / "results.json"), replay_pairs=4 * len(CASES),
            recorded_unchanged_artifacts=len(meta["artifacts_sha256"]),
            current_artifacts_checked=check_current_artifacts,
            torch_probe_control_ratio_range=[min(perturbations), max(perturbations)],
            modeled_candidate_better=sum(r["candidate_kernel_score"] < r["baseline_kernel_score"] for r in records),
            summaries={metric: dict(positive_medians=sum(r[metric]["gain"] > 1 for r in records),
                                   all_pairs_positive=sum(r[metric]["gain_min"] > 1 for r in records),
                                   all_pairs_negative=sum(r[metric]["gain_max"] < 1 for r in records),
                                   faster_torch_all_pairs=sum(r[metric]["ratio_max"] < 1 for r in records),
                                   slower_torch_all_pairs=sum(r[metric]["ratio_min"] > 1 for r in records),
                                   baseline_faster_torch_all_pairs=sum(r[metric]["reference_ratio_max"] < 1 for r in records),
                                   baseline_slower_torch_all_pairs=sum(r[metric]["reference_ratio_min"] > 1 for r in records))
                       for metric in METRICS}, records=records)
    assert artifact_sets[0] == artifact_sets[1], "Binary/library change between the two experiments"
    assert all(env == environments[0] for env in environments), "Recorded environment differs between runs"
    result["environment"] = environments[0]
    assert result["executed_output_validations"] == 456
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilots-only", action="store_true")
    parser.add_argument("--check-current-artifacts", action="store_true")
    parser.add_argument("--receipt", type=Path, help="new JSON receipt, never overwrite existing evidence")
    args = parser.parse_args()
    result = audit(args.pilots_only, args.check_current_artifacts)
    if args.receipt:
        with args.receipt.open("x") as output:
            json.dump(result, output, indent=2, allow_nan=False)
            output.write("\n")
    summary = {k: v for k, v in result.items() if k not in ("automatic_candidates", "replays")}
    summary["replays"] = {k: {field: value for field, value in v.items() if field != "records"}
                          for k, v in result["replays"].items()}
    print(json.dumps(summary, indent=2, allow_nan=False))
