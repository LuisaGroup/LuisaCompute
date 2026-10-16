"""Audit the fixed cooperating-program packing experiment independently.

No production benchmark statistics, cost or selection helper is imported.
Run without -O. --check-current-artifacts additionally checks the local binaries;
the saved evidence can be re-audited without those machine-specific paths.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median


HERE = Path(__file__).resolve().parent
SHAPES = ((37, 1537), (256, 3072), (768, 6144), (1024, 4096))
OPERATIONS = ("softmax", "rmsnorm", "layernorm")
CASES = tuple(f"{op}_{m}x{n}" for op in OPERATIONS for m, n in SHAPES)
PACKING = {"reference": 1, "candidate": 2}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu(item, phase):
    control = item["device_timing"]["control"]
    repetitions = control["repetitions"] if phase == "throughput" else 1
    return median(s["command_buffer_ns"] for s in control[phase]) / repetitions / 1000


def validate(row, directory, samples, variant, require_gpu):
    assert row["valid"] is True and row["backend"] == "metal" and row["name"] in CASES
    case, native = row["case"], row["native"]
    m, n, op = case["m"], case["n"], case["operation"]
    assert (m, n) in SHAPES and op in OPERATIONS and case["k"] == 1
    assert row["name"] == f"{op}_{m}x{n}"
    packing, width, subgroups = PACKING[variant], 256, 8
    assert native["planner_threads"] == width * packing
    assert native["reduction_programs_per_group"] == packing
    assert native["reduction_unroll_factor"] == 1 and native["reduction_lane_elements"] == 4
    assert native["cache_reduction_inputs"] is True
    assert native["reduction_cost_profile"] == "analytic"
    assert native["metal_subgroup_reductions"] is True and native["metal_max_threads"] == 1024
    assert native["max_reduction_striped_scalars_per_worker"] == 64
    assert native["output_elements"] == m * n
    assert len(native["execution_plans"]) == 1
    plan = native["execution_plans"][0]
    assert plan["optimized"] is True and plan["cost_basis"] == "metal_subgroup_reduction_v1"
    assert plan["candidates_considered"] == 1 and plan["candidates_rejected"] == 0
    assert plan["programs"] == m and plan["threads"] == width * packing
    assert plan["reduction_subgroups_per_program"] == subgroups
    assert plan["reduction_programs_per_group"] == packing
    assert plan["reduction_threadgroups"] == (m + packing - 1) // packing
    assert plan["reduction_unroll_factor"] == 1 and plan["reduction_lane_elements"] == 4
    assert plan["independent_subgroups"] is False

    # Enumerate this fixture's ownership independently of the compiler formulas.
    counts = [0] * width
    for element in range(n):
        counts[(element // 4) % width] += 1
    slots = max(counts)
    reductions = 1 if op == "rmsnorm" else 2
    rounds = slots * (3 if op == "rmsnorm" else 5)
    storage = slots * (1 if op == "rmsnorm" else 2)
    accesses = dict(global_read_bytes={"softmax": 4, "rmsnorm": 8, "layernorm": 12}[op],
                    global_write_bytes=4,
                    private_read_bytes=12 if op == "rmsnorm" else 24,
                    private_write_bytes=4 if op == "rmsnorm" else 8)
    assert plan["reduction_scalar_rounds"] == rounds
    assert plan["striped_storage_scalars_per_worker"] == storage <= 64
    assert plan["reduction_operations"] == reductions
    assert plan["shared_memory_bytes"] == reductions * subgroups * packing * 4
    assert plan["group_barrier_sites_before"] == plan["group_barrier_sites_after"] == reductions
    assert plan["reduction_payload_accesses_known"] is True
    for field, scale in (("reduction_payload_accesses_per_program", n),
                         ("reduction_payload_accesses_per_worker", slots)):
        assert plan[field] == {k: v * scale for k, v in accesses.items()}
    score = rounds + reductions * subgroups * 2 + 16 / packing
    waves = max(1, (m + 63) // 64)
    assert plan["normalized_cost"] == score and plan["concurrent_waves"] == waves
    assert plan["normalized_kernel_cost"] == score * waves
    assert math.isclose(plan["reduction_lane_utilization"], n / (width * slots), rel_tol=1e-12)

    source = directory / "sources" / (row["native_source_sha256"] + ".metal")
    assert digest(source) == row["native_source_sha256"]
    emitted = source.read_text()
    assert "simd_sum(" in emitted
    assert ("simd_max(" in emitted) == (op == "softmax")
    assert emitted.count("threadgroup_barrier(") == reductions
    assert row["torch"]["output_policy"] == ("preallocated_out" if op == "softmax" else "framework_return_value")
    for provider in ("native", "torch"):
        item = row[provider]
        # The runner checked every output against its independent FP64 oracle.
        # Raw output arrays are not archived: do not claim to re-run that oracle here.
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


def audit(check_current_artifacts):
    frozen = {}
    for variant in PACKING:
        directory = HERE / variant
        report = json.loads((directory / "results.json").read_text())
        assert len(report["results"]) == len(CASES)
        assert {row["name"] for row in report["results"]} == set(CASES)
        frozen[variant] = {row["name"]: row for row in report["results"]}
        for row in report["results"]:
            validate(row, directory, 3, variant, require_gpu=False)

    report = json.loads((HERE / "replay/results.json").read_text())
    meta = report["metadata"]
    assert meta["rounds"] == 4 and meta["samples"] == 9
    assert meta["sample_ms"] == 30 and meta["warmup_ms"] == 200
    assert meta["artifacts_unchanged"] is True
    assert meta["native_variants"]["reference"] == meta["native_variants"]["candidate"]
    assert {"benchmark_tile_tirx", "libluisa-tile-bridge-tirx.dylib", "libluisa-benchmark-metal-timing.dylib",
            "libtvm_compiler.dylib", "libtvm_runtime.dylib", "libtvm_runtime_metal.dylib", "libtvm_ffi.dylib"} <= {
                Path(p).name for p in meta["artifacts_sha256"]}
    if check_current_artifacts:
        for path, expected in meta["artifacts_sha256"].items():
            assert digest(Path(path)) == expected, path
    for variant in PACKING:
        assert digest(HERE / variant / "results.json") == meta["source_reports"][variant]["sha256"]
    assert len(report["results"]) == 96
    perturbations = []
    for row in report["results"]:
        validate(row, HERE / "replay", 9, row["variant"], require_gpu=True)
        original = frozen[row["variant"]][row["name"]]
        assert row["native_source_sha256"] == original["native_source_sha256"]
        assert row["native"]["execution_plans"] == original["native"]["execution_plans"]
        timing = row["torch"]["device_timing"]
        perturbations.append(timing["command_buffer_throughput_us_p50"] / gpu(row["torch"], "throughput"))

    records = []
    metrics = ("gpu_throughput", "e2e_throughput", "gpu_latency", "e2e_latency")
    for name in CASES:
        rows = [row for row in report["results"] if row["name"] == name]
        assert len(rows) == 8
        first = [next(row["variant"] for row in rows if row["round"] == r) for r in range(4)]
        assert first.count("reference") == first.count("candidate") == 2
        for variant in PACKING:
            orders = [row["implementation_order"] for row in rows if row["variant"] == variant]
            assert orders.count(["native", "torch"]) == orders.count(["torch", "native"]) == 2
        pairs = [{row["variant"]: row for row in rows if row["round"] == r} for r in range(4)]
        assert all(set(pair) == set(PACKING) for pair in pairs)
        record = {"name": name}
        for metric in metrics:
            kind, phase = metric.split("_")

            def read(row, provider):
                item = row[provider]
                return gpu(item, phase) if kind == "gpu" else median(item[phase + "_us"])

            a = [read(pair["reference"], "native") for pair in pairs]
            b = [read(pair["candidate"], "native") for pair in pairs]
            t = [read(pair["candidate"], "torch") for pair in pairs]
            gains, ratios = [x / y for x, y in zip(a, b)], [x / y for x, y in zip(b, t)]
            record[metric] = dict(reference_us=median(a), candidate_us=median(b), torch_us=median(t),
                                  reference_rounds_us=a, candidate_rounds_us=b, torch_rounds_us=t,
                                  gain=median(gains), gain_min=min(gains), gain_max=max(gains),
                                  native_torch_ratio=median(ratios), ratio_min=min(ratios), ratio_max=max(ratios))
        records.append(record)
    return dict(status="pass", audit_sha256=digest(Path(__file__)),
                source_reports_sha256={v: digest(HERE / v / "results.json") for v in PACKING},
                replay_sha256=digest(HERE / "replay/results.json"),
                executed_output_validations=240, replay_pairs=48,
                recorded_unchanged_artifacts=len(meta["artifacts_sha256"]),
                current_artifacts_checked=check_current_artifacts,
                torch_probe_control_ratio_range=[min(perturbations), max(perturbations)],
                summaries={metric: dict(positive_medians=sum(r[metric]["gain"] > 1 for r in records),
                                       all_pairs_positive=sum(r[metric]["gain_min"] > 1 for r in records),
                                       all_pairs_negative=sum(r[metric]["gain_max"] < 1 for r in records),
                                       faster_torch_all_pairs=sum(r[metric]["ratio_max"] < 1 for r in records))
                           for metric in metrics}, records=records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, help="new JSON receipt; never overwrite an existing result")
    parser.add_argument("--check-current-artifacts", action="store_true")
    args = parser.parse_args()
    result = audit(args.check_current_artifacts)
    if args.receipt:
        with args.receipt.open("x") as output:
            json.dump(result, output, indent=2, allow_nan=False)
            output.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2, allow_nan=False))
