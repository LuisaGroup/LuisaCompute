"""Independently audit the fixed-width input-cache experiment (read-only).

Uses only the standard library, not benchmark statistics/selection helpers.
Run without -O; assertions check raw timing medians, plans, sources, controls,
cohort completeness, source catalogs and replay order. The executed harness
compared complete outputs with an FP64 oracle; the raw output arrays are not
archived, so this offline audit checks recorded validation, not those arrays.
"""

import hashlib
import json
import math
from pathlib import Path
from statistics import median


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREFIX = "m1-max-20260905-input-cache-"
SHAPES = ((17, 257), (7, 1537), (64, 4096), (1024, 4096), (128, 8192))
OPS = ("sum", "softmax", "rmsnorm", "layernorm", "residual_layernorm")
CASES = {f"{op}_{m}x{n}" for op in OPS for m, n in SHAPES}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu(item, phase="throughput"):
    control = item["device_timing"]["control"]
    divisor = control["repetitions"] if phase == "throughput" else 1
    return median(sample["command_buffer_ns"] for sample in control[phase]) / divisor / 1000


def validate(row, directory, samples, cache):
    assert row["valid"] is True and row["backend"] == "metal" and row["name"] in CASES
    native = row["native"]
    assert native["cache_reduction_inputs"] is cache
    assert native["metal_subgroup_reductions"] is True and native["metal_max_threads"] == 1024
    assert native["planner_threads"] == 512 and native["reduction_lane_elements"] == 4
    assert native["reduction_unroll_factor"] == 1 and native["reduction_programs_per_group"] == 0
    assert native["shared_tile_materialization"] == "preserve"
    assert len(native["execution_plans"]) == 1
    plan = native["execution_plans"][0]
    assert plan["optimized"] is True and plan["threads"] == 512
    assert plan["reduction_subgroups_per_program"] == 16 and plan["reduction_programs_per_group"] == 1
    assert plan["programs"] == plan["reduction_threadgroups"] == row["case"]["m"]
    n = row["case"]["n"]
    slots = n // 2048 * 4 + min(n % 2048, 4)
    stripes = {"sum": 0, "rmsnorm": int(cache), "softmax": 1 + int(cache),
               "layernorm": 1 + int(cache), "residual_layernorm": 2}[row["case"]["operation"]]
    assert plan["striped_storage_scalars_per_worker"] == slots * stripes <= native["max_reduction_striped_scalars_per_worker"] == 64
    for provider in ("native", "torch"):
        item = row[provider]
        assert math.isfinite(item["correctness"]["max_abs_error"])
        timing = item["device_timing"]
        control = timing["control"]
        assert timing["host_samples_instrumented"] is False and control["encoder_instrumentation"] is False
        assert control["method"] == "metal_command_buffer_timestamps_v1"
        assert control["scope"] == "sum_of_command_buffer_gpu_intervals"
        assert type(control["repetitions"]) is int and control["repetitions"] > 0
        for phase in ("throughput", "latency"):
            raw = control[phase]
            assert len(raw) == samples and all(math.isfinite(s["command_buffer_ns"]) and s["command_buffer_ns"] > 0 and s["command_buffers"] > 0 for s in raw)
            assert math.isclose(gpu(item, phase), control[f"command_buffer_{phase}_us_p50"], rel_tol=1e-12)
            host = item[f"{phase}_us"]
            assert len(host) == samples and all(math.isfinite(v) and v > 0 for v in host)
            assert math.isclose(median(host), item[f"{phase}_us_p50"], rel_tol=1e-10)
    source = row["native_source_sha256"]
    assert digest(directory / "sources" / f"{source}.metal") == source


def audit():
    frozen = {}
    for variant in ("reference", "candidate"):
        directory = ROOT / (PREFIX + variant)
        report = json.loads((directory / "results.json").read_text())
        assert len(report["results"]) == 25 and {row["name"] for row in report["results"]} == CASES
        for row in report["results"]:
            validate(row, directory, 5, variant == "candidate")
        frozen[variant] = {row["name"]: row for row in report["results"]}
    controls = {name for name in CASES if frozen["reference"][name]["native_source_sha256"] == frozen["candidate"][name]["native_source_sha256"]}
    assert controls == {f"{op}_{m}x{n}" for op in ("sum", "residual_layernorm") for m, n in SHAPES}
    print("Pilot: 50 measurements / 100 complete recorded outputs; 15 changed-source pairs and 10 identical-source controls.")
    directory = ROOT / (PREFIX + "replay")
    replay = json.loads((directory / "results.json").read_text())
    meta = replay["metadata"]
    assert meta["rounds"] == 4 and meta["samples"] == 9 and meta["artifacts_unchanged"] is True
    assert meta["native_variants"]["reference"] == meta["native_variants"]["candidate"]
    required = {"libtvm_compiler.dylib", "libtvm_runtime.dylib", "libtvm_runtime_metal.dylib", "libluisa-tile-bridge-tirx.dylib", "libluisa-benchmark-metal-timing.dylib"}
    assert required <= {Path(path).name for path in meta["artifacts_sha256"]}
    assert len(replay["results"]) == 200 and {row["name"] for row in replay["results"]} == CASES
    for variant in frozen:
        assert meta["source_reports"][variant]["sha256"] == digest(ROOT / (PREFIX + variant) / "results.json")
    for row in replay["results"]:
        validate(row, directory, 9, row["variant"] == "candidate")
        original = frozen[row["variant"]][row["name"]]
        assert row["native_source_sha256"] == original["native_source_sha256"]
        assert row["native"]["execution_plans"] == original["native"]["execution_plans"]
    print("Replay: 200 measurements / 400 complete recorded outputs; all raw GPU/E2E medians, frozen plans and source hashes checked.")
    for name in frozen["reference"]:
        rows = [row for row in replay["results"] if row["name"] == name]
        assert len(rows) == 8
        first = [next(row["variant"] for row in rows if row["round"] == r) for r in range(4)]
        assert first.count("reference") == first.count("candidate") == 2
        for variant in frozen:
            orders = [row["implementation_order"] for row in rows if row["variant"] == variant]
            assert orders.count(["native", "torch"]) == orders.count(["torch", "native"]) == 2
        pairs = [{row["variant"]: row for row in rows if row["round"] == r} for r in range(4)]
        assert all(set(pair) == set(frozen) for pair in pairs)
        a = [gpu(pair["reference"]["native"]) for pair in pairs]
        b = [gpu(pair["candidate"]["native"]) for pair in pairs]
        t = [gpu(pair["candidate"]["torch"]) for pair in pairs]
        gains = [x / y for x, y in zip(a, b)]
        host_gains = [pair["reference"]["native"]["throughput_us_p50"] / pair["candidate"]["native"]["throughput_us_p50"] for pair in pairs]
        ratios = [x / y for x, y in zip(b, t)]
        host = lambda provider, phase: median(pair["candidate"][provider][f"{phase}_us_p50"] for pair in pairs)
        single = lambda provider: median(gpu(pair["candidate"][provider], "latency") for pair in pairs)
        print(f"{name}{' CONTROL' if name in controls else ''}: GPU {median(a):.6f}->{median(b):.6f} us; "
              f"gain {median(gains):.6f} [{min(gains):.6f},{max(gains):.6f}]; Torch {median(t):.6f} us; "
              f"native/Torch {median(ratios):.6f} [{min(ratios):.6f},{max(ratios):.6f}]; "
              f"E2E gain {median(host_gains):.6f} [{min(host_gains):.6f},{max(host_gains):.6f}]; "
              f"single GPU {single('native'):.6f}/{single('torch'):.6f} us; "
              f"single E2E {host('native', 'latency'):.6f}/{host('torch', 'latency'):.6f} us")


if __name__ == "__main__":
    audit()
