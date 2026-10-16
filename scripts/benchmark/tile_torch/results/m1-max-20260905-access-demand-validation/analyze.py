"""Freeze joint resource/mapping selections, then independently audit replay.

`select` writes two new catalogs of unmodified search rows. `audit` checks the
completed replay and prints paired statistics from raw samples. No benchmark
selection/statistics helper is imported. Run without Python's -O option.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median


HERE = Path(__file__).resolve().parent
SEARCH = HERE.parent / "m1-max-20260905-resource-map-search"
REPLAY = HERE.parent / "m1-max-20260905-resource-map-replay"
WIDTHS = (32, 128, 256, 512, 1024)
SHAPES = ((23, 769), (128, 2048), (1024, 4096), (128, 8193))
OPERATIONS = ("softmax", "rmsnorm", "layernorm")
CASES = {f"{op}_{m}x{n}" for op in OPERATIONS for m, n in SHAPES}
VARIANTS = ("reference", "candidate")
ACCESS_FIELDS = {"global_read_bytes", "global_write_bytes", "private_read_bytes", "private_write_bytes"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu(item, phase="throughput"):
    control = item["device_timing"]["control"]
    divisor = control["repetitions"] if phase == "throughput" else 1
    return median(sample["command_buffer_ns"] for sample in control[phase]) / divisor / 1000.0


def stripe_slots(n, width):
    full, tail = divmod(n, width * 4)
    return full * 4 + min(tail, 4)


def storage_scalars(case, width, cache):
    stripes = int(case["operation"] != "rmsnorm") + int(cache)
    return stripes * stripe_slots(case["n"], width)


def validate(row, directory, samples):
    assert row["valid"] is True and row["backend"] == "metal" and row["name"] in CASES
    native, case = row["native"], row["case"]
    assert native["metal_max_threads"] == 1024 and native["metal_subgroup_reductions"] is True
    assert native["reduction_lane_elements"] == 4 and native["reduction_unroll_factor"] == 1
    assert native["reduction_programs_per_group"] == 1 and native["planner_threads"] in WIDTHS
    assert type(native["cache_reduction_inputs"]) is bool
    assert len(native["execution_plans"]) == 1
    plan = native["execution_plans"][0]
    assert plan["optimized"] is True and plan["candidates_considered"] == 1
    assert plan["threads"] == native["planner_threads"] == plan["reduction_subgroups_per_program"] * 32
    assert plan["reduction_programs_per_group"] == 1
    assert plan["programs"] == plan["reduction_threadgroups"] == case["m"]
    assert plan["reduction_lane_elements"] == 4 and plan["reduction_unroll_factor"] == 1
    assert 0 < plan["reduction_lane_utilization"] <= 1
    assert plan["striped_storage_scalars_per_worker"] <= native["max_reduction_striped_scalars_per_worker"] == 64
    assert plan["striped_storage_scalars_per_worker"] == storage_scalars(
        case, native["planner_threads"], native["cache_reduction_inputs"])
    assert plan["reduction_payload_accesses_known"] is True
    program = plan["reduction_payload_accesses_per_program"]
    worker = plan["reduction_payload_accesses_per_worker"]
    for demand in (program, worker):
        assert set(demand) == ACCESS_FIELDS
        assert all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in demand.values())
    # These admitted kernels use one common row domain. The longest stripe
    # has ceil(N/W/V) full packs plus up to V elements of its final pack.
    n, width = case["n"], plan["threads"]
    slots = stripe_slots(n, width)
    for key in ACCESS_FIELDS:
        assert math.isclose(worker[key], program[key] / n * slots, rel_tol=1e-12)
    global_reads_per_element = {"softmax": 8, "rmsnorm": 12, "layernorm": 16}[case["operation"]]
    if native["cache_reduction_inputs"]:
        global_reads_per_element -= 4
    assert program["global_read_bytes"] == n * global_reads_per_element
    assert program["global_write_bytes"] == n * 4
    for provider in ("native", "torch"):
        item = row[provider]
        assert math.isfinite(item["correctness"]["max_abs_error"])
        timing, control = item["device_timing"], item["device_timing"]["control"]
        assert timing["host_samples_instrumented"] is False
        assert control["encoder_instrumentation"] is False
        assert control["method"] == "metal_command_buffer_timestamps_v1"
        assert control["scope"] == "sum_of_command_buffer_gpu_intervals"
        assert type(control["repetitions"]) is int and control["repetitions"] > 0
        for phase in ("throughput", "latency"):
            raw = control[phase]
            assert len(raw) == samples
            assert all(math.isfinite(sample["command_buffer_ns"]) and sample["command_buffer_ns"] > 0
                       and sample["command_buffers"] > 0 for sample in raw)
            assert math.isclose(gpu(item, phase), control[f"command_buffer_{phase}_us_p50"], rel_tol=1e-12)
            host = item[f"{phase}_us"]
            assert len(host) == samples and all(math.isfinite(value) and value > 0 for value in host)
            assert math.isclose(median(host), item[f"{phase}_us_p50"], rel_tol=1e-10)
    source_hash = row["native_source_sha256"]
    assert digest(directory / "sources" / (source_hash + ".metal")) == source_hash


def search_selections():
    report = json.loads((SEARCH / "results.json").read_text())
    assert len(report["results"]) == 12 and {row["name"] for row in report["results"]} == CASES
    assert report["metadata"]["samples"] == 5 and report["metadata"]["tuning_metric"] == "gpu-control"
    selected = {variant: [] for variant in VARIANTS}
    valid_count = rejected_count = 0
    for row in report["results"]:
        validate(row, SEARCH, 5)
        trials, tuning = row["tuning"]["trials"], row["tuning"]
        assert tuning["selection_metric"] == "native_gpu_command_buffer_throughput_us_p50"
        assert len(trials) == 10
        assert {(trial["group_threads"], trial["cache_reduction_inputs"]) for trial in trials} == {
            (width, cache) for width in WIDTHS for cache in (False, True)}
        for trial in trials:
            if not trial["valid"]:
                rejected_count += 1
                assert "cannot realize the exact reduction mapping" in trial["error"]
                assert "measurement" not in trial
                assert storage_scalars(row["case"], trial["group_threads"], trial["cache_reduction_inputs"]) > 64
                continue
            valid_count += 1
            measured = trial["measurement"]
            validate(measured, SEARCH, 5)
            assert measured["case"] == row["case"]
            assert trial["group_threads"] == measured["native"]["planner_threads"]
            assert trial["cache_reduction_inputs"] == measured["native"]["cache_reduction_inputs"]
            assert math.isclose(gpu(measured["native"]), trial["selection_score"], rel_tol=1e-12)
        for variant in VARIANTS:
            index = min((i for i, trial in enumerate(trials) if trial["valid"] and
                         (variant == "candidate" or not trial["cache_reduction_inputs"])),
                        key=lambda i: gpu(trials[i]["measurement"]["native"]))
            measured = trials[index]["measurement"]
            selected[variant].append((index, measured))
            if variant == "candidate":
                assert index == tuning["selected_trial"]
                assert row["native"]["execution_plans"] == measured["native"]["execution_plans"]
                assert row["native_source_sha256"] == measured["native_source_sha256"]
                assert row["native"]["cache_reduction_inputs"] == measured["native"]["cache_reduction_inputs"]
    print(f"Search: {valid_count} valid, {rejected_count} rejected trials; 12 fresh winners; "
          f"{2 * (valid_count + 12)} complete native/Torch outputs.")
    # Feature extraction alone must not alter the existing fixed-width source.
    for cache, previous_dir in ((False, "m1-max-20260905-input-cache-reference"),
                                (True, "m1-max-20260905-input-cache-candidate")):
        previous = json.loads((HERE.parent / previous_dir / "results.json").read_text())
        for op in OPERATIONS:
            name = f"{op}_1024x4096"
            old = next(row for row in previous["results"] if row["name"] == name)
            row = next(row for row in report["results"] if row["name"] == name)
            trial = next(trial for trial in row["tuning"]["trials"] if trial["group_threads"] == 512 and
                         trial["cache_reduction_inputs"] == cache)
            assert trial["measurement"]["native_source_sha256"] == old["native_source_sha256"]
    print("Six W=512 anchor sources match the preceding input-cache implementation exactly.")
    return report, selected


def select(report, selected):
    paths = {variant: HERE / f"{variant}-plan.json" for variant in VARIANTS}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Frozen catalogs already exist; do not overwrite them.")
    for variant, path in paths.items():
        catalog = {"metadata": {
            **report["metadata"], "artifact_kind": "frozen_plan_catalog_from_search_trials",
            "source_report": {"path": str(SEARCH / "results.json"), "sha256": digest(SEARCH / "results.json")},
            "allowed_widths": WIDTHS, "allowed_input_caches": [False] if variant == "reference" else [False, True],
            "selection": [{"name": row["name"], "trial_index": index} for index, row in selected[variant]],
            "timing_warning": "Unmodified historical search rows, not new measurements or independent acceptance evidence.",
        }, "results": [row for _, row in selected[variant]]}
        with path.open("x") as output:
            json.dump(catalog, output, indent=2, allow_nan=False)
            output.write("\n")
        print(f"{variant}: {[(row['name'], row['native']['planner_threads'], row['native']['cache_reduction_inputs']) for _, row in selected[variant]]}")


def audit(report, selected):
    replay = json.loads((REPLAY / "results.json").read_text())
    meta = replay["metadata"]
    assert meta["rounds"] == 4 and meta["samples"] == 9 and meta["artifacts_unchanged"] is True
    assert len(replay["results"]) == 96 and {row["name"] for row in replay["results"]} == CASES
    assert meta["native_sha256"] == report["metadata"]["native_sha256"]
    assert meta["adjacent_tile_library_sha256"] == report["metadata"]["adjacent_tile_library_sha256"]
    assert meta["native_variants"]["reference"] == meta["native_variants"]["candidate"]
    assert {"libtvm_compiler.dylib", "libtvm_runtime.dylib", "libtvm_runtime_metal.dylib",
            "libluisa-benchmark-metal-timing.dylib", "libluisa-tile-bridge-tirx.dylib"} <= {
                Path(path).name for path in meta["artifacts_sha256"]}
    frozen = {}
    for variant in VARIANTS:
        path = HERE / f"{variant}-plan.json"
        catalog = json.loads(path.read_text())
        assert meta["source_reports"][variant]["sha256"] == digest(path)
        assert catalog["metadata"]["source_report"]["sha256"] == digest(SEARCH / "results.json")
        assert catalog["results"] == [row for _, row in selected[variant]]
        frozen[variant] = {row["name"]: row for row in catalog["results"]}
    for row in replay["results"]:
        validate(row, REPLAY, 9)
        original = frozen[row["variant"]][row["name"]]
        assert row["native"]["planner_threads"] == original["native"]["planner_threads"]
        assert row["native"]["cache_reduction_inputs"] == original["native"]["cache_reduction_inputs"]
        assert row["native_source_sha256"] == original["native_source_sha256"]
        assert row["native"]["execution_plans"] == original["native"]["execution_plans"]
    print("Replay: 96 measurements / 192 complete outputs; fixed plans, raw GPU/E2E p50s and source hashes checked.")
    print("One executable; bridge, timing and explicit TVM compiler/runtime artifacts unchanged.")
    for name in (row["name"] for row in report["results"]):
        rows = [row for row in replay["results"] if row["name"] == name]
        assert len(rows) == 8
        first = [next(row["variant"] for row in rows if row["round"] == r) for r in range(4)]
        assert first.count("reference") == first.count("candidate") == 2
        for variant in VARIANTS:
            orders = [row["implementation_order"] for row in rows if row["variant"] == variant]
            assert orders.count(["native", "torch"]) == orders.count(["torch", "native"]) == 2
        pairs = [{row["variant"]: row for row in rows if row["round"] == r} for r in range(4)]
        assert all(set(pair) == set(VARIANTS) for pair in pairs)
        a = [gpu(pair["reference"]["native"]) for pair in pairs]
        b = [gpu(pair["candidate"]["native"]) for pair in pairs]
        t = [gpu(pair["candidate"]["torch"]) for pair in pairs]
        gains = [x / y for x, y in zip(a, b)]
        ratios = [x / y for x, y in zip(b, t)]
        host_gains = [pair["reference"]["native"]["throughput_us_p50"] /
                      pair["candidate"]["native"]["throughput_us_p50"] for pair in pairs]
        config = [(frozen[v][name]["native"]["planner_threads"], frozen[v][name]["native"]["cache_reduction_inputs"]) for v in VARIANTS]
        control = frozen["reference"][name]["native_source_sha256"] == frozen["candidate"][name]["native_source_sha256"]
        host = lambda provider, phase: median(pair["candidate"][provider][f"{phase}_us_p50"] for pair in pairs)
        single = lambda provider: median(gpu(pair["candidate"][provider], "latency") for pair in pairs)
        print(f"{name}, {config}{' SAME-SOURCE CONTROL' if control else ''}: "
              f"GPU {median(a):.6f}->{median(b):.6f} us; paired gain {median(gains):.6f} "
              f"[{min(gains):.6f},{max(gains):.6f}]; Torch {median(t):.6f} us; native/Torch {median(ratios):.6f} "
              f"[{min(ratios):.6f},{max(ratios):.6f}]; E2E throughput gain {median(host_gains):.6f} "
              f"[{min(host_gains):.6f},{max(host_gains):.6f}]; "
              f"native/Torch E2E {host('native', 'throughput'):.6f}/{host('torch', 'throughput'):.6f} us; "
              f"GPU single {single('native'):.6f}/{single('torch'):.6f} us; "
              f"E2E single {host('native', 'latency'):.6f}/{host('torch', 'latency'):.6f} us")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("select", "audit"))
    args = parser.parse_args()
    report, selected = search_selections()
    (select if args.mode == "select" else audit)(report, selected)
