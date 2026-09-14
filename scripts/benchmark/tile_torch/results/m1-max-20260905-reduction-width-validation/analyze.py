"""Freeze search selections, then independently audit the width experiment.

Run `analyze.py select` once before repeat.py, and `analyze.py audit`
after the four-round replay. Only `select` writes files: two new plan catalogs
containing unmodified search measurements, not new performance evidence.
The audit uses only the standard library, not the benchmark's statistics or
selection helpers. Run without Python's -O option so assertions are checked.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median


HERE = Path(__file__).resolve().parent
SEARCH = HERE.parent / "m1-max-20260905-target-width-search"
REPLAY = HERE.parent / "m1-max-20260905-target-width-replay"
WIDTHS = {"reference": (32, 128, 256), "candidate": (32, 96, 128, 256, 512, 1024)}
SHAPES = ((17, 257), (64, 4096), (1024, 4096), (7, 1537), (128, 8192))
CASES = {f"{operation}_{m}x{n}" for operation in ("sum", "softmax", "rmsnorm") for m, n in SHAPES}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu(item, phase="throughput"):
    control = item["device_timing"]["control"]
    divisor = control["repetitions"] if phase == "throughput" else 1
    return median(sample["command_buffer_ns"] for sample in control[phase]) / divisor / 1000.0


def validate(row, directory, samples):
    assert row["valid"] is True and row["backend"] == "metal" and row["name"] in CASES
    native = row["native"]
    assert native["metal_max_threads"] == 1024
    assert native["metal_subgroup_reductions"] is True
    assert native["reduction_lane_elements"] == 4 and native["reduction_unroll_factor"] == 1
    assert native["reduction_programs_per_group"] == 0  # Exact widths realize P=1.
    assert native["planner_threads"] in WIDTHS["candidate"]
    assert len(native["execution_plans"]) == 1
    plan = native["execution_plans"][0]
    assert plan["optimized"] is True and plan["candidates_considered"] == 1
    assert plan["threads"] == native["planner_threads"]
    assert plan["reduction_subgroups_per_program"] * 32 == plan["threads"]
    assert plan["reduction_programs_per_group"] == 1
    assert plan["programs"] == plan["reduction_threadgroups"] == row["case"]["m"]
    assert plan["reduction_lane_elements"] == 4 and plan["reduction_unroll_factor"] == 1
    assert math.isfinite(plan["reduction_scalar_rounds"]) and plan["reduction_scalar_rounds"] > 0
    assert 0 < plan["reduction_lane_utilization"] <= 1
    assert plan["striped_storage_scalars_per_worker"] <= native["max_reduction_striped_scalars_per_worker"]
    for provider in ("native", "torch"):
        item = row[provider]
        assert math.isfinite(item["correctness"]["max_abs_error"])
        timing = item["device_timing"]
        control = timing["control"]
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
    assert len(report["results"]) == 15 and {row["name"] for row in report["results"]} == CASES
    assert report["metadata"]["samples"] == 5
    assert report["metadata"]["tuning_metric"] == "gpu-control"
    selected = {variant: [] for variant in WIDTHS}
    counts = {"valid_trials": 0, "rejected_trials": 0, "fresh_winners": 0}
    for row in report["results"]:
        validate(row, SEARCH, 5)
        counts["fresh_winners"] += 1
        tuning = row["tuning"]
        assert tuning["selection_metric"] == "native_gpu_command_buffer_throughput_us_p50"
        trials = tuning["trials"]
        assert len(trials) == 6 and {trial["group_threads"] for trial in trials} == set(WIDTHS["candidate"])
        for trial in trials:
            if not trial["valid"]:
                counts["rejected_trials"] += 1
                assert "cannot realize the exact reduction mapping" in trial["error"]
                assert "measurement" not in trial
                continue
            counts["valid_trials"] += 1
            validate(trial["measurement"], SEARCH, 5)
            assert trial["measurement"]["case"] == row["case"]
            assert trial["group_threads"] == trial["measurement"]["native"]["planner_threads"]
            assert math.isclose(gpu(trial["measurement"]["native"]), trial["selection_score"], rel_tol=1e-12)
        for variant, widths in WIDTHS.items():
            index = min((i for i, trial in enumerate(trials) if trial["valid"] and trial["group_threads"] in widths),
                        key=lambda i: gpu(trials[i]["measurement"]["native"]))
            selected[variant].append((index, trials[index]["measurement"]))
            if variant == "candidate":
                assert index == tuning["selected_trial"]
                assert row["native"]["planner_threads"] == trials[index]["group_threads"]
    assert counts == {"valid_trials": 86, "rejected_trials": 4, "fresh_winners": 15}
    print(f"Search: {counts}; 202 complete native/Torch outputs; raw GPU/E2E p50s, plans and sources checked.")
    return report, selected


def select(report, selected):
    paths = {variant: HERE / f"{variant}-plan.json" for variant in WIDTHS}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Plan catalogs already exist; selection must not overwrite frozen plans.")
    for variant, path in paths.items():
        catalog = {"metadata": {
            **report["metadata"], "artifact_kind": "frozen_plan_catalog_from_search_trials",
            "source_report": {"path": str(SEARCH / "results.json"), "sha256": digest(SEARCH / "results.json")},
            "allowed_widths": WIDTHS[variant],
            "selection": [{"name": row["name"], "trial_index": index} for index, row in selected[variant]],
            "timing_warning": "Unmodified historical search rows, not new measurements or independent acceptance evidence.",
        }, "results": [row for _, row in selected[variant]]}
        with path.open("x") as output:
            json.dump(catalog, output, indent=2, allow_nan=False)
            output.write("\n")
        print(f"{variant}: {[(row['name'], row['native']['planner_threads']) for _, row in selected[variant]]}")


def audit(report, selected):
    replay = json.loads((REPLAY / "results.json").read_text())
    meta = replay["metadata"]
    assert meta["rounds"] == 4 and meta["samples"] == 9 and meta["artifacts_unchanged"] is True
    assert len(replay["results"]) == 120 and {row["name"] for row in replay["results"]} == CASES
    assert meta["native_sha256"] == report["metadata"]["native_sha256"]
    assert meta["adjacent_tile_library_sha256"] == report["metadata"]["adjacent_tile_library_sha256"]
    assert meta["native_variants"]["reference"] == meta["native_variants"]["candidate"]
    required_artifacts = {"libtvm_compiler.dylib", "libtvm_runtime.dylib", "libtvm_runtime_metal.dylib",
                          "libluisa-benchmark-metal-timing.dylib", "libluisa-tile-bridge-tirx.dylib"}
    assert required_artifacts <= {Path(path).name for path in meta["artifacts_sha256"]}
    frozen = {}
    for variant in WIDTHS:
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
        assert row["native_source_sha256"] == original["native_source_sha256"]
        assert row["native"]["execution_plans"] == original["native"]["execution_plans"]
    print("Replay: 120 measurements / 240 complete outputs; fixed plans, GPU/E2E raw p50s and source hashes checked.")
    print("Both variants use one executable; bridge and explicit TVM compiler/runtime artifacts remained unchanged.")
    for name in (row["name"] for row in report["results"]):
        rows = [row for row in replay["results"] if row["name"] == name]
        assert len(rows) == 8
        first = [next(row["variant"] for row in rows if row["round"] == r) for r in range(4)]
        assert first.count("reference") == first.count("candidate") == 2
        for variant in WIDTHS:
            orders = [row["implementation_order"] for row in rows if row["variant"] == variant]
            assert orders.count(["native", "torch"]) == orders.count(["torch", "native"]) == 2
        pairs = [{row["variant"]: row for row in rows if row["round"] == r} for r in range(4)]
        assert all(set(pair) == set(WIDTHS) for pair in pairs)
        a = [gpu(pair["reference"]["native"]) for pair in pairs]
        b = [gpu(pair["candidate"]["native"]) for pair in pairs]
        t = [gpu(pair["candidate"]["torch"]) for pair in pairs]
        gains = [x / y for x, y in zip(a, b)]
        time_ratios = [y / x for x, y in zip(a, b)]
        ratios = [x / y for x, y in zip(b, t)]
        host_gains = [pair["reference"]["native"]["throughput_us_p50"] /
                      pair["candidate"]["native"]["throughput_us_p50"] for pair in pairs]
        width_a, width_b = [frozen[variant][name]["native"]["planner_threads"] for variant in WIDTHS]
        host = lambda provider, phase: median(pair["candidate"][provider][f"{phase}_us_p50"] for pair in pairs)
        single = lambda provider: median(gpu(pair["candidate"][provider], "latency") for pair in pairs)
        print(f"{name}, W={width_a}->{width_b}{' SAME-PLAN CONTROL' if width_a == width_b else ''}: "
              f"GPU {median(a):.6f}->{median(b):.6f} us; paired gain {median(gains):.6f} "
              f"[{min(gains):.6f},{max(gains):.6f}]; Torch {median(t):.6f} us; native/Torch {median(ratios):.6f} "
              f"[{min(ratios):.6f},{max(ratios):.6f}]; E2E throughput gain {median(host_gains):.6f}; "
              f"candidate/reference GPU time {median(time_ratios):.6f}; "
              f"native/Torch E2E throughput {host('native', 'throughput'):.6f}/{host('torch', 'throughput'):.6f} us; "
              f"single GPU {single('native'):.6f}/{single('torch'):.6f} us, "
              f"single E2E {host('native', 'latency'):.6f}/{host('torch', 'latency'):.6f} us")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("select", "audit"))
    args = parser.parse_args()
    report, selected = search_selections()
    (select if args.mode == "select" else audit)(report, selected)
