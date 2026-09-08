"""Independently check the frozen fit, model choices and paired replay.

No production benchmark statistics, selection or cost helper is imported.
`plans` can run before the replay; `replay` additionally checks all raw rounds.
Run without Python's -O option.
"""

import argparse
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import median


HERE = Path(__file__).resolve().parent
DIRECTORIES = {"reference": HERE.parent / "m1-max-20260905-service-policy-reference",
               "candidate": HERE.parent / "m1-max-20260905-service-policy-plan"}
REPLAY = HERE.parent / "m1-max-20260905-service-policy-replay"
FIT = json.loads((HERE / "calibration.json").read_text())
SHAPES = {(37, 1537), (256, 3072), (768, 6144), (64, 12289)}
OPERATIONS = ("softmax", "rmsnorm", "layernorm")
CASES = {f"{op}_{m}x{n}" for op in OPERATIONS for m, n in SHAPES}
COEFFICIENTS = next(row["coefficients"] for row in FIT["candidates"] if row["capacity"] == FIT["selected_capacity"])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu(item, phase="throughput"):
    control = item["device_timing"]["control"]
    return median(sample["command_buffer_ns"] for sample in control[phase]) / (control["repetitions"] if phase == "throughput" else 1) / 1000


def facts(case, width, cache):
    # Independent fixture oracle. Operation names are used here to verify
    # expected IR demand, never as inputs to the production cost policy.
    n, op = case["n"], case["operation"]
    counts = [0] * width
    for i in range(n):
        counts[(i // 4) % width] += 1
    slots = max(counts)
    return dict(slots=slots, storage=slots * (int(op != "rmsnorm") + int(cache)),
                rounds=slots * ((2 if op == "rmsnorm" else 4) + int(cache)),
                reductions=1 if op == "rmsnorm" else 2,
                global_read={"softmax": 8, "rmsnorm": 12, "layernorm": 16}[op] - 4 * cache,
                private_read=(4 if op == "rmsnorm" else 16) + 8 * cache,
                private_write=(0 if op == "rmsnorm" else 4) + 4 * cache)


def score(case, width, cache, variant):
    f = facts(case, width, cache)
    if f["storage"] > 64:
        return math.inf
    groups = width // 32
    if variant == "reference":
        return (f["rounds"] + f["reductions"] * groups * 2 + 16) * max(1, math.ceil(case["m"] / 64))
    d, r, k, g, w, p = COEFFICIENTS
    waves = max(1, case["m"] * groups / FIT["selected_capacity"])
    local = f["rounds"] * r + f["reductions"] * groups * k + (f["private_read"] + f["private_write"]) * f["slots"] * p
    return d + local * waves + case["m"] * case["n"] * (f["global_read"] + 4) * g + f["slots"] * (f["global_read"] + 4) * w


def validate(row, directory, samples, variant):
    assert row["valid"] is True and row["backend"] == "metal" and row["name"] in CASES
    native, case = row["native"], row["case"]
    assert (case["m"], case["n"]) in SHAPES and case["operation"] in OPERATIONS
    assert native["reduction_cost_profile"] == ("analytic" if variant == "reference" else FIT["native_argument"])
    assert native["metal_subgroup_reductions"] is True and native["metal_max_threads"] == 1024
    assert native["planner_threads"] == 0 and native["reduction_programs_per_group"] == 1
    assert native["reduction_unroll_factor"] == 1 and native["reduction_lane_elements"] == 4
    assert native["max_reduction_striped_scalars_per_worker"] == 64
    assert len(native["execution_plans"]) == 1
    plan = native["execution_plans"][0]
    width, cache = plan["threads"], native["cache_reduction_inputs"]
    assert type(cache) is bool
    assert plan["optimized"] is True and plan["candidates_considered"] == 32
    assert plan["programs"] == plan["reduction_threadgroups"] == case["m"]
    assert width == plan["reduction_subgroups_per_program"] * 32 and 32 <= width <= 1024
    assert plan["reduction_programs_per_group"] == 1 and plan["reduction_payload_accesses_known"] is True
    f = facts(case, width, cache)
    assert plan["striped_storage_scalars_per_worker"] == f["storage"] <= 64
    assert plan["reduction_scalar_rounds"] == f["rounds"] and plan["reduction_operations"] == f["reductions"]
    expected_bytes = {"global_read_bytes": f["global_read"], "global_write_bytes": 4,
                      "private_read_bytes": f["private_read"], "private_write_bytes": f["private_write"]}
    for field, scale in (("reduction_payload_accesses_per_program", case["n"]), ("reduction_payload_accesses_per_worker", f["slots"])):
        assert plan[field] == {key: value * scale for key, value in expected_bytes.items()}
    assert math.isclose(plan["normalized_kernel_cost"], score(case, width, cache, variant), rel_tol=1e-9)
    expected_width = min(range(32, 1025, 32), key=lambda w: score(case, w, cache, variant))
    assert width == expected_width, (row["name"], width, expected_width)
    assert digest(directory / "sources" / (row["native_source_sha256"] + ".metal")) == row["native_source_sha256"]
    for provider in ("native", "torch"):
        item = row[provider]
        assert math.isfinite(item["correctness"]["max_abs_error"])
        control = item["device_timing"]["control"]
        assert item["device_timing"]["host_samples_instrumented"] is False
        assert control["encoder_instrumentation"] is False and control["method"] == "metal_command_buffer_timestamps_v1"
        for phase in ("throughput", "latency"):
            assert len(control[phase]) == samples and len(item[phase + "_us"]) == samples
            assert all(s["command_buffer_ns"] > 0 and s["command_buffers"] > 0 for s in control[phase])
            assert all(math.isfinite(s) and s > 0 for s in item[phase + "_us"])
            assert math.isclose(gpu(item, phase), control[f"command_buffer_{phase}_us_p50"], rel_tol=1e-12)
            assert math.isclose(median(item[phase + "_us"]), item[phase + "_us_p50"], rel_tol=1e-9)


def plans():
    assert {tuple(s) for s in FIT["training_shapes"]}.isdisjoint(SHAPES)
    assert {tuple(s) for s in FIT["holdout_shapes"]} == SHAPES
    assert digest(HERE / "fit.py") == FIT["fit_script_sha256"]
    assert digest(HERE.parent / FIT["source"]) == FIT["source_sha256"]
    frozen = {}
    for variant, directory in DIRECTORIES.items():
        report = json.loads((directory / "results.json").read_text())
        assert len(report["results"]) == 12 and {row["name"] for row in report["results"]} == CASES
        frozen[variant] = {row["name"]: row for row in report["results"]}
        for row in report["results"]:
            validate(row, directory, 5, variant)
            if variant == "reference":
                assert row["native"]["cache_reduction_inputs"] is False
            else:
                trials = row["tuning"]["trials"]
                assert len(trials) == 2 and {t["cache_reduction_inputs"] for t in trials} == {False, True}
                assert row["tuning"]["selection_metric"] == "sum_execution_plan_normalized_kernel_cost"
                assert "model_regret" not in row["tuning"]
                for trial in trials:
                    assert trial["valid"] is True
                    validate(trial["measurement"], directory, 5, variant)
                    assert math.isclose(trial["selection_score"], trial["measurement"]["native"]["execution_plans"][0]["normalized_kernel_cost"], rel_tol=1e-12)
                selected = min(range(2), key=lambda i: trials[i]["selection_score"])
                assert row["tuning"]["selected_trial"] == selected
                winner = trials[selected]["measurement"]
                assert row["native_source_sha256"] == winner["native_source_sha256"]
                assert row["native"]["execution_plans"] == winner["native"]["execution_plans"]
            print(variant, row["name"], row["native"]["execution_plans"][0]["threads"], row["native"]["cache_reduction_inputs"])
    print("Plan collection: 48 measurements / 96 outputs; independent exhaustive width and model-only resource selection checks pass.")
    return frozen


def replay(frozen):
    report = json.loads((REPLAY / "results.json").read_text())
    meta = report["metadata"]
    assert meta["rounds"] == 4 and meta["samples"] == 9 and meta["sample_ms"] == 30 and meta["warmup_ms"] == 200
    assert meta["artifacts_unchanged"] is True
    assert len(report["results"]) == 96 and {r["name"] for r in report["results"]} == CASES
    assert meta["native_variants"]["reference"] == meta["native_variants"]["candidate"]
    assert {"libtvm_compiler.dylib", "libtvm_runtime.dylib", "libtvm_runtime_metal.dylib",
            "libluisa-benchmark-metal-timing.dylib", "libluisa-tile-bridge-tirx.dylib", "calibration.json"} <= {
                Path(path).name for path in meta["artifacts_sha256"]}
    assert meta["artifacts_sha256"][str(HERE / "calibration.json")] == digest(HERE / "calibration.json")
    for variant, directory in DIRECTORIES.items():
        assert meta["source_reports"][variant]["sha256"] == digest(directory / "results.json")
        source_meta = json.loads((directory / "results.json").read_text())["metadata"]
        assert meta["native_sha256"] == source_meta["native_sha256"]
        assert meta["adjacent_tile_library_sha256"] == source_meta["adjacent_tile_library_sha256"]
    perturbations = []
    for row in report["results"]:
        validate(row, REPLAY, 9, row["variant"])
        original = frozen[row["variant"]][row["name"]]
        assert row["native_source_sha256"] == original["native_source_sha256"]
        assert row["native"]["execution_plans"] == original["native"]["execution_plans"]
        assert row["native"]["cache_reduction_inputs"] == original["native"]["cache_reduction_inputs"]
        timing = row["torch"]["device_timing"]
        perturbations.append(timing["command_buffer_throughput_us_p50"] / timing["control"]["command_buffer_throughput_us_p50"])
    print(f"Replay: 96 measurements / 192 validated outputs; unchanged executable, bridge, TVM, timing and calibration artifacts ({len(meta['artifacts_sha256'])} fingerprints).")
    print(f"Torch probe/control throughput ratio range {min(perturbations):.6f}..{max(perturbations):.6f}; probe is not selection/acceptance evidence.")
    records = []
    for name in (f"{op}_{m}x{n}" for op in OPERATIONS for m, n in sorted(SHAPES)):
        rows = [row for row in report["results"] if row["name"] == name]
        assert len(rows) == 8
        first = [next(row["variant"] for row in rows if row["round"] == r) for r in range(4)]
        assert first.count("reference") == first.count("candidate") == 2
        for variant in DIRECTORIES:
            orders = [row["implementation_order"] for row in rows if row["variant"] == variant]
            assert orders.count(["native", "torch"]) == orders.count(["torch", "native"]) == 2
        pairs = [{row["variant"]: row for row in rows if row["round"] == r} for r in range(4)]
        assert all(set(pair) == set(DIRECTORIES) for pair in pairs)
        record = {"name": name, "widths": [frozen[v][name]["native"]["execution_plans"][0]["threads"] for v in DIRECTORIES]}
        for metric in ("gpu_throughput", "e2e_throughput", "gpu_latency", "e2e_latency"):
            kind, phase = metric.split("_")
            def read(row, provider):
                item = row[provider]
                return gpu(item, phase) if kind == "gpu" else median(item[f"{phase}_us"])
            a = [read(pair["reference"], "native") for pair in pairs]
            b = [read(pair["candidate"], "native") for pair in pairs]
            t = [read(pair["candidate"], "torch") for pair in pairs]
            gain, ratio = [x / y for x, y in zip(a, b)], [x / y for x, y in zip(b, t)]
            record[metric] = dict(reference_us=median(a), candidate_us=median(b), torch_us=median(t),
                                  gain=median(gain), gain_min=min(gain), gain_max=max(gain),
                                  native_torch_ratio=median(ratio), ratio_min=min(ratio), ratio_max=max(ratio))
        records.append(record)
        print(json.dumps(record, allow_nan=False))
    for metric in ("gpu_throughput", "e2e_throughput", "gpu_latency", "e2e_latency"):
        print(metric, "positive_medians", sum(r[metric]["gain"] > 1 for r in records),
              "all_pairs_positive", sum(r[metric]["gain_min"] > 1 for r in records),
              "all_pairs_negative", sum(r[metric]["gain_max"] < 1 for r in records),
              "faster_torch_all_pairs", sum(r[metric]["ratio_max"] < 1 for r in records))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plans", "replay"))
    parser.add_argument("--receipt", type=Path, help="new file for the complete passing audit receipt")
    args = parser.parse_args()
    receipt = io.StringIO()
    with contextlib.redirect_stdout(receipt):
        frozen = plans()
        if args.mode == "replay":
            replay(frozen)
    if args.receipt:
        with args.receipt.open("x") as file:
            file.write(receipt.getvalue())
    print(receipt.getvalue(), end="")
