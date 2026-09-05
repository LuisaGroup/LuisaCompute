"""Independent standard-library audit of the archived lane-layout experiment.

Run from any directory; prints recomputed paired medians and validates the
complete cohort, timing scopes, frozen layouts, ordering and source hashes.
No benchmark execution and no artifact writes.
"""

import hashlib
import json
import math
from pathlib import Path
from statistics import median


root = Path(__file__).resolve().parent.parent
prefix = "m1-max-20260905-reduction-lane-"
reports = {}
for name, expected_rows in (("reference", 4), ("search", 4), ("replay", 32), ("operators", 24)):
    directory = root / (prefix + name)
    report = json.loads((directory / "results.json").read_text())
    reports[name] = report
    assert len(report["results"]) == expected_rows
    measurements = list(report["results"])
    if name == "search":
        for row in report["results"]:
            tuning = row["tuning"]
            trials = tuning["trials"]
            assert len(trials) == 12 and all(trial["valid"] for trial in trials)
            assert tuning["selection_metric"] == "native_gpu_command_buffer_throughput_us_p50"
            measurements.extend(trial["measurement"] for trial in trials)
    for row in measurements:
        assert row["valid"]
        for provider in ("native", "torch"):
            timing = row[provider]["device_timing"]
            control = timing["control"]
            assert timing["host_samples_instrumented"] is False
            assert control["encoder_instrumentation"] is False
            assert control["method"] == "metal_command_buffer_timestamps_v1"
            assert control["scope"] == "sum_of_command_buffer_gpu_intervals"
            values = [sample["command_buffer_ns"] / control["repetitions"] / 1000.0
                      for sample in control["throughput"]]
            assert len(values) == report["metadata"]["samples"]
            assert all(math.isfinite(value) and value > 0 for value in values)
            assert math.isclose(median(values), control["command_buffer_throughput_us_p50"], rel_tol=1e-12)
        width = row["native"]["reduction_lane_elements"]
        assert all(plan["reduction_lane_elements"] == width for plan in row["native"]["execution_plans"])
        digest = row["native_source_sha256"]
        assert hashlib.sha256((directory / "sources" / (digest + ".metal")).read_bytes()).hexdigest() == digest
    print(f"{name}: {len(measurements) * 2} complete valid outputs; timing scopes and source hashes checked")

replay = reports["replay"]
assert len({report["metadata"]["native_sha256"] for report in reports.values()}) == 1
case_names = {row["name"] for row in reports["reference"]["results"]}
assert len(case_names) == 4
assert case_names == {row["name"] for row in reports["search"]["results"]}
assert case_names == {row["name"] for row in replay["results"]}
assert len({row["name"] for row in reports["operators"]["results"]}) == 24
assert replay["metadata"]["artifacts_unchanged"] is True
assert replay["metadata"]["rounds"] == 4
variants = replay["metadata"]["native_variants"]
assert variants["reference"]["sha256"] == variants["candidate"]["sha256"]
for name in dict.fromkeys(row["name"] for row in replay["results"]):
    selected = [row for row in replay["results"] if row["name"] == name]
    assert len(selected) == 8
    first_variants = [next(row["variant"] for row in selected if row["round"] == index) for index in range(4)]
    assert first_variants.count("reference") == first_variants.count("candidate") == 2
    reference, candidate, torch, host_speedup, gpu_speedup, native_torch = [], [], [], [], [], []
    for round_index in range(4):
        pair = {row["variant"]: row for row in selected if row["round"] == round_index}
        assert set(pair) == {"reference", "candidate"}
        a, b = pair["reference"]["native"], pair["candidate"]["native"]
        gpu = lambda item: median(sample["command_buffer_ns"] for sample in item["device_timing"]["control"]["throughput"]) / item["device_timing"]["control"]["repetitions"] / 1000.0
        ag, bg, tg = gpu(a), gpu(b), gpu(pair["candidate"]["torch"])
        reference.append(ag)
        candidate.append(bg)
        torch.append(tg)
        host_speedup.append(a["throughput_us_p50"] / b["throughput_us_p50"])
        gpu_speedup.append(ag / bg)
        native_torch.append(bg / tg)
    for variant in ("reference", "candidate"):
        rows = [row for row in selected if row["variant"] == variant]
        assert sum(row["implementation_order"] == ["native", "torch"] for row in rows) == 2
        assert sum(row["implementation_order"] == ["torch", "native"] for row in rows) == 2
        assert len({row["native_source_sha256"] for row in rows}) == 1
        assert len({(row["native"]["planner_threads"], row["native"]["reduction_lane_elements"])
                    for row in rows}) == 1
    print(f"{name}: GPU {median(reference):.6f} -> {median(candidate):.6f} us; "
          f"paired gain {median(gpu_speedup):.6f} [{min(gpu_speedup):.6f}, {max(gpu_speedup):.6f}]; "
          f"Torch {median(torch):.6f} us; paired native/Torch {median(native_torch):.6f} "
          f"[{min(native_torch):.6f}, {max(native_torch):.6f}]; E2E gain {median(host_speedup):.6f}")
