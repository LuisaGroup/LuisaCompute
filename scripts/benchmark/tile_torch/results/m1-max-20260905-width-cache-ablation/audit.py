"""Recompute the fixed factorial evidence; import no benchmark statistics.

The recorded valid flags attest to executed FP64-oracle checks. Output arrays
are not archived, so this audit does not independently reexecute the kernels.
All cells and all rounds remain included, including unstable GPU samples.
"""

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median


HERE = Path(__file__).resolve().parent
OPERATIONS = ("softmax", "rmsnorm", "layernorm")
CELLS = {(192, False), (192, True), (416, False), (416, True)}


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9), (actual, expected)


def validate(row, folder):
    assert row["valid"] is True and row["backend"] == "metal"
    assert row["case"]["m"] == 37 and row["case"]["n"] == 1537
    assert row["case"]["operation"] in OPERATIONS
    native = row["native"]
    assert native["metal_subgroup_reductions"] is True
    assert native["reduction_programs_per_group"] == 1
    assert native["reduction_lane_elements"] == 4 and native["reduction_unroll_factor"] == 1
    assert len(native["execution_plans"]) == 1
    plan = native["execution_plans"][0]
    cell = native["planner_threads"], native["cache_reduction_inputs"]
    assert cell in CELLS and plan["threads"] == cell[0]
    assert plan["programs"] == plan["reduction_threadgroups"] == 37
    assert plan["reduction_subgroups_per_program"] * 32 == cell[0]
    assert plan["candidates_considered"] == 1 and plan["candidates_rejected"] == 0
    source_hash = row["native_source_sha256"]
    source = folder / "sources" / f"{source_hash}.metal"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    metrics = {}
    for provider in ("native", "torch"):
        measurement = row[provider]
        assert math.isfinite(measurement["correctness"]["max_abs_error"])
        for phase in ("throughput", "latency"):
            values = measurement[f"{phase}_us"]
            assert len(values) == 9 and all(math.isfinite(x) and x > 0 for x in values)
            value = median(values)
            close(value, measurement[f"{phase}_us_p50"])
            metrics[f"{provider}_e2e_{phase}"] = value
        control = measurement["device_timing"]["control"]
        assert control["encoder_instrumentation"] is False
        for phase in ("throughput", "latency"):
            samples = control[phase]
            assert len(samples) == 9
            repetitions = control["repetitions"] if phase == "throughput" else 1
            assert repetitions >= 1
            values = [sample["command_buffer_ns"] / repetitions / 1000 for sample in samples]
            assert all(math.isfinite(x) and x > 0 for x in values)
            value = median(values)
            close(value, control[f"command_buffer_{phase}_us_p50"])
            metrics[f"{provider}_gpu_{phase}"] = value
    return cell, source_hash, metrics


def main():
    manifest = json.loads((HERE / "manifest.json").read_text())
    assert len(manifest["rounds"]) == 4
    assert all(r["exit_code"] == 0 and r["artifacts_unchanged"] for r in manifest["rounds"])
    values, slots, orders, hashes = {}, defaultdict(list), defaultdict(list), defaultdict(set)
    count = 0
    for index in range(4):
        folder = HERE / f"round-{index + 1}"
        report = json.loads((folder / "results.json").read_text())
        assert len(report["results"]) == 3
        assert report["metadata"]["samples"] == 9 and report["metadata"]["tuning_metric"] == "model"
        seen = set()
        for row in report["results"]:
            op = row["case"]["operation"]
            assert op not in seen
            seen.add(op)
            validate(row, folder)  # Extra fresh model-selected run, not an effect sample.
            count += 2
            trials = row["tuning"]["trials"]
            assert len(trials) == 4
            cells = set()
            for slot, trial in enumerate(trials):
                assert trial["valid"] is True
                cell, source, metrics = validate(trial["measurement"], folder)
                assert cell not in cells
                cells.add(cell)
                key = op, cell
                values[index, *key] = metrics
                slots[key].append(slot)
                orders[key].append(trial["measurement"]["implementation_order"][0])
                hashes[key].add(source)
                count += 2
            assert cells == CELLS
        assert seen == set(OPERATIONS)
    assert count == 120
    for key in slots:
        assert sorted(slots[key]) == [0, 1, 2, 3]
        assert sorted(orders[key]) == ["native", "native", "torch", "torch"]
        assert len(hashes[key]) == 1
    summary = {"validated_outputs": count, "trial_measurements": 48, "extra_fresh_measurements": 12,
               "cells": [], "paired_gains": []}
    for op in OPERATIONS:
        for cell in sorted(CELLS):
            metrics = [values[index, op, cell] for index in range(4)]
            summary["cells"].append({"operation": op, "width": cell[0], "cache": cell[1],
                                     "source_sha256": next(iter(hashes[op, cell])),
                                     "medians": {key: median(m[key] for m in metrics) for key in metrics[0]}})
        for label, reference, candidate in (
                ("width_reload", (192, False), (416, False)),
                ("width_cache", (192, True), (416, True)),
                ("cache_192", (192, False), (192, True)),
                ("cache_416", (416, False), (416, True)),
                ("combined", (192, False), (416, True))):
            gains = {}
            for metric in ("native_gpu_throughput", "native_e2e_throughput", "native_e2e_latency"):
                pairs = [values[i, op, reference][metric] / values[i, op, candidate][metric] for i in range(4)]
                gains[metric] = {"median": median(pairs), "min": min(pairs), "max": max(pairs), "pairs": pairs}
            summary["paired_gains"].append({"operation": op, "effect": label, "gains": gains})
    (HERE / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"PASS: {count} executed output validations; 48 fixed trial measurements; balanced orders; 12 stable sources")
    for row in summary["paired_gains"]:
        gain = row["gains"]["native_e2e_throughput"]
        print(f"{row['operation']} {row['effect']}: E2E {gain['median']:.4f} [{gain['min']:.4f}, {gain['max']:.4f}]")


if __name__ == "__main__":
    main()
