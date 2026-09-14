#!/usr/bin/env python3
"""Reconcile this fixed replay from raw samples, including negative audits."""
import copy
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "llm_reference_audit", ROOT.parent / "m1-max-20260907-llm-coverage" / "audit.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
CASES = {"prefill": [1, 4, 2, 64, 128, 64, 64], "decode": [1, 8, 2, 1, 2048, 64, 64]}
PATHS = ("native", "baseline", "torch")


def near(a, b):
    assert math.isclose(a, b, rel_tol=1e-12), (a, b)


def validate(name, report):
    metadata, rows = report["metadata"], report["results"]
    assert metadata["rounds"] == 6 and metadata["samples"] == 5
    assert metadata["sample_ms"] == 20 and metadata["warmup_ms"] == 100
    assert metadata["artifacts_unchanged"] and len(metadata["artifacts_sha256"]) > 20
    assert metadata["backend"] == "metal"
    assert len(rows) == 18
    assert {(r["round"], r["path"]) for r in rows} == set(itertools.product(range(6), PATHS))
    elements = math.prod(CASES[name][i] for i in (0, 1, 3, 6))
    assert len({tuple(r["input_sha256"]) for r in rows}) == 1
    hashes = {path: set() for path in PATHS[:-1]}
    for row in rows:
        assert row["valid"] and row["operation"] == "attention" and row["dimensions"] == CASES[name]
        assert row["order"] == list(list(itertools.permutations(PATHS))[row["round"]])
        check, measurement = row["correctness"], row["measurement"]
        assert check["elements"] == elements and check["atol"] == check["rtol"] == 5e-5
        assert math.isfinite(check["max_abs_error"]) and check["max_abs_error"] >= 0
        assert len(row["input_sha256"]) == 3 and all(len(h) == 64 for h in row["input_sha256"])
        assert len(row["output_sha256"]) == 64
        if row["path"] != "torch":
            assert not measurement["fast_math"] and not measurement["relaxed_precision"]
            assert measurement["correctness"]["checks"] == 2
            assert measurement["correctness"]["elements_per_check"] == elements
            assert measurement["correctness"]["guard_elements_per_check"] == 34
            source = (ROOT / name / row["source"]).read_bytes()
            assert hashlib.sha256(source).hexdigest() == row["source_sha256"]
            hashes[row["path"]].add(row["source_sha256"])
            if row["path"] == "native":
                threads = 64 if name == "prefill" else 1024
                assert f"{threads} threads/group; 1 group plans" in measurement["realization"]
                assert (b"simdgroup_multiply_accumulate(" in source) == (name == "prefill")
            else:
                assert "0 group plans" in measurement["realization"]
        device = measurement["device_timing"]
        for phase in ("throughput", "latency"):
            assert len(device[phase]) == len(device["control"][phase]) == 5
        for metric in reference.METRICS:
            value = reference.raw_value(row, metric)
            if not metric.startswith("gpu_"):
                assert len(measurement[metric.removesuffix("_p50")]) == 5
                near(value, measurement[metric])
    assert all(len(h) == 1 for h in hashes.values()) and hashes["native"] != hashes["baseline"]
    assert len(report["summary"]) == 1 and report["summary"][0]["complete"]
    summary = report["summary"][0]
    output = {}
    for metric in reference.METRICS:
        data = summary[metric]
        for path in PATHS:
            near(statistics.median(reference.raw_value(r, metric) for r in rows if r["path"] == path), data["median_us"][path])
        ratios = {path: [] for path in ("torch", "baseline")}
        for round_index in range(6):
            pair = {r["path"]: reference.raw_value(r, metric) for r in rows if r["round"] == round_index}
            for path in ratios:
                ratios[path].append(pair["native"] / pair[path])
        for path, values in ratios.items():
            observed = data if path == "torch" else data["baseline_comparison"]
            near(statistics.median(values), observed[f"paired_native_over_{path}_median"])
            near(min(values), observed["min_ratio"])
            near(max(values), observed["max_ratio"])
            assert observed["slower_rounds"] == sum(v > 1 for v in values)
        output[metric] = data
    diagnostics = {}
    for path in PATHS:
        selected = [r["measurement"]["device_timing"] for r in rows if r["path"] == path]
        compute = [statistics.median(s["compute_ns"] / (1000 * d["repetitions"]) for s in d["throughput"]) for d in selected]
        diagnostics[path] = {"instrumented_compute_us": statistics.median(compute),
                             "counter_over_control": statistics.median(d["counter_control_throughput_ratio"] for d in selected)}
    return dict(valid_outputs=18, output_elements=18 * elements, metrics=output, diagnostics=diagnostics)


def main():
    reports = {name: json.loads((ROOT / name / "results.json").read_text()) for name in CASES}
    results = {name: validate(name, report) for name, report in reports.items()}
    for mutation in ("missing", "ratio", "gpu_divisor", "source", "order"):
        bad = copy.deepcopy(reports["prefill"])
        if mutation == "missing":
            bad["results"].pop()
        elif mutation == "ratio":
            bad["summary"][0]["throughput_us_p50"]["paired_native_over_torch_median"] *= 2
        elif mutation == "gpu_divisor":
            bad["results"][0]["measurement"]["device_timing"]["control"]["repetitions"] += 1
        elif mutation == "source":
            bad["results"][0]["source_sha256"] = "0" * 64
        else:
            bad["results"][0]["order"].reverse()
        try:
            validate("prefill", bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("auditor accepted " + mutation)
    result = dict(status="pass", negative_audits=5, cohorts=results)
    (ROOT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
