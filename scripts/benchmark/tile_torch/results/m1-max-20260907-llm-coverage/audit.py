#!/usr/bin/env python3
"""Independent coverage, provenance and raw-timing reconciliation (stdlib only)."""
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent
METRICS = ("throughput_us_p50", "latency_us_p50", "gpu_control_throughput_us_p50", "gpu_control_latency_us_p50")
COHORTS = {
    "metal-screen": (2, 3, ["swiglu:37,1537", "rope:37,1538", "attention:1,2,2,4,5,4,3", "attention:1,4,2,1,128,64,64"], 0),
    "simd-screen": (2, 3, ["swiglu:17,257", "rope:17,258", "attention:1,4,2,1,128,64,64"], 2),
    "metal-replay": (6, 5, ["rope:1,128", "rope:37,1538", "rope:1024,4096", "rope:4096,4096",
                             "swiglu:1,127", "swiglu:37,1537", "swiglu:1024,4096", "swiglu:4096,4096"], 0),
    "simd-operators": (2, 5, ["swiglu:1024,256", "rope:1024,256", "rmsnorm:64,256", "layernorm:64,256",
                                "gelu_residual:64,256", "masked_softmax:64,256"], 0),
    "metal-prefill": (2, 3, ["attention:1,4,2,64,128,64,64"], 0),
    "metal-decode": (2, 3, ["attention:1,8,2,1,2048,64,64"], 0),
}


def key(row):
    return row["operation"] + ":" + ",".join(map(str, row["dimensions"]))


def raw_value(row, metric):
    data = row["measurement"]
    if metric.startswith("gpu_control_"):
        device = data["device_timing"]
        control = device["control"]
        assert control["encoder_instrumentation"] is False
        assert control["scope"] == "sum_of_command_buffer_gpu_intervals"
        phase = "throughput" if "throughput" in metric else "latency"
        divisor = control["repetitions"] if phase == "throughput" else 1
        assert control["repetitions"] == device["repetitions"] == min(data["repetitions"], 64)
        values = [s["command_buffer_ns"] / (1000 * divisor) for s in control[phase]]
    else:
        values = data[metric.removesuffix("_p50")]
    assert values and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in values)
    return statistics.median(values)


def validate(name, report, sources=True):
    rounds, samples, cases, failures = COHORTS[name]
    metadata, rows = report["metadata"], report["results"]
    assert metadata["rounds"] == rounds and metadata["samples"] == samples
    assert metadata["artifacts_unchanged"] is True and len(metadata["artifacts_sha256"]) >= 10
    assert metadata["requested_threads"] == 8
    paths = ("native", "baseline", "torch") if name == "metal-replay" else ("native", "torch")
    expected = set(itertools.product(cases, range(rounds), paths))
    assert len(rows) == len(expected)
    assert {(key(r), r["round"], r["path"]) for r in rows} == expected
    assert sum(not r["valid"] for r in rows) == failures
    checks, elements = 0, 0
    for row in rows:
        order = list(list(itertools.permutations(paths))[row["round"] % 6]) if len(paths) == 3 else list(paths if row["round"] % 2 == 0 else reversed(paths))
        assert row["order"] == order
        if not row["valid"]:
            assert name == "simd-screen" and row["operation"] == "attention" and row["path"] == "native"
            assert "timed out after 90 seconds" in row["error"]
            continue
        dims = row["dimensions"]
        count = dims[0] * dims[1] * dims[3] * dims[6] if row["operation"] == "attention" else math.prod(dims)
        check = row["correctness"]
        assert check["elements"] == count and check["atol"] == check["rtol"] == 5e-5
        assert math.isfinite(check["max_abs_error"]) and check["max_abs_error"] >= 0
        assert len(row["input_sha256"]) == 3 and all(len(h) == 64 for h in row["input_sha256"])
        assert len(row["output_sha256"]) == 64
        checks += 1
        elements += count
        m = row["measurement"]
        if row["path"] != "torch":
            assert m["correctness"]["checks"] == 2 and m["correctness"]["elements_per_check"] == count
            assert m["correctness"]["guard_elements_per_check"] == 34
            if sources:
                source = ROOT / name / row["source"]
                assert hashlib.sha256(source.read_bytes()).hexdigest() == row["source_sha256"]
        for metric in METRICS:
            if metric.startswith("gpu_") and metadata["backend"] != "metal":
                continue
            value = raw_value(row, metric)
            if not metric.startswith("gpu_"):
                assert math.isclose(value, m[metric], rel_tol=1e-12)
                assert len(m[metric.removesuffix("_p50")]) == samples
        if "device_timing" in m:
            for phase in ("throughput", "latency"):
                assert len(m["device_timing"][phase]) == samples == len(m["device_timing"]["control"][phase])
    for case in cases:
        selected = [r for r in rows if key(r) == case]
        assert len({tuple(r["input_sha256"]) for r in selected if r["valid"]}) == 1
        summary = next(s for s in report["summary"] if key(s) == case)
        assert summary["complete"] == all(r["valid"] for r in selected)
        if not summary["complete"]:
            continue
        for metric in METRICS:
            if metric not in summary:
                assert metric.startswith("gpu_") and metadata["backend"] == "cpu"
                continue
            s = summary[metric]
            for path in paths:
                assert math.isclose(statistics.median(raw_value(r, metric) for r in selected if r["path"] == path), s["median_us"][path], rel_tol=1e-12)
            ratios = []
            old = []
            for index in range(rounds):
                pair = {r["path"]: raw_value(r, metric) for r in selected if r["round"] == index}
                ratios.append(pair["native"] / pair["torch"])
                if len(paths) == 3:
                    old.append(pair["native"] / pair["baseline"])
            assert math.isclose(statistics.median(ratios), s["paired_native_over_torch_median"], rel_tol=1e-12)
            assert s["slower_rounds"] == sum(v > 1 for v in ratios)
            if old:
                assert math.isclose(statistics.median(old), s["baseline_comparison"]["paired_native_over_baseline_median"], rel_tol=1e-12)
                assert s["baseline_comparison"]["slower_rounds"] == sum(v > 1 for v in old)
    return dict(rows=len(rows), valid_outputs=checks, output_elements=elements, failures=failures)


def main():
    reports = {name: json.loads((ROOT / name / "results.json").read_text()) for name in COHORTS}
    totals = {name: validate(name, report) for name, report in reports.items()}
    replay = reports["metal-replay"]
    for case in COHORTS["metal-replay"][2]:
        hashes = {p: {r["source_sha256"] for r in replay["results"] if key(r) == case and r["path"] == p} for p in ("native", "baseline")}
        assert all(len(h) == 1 for h in hashes.values())
        assert (hashes["native"] == hashes["baseline"]) == case.startswith("swiglu:")
        for row in replay["results"]:
            if key(row) == case and row["path"] == "native":
                assert "thread float tile_storage_" not in (ROOT / "metal-replay" / row["source"]).read_text()
    # The auditor must reject missing rows, forged ratios and wrong GPU units.
    for mutation in ("missing", "ratio", "gpu_divisor"):
        bad = copy.deepcopy(replay)
        if mutation == "missing":
            bad["results"].pop()
        elif mutation == "ratio":
            bad["summary"][0]["throughput_us_p50"]["paired_native_over_torch_median"] *= 2
        else:
            bad["results"][0]["measurement"]["device_timing"]["control"]["repetitions"] += 1
        try:
            validate("metal-replay", bad, sources=False)
        except AssertionError:
            pass
        else:
            raise AssertionError("auditor accepted " + mutation)
    result = dict(status="pass", cohorts=totals, negative_audits=3, control="four SwiGLU old/new sources identical; four RoPE sources changed; no full private arrays")
    (ROOT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
