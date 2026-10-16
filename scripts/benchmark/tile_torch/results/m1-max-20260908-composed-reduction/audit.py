#!/usr/bin/env python3
"""Recompute recorded timings independently; no benchmark runner imports or tuning."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

HERE = Path(__file__).resolve().parent
SHAPES = ((1, 8, 2, 1, 2048, 64, 64), (1, 8, 2, 1, 2053, 80, 96),
          (1, 16, 4, 1, 4096, 128, 128))
PILOTS = ("off-0", "on-64", "on-0", "off-64", "on-32", "on-128", "on-256")
CONTROLS = ("views-off-64", "views-on-64", "views-off-1024", "views-on-1024")
PROBES = ("qk-reduce-64", "qk-reduce-1024")
METRICS = {"e2e_throughput": "throughput_us_p50", "e2e_latency": "latency_us_p50",
           "gpu_throughput": "gpu_control_throughput_us_p50", "gpu_latency": "gpu_control_latency_us_p50"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    return type(a) in (int, float) and type(b) in (int, float) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-10)


def p50(values):
    require(values and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values), "invalid timing samples")
    return statistics.median(values)


def raw_matches(raw, augmented):
    if isinstance(raw, dict):
        return isinstance(augmented, dict) and all(k in augmented and raw_matches(v, augmented[k]) for k, v in raw.items())
    return raw == augmented


def timing(row, samples):
    m = row["measurement"]
    require(type(m["repetitions"]) is int and m["repetitions"] > 0, "invalid host divisor")
    result = {}
    counter = m["device_timing"]
    control = counter["control"]
    require(counter["method"] == "metal_compute_pass_timestamps_v1" and
            counter["scope"] == "sum_of_compute_encoder_gpu_intervals" and counter["host_samples_instrumented"] is False,
            "wrong counter timing scope")
    require(control["method"] == "metal_command_buffer_timestamps_v1" and
            control["scope"] == "sum_of_command_buffer_gpu_intervals" and control["encoder_instrumentation"] is False,
            "wrong GPU control scope")
    for phase in ("throughput", "latency"):
        values = m[phase + "_us"]
        require(len(values) == samples and close(p50(values), m[phase + "_us_p50"]), "wrong host summary")
        result["e2e_" + phase] = p50(values)
        for label, record, raw_field, saved_field in (
                ("gpu", control, "command_buffer_ns", "command_buffer_"),
                ("compute_pass", counter, "compute_ns", "compute_")):
            require(type(record["repetitions"]) is int and record["repetitions"] > 0, "invalid GPU divisor")
            divisor = record["repetitions"] if phase == "throughput" else 1
            raw = record[phase]
            require(len(raw) == samples and all(r["command_buffers"] > 0 for r in raw), "incomplete GPU samples")
            if label == "compute_pass":
                require(all(r["compute_passes"] > 0 and r["calibration_cpu_ns"] > 0 and r["calibration_gpu_ticks"] > 0 for r in raw), "invalid counter receipt")
                if row["path"] == "native":
                    require(all(r["compute_passes"] == divisor for r in raw), "not one native compute pass per invocation")
            normalized = [r[raw_field] / (1000 * divisor) for r in raw]
            saved = record[saved_field + phase + "_us"]
            require(len(saved) == samples and all(close(a, b) for a, b in zip(normalized, saved)) and
                    close(p50(normalized), record[saved_field + phase + "_us_p50"]), "wrong GPU normalization or summary")
            result[label + "_" + phase] = p50(normalized)
    return result


def oracle(receipt, elements):
    require(receipt["elements"] == elements and receipt["atol"] == receipt["rtol"] == 5e-5 and
            math.isfinite(receipt["max_abs_error"]) and receipt["max_abs_error"] >= 0, "incomplete FP64 oracle receipt")
    # Complete elementwise comparisons ran before export removal. A scalar
    # maximum and hashes cannot reproduce those comparisons after the fact.


def cohort(name, report, input_hashes):
    pilot, probe = name in PILOTS, name in PROBES
    rounds, samples = (2, 5) if pilot else (4, 7)
    shapes = SHAPES[:1] if pilot else SHAPES
    views = not pilot
    enabled = "on-" in name or probe
    requested = int(name.rsplit("-", 1)[1])
    meta = report["metadata"]
    require(meta["backend"] == "metal" and meta["rounds"] == rounds and meta["samples"] == samples and
            meta["sample_ms"] == (20 if pilot else 30) and meta["warmup_ms"] == 100 and
            meta["artifacts_unchanged"] is True and meta["baseline"] is None, "wrong or unstable cohort")
    require(meta["reduction_tree"] is enabled and meta["requested_group_threads"] == requested and
            meta.get("requested_input_views", False) is views and meta.get("attention_qk", "mma") == ("reduce" if probe else "mma"), "wrong requested control")
    require(len(meta["source_sha256"]) == 4 and
            sum(Path(p).name.startswith("libtvm_") for p in meta["artifacts_sha256"]) == 5, "missing compiler/protocol provenance")
    require((HERE / name / "build.log").is_file(), "missing build record")
    rows = report["results"]
    expected = [(s, r, p) for s in shapes for r in range(rounds)
                for p in (("native", "torch") if r % 2 == 0 else ("torch", "native"))]
    require(len(rows) == len(expected), "missing or extra row")
    records, sources, widths = {}, {}, {}
    for row, (shape, r, path) in zip(rows, expected):
        failed = name == "qk-reduce-64" and shape == SHAPES[1] and r == 3 and path == "torch"
        require(tuple(row["dimensions"]) == shape and row["round"] == r and row["path"] == path and
                row["operation"] == "attention" and row["valid"] is not failed and
                row["order"] == (["native", "torch"] if r % 2 == 0 else ["torch", "native"]), "wrong, duplicate or failed visit")
        if failed:
            require(row["error"] == "Metal compute-pass duration exceeds enclosing command-buffer time" and
                    "measurement" not in row, "unexpected failed-probe receipt")
            continue
        require(len(row["input_sha256"]) == 3 and all(re.fullmatch("[a-f0-9]{64}", h) for h in row["input_sha256"] + [row["output_sha256"]]), "invalid tensor fingerprint")
        require(shape not in input_hashes or input_hashes[shape] == row["input_sha256"], "changed inputs")
        input_hashes[shape] = row["input_sha256"]
        elements = shape[0] * shape[1] * shape[3] * shape[6]
        oracle(row["correctness"], elements)
        m = row["measurement"]
        if path == "native":
            require(m["implementation"] == "tile_tirx_metal" and m["backend"] == "metal" and
                    m["precision"] == "fp32" and m["fast_math"] is False and m["relaxed_precision"] is False and
                    m["runtime"] == "luisa" and m["source_reduction_policy"] == "unordered_tree" and
                    m["attention_block"] == [1, 32] and m["requested_group_threads"] == requested and
                    m["reduction_tree"] is enabled and m.get("requested_input_views", False) is views and
                    m.get("attention_qk", "mma") == ("reduce" if probe else "mma"), "native policy not acknowledged")
            c = m["correctness"]
            require(c["checks"] == 2 and c["elements_per_check"] == elements and c["guard_elements_per_check"] == 34 and
                    c["atol"] == c["rtol"] == 5e-5, "missing native full-output/guard checks")
            width = int(re.search(r"(\d+) threads/group", m["realization"])[1])
            require(width == (requested or (64 if enabled else 1024)), "wrong actual width")
            code_path = HERE / name / row["source"]
            code = code_path.read_text()
            require(hashlib.sha256(code_path.read_bytes()).hexdigest() == row["source_sha256"], "modified generated source")
            require(shape not in sources or code == sources[shape], "source changed between rounds")
            require(code.count("kernel void ") == 1 and "mem_flags(2)" not in code and
                    code.count("simd_max(") == int(enabled) and code.count("simd_sum(") == int(enabled) + int(probe), "wrong emitted collective family")
            require(("tile_storage_0_shared" not in code) == (views or enabled), "unexpected input staging")
            sources[shape], widths[shape] = code, width
            raw = json.loads(code_path.with_suffix(".log").read_text())
            for key in ("device_timing", "throughput_us", "latency_us", "correctness", "realization"):
                require(raw_matches(raw[key], m[key]), "raw native log mismatch")
        else:
            oracle(m["pre_timing_correctness"], elements)
            require(m["expression"] == "functional_sdpa_explicit_bottom_right_causal_mask_output_allocation_included" and
                    m["precision"] == "fp32" and m["fast_math_policy"] == "torch_default", "unmatched Torch baseline")
        records[shape, r, path] = timing(row, samples)
    require(len(report["summary"]) == len(shapes), "missing published summary")
    summary = []
    for shape in shapes:
        saved, = [s for s in report["summary"] if tuple(s["dimensions"]) == shape]
        complete = all((shape, r, p) in records for r in range(rounds) for p in ("native", "torch"))
        require(saved["complete"] is complete, "wrong published completeness")
        item = dict(shape=shape, threads=widths[shape], complete=complete, metrics={})
        if not complete:
            item["failure"] = "Torch round 3 counter validation failed; no complete native/Torch ratio"
            item["native_only_us"] = {metric: p50([records[shape, r, "native"][metric] for r in range(rounds)])
                                      for metric in (*METRICS, "compute_pass_throughput", "compute_pass_latency")}
            summary.append(item)
            continue
        for metric in (*METRICS, "compute_pass_throughput", "compute_pass_latency"):
            values = {p: [records[shape, r, p][metric] for r in range(rounds)] for p in ("native", "torch")}
            ratio = [a / b for a, b in zip(values["native"], values["torch"])]
            medians = {p: p50(v) for p, v in values.items()}
            item["metrics"][metric] = dict(median_us=medians, paired_native_over_torch=p50(ratio),
                                            min_ratio=min(ratio), max_ratio=max(ratio), slower_rounds=sum(v > 1 for v in ratio))
            if metric in METRICS:
                published = saved[METRICS[metric]]
                require(all(close(medians[p], published["median_us"][p]) for p in values) and
                        close(p50(ratio), published["paired_native_over_torch_median"]) and
                        close(min(ratio), published["min_ratio"]) and close(max(ratio), published["max_ratio"]) and
                        sum(v > 1 for v in ratio) == published["slower_rounds"], "forged published ratio/summary")
        summary.append(item)
    return summary, sources


def normalize_closed_reductions(code):
    # Only the two barrier-delimited closed sum/max phases are replaced.
    # Everything else (including QK/PV accesses and resource declarations)
    # must compare byte-for-byte between the fixed-width/view controls.
    chunks = code.split("metal::threadgroup_barrier(metal::mem_flags(3));")
    count = 0
    for i, chunk in enumerate(chunks):
        if "for (int n_" in chunk or "_subgroup_value" in chunk:
            chunks[i] = "<closed reduction>"
            count += 1
    require(count == 2, "not the two closed reduction phases")
    return chunks


def audit(reports):
    input_hashes, summaries, sources = {}, {}, {}
    for name in (*PILOTS, *CONTROLS, *PROBES):
        summaries[name], sources[name] = cohort(name, reports[name], input_hashes)
    for width in (64, 1024):
        for shape in SHAPES:
            require(normalize_closed_reductions(sources[f"views-off-{width}"][shape]) ==
                    normalize_closed_reductions(sources[f"views-on-{width}"][shape]), "non-reduction source changed in controlled comparison")
    for group in (PILOTS, CONTROLS, PROBES):
        first = reports[group[0]]["metadata"]
        require(all(reports[n]["metadata"]["artifacts_sha256"] == first["artifacts_sha256"] and
                    reports[n]["metadata"]["source_sha256"] == first["source_sha256"] for n in group), "binary/protocol changed within experiment")
    return dict(verdict="share_with_caveats", rows=sum(len(r["results"]) for r in reports.values()),
                limitations=["Configurations were not interleaved; cross-configuration deltas are descriptive, not paired trials.",
                             "Initial pilot also changes input storage and automatic width; it does not isolate collectives.",
                             "QK reduce is a benchmark-only decomposition probe, not a production planner change.",
                             "GPU controls include command-buffer gaps; compute-pass probes are instrumented, not per-dispatch hardware counters.",
                             "Full-output FP64/guard checks are execution receipts; temporary tensors were removed by the driver.",
                             "Artifact stability is the driver's before/after receipt; old executables are not archived."], summaries=summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reports = {n: json.loads((HERE / n / "results.json").read_text()) for n in (*PILOTS, *CONTROLS, *PROBES)}
    result = audit(reports)
    mutations = (
        lambda r: r["results"].pop(),
        lambda r: r["summary"][0]["gpu_control_throughput_us_p50"].update(paired_native_over_torch_median=0.01),
        lambda r: r["results"][0]["measurement"]["device_timing"]["control"].update(repetitions=1),
        lambda r: r["results"][0]["measurement"].update(requested_input_views=False),
        lambda r: r["results"][0]["correctness"].update(elements=1),
        lambda r: r["results"][0].update(source_sha256="0" * 64),
    )
    for mutate in mutations:
        bad = copy.deepcopy(reports["views-on-64"])
        mutate(bad)
        try:
            cohort("views-on-64", bad, {})
        except ValueError:
            continue
        raise ValueError("adversarial mutation escaped audit")
    result["rejected_mutations"] = len(mutations)
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded if not args.output else f"PASS: {result['rows']} rows; {len(mutations)} rejected corruptions; {result['verdict']}")


if __name__ == "__main__":
    main()
