#!/usr/bin/env python3
"""Recompute the frozen multi-output comparison from raw samples, fail closed."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent
SHAPES = ((1, 127), (37, 1537), (1024, 4096), (4096, 4096))
OPERATIONS = ("sigmoid_pair", "gelu_pair")
NAMES = {f"{op}_{m}x{n}": (op, m, n) for op in OPERATIONS for m, n in SHAPES}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def timings(measurement):
    result = {}
    for phase in ("throughput", "latency"):
        values = measurement[f"{phase}_us"]
        require(len(values) == 9 and all(map(positive, values)), "host sample coverage")
        value = statistics.median(values)
        require(close(value, measurement[f"{phase}_us_p50"]), "host p50 mismatch")
        result[f"host_{phase}"] = value
    timing = measurement["device_timing"]
    require(timing["host_samples_instrumented"] is False, "host timing instrumentation")
    require(timing["method"] == "metal_compute_pass_timestamps_v1", "diagnostic timing method")
    control = timing["control"]
    require(control["method"] == "metal_command_buffer_timestamps_v1" and
            control["scope"] == "sum_of_command_buffer_gpu_intervals" and
            control["encoder_instrumentation"] is False, "GPU control scope")
    repetitions = control["repetitions"]
    require(type(repetitions) is int and 1 <= repetitions <= 64, "GPU denominator")
    for phase in ("throughput", "latency"):
        samples = control[phase]
        require(len(samples) == 9 and all(positive(s["command_buffer_ns"]) and
                type(s["command_buffers"]) is int and s["command_buffers"] > 0 for s in samples), "GPU sample coverage")
        denominator = 1000 * (repetitions if phase == "throughput" else 1)
        value = statistics.median(s["command_buffer_ns"] / denominator for s in samples)
        require(close(value, control[f"command_buffer_{phase}_us_p50"]), "GPU p50 mismatch")
        result[f"gpu_{phase}"] = value
    return result


def audit(report):
    meta = report["metadata"]
    require(meta["rounds"] == 6 and meta["samples"] == 9 and meta["sample_ms"] == 30 and
            meta["warmup_ms"] == 100, "frozen protocol")
    require(meta["artifacts_unchanged"] is True, "compiler/runtime artifacts changed")
    require(meta["capture_sources"] is True, "missing generated sources")
    for variant, folder in (("reference", "new-reference"), ("candidate", "new-auto")):
        require(digest(ROOT / folder / "results.json") == meta["source_reports"][variant]["sha256"], "plan report hash")
    require(meta["native_variants"]["reference"] == meta["native_variants"]["candidate"], "not same-binary comparison")
    expected = {(name, round_index, variant) for name in NAMES for round_index in range(6)
                for variant in ("reference", "candidate")}
    rows = report["results"]
    keys = [(r["name"], r["round"], r["variant"]) for r in rows]
    require(len(keys) == len(expected) and set(keys) == expected, "missing/duplicate/extra replay rows")
    indexed = {}
    sources = {}
    for row in rows:
        key = row["name"], row["round"], row["variant"]
        operation, m, n = NAMES[row["name"]]
        require(row["valid"] is True and row["backend"] == "metal", "failed measurement")
        require(row["case"] == dict(operation=operation, m=m, n=n, k=1) and row["block"] == [1, 256, 1], "case/block drift")
        native, torch = row["native"], row["torch"]
        require(native["output_elements"] == 2 * m * n, "only one output validated")
        require(native["execution_scope"] == "auto" and native["planner_threads"] == 0 and
                native["pipeline_window"] == 2 and native["vectorize"] is True and
                native["cooperative_matrix"] is False and native["metal_subgroup_reductions"] is False,
                "unexpected execution policy")
        candidate = row["variant"] == "candidate"
        require(native["fuse_gpu_elementwise"] is candidate, "wrong fusion policy")
        plans = native["execution_plans"]
        require(len(plans) == (1 if candidate else 0), "unexpected realized plan")
        if candidate:
            require(plans[0]["elementwise_scalar_temporaries"] == 1 and
                    plans[0]["elementwise_elements_per_program"] == 256 and
                    plans[0]["threads"] == 256, "shared scalar/grid not realized")
        sequence = (["sigmoid.out", "sub.out", "mul.out"] if operation == "sigmoid_pair" else
                    ["gelu.out(approximate=tanh)", "gelu_backward.grad_input(ones, approximate=tanh)"])
        require(torch["operator_sequence"] == sequence and torch["output_policy"] == "preallocated_out" and
                torch["output_order"] == ["value", "derivative"], "Torch graph/allocation mismatch")
        for provider in (native, torch):
            correctness = provider["correctness"]
            require(correctness["atol"] == 2e-6 and correctness["rtol"] == 2e-5 and
                    0 <= correctness["max_abs_error"] <= 2e-6, "full-output accuracy receipt")
        source_hash = row["native_source_sha256"]
        source = ROOT / "replay/sources" / (source_hash + ".metal")
        require(digest(source) == source_hash, "generated source hash")
        text = source.read_text()
        require("arg1_ptr[" in text and "arg2_ptr[" in text, "missing output store")
        if candidate:
            require("thread float tile_storage_" not in text and "_element =" in text,
                    "serial Tile arrays survived scalarization")
        sources.setdefault((row["name"], row["variant"]), set()).add(source_hash)
        indexed[key] = dict(row=row, native=timings(native), torch=timings(torch))
    require(all(len(hashes) == 1 for hashes in sources.values()), "source changed across fresh JITs")
    # The preliminary old compiler and the new disabled-fusion control must
    # agree on generated source, not just on a Boolean command-line switch.
    for folder in ("old-auto", "old-reference", "control-large-old-old"):
        for old in json.loads((ROOT / folder / "results.json").read_text())["results"]:
            require(old["valid"] is True and old["native_source_sha256"] in sources[old["name"], "reference"],
                    "old compiler does not match reference source")
    findings = []
    for name in NAMES:
        pairs = [(indexed[name, i, "reference"], indexed[name, i, "candidate"]) for i in range(6)]
        for variant in ("reference", "candidate"):
            orders = [indexed[name, i, variant]["row"]["implementation_order"] for i in range(6)]
            require(orders.count(["native", "torch"]) == orders.count(["torch", "native"]) == 3, "unbalanced framework order")
        native_first = [keys.index((name, i, "reference")) < keys.index((name, i, "candidate")) for i in range(6)]
        require(sum(native_first) == 3, "unbalanced mapper order")
        values = {}
        for metric in ("host_throughput", "gpu_throughput", "host_latency", "gpu_latency"):
            reference = [p[0]["native"][metric] for p in pairs]
            candidate = [p[1]["native"][metric] for p in pairs]
            torch = [p[1]["torch"][metric] for p in pairs]
            ratios = [a / b for a, b in zip(reference, candidate)]
            torch_ratios = [a / b for a, b in zip(candidate, torch)]
            values[metric] = dict(reference_us=statistics.median(reference), candidate_us=statistics.median(candidate),
                                  torch_us=statistics.median(torch), paired_reference_over_candidate=statistics.median(ratios),
                                  speedup_range=[min(ratios), max(ratios)], candidate_slower_rounds=sum(r < 1 for r in ratios),
                                  paired_candidate_over_torch=statistics.median(torch_ratios),
                                  torch_ratio_range=[min(torch_ratios), max(torch_ratios)],
                                  torch_slower_rounds=sum(r > 1 for r in torch_ratios))
        findings.append(dict(name=name, **values))
    return dict(passed=True, graph_comparisons=len(rows), complete_value_derivative_pairs=2 * len(rows),
                complete_output_planes=4 * len(rows), frozen_artifacts=len(meta["artifacts_sha256"]),
                unique_generated_sources=len({h for values in sources.values() for h in values}),
                findings=findings, interpretation="paired medians and min-max ranges, not confidence intervals; eager Torch")


def self_test(report):
    def missing(r): r["results"].pop()
    def duplicate(r): r["results"][-1] = copy.deepcopy(r["results"][0])
    def failed(r): r["results"][0]["valid"] = False
    def partial(r): r["results"][0]["native"]["output_elements"] //= 2
    def mutated(r): r["metadata"]["artifacts_unchanged"] = False
    def timing(r): r["results"][0]["native"]["device_timing"]["control"]["repetitions"] = 1
    def source(r): r["results"][0]["native_source_sha256"] = "0" * 64
    def order(r): r["results"][0]["implementation_order"] = ["torch", "native"]
    for mutate in (missing, duplicate, failed, partial, mutated, timing, source, order):
        changed = copy.deepcopy(report)
        mutate(changed)
        try:
            audit(changed)
        except (ValueError, FileNotFoundError):
            continue
        raise AssertionError(f"audit accepted {mutate.__name__}")
    return 8


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-local-artifacts", action="store_true")
    args = parser.parse_args()
    report = json.loads((ROOT / "replay/results.json").read_text())
    result = audit(report)
    result["adversarial_checks"] = self_test(report)
    if args.check_local_artifacts:
        require(all(digest(Path(p)) == h for p, h in report["metadata"]["artifacts_sha256"].items()), "local artifact drift")
        result["local_artifacts_rechecked"] = True
    (ROOT / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
