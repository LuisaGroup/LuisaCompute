#!/usr/bin/env python3
"""Audit complete cohorts, source controls and paired raw timings; never tune."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

HERE = Path(__file__).resolve().parent
OPS = ("gemm", "gemm_relu", "gemm_gelu")
SHAPES = ((128, 128, 128), (127, 193, 61), (1024, 1024, 1024),
          (4096, 4096, 4096), (128, 2048, 512), (2048, 128, 512))
METRICS = ("gpu_throughput", "e2e_throughput", "gpu_latency", "e2e_latency",
           "counter_throughput", "counter_latency")


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def median(values):
    require(values and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in values), "invalid samples")
    return statistics.median(values)


def ratios(values):
    return dict(median=median(values), minimum=min(values), maximum=max(values),
                faster_rounds=sum(x < 1 for x in values), slower_rounds=sum(x > 1 for x in values), values=values)


def gate(report, phase, success=True):
    meta = report["metadata"]
    require(meta["phase"] == phase and meta["build"]["exit_code"] == 0, "wrong phase or missing build gate")
    require(meta["artifacts_unchanged"] is True and meta["artifacts_before"] == meta["artifacts_after"], "changed compiler/protocol")
    require(any(p.endswith("libtvm_compiler.dylib") for p in meta["artifacts_before"]), "missing compiler fingerprint")
    if success:
        require(report["passed"] is True, "incomplete or failed cohort")
    return meta


def source_code(row, directory):
    sha = row["native_source_sha256"]
    path = directory / "sources" / (sha + (".metal" if row["backend"] == "metal" else ".ll"))
    require(digest(path) == sha, "missing or modified generated source")
    return path.read_text()


def correctness(row):
    op, m, n = (row["case"][key] for key in ("operation", "m", "n"))
    count = m if op == "sum" else m * n * (2 if op == "gelu_pair" else 1)
    require(row["valid"] is True and row["native"]["output_elements"] == count, "invalid or partial output")
    tolerance = (1e-4, 1e-4) if op in OPS else (0, 0) if op == "add" else (1e-5, 1e-5) if op == "sum" else (2e-6, 2e-5)
    for provider in ("native", "torch"):
        proof = row[provider]["correctness"]
        require((proof["atol"], proof["rtol"]) == tolerance and
                math.isfinite(proof["max_abs_error"]) and proof["max_abs_error"] >= 0, "wrong oracle receipt")
    # The driver checked every element against atol + rtol * abs(FP64 oracle).
    # Its scalar maximum alone cannot reconstruct that elementwise comparison.
    return 2 * count


def timing(values, samples):
    result = {}
    counter = values["device_timing"]
    control = counter["control"]
    require(counter["method"] == "metal_compute_pass_timestamps_v1" and counter["host_samples_instrumented"] is False and
            control["method"] == "metal_command_buffer_timestamps_v1" and control["encoder_instrumentation"] is False, "wrong timing scope")
    for record in (control, counter):
        require(type(record["repetitions"]) is int and record["repetitions"] > 0, "wrong GPU divisor")
    for phase in ("throughput", "latency"):
        host = values[phase + "_us"]
        require(len(host) == samples and math.isclose(median(host), values[phase + "_us_p50"], rel_tol=1e-10), "wrong host p50")
        result["e2e_" + phase] = median(host)
        for record, prefix, field, saved in ((control, "gpu_", "command_buffer_ns", "command_buffer_"),
                                             (counter, "counter_", "compute_ns", "compute_")):
            divisor = record["repetitions"] if phase == "throughput" else 1
            raw = record[phase]
            require(len(raw) == samples and all(x["command_buffers"] > 0 for x in raw), "missing GPU samples")
            if prefix == "counter_":
                require(all(x["compute_passes"] > 0 and x["calibration_cpu_ns"] > 0 and x["calibration_gpu_ticks"] > 0 for x in raw), "invalid counters")
            derived = [x[field] / (1000 * divisor) for x in raw]
            published = record[saved + phase + "_us"]
            require(len(published) == samples and all(math.isclose(a, b, rel_tol=1e-10) for a, b in zip(derived, published)) and
                    math.isclose(median(derived), record[saved + phase + "_us_p50"], rel_tol=1e-10), "wrong GPU normalization/p50")
            result[prefix + phase] = median(derived)
    return result


def replay(report, directory):
    meta = gate(report, "replay")
    require(meta["rounds"] == 4 and tuple(map(tuple, meta["shapes"])) == SHAPES and len(report["results"]) == 144, "wrong replay cohort")
    cases = [(op, *shape) for shape in SHAPES for op in OPS]
    expected = [(r, case, variant) for r in range(4) for case in (cases if r % 2 == 0 else cases[::-1])
                for variant in (("reference", "fused") if r % 2 == 0 else ("fused", "reference"))]
    records, sources, plans = {}, {}, {}
    elements = 0
    for row, (r, case, variant) in zip(report["results"], expected):
        require(row["round"] == r and row["variant"] == variant and row["backend"] == "metal" and
                tuple(row["case"][a] for a in ("operation", "m", "n", "k")) == case, "wrong or duplicated visit")
        order = ["native", "torch"] if r // 2 == 0 else ["torch", "native"]
        require(row["implementation_order"] == order, "unbalanced framework order")
        elements += correctness(row)
        op, m, n, k = case
        native = row["native"]
        plan, = native["execution_plans"]
        matrix, = plan["matrices"]
        require(row["block"] == [64, 64, 4096] and native["planner_threads"] == plan["threads"] == 256 and
                native["execution_scope"] == "group" and native["pipeline_window"] == native["copy_batch"] == 1 and
                native["fuse_matrix_epilogues"] == (variant == "fused") and native["metal_mpp"] and
                native["forward_readonly_tile_loads"] and not native["elide_independent_subgroup_barriers"], "retuned realization")
        require(native["program_order"] == plan["program_order"] == [1, 1] and
                plan["program_grid"] == [(m + 63) // 64, (n + 63) // 64] and
                plan["programs"] == math.prod(plan["program_grid"]) and matrix["persistent_accumulator"], "wrong launch/recurrence")
        require(matrix["subgroups_m"] * matrix["atom_rows"] * 8 == 64 and
                matrix["subgroups_n"] * matrix["atom_columns"] * 8 == 64 and
                matrix["subgroups_m"] * matrix["subgroups_n"] * 32 == 256, "invalid ownership")
        enabled = variant == "fused" and op != "gemm"
        direct = op == "gemm" or enabled
        require(matrix["direct_accumulator_store"] == direct and (plan["shared_memory_bytes"] == 0) == direct,
                "wrong direct output/storage plan")
        require(op == "gemm" or plan["independent_elements"] > 0, "erased scalar epilogue cost")
        code = source_code(row, directory)
        require(("mpp_element_index" in code) == enabled and ("threadgroup float" not in code) == direct and
                "mpp::tensor_ops::matmul2d<" in code, "wrong emitted family")
        key = case, variant
        require(key not in sources or sources[key] == row["native_source_sha256"], "source changed across rounds")
        require(key not in plans or plans[key] == plan, "plan changed across rounds")
        sources[key], plans[key] = row["native_source_sha256"], plan
        sequence = ["mm.out"] if op == "gemm" else ["mm.out", "mul_(0.125)", "add_(0.25)",
                                                     "clamp_min.out(0)" if op == "gemm_relu" else "gelu.out(approximate=tanh)"]
        require(row["torch"]["output_policy"] == "preallocated_out" and row["torch"]["operator_sequence"] == sequence and
                row["torch"]["intermediate_policy"] == ("none" if op == "gemm" else "preallocated_matrix_result"), "unmatched eager graph")
        records[case, r, variant] = {p: timing(row[p], 7) for p in ("native", "torch")}
    summary = []
    for case in cases:
        same = sources[case, "reference"] == sources[case, "fused"]
        require(same == (case[0] == "gemm"), "missing plain-GEMM source control")
        item = dict(operation=case[0], shape=case[1:], source_identical=same,
                    sources={v: sources[case, v] for v in ("reference", "fused")},
                    shared_bytes={v: plans[case, v]["shared_memory_bytes"] for v in ("reference", "fused")}, metrics={})
        for metric in METRICS:
            before = [records[case, r, "reference"]["native"][metric] for r in range(4)]
            after = [records[case, r, "fused"]["native"][metric] for r in range(4)]
            torch = [records[case, r, "fused"]["torch"][metric] for r in range(4)]
            item["metrics"][metric] = dict(reference_us=median(before), fused_us=median(after), torch_us=median(torch),
                new_old=ratios([b / a for a, b in zip(before, after)]), new_torch=ratios([b / a for a, b in zip(torch, after)]))
        summary.append(item)
    return dict(passed=True, complete_outputs=288, checked_elements=elements, pairs=72,
                unique_sources=len(set(sources.values())), summary=summary,
                use="fixed-schedule exploratory profitability; default off, no cross-route or MPS parity claim")


def controls(report, directory):
    gate(report, "controls")
    cases = [("metal", op, *shape) for op in OPS for shape in SHAPES[:2]]
    cases += [(backend, op, 17, 257, 1) for backend in ("metal", "cpu") for op in ("add", "gelu_pair", "sum", "softmax")]
    require(len(report["results"]) == 28, "wrong controls cohort")
    pattern = re.compile(r'(!\d+ = !\{!")(0x[0-9a-f]+)((?:\.w\d+\.b\d+)?", !\d+, i64 0\})')
    summary, elements = [], 0
    for i, (backend, op, m, n, k) in enumerate(cases):
        codes = []
        for row, variant in zip(report["results"][i * 2:i * 2 + 2], ("baseline", "reference")):
            require(row["backend"] == backend and row["variant"] == variant and row["round"] == 0 and
                    row["case"] == dict(operation=op, m=m, n=n, k=k) and
                    not row["native"].get("fuse_matrix_epilogues", False), "wrong compatibility control")
            elements += correctness(row)
            codes.append(source_code(row, directory))
        left, right = (code.splitlines(keepends=True) for code in codes)
        require(len(left) == len(right), "changed default instruction count")
        forward, inverse = {}, {}
        for a, b in zip(left, right):
            am, bm = pattern.fullmatch(a.rstrip("\n")), pattern.fullmatch(b.rstrip("\n"))
            if backend == "cpu" and am and bm:
                require(am[1] == bm[1] and am[3] == bm[3] and forward.setdefault(am[2], bm[2]) == bm[2] and
                        inverse.setdefault(bm[2], am[2]) == am[2], "changed alias graph/width/offset or nonbijective identity")
            else:
                require(a == b, "changed default executable IR")
        summary.append(dict(backend=backend, operation=op, shape=[m, n, k], raw_identical=codes[0] == codes[1],
                            equivalent=True, tbaa_identities=len(forward)))
    return dict(passed=True, complete_outputs=56, checked_elements=elements, summary=summary, use="compatibility, not speed")


def checks(report, directory):
    gate(report, "check", success=False)
    expected = [("test_tile_tirx", "unit"), ("test_tile_tirx_planner", "unit")]
    expected += [("test_tile_tirx_" + name, backend) for name in ("execution", "matrix", "poc", "poc_neural",
                 "poc_algorithms", "pipeline", "memory", "cooperative") for backend in ("metal", "cpu")]
    expected += [("test_tile_native_runtime", "metal")]
    require(len(report["results"]) == len(expected) == 19 and report["passed"] is False, "wrong check status/cohort")
    known = {"test_tile_tirx_memory-metal.log": (168, 2), "test_tile_tirx_cooperative-metal.log": (100, 1)}
    assertions, failures = 0, []
    for row, (name, backend) in zip(report["results"], expected):
        require(row["log"] == name + "-" + backend + ".log", "missing/repeated test")
        log = re.sub(r"\x1b\[[0-9;]*m", "", (directory / row["log"]).read_text())
        if row["log"] in known:
            line, failed = known[row["log"]]
            require(not row["passed"] and row["exit_code"] != 0 and
                    set(re.findall(r"test_tirx_\w+\.cpp:(\d+)", log)) == {str(line)} and
                    re.search(r"asserts:\s+\d+.*\b" + str(failed) + r" failed", log), "unexpected test failure")
            failures.append(dict(log=row["log"], line=line, failed_assertions=failed))
        else:
            match = re.search(r"all tests passed \((\d+) asserts in (\d+) tests\)", log)
            require(row["passed"] and row["exit_code"] == 0 and match and row["assertions"] == int(match[1]) > 0, "empty/failed test")
            assertions += row["assertions"]
    return dict(passed=True, all_suites_green=False, passing_suites=17, successful_suite_assertions=assertions,
                known_failures=failures, use="new tests pass; two pre-existing barrier source-string suites remain red")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("replay", "controls", "check"))
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    directory = args.directory or HERE / args.phase
    report = json.loads((directory / "results.json").read_text())
    analyze = dict(replay=replay, controls=controls, check=checks)[args.phase]
    result = analyze(report, directory)
    mutations = {"missing_row": lambda r: r["results"].pop(),
                 "duplicate_row": lambda r: r["results"].__setitem__(1, r["results"][0]),
                 "changed_artifacts": lambda r: r["metadata"].update(artifacts_unchanged=False)}
    if args.phase != "check":
        mutations.update(partial_output=lambda r: r["results"][0]["native"].update(output_elements=1),
                         bad_oracle=lambda r: r["results"][0]["native"]["correctness"].update(atol=1),
                         missing_source=lambda r: r["results"][0].update(native_source_sha256="missing"))
    else:
        mutations.update(empty_test=lambda r: r["results"][0].update(assertions=0),
                         concealed_failure=lambda r: r.update(passed=True))
    if args.phase == "replay":
        mutations.update(wrong_divisor=lambda r: r["results"][0]["native"]["device_timing"]["control"].update(repetitions=0),
                         unbalanced_order=lambda r: r["results"][0].update(implementation_order=["torch", "native"]),
                         retuned_workers=lambda r: r["results"][0]["native"].update(planner_threads=1024),
                         erased_math=lambda r: r["results"][3]["native"]["execution_plans"][0].update(independent_elements=0))
    for name, mutate in mutations.items():
        bad = copy.deepcopy(report)
        mutate(bad)
        try:
            analyze(bad, directory)
        except (ValueError, KeyError, FileNotFoundError, ZeroDivisionError):
            continue
        raise ValueError("auditor accepted corruption: " + name)
    result.update(input_sha256=digest(directory / "results.json"), auditor_sha256=digest(__file__), rejected_mutations=list(mutations))
    (directory / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.phase == "replay":
        lines = ["# Fixed-schedule matrix epilogue timing", "",
                 "All times are microseconds/op. Ratios are medians of four within-round pairs; below one favors the fragment candidate.",
                 "Ranges and slower counts retain every round, not confidence intervals. Plain GEMM is an identical-source control.",
                 "No-counter GPU intervals include command-buffer work and gaps; counter intervals are instrumented diagnostics.",
                 "See [methods and limitations](../notes.md) before comparing frameworks. Fusion remains off by default.", ""]
        for metric in METRICS:
            lines += ["## " + metric, "",
                      "| Graph / M×N×K | Reference µs | Fragment µs | Torch µs | New/old median [min, max] | Slower/4 | New/Torch median [min, max] | Slower/4 |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for item in result["summary"]:
                v = item["metrics"][metric]
                a, b = v["new_old"], v["new_torch"]
                label = item["operation"] + " / " + "×".join(map(str, item["shape"]))
                lines.append(f"| {label} | {v['reference_us']:.3f} | {v['fused_us']:.3f} | {v['torch_us']:.3f} | "
                             f"{a['median']:.3f} [{a['minimum']:.3f}, {a['maximum']:.3f}] | {a['slower_rounds']} | "
                             f"{b['median']:.3f} [{b['minimum']:.3f}, {b['maximum']:.3f}] | {b['slower_rounds']} |")
            lines += [""]
        (directory / "results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "summary"}))


if __name__ == "__main__":
    main()
