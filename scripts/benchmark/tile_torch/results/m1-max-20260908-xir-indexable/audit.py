#!/usr/bin/env python3
"""Independently recompute descriptive A/B summaries, including incomplete probes."""
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct

HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def p50(values):
    require(bool(values) and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in values), "invalid timings")
    return statistics.median(values)


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def code_evidence():
    data = gzip.decompress((HERE / "code/kernel.o.gz").read_bytes())
    magic, cpu, _, _, commands, _, _, _ = struct.unpack_from("<8I", data)
    require(magic == 0xFEEDFACF and cpu == 0x0100000C, "not ARM64 Mach-O")
    offset, text_size = 32, None
    for _ in range(commands):
        command, size = struct.unpack_from("<2I", data, offset)
        require(size >= 8 and offset + size <= len(data), "invalid object command")
        if command == 0x19:
            count = struct.unpack_from("<I", data, offset + 64)[0]
            require(72 + count * 80 <= size, "invalid sections")
            for index in range(count):
                section = offset + 72 + index * 80
                if data[section:section + 16].rstrip(b"\0") == b"__text":
                    text_size = struct.unpack_from("<Q", data, section + 40)[0]
                    start = struct.unpack_from("<I", data, section + 48)[0]
                    require(start + text_size <= len(data), "truncated code")
        offset += size
    require(text_size == 262772, "unexpected object text size")
    assembly = gzip.decompress((HERE / "code/object.asm.gz").read_bytes()).decode()
    loop = assembly.split("   13724:", 1)[1].split("   13810:", 1)[0]
    require(loop.count("fadd.4s") == 2 and "add\tx9, x9, #0x4" in loop and "cmp\tx9, #0x400" in loop,
            "wrong contribution loop")
    require(loop.count("\tcmp\t") == 1 and loop.count("\tldr\ts") + loop.count("\tld1.s\t") == 8 and
            "b\t0x13724" in loop, "indexed loop has changed")
    require("[sp, #-0x50]!" in assembly and "sub\tsp, sp, #0x9, lsl #12" in assembly and
            "sub\tsp, sp, #0x1b0" in assembly, "stack frame changed")
    llvm = gzip.decompress((HERE / "code/kernel.ll.gz").read_bytes()).decode()
    body = llvm.split("\ndirect.schedule.1:", 1)[1].split("\ndirect.schedule.5:", 1)[0]
    require(body.count("call <8 x float> @llvm.masked.gather") == 1, "missing direct local gather")
    for variant, hotspot in (("baseline", "MachineSinking"), ("candidate", "MachineCSE")):
        sample = (HERE / f"code/{variant}-4096.sample.txt").read_text()
        require("LLVMJIT::lookup" in sample and hotspot in sample and "emit_assembly_copy" not in sample,
                "wrong normal-JIT hotspot")
    diagnostic = json.loads((HERE / "code/measurement.json").read_text())
    require(diagnostic["correctness"]["checks"] == 2 and diagnostic["correctness"]["elements_per_check"] == 64 * 256,
            "unverified diagnostic output")
    return dict(object_text_bytes=text_size, kernel_frame_bytes=80 + 9 * 4096 + 432,
                loop_static_instruction_slots=(0x13810 - 0x13724) // 4,
                loop_contribution_iterations=256, loop_lane_load_sites=8,
                normal_jit_hotspots=["MachineSinking", "MachineCSE"])


def inspect(report, directory, raw=True):
    meta = report["metadata"]
    require(meta["artifacts_unchanged"] is True and meta["packet_width"] == meta["workers"] == 8,
            "changed artifacts or controls")
    require(meta["timing"] == "synchronized_host_wall" and meta["kernel_only"] is False, "wrong timing scope")
    require(meta["samples"] == 7 and meta["sample_ms"] == 30 and meta["warmup_ms"] == 100, "changed sampling protocol")
    expected = [(i, op, dims, r, v, ["baseline", "candidate"] if r % 2 == 0 else ["candidate", "baseline"])
                for i, (op, dims) in enumerate(meta["cases"]) for r in range(meta["rounds"])
                for v in (["baseline", "candidate"] if r % 2 == 0 else ["candidate", "baseline"])]
    require(len(expected) == len(report["results"]), "missing visits")
    values, inputs, valid, failures = {}, {}, 0, []
    for row, (i, op, dims, r, variant, order) in zip(report["results"], expected):
        require((row["operation"], row["dimensions"], row["round"], row["variant"], row["order"]) ==
                (op, dims, r, variant, order), "wrong case/order")
        if not row["valid"]:
            require(row.get("error") and "measurement" not in row, "unrecorded failure")
            failures.append(dict(operation=op, dimensions=dims, variant=variant, error=row["error"]))
            continue
        m = row["measurement"]
        require(row["returncode"] == 0 and m["implementation"] == "tile_xir_simd" and m["backend"] == "cpu" and
                m["timing"] == meta["timing"] and m["precision"] == "fp32" and m["fast_math"] is False and
                m["relaxed_precision"] is False and m["dimensions"] == dims and m["operation"] == op, "wrong native metadata")
        require("W8," in m["realization"] and "8 CPU workers;" in m["realization"], "wrong execution controls")
        require(m["source_reduction_policy"] == "unordered_tree" and
                m["batch_policy"] == "one_runtime_command_list_per_batch", "changed policy")
        check = m["correctness"]
        require(check["checks"] == 2 and check["elements_per_check"] == math.prod(dims) and
                check["guard_elements_per_check"] == 34 and check["atol"] == check["rtol"] == 5e-5, "incomplete native checks")
        oracle = row["oracle"]
        require(oracle["elements"] == math.prod(dims) and oracle["atol"] == oracle["rtol"] == 5e-5 and
                math.isfinite(oracle["max_abs_error"]), "incomplete independent oracle")
        key = op, tuple(dims)
        require(key not in inputs or inputs[key] == row["input_sha256"], "mismatched inputs")
        inputs[key] = row["input_sha256"]
        require(len(inputs[key]) == 3 and all(len(h) == 64 for h in inputs[key] + [row["output_sha256"]]), "missing tensor fingerprints")
        require(1 <= m["repetitions"] <= 100000, "wrong batch divisor")
        for metric in ("throughput_us", "latency_us"):
            require(len(m[metric]) == 7 and close(p50(m[metric]), m[metric + "_p50"]), "bad sample median")
        if raw:
            tag = f"{i:02}-{op}-{'x'.join(map(str, dims))}-{r}-{variant}"
            source = json.loads((directory / (tag + ".json")).read_text())
            require(all(m[k] == v for k, v in source.items()), "raw measurement changed")
        values[op, tuple(dims), r, variant] = m
        valid += 1
    summaries = []
    require(len(report["summaries"]) == len(meta["cases"]), "missing case summaries")
    for (op, dims), saved in zip(meta["cases"], report["summaries"]):
        ratios = []
        for r in range(meta["rounds"]):
            key = op, tuple(dims), r
            if (*key, "baseline") in values and (*key, "candidate") in values:
                ratios.append(p50(values[*key, "baseline"]["throughput_us"]) / p50(values[*key, "candidate"]["throughput_us"]))
        require(saved["operation"] == op and saved["dimensions"] == dims and saved["complete_pairs"] == len(ratios) and
                len(saved["baseline_over_candidate_throughput_ratios"]) == len(ratios) and
                all(close(a, b) for a, b in zip(ratios, saved["baseline_over_candidate_throughput_ratios"])), "bad paired ratio")
        require(close(saved["median_ratio"], p50(ratios)) if ratios else saved["median_ratio"] is None, "bad paired median")
        result = dict(operation=op, dimensions=dims, complete_pairs=len(ratios), paired_ratios=ratios)
        for variant in ("baseline", "candidate"):
            rows = [v for (o, d, _, n), v in values.items() if (o, d, n) == (op, tuple(dims), variant)]
            if rows:
                result[variant] = dict(throughput_us=p50([p50(v["throughput_us"]) for v in rows]),
                                       latency_us=p50([p50(v["latency_us"]) for v in rows]),
                                       compile_ms=p50([v["compile_ms"] for v in rows]))
        summaries.append(result)
    return dict(valid_visits=valid, failed_visits=failures, summaries=summaries)


def main():
    reports = {name: json.loads((HERE / name / "report.json").read_text()) for name in ("ab", "large")}
    require(reports["ab"]["metadata"]["cases"] == [["rmsnorm", [17, 7]], ["rmsnorm", [17, 127]],
            ["rmsnorm", [64, 256]], ["rmsnorm", [1024, 256]], ["rmsnorm", [64, 513]],
            ["layernorm", [64, 256]], ["masked_softmax", [17, 65]], ["swiglu", [17, 65]]], "changed A/B cohort")
    require(reports["large"]["metadata"]["cases"] == [["rmsnorm", [17, 1537]], ["rmsnorm", [1024, 4096]],
            ["rmsnorm", [64, 16384]]], "changed large-shape cohort")
    require(reports["ab"]["metadata"]["rounds"] == 2 and reports["large"]["metadata"]["rounds"] == 1, "changed order count")
    results = {name: inspect(r, HERE / name) for name, r in reports.items()}
    require(results["ab"]["valid_visits"] == 32 and not results["ab"]["failed_visits"] and
            results["large"]["valid_visits"] == 0 and len(results["large"]["failed_visits"]) == 6, "changed validation coverage")
    results["code"] = code_evidence()
    mutations = [lambda r: r["metadata"].update(kernel_only=True),
                 lambda r: r["metadata"].update(artifacts_unchanged=False),
                 lambda r: r["results"][0]["measurement"]["throughput_us"].__setitem__(0, float("nan")),
                 lambda r: r["results"][0]["oracle"].update(elements=1),
                 lambda r: r["results"][1]["input_sha256"].__setitem__(0, "a" * 64),
                 lambda r: r["summaries"][0].update(median_ratio=999.0)]
    for mutate in mutations:
        broken = copy.deepcopy(reports["ab"])
        mutate(broken)
        try:
            inspect(broken, HERE / "ab", raw=False)
        except ValueError:
            continue
        raise ValueError("failed to reject a corrupted report")
    results["rejected_mutations"] = len(mutations)
    results["scope"] = "descriptive_frozen_xir_ab_not_torch_comparison_or_kernel_only"
    results["sha256"] = {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(HERE.rglob("*")) if p.is_file() and p.name != "audit.json" and "__pycache__" not in p.parts}
    (HERE / "audit.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({k: v for k, v in results.items() if k != "sha256"}, indent=2))


if __name__ == "__main__":
    main()
