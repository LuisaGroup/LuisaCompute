#!/usr/bin/env python3
"""Recompute each timing boundary separately; never discard failed probes."""
import copy
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import struct

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("snapshot_audit", HERE.parent / "m1-max-20260908-xir-indexable/audit.py")
snapshot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snapshot)
require, p50 = snapshot.require, snapshot.p50


def read(path):
    return json.loads(path.read_text())


def large_report(directory):
    report = read(directory / "results.json")
    meta = report["metadata"]
    require(meta["backend"] == "cpu" and meta["requested_threads"] == 8 and meta["rounds"] == 2 and
            meta["samples"] == 7 and meta["artifacts_unchanged"], "wrong large-run controls")
    values, inputs, failures = {}, {}, []
    for row in report["results"]:
        key = row["operation"], tuple(row["dimensions"]), row["round"], row["path"]
        require(key not in values, "duplicate case visit")
        values[key] = row
        require(row["order"] == (["native", "torch"] if row["round"] == 0 else ["torch", "native"]), "unbalanced large-run order")
        if not row["valid"]:
            require(bool(row.get("error")), "unrecorded failure")
            failures.append(dict(operation=row["operation"], dimensions=row["dimensions"], path=row["path"], error=row["error"]))
            continue
        m, check = row["measurement"], row["correctness"]
        dims = row["dimensions"]
        count = math.prod(dims) if len(dims) == 2 else dims[0] * dims[1] * dims[3] * dims[6]
        require(check["elements"] == count and check["atol"] == check["rtol"] == 5e-5 and math.isfinite(check["max_abs_error"]), "incomplete independent FP64 check")
        case = key[:2]
        require(case not in inputs or inputs[case] == row["input_sha256"], "input bits changed")
        inputs[case] = row["input_sha256"]
        require(len(inputs[case]) == 3 and all(len(v) == 64 for v in inputs[case]), "missing input fingerprints")
        for metric in ("throughput_us", "latency_us"):
            require(len(m[metric]) == 7 and snapshot.close(p50(m[metric]), m[metric + "_p50"]), "bad median")
        if key[3] == "native":
            require(m["timing"] == "synchronized_host_wall" and m["implementation"] == "tile_xir_simd" and
                    m["fast_math"] is False and m["correctness"]["checks"] == 2 and
                    m["correctness"]["guard_elements_per_check"] == 34, "wrong native boundary or checks")
        else:
            require(m["pre_timing_correctness"]["elements"] == count and bool(m["expression"]), "missing eager API/precheck")
    summaries = []
    for saved in report["summary"]:
        op, dims = saved["operation"], tuple(saved["dimensions"])
        rows = [values[op, dims, r, name] for r in range(2) for name in ("native", "torch")]
        complete = all(row["valid"] for row in rows)
        require(saved["complete"] == complete, "failure lost from summary")
        item = dict(operation=op, dimensions=dims, complete=complete)
        if complete:
            for metric in ("throughput_us", "latency_us"):
                medians = {name: p50([p50(values[op, dims, r, name]["measurement"][metric]) for r in range(2)]) for name in ("native", "torch")}
                ratios = [p50(values[op, dims, r, "native"]["measurement"][metric]) / p50(values[op, dims, r, "torch"]["measurement"][metric]) for r in range(2)]
                require(snapshot.close(saved[metric + "_p50"]["paired_native_over_torch_median"], p50(ratios)), "bad paired large-run ratio")
                require(all(snapshot.close(saved[metric + "_p50"]["median_us"][name], value) for name, value in medians.items()), "bad large-run summary")
                item[metric] = dict(medians=medians, native_over_torch=p50(ratios))
            item["compile_ms"] = p50([values[op, dims, r, "native"]["measurement"]["compile_ms"] for r in range(2)])
        summaries.append(item)
    require(len(values) == 4 * len(summaries), "missing large-run visits")
    return dict(valid_visits=sum(r["valid"] for r in values.values()), failures=failures, summaries=summaries)


def native_replay():
    report = read(HERE / "native/results.json")
    require(report["metric"] == "single_thread_native_entry_host_wall_us" and report["runtime_excluded"] and
            report["cpu_threads"] == 1 and not report["hardware_cycles"], "native-entry boundary changed")
    require(len(report["results"]) == 18, "missing native replay visits")
    require(hashlib.sha256((HERE / "native/inductor.cpp").read_bytes()).hexdigest() ==
            report["artifacts"]["inductor"]["source_sha256"], "generated Torch source changed")
    require(hashlib.sha256((HERE / "replay_native.cpp").read_bytes()).hexdigest() == report["helper_source_sha256"], "replay helper changed")
    for name, path in (("baseline", "native/baseline.o.gz"), ("candidate", "code/kernel.o.gz")):
        require(hashlib.sha256(gzip.decompress((HERE / path).read_bytes())).hexdigest() ==
                report["artifacts"][name]["object_sha256"], "replayed native object changed")
    for name in ("baseline", "candidate", "inductor"):
        rows = [r for r in report["results"] if r["variant"] == name]
        require(len(rows) == 6 and sorted(r["round"] for r in rows) == list(range(6)), "unbalanced native visits")
        for row in rows:
            require(len(row["samples_us"]) == 7 and snapshot.close(p50(row["samples_us"]), row["median_us"]), "bad replay samples")
            require(row["guard_elements"] == 68 and row["correctness"]["elements"] == 16384, "incomplete replay checks")
        require(len({tuple(row["order"]) for row in rows}) == 6, "native orders were not fully balanced")
        require(snapshot.close(report["summary_us"][name], p50([r["median_us"] for r in rows])), "bad replay summary")
    return report["summary_us"]


def code_size():
    data = gzip.decompress((HERE / "code/kernel.o.gz").read_bytes())
    require(struct.unpack_from("<2I", data) == (0xFEEDFACF, 0x0100000C), "not ARM64 Mach-O")
    commands = struct.unpack_from("<I", data, 16)[0]
    offset, result = 32, None
    for _ in range(commands):
        command, size = struct.unpack_from("<2I", data, offset)
        require(size >= 8 and offset + size <= len(data), "bad Mach-O command")
        if command == 0x19:
            sections = struct.unpack_from("<I", data, offset + 64)[0]
            require(72 + sections * 80 <= size, "bad Mach-O sections")
            for index in range(sections):
                section = offset + 72 + index * 80
                if data[section:section + 16].rstrip(b"\0") == b"__text":
                    result = struct.unpack_from("<Q", data, section + 40)[0]
        offset += size
    require(result is not None, "no machine code")
    assembly = gzip.decompress((HERE / "code/object.asm.gz").read_bytes()).decode()
    require("[sp, #-0xa0]!" in assembly and "sub\tsp, sp, #0x4, lsl #12" in assembly and
            "sub\tsp, sp, #0x200" in assembly, "unexpected captured frame")
    return result


def main():
    ab = read(HERE / "ab/report.json")
    result = dict(ab=snapshot.inspect(ab, HERE / "ab"),
                  pre_workspace=large_report(HERE / "pre-workspace"),
                  large=large_report(HERE / "large"), native_entry_us=native_replay(),
                  actual_object_text_bytes=code_size())
    mutations = [lambda r: r["metadata"].update(kernel_only=True),
                 lambda r: r["metadata"].update(artifacts_unchanged=False),
                 lambda r: r["results"][0]["measurement"]["throughput_us"].__setitem__(0, float("nan")),
                 lambda r: r["results"][0]["oracle"].update(elements=1),
                 lambda r: r["summaries"][0].update(median_ratio=999.0)]
    for mutate in mutations:
        broken = copy.deepcopy(ab)
        mutate(broken)
        try:
            snapshot.inspect(broken, HERE / "ab", raw=False)
        except ValueError:
            continue
        raise ValueError("corrupted A/B report accepted")
    result["rejected_mutations"] = len(mutations)
    result["sha256"] = {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(HERE.rglob("*")) if p.is_file() and p.name != "audit.json" and "__pycache__" not in p.parts}
    (HERE / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "sha256"}, indent=2))


if __name__ == "__main__":
    main()
