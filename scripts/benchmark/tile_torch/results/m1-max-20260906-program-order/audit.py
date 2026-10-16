#!/usr/bin/env python3
"""Independently check the two exploratory cohorts; never select a default."""
import argparse
import copy
import importlib.util
import itertools
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
TIMING = HERE.parent / "m1-max-20260906-mpp-state-budget/audit.py"
spec = importlib.util.spec_from_file_location("timing_receipts", TIMING)
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)
require, digest, median, ratio = timing.require, timing.digest, timing.median, timing.ratio
SHAPES = ((512, 512, 512), (4096, 4096, 4096), (8192, 8192, 8192),
          (4096, 4096, 11008), (2049, 4097, 1025), (257, 769, 113))
CONFIGS = tuple((k, r, c) for k in (512, 4096) for r, c in ((1, 1), (2, 4), (4, 8), (8, 16)))
MPP_SHAPES = SHAPES[:3] + ((256, 11008, 4096), SHAPES[4])
MPP_CONFIGS = {
    "independent_128x64": (32, 32, 1, 1, 0, 1, 8, 4),
    "collective_128x64": (128, 64, 8, 1, 0, 1, 8, 1),
    "independent_128x32": (32, 32, 1, 1, 0, 1, 4, 4),
    "collective_128x32": (128, 32, 4, 1, 0, 1, 4, 1),
    "independent_64x64": (32, 32, 1, 1, 0, 1, 4, 2),
    "collective_64x64": (64, 64, 4, 1, 0, 1, 4, 1),
    "mps": None,
}
METRICS = ("gpu_throughput", "e2e_throughput", "gpu_latency", "e2e_latency")


def gate(report):
    meta = report["metadata"]
    require(report["passed"] is True and meta["build"]["exit_code"] == 0, "incomplete/build-failed run")
    require(meta["artifacts_unchanged"] is True and meta["artifacts_before"] == meta["artifacts_after"], "changed artifact")
    require(meta["rounds"] == 2 and meta["samples"] == 5 and meta["sample_ms"] == 20 and meta["warmup_ms"] == 100, "altered protocol")
    return meta


def expected_visits(shapes, configurations):
    for round_index in range(2):
        for shape in shapes if round_index == 0 else shapes[::-1]:
            shift = shapes.index(shape)
            order = configurations[shift:] + configurations[:shift]
            if round_index:
                order = order[::-1]
            for config in order:
                yield round_index, shape, config, order


def program_screen(report, directory):
    meta = gate(report)
    require(meta["phase"] == "screen" and len(report["results"]) == 96, "wrong traversal cohort")
    require(any(p.endswith("libtvm_compiler.dylib") for p in meta["artifacts_before"]), "unfingerprinted compiler")
    records, sources, plans = {}, {}, {}
    for row, (r, shape, cfg, order) in zip(report["results"], expected_visits(SHAPES, CONFIGS)):
        require(row["round"] == r and tuple(row["configuration"]) == cfg and
                tuple(tuple(v) for v in row["order"]) == order and
                tuple(row["case"][a] for a in ("m", "n", "k")) == shape, "wrong visit/configuration")
        m, n, k = shape
        bk, pr, pc = cfg
        native = row["native"]
        plan, = native["execution_plans"]
        matrix, = plan["matrices"]
        require(row["valid"] is True and row["backend"] == "metal" and row["case"]["operation"] == "gemm" and
                row["block"] == [128, 64, bk] and native["output_elements"] == m * n, "wrong case/output")
        require(native["execution_scope"] == "group" and native["program_order"] == plan["program_order"] == [pr, pc], "unrealized traversal")
        grid = [(m + 127) // 128, (n + 63) // 64]
        require(plan["program_grid"] == grid and plan["programs"] == grid[0] * grid[1], "changed launch coverage")
        require(native["planner_threads"] == plan["threads"] == 256 and
                native["pipeline_window"] == native["copy_batch"] == 1 and native["metal_mpp"] and
                native["forward_readonly_tile_loads"] and not native["elide_independent_subgroup_barriers"], "wrong execution policy")
        require(matrix["subgroups_m"] == 4 and matrix["subgroups_n"] == 2 and
                matrix["atom_rows"] == matrix["atom_columns"] == 4 and
                matrix["persistent_accumulator"] and matrix["direct_accumulator_store"] and
                plan["shared_memory_bytes"] == 0, "wrong local realization")
        key = shape, cfg
        sha = row["native_source_sha256"]
        source = directory / "sources" / (sha + ".metal")
        require(digest(source) == sha and (key not in sources or sources[key] == sha), "changed/missing source")
        code = source.read_text()
        require("mpp::tensor_ops::matmul2d<" in code and "threadgroup float" not in code and
                ("mode::multiply_accumulate" in code) == (k > bk), "unexpected matrix/storage mode")
        sources[key] = sha
        comparable = {name: value for name, value in plan.items() if name != "program_order"}
        require((shape, bk) not in plans or plans[shape, bk] == comparable, "traversal changed the local plan")
        plans[shape, bk] = comparable
        records[shape, cfg, r] = timing.metrics(row, 5)
    summary = []
    for shape, cfg in itertools.product(SHAPES, CONFIGS):
        control = cfg[0], 1, 1
        result = dict(shape=shape, configuration=cfg, source=sources[shape, cfg],
                      source_same_as_row_major=sources[shape, cfg] == sources[shape, control], metrics={})
        for metric in METRICS:
            current = [records[shape, cfg, r]["native"][metric] for r in range(2)]
            base = [records[shape, control, r]["native"][metric] for r in range(2)]
            torch = [records[shape, cfg, r]["torch"][metric] for r in range(2)]
            mps = [records[shape, cfg, r]["system"][metric] for r in range(2)]
            result["metrics"][metric] = dict(native_us=current, row_major_us=base, torch_us=torch, mps_us=mps,
                versus_row_major=ratio([b / a for a, b in zip(base, current)]),
                versus_torch=ratio([b / a for a, b in zip(torch, current)]),
                versus_mps=ratio([b / a for a, b in zip(mps, current)]))
        summary.append(result)
    return dict(passed=True, use="exploratory sensitivity only; no default or fitted cost promotion", complete_outputs=288,
                checked_elements=sum(m * n for m, n, _ in SHAPES) * 48, unique_sources=len(set(sources.values())), summary=summary)


def participation(report, directory):
    gate(report)
    require(len(report["results"]) == 70, "wrong participation cohort")
    records, sources = {}, {}
    for row, (r, shape, label, order) in zip(report["results"], expected_visits(MPP_SHAPES, tuple(MPP_CONFIGS))):
        require(row["valid"] is True and row["round"] == r and tuple(row["shape"]) == shape and
                row["variant"] == label and tuple(row["order"]) == order, "wrong participation visit")
        proof = row["correctness"]
        require(proof == dict(max_abs_error=0.0, atol=1e-4, rtol=1e-4, checked_elements=shape[0] * shape[1]), "wrong complete oracle")
        cfg = MPP_CONFIGS[label]
        values = row["measurement"]
        require(tuple(values[a] for a in ("m", "n", "k")) == shape, "wrong measured dimensions")
        if cfg is not None:
            require(tuple(row["config"]) == cfg, "wrong MPP configuration")
            bm, bn, participants, cooperative, wr, wc, groups, gr = cfg
            require(values["block"] == [bm, bn] and values["execution_simdgroups"] == participants and
                    values["group_simdgroups"] == groups and values["cohort_rows"] == gr and
                    values["walk_rows"] == wr and values["walk_columns"] == wc and
                    values["cooperative_output"] and values["inline_tensors"] and not values["relaxed_precision"] and
                    not values["static_reduction"] and not values["fast_math"], "wrong MPP realization")
            require(values["max_threads_per_group"] >= groups * 32, "invalid thread count")
            source = directory / row["source"]
            require(digest(source) == row["source_sha256"], "missing/changed MPP source")
            key = shape, label
            require(key not in sources or sources[key] == row["source_sha256"], "source changed across rounds")
            sources[key] = row["source_sha256"]
        else:
            require(row["config"] is None and values["implementation"] == "mps_matrix_multiplication" and
                    values["alpha"] == 1 and values["beta"] == 0 and values["dtype"] == "float32", "wrong MPS baseline")
        record = {}
        for metric in METRICS:
            field = metric.replace("e2e_", "") + "_us"
            require(len(values[field]) == 5 and math.isclose(median(values[field]), values[field + "_p50"], rel_tol=1e-10), "wrong median/sample count")
            record[metric] = median(values[field])
        records[shape, label, r] = record
    summary = []
    for shape, geometry in itertools.product(MPP_SHAPES, ("128x64", "128x32", "64x64")):
        result = dict(shape=shape, group_rectangle=geometry, metrics={})
        for metric in METRICS:
            independent = [records[shape, "independent_" + geometry, r][metric] for r in range(2)]
            collective = [records[shape, "collective_" + geometry, r][metric] for r in range(2)]
            mps = [records[shape, "mps", r][metric] for r in range(2)]
            result["metrics"][metric] = dict(independent_us=independent, collective_us=collective, mps_us=mps,
                                              collective_independent=ratio([b / a for a, b in zip(independent, collective)]))
        summary.append(result)
    return dict(passed=True, complete_outputs=70, checked_elements=sum(m * n for m, n, _ in MPP_SHAPES) * 14,
                source_files=60, use="matched-geometry exploratory diagnostic, not TIRx or accepted performance", summary=summary)


def default_controls(report, directory):
    meta = report["metadata"]
    require(meta["phase"] == "defaults" and meta["build"]["exit_code"] == 0 and
            meta["artifacts_unchanged"] and meta["artifacts_before"] == meta["artifacts_after"], "invalid default-control run")
    cohort = [("metal", "gemm", *shape) for shape in (SHAPES[0], SHAPES[1], SHAPES[4])]
    cohort += [(backend, op, m, n, 1) for backend in ("metal", "cpu")
               for op, m, n in (("add", 17, 257), ("gelu_pair", 17, 257), ("softmax", 37, 1537))]
    require(len(report["results"]) == 2 * len(cohort), "missing controls")
    summary = []
    # Check differing lines pairwise with a bijective address correspondence.
    # This is independent of the experiment's first-occurrence canonicalizer.
    pattern = re.compile(r'(!\d+ = !\{!")(0x[0-9a-f]+)((?:\.w\d+\.b\d+)?", !\d+, i64 0\})')
    for index, (backend, op, m, n, k) in enumerate(cohort):
        rows = report["results"][2 * index:2 * index + 2]
        codes = []
        for row, variant in zip(rows, ("baseline", "current")):
            require(row["valid"] and row["backend"] == backend and row["variant"] == variant and
                    row["case"] == dict(operation=op, m=m, n=n, k=k), "wrong control identity")
            require(row["native"]["output_elements"] == m * n * (2 if op == "gelu_pair" else 1), "partial control output")
            for provider in ("native", "torch"):
                proof = row[provider]["correctness"]
                atol, rtol = (1e-4, 1e-4) if op == "gemm" else (0.0, 0.0) if op == "add" else (2e-6, 2e-5)
                require(proof["atol"] == atol and proof["rtol"] == rtol and math.isfinite(proof["max_abs_error"]) and
                        0 <= proof["max_abs_error"] <= atol, "wrong oracle policy/error")
            source = directory / "sources" / (row["native_source_sha256"] + (".metal" if backend == "metal" else ".ll"))
            require(digest(source) == row["native_source_sha256"], "changed control source")
            codes.append(source.read_text())
        a, b = (code.splitlines(keepends=True) for code in codes)
        require(len(a) == len(b), "instruction/metadata line count changed")
        forward, inverse = {}, {}
        for left, right in zip(a, b):
            lm, rm = pattern.fullmatch(left.rstrip("\n")), pattern.fullmatch(right.rstrip("\n"))
            if backend == "cpu" and lm and rm:
                require(lm[1] == rm[1] and lm[3] == rm[3], "changed alias graph/offset/width")
                require(forward.setdefault(lm[2], rm[2]) == rm[2] and inverse.setdefault(rm[2], lm[2]) == lm[2], "nonbijective alias labels")
            else:
                require(left == right, "default executable IR changed")
        summary.append(dict(backend=backend, operation=op, shape=[m, n, k], raw_identical=codes[0] == codes[1],
                            equivalent=True, tbaa_identities=len(forward)))
    return dict(passed=True, use="source/numerical compatibility, not performance", complete_outputs=36, summary=summary,
                raw_driver_passed=report["passed"],
                raw_failure_explanation="Initial driver required raw CPU source equality; only bijective TBAA labels differ.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("participation", "screen", "defaults"))
    args = parser.parse_args()
    directory = HERE.parent / "m1-max-20260906-mpp-participation-geometry/screen" if args.phase == "participation" else HERE / args.phase
    source = directory / "results.json"
    report = json.loads(source.read_text())
    analyze = {"screen": program_screen, "participation": participation, "defaults": default_controls}[args.phase]
    result = analyze(report, directory)
    mutations = {"missing_row": lambda r: r["results"].pop(),
                 "duplicate_row": lambda r: r["results"].__setitem__(1, r["results"][0]),
                 "changed_artifact": lambda r: r["metadata"].update(artifacts_unchanged=False)}
    if args.phase == "screen":
        mutations.update(partial_output=lambda r: r["results"][0]["native"].update(output_elements=1),
                         ignored_order=lambda r: r["results"][1]["native"].update(program_order=[1, 1]),
                         wrong_divisor=lambda r: r["results"][0]["native"]["device_timing"]["control"].update(repetitions=0))
    elif args.phase == "participation":
        mutations.update(partial_output=lambda r: r["results"][0]["correctness"].update(checked_elements=1),
                         wrong_participation=lambda r: r["results"][0]["measurement"].update(execution_simdgroups=2))
    else:
        mutations.update(partial_output=lambda r: r["results"][0]["native"].update(output_elements=1))
    for name, mutate in mutations.items():
        bad = copy.deepcopy(report)
        mutate(bad)
        try:
            analyze(bad, directory)
        except (ValueError, KeyError, ZeroDivisionError):
            continue
        raise ValueError("auditor accepted mutation: " + name)
    result.update(input_sha256=digest(source), timing_auditor_sha256=digest(TIMING), rejected_mutations=list(mutations))
    (directory / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "summary"}), flush=True)


if __name__ == "__main__":
    main()
