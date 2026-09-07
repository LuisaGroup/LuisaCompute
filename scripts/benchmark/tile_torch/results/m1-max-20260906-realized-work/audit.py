#!/usr/bin/env python3
"""Audit saved model choices, realized work, full outputs and paired timings."""
import argparse
import copy
import importlib.util
import itertools
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHAPES = ((512, 512, 512), (4096, 4096, 4096), (1025, 1025, 1024), (4096, 4096, 11008),
          (257, 769, 113), (2049, 4097, 1025), (4097, 4097, 4096), (8192, 8192, 8192))
BLOCKS = tuple((m, n, k) for m, n in ((32, 64), (64, 64), (128, 32), (128, 64), (64, 128)) for k in (128, 512, 4096))
# Reuse only the independent raw GPU/E2E denominator checker, not benchmark
# validation, selection or percentile implementations.
TIMING_AUDIT = HERE.parent / "m1-max-20260906-mpp-state-budget/audit.py"
spec = importlib.util.spec_from_file_location("prior_timing_audit", TIMING_AUDIT)
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)
require, digest, median, ratio = timing.require, timing.digest, timing.median, timing.ratio


def gate(report, phase):
    meta = report["metadata"]
    require(report["passed"] is True and meta["phase"] == phase, "incomplete phase")
    require(meta["artifacts_unchanged"] and meta["artifacts_before"] == meta["artifacts_after"], "changed compiler")
    require(len(meta["builds"]) == 2 and all(row["exit_code"] == 0 for row in meta["builds"]), "missing full builds")
    return meta


def realization(row, variant, directory, samples):
    m, n, k = (row["case"][a] for a in ("m", "n", "k"))
    bm, bn, bk = row["block"]
    require(row["valid"] is True and row["backend"] == "metal" and row["case"]["operation"] == "gemm", "wrong case")
    require((m, n, k) in SHAPES and (bm, bn, bk) in BLOCKS, "unregistered shape/block")
    native = row["native"]
    plan, = native["execution_plans"]
    mapping, = plan["matrices"]
    threads = plan["threads"]
    gm, gn, rm, rn = (mapping[a] for a in ("subgroups_m", "subgroups_n", "atom_rows", "atom_columns"))
    require(native["planner_threads"] in (0, threads) and type(threads) is int and threads > 0 and
            threads % 32 == 0 and threads <= native["metal_max_threads"], "wrong thread request")
    require(gm * gn * 32 == threads and gm * rm * 8 == bm and gn * rn * 8 == bn and (rm % 2 == 0 or rn % 2 == 0), "wrong coverage")
    require(native["pipeline_window"] == native["copy_batch"] == 1 and native["metal_mpp"] is True and
            native["forward_readonly_tile_loads"] is True and native["elide_independent_subgroup_barriers"] is False,
            "wrong realization policy")
    require(plan["shared_memory_bytes"] == 0 and plan["fragment_scalars_per_lane"] == 2 * rm * rn <= 64 and
            plan["optimized"] is True and mapping["persistent_accumulator"] and mapping["direct_accumulator_store"], "wrong resource path")
    steps = (k + bk - 1) // bk
    scalar = bm * bn * (steps + 2)
    nominal = bm * bn / 64 * steps * bk / 8
    physical = bm * bn / 64 * k / 8
    expected = {"matrix_issues": physical if variant == "candidate" else nominal,
                "independent_elements": 0 if variant == "candidate" else scalar}
    require(plan["cost_basis"] == ("metal_mpp_memory_v3" if variant == "candidate" else "metal_mpp_memory_v2"), "wrong cost basis")
    if variant == "candidate":
        expected.update(nominal_matrix_issues=nominal, elided_independent_elements=scalar)
    for name, count in expected.items():
        require(math.isclose(plan[name], count, rel_tol=1e-6, abs_tol=1e-6), "wrong realized work: " + name)
    for provider in ("native", "torch", "system"):
        values = row[provider]
        require(values["correctness"] == dict(atol=1e-4, rtol=1e-4, max_abs_error=0.0), "incorrect/incomplete oracle")
        require(len(values["throughput_us"]) == len(values["latency_us"]) == samples, "wrong sample count")
        require(all(math.isclose(median(values[phase + "_us"]), values[phase + "_us_p50"], rel_tol=1e-10)
                    for phase in ("throughput", "latency")), "wrong host median")
    require(native["output_elements"] == m * n and row["torch"]["output_policy"] == "preallocated_out" and
            row["system"]["implementation"] == "mps_matrix_multiplication", "wrong outputs/baseline")
    sha = row["native_source_sha256"]
    source = directory / "sources" / (sha + ".metal")
    require(source.is_file() and digest(source) == sha, "missing/changed source")
    code = source.read_text()
    require("mpp::tensor_ops::matmul2d<" in code and "threadgroup float" not in code and
            ("mode::multiply_accumulate" in code) == (k > bk), "wrong emitted matrix/storage mode")
    return dict(block=row["block"], threads=threads, subgroup=[gm, gn], local=[8 * rm, 8 * rn],
                source=sha, physical_equivalent_issues=plan["matrix_issues"], nominal_issues=nominal,
                retained_scalar_elements=plan["independent_elements"], elided_scalar_elements=plan.get("elided_independent_elements", 0))


def selection(report, directory):
    meta = gate(report, "select")
    require(len(report["results"]) == 16 and not meta["coefficients_fitted"], "altered selection protocol")
    summary, seen, controls = [], set(), {}
    outputs = elements = 0
    for record in report["results"]:
        shape, variant = tuple(record["shape"]), record["variant"]
        require(shape in SHAPES and variant in ("reference", "candidate") and (shape, variant) not in seen and record["passed"], "duplicate/missing selection")
        seen.add((shape, variant))
        path = directory / record["report"]
        row, = json.loads(path.read_text())["results"]
        require(tuple(row["case"][a] for a in ("m", "n", "k")) == shape, "wrong selection shape")
        trials = row["tuning"]["trials"]
        require(len(trials) == 15 and {tuple(t["block"]) for t in trials} == set(BLOCKS), "altered candidate set")
        require(row["tuning"]["selection_metric"] == "sum_execution_plan_normalized_kernel_cost", "timing-selected plan")
        mappings = []
        for trial in trials:
            require(trial["valid"] is True and trial["group_threads"] == 0, "unexpected admission/numerical failure")
            measured = trial["measurement"]
            require(measured["case"] == row["case"] and measured["block"] == trial["block"] and
                    measured["native_command"][0] == meta["variants"][variant]["binary"], "wrong trial identity")
            mapping = realization(measured, variant, path.parent, 1)
            mappings.append(mapping)
            cost = sum(p["normalized_kernel_cost"] for p in measured["native"]["execution_plans"])
            require(math.isclose(cost, trial["model_cost"]) and math.isclose(cost, trial["selection_score"]), "wrong model score")
            controls[shape, tuple(trial["block"]), variant] = mapping
        selected = min(range(15), key=lambda i: trials[i]["model_cost"])
        require(selected == row["tuning"]["selected_trial"] == row["tuning"]["model_selected_trial"], "wrong selection")
        final = realization(row, variant, path.parent, 1)
        require(final == mappings[selected], "fresh selection changed source/mapping")
        outputs += 16 * 3
        elements += shape[0] * shape[1] * 16 * 3
        summary.append(dict(shape=shape, variant=variant, heldout=shape in SHAPES[4:], selected=final,
                            model_score=trials[selected]["model_cost"]))
    unchanged = remapped = 0
    for shape, block in itertools.product(SHAPES, BLOCKS):
        a, b = (controls[shape, block, variant] for variant in ("reference", "candidate"))
        same = all(a[field] == b[field] for field in ("threads", "subgroup", "local"))
        if same:
            require(a["source"] == b["source"], "same mapping changed emitted code")
            unchanged += 1
        else:
            remapped += 1
    return dict(passed=True, attempted_candidates=240, complete_outputs=outputs, checked_elements=elements,
                unchanged_fixed_mappings=unchanged, remapped_fixed_blocks=remapped, summary=summary)


def replay(report, directory):
    meta = gate(report, "replay")
    selected = json.loads((directory.parent / "selection/audit.json").read_text())
    selected_meta = json.loads((directory.parent / "selection/results.json").read_text())["metadata"]
    require(all(value == meta["artifacts_before"][name] for name, value in selected_meta["artifacts_after"].items()),
            "compiler/driver changed after model selection")
    selected_maps = {(tuple(row["shape"]), row["variant"]): row["selected"] for row in selected["summary"]}
    require(meta["rounds"] == 6 and meta["samples"] == 9 and meta["sample_ms"] == 30 and meta["warmup_ms"] == 100, "altered replay")
    require(len(report["results"]) == 96, "missing replay rows")
    frozen, records, mappings = {}, {}, {}
    for variant, plans in meta["frozen_plans"].items():
        for p in plans:
            shape = tuple(p["case"][a] for a in ("m", "n", "k"))
            require(shape in SHAPES and (shape, variant) not in frozen, "duplicate frozen plan")
            frozen[shape, variant] = p
    require(len(frozen) == 16, "missing frozen plan")
    for row in report["results"]:
        shape = tuple(row["case"][a] for a in ("m", "n", "k"))
        variant, r = row["variant"], row["round"]
        require(type(r) is int and 0 <= r < 6 and (shape, r, variant) not in records, "duplicate/invalid replay row")
        p = frozen[shape, variant]
        require(row["block"] == p["gemm_block"] and row["native"]["planner_threads"] == p["group_threads"] != 0, "retuned plan")
        require(row["loader"] == meta["variants"][variant]["loader"] and row["native_command"][0] == meta["variants"][variant]["binary"], "wrong loader")
        mapped = realization(row, variant, directory, 9)
        require(mapped == selected_maps[shape, variant], "frozen source differs from selected source")
        require((shape, variant) not in mappings or mappings[shape, variant] == mapped, "unstable realization")
        mappings[shape, variant] = mapped
        records[shape, r, variant] = dict(data=timing.metrics(row, 9), order=tuple(row["implementation_order"]))
    summary = []
    for i, shape in enumerate(SHAPES):
        for variant in ("reference", "candidate"):
            require({records[shape, r, variant]["order"] for r in range(6)} == set(itertools.permutations(("native", "torch", "system"))), "unbalanced providers")
        for r in range(6):
            visits = [row["variant"] for row in report["results"] if row["round"] == r and tuple(row["case"][a] for a in ("m", "n", "k")) == shape]
            require(visits == (["reference", "candidate"] if (r + i) % 2 == 0 else ["candidate", "reference"]), "unbalanced variants")
        result = dict(shape=shape, heldout=shape in SHAPES[4:], mappings={v: mappings[shape, v] for v in ("reference", "candidate")}, metrics={})
        for metric in ("gpu_throughput", "e2e_throughput", "gpu_latency", "e2e_latency"):
            old = [records[shape, r, "reference"]["data"]["native"][metric] for r in range(6)]
            new = [records[shape, r, "candidate"]["data"]["native"][metric] for r in range(6)]
            torch = [records[shape, r, "candidate"]["data"]["torch"][metric] for r in range(6)]
            mps = [records[shape, r, "candidate"]["data"]["system"][metric] for r in range(6)]
            result["metrics"][metric] = dict(old_us=median(old), new_us=median(new), torch_us=median(torch), mps_us=median(mps),
                                               new_old=ratio([b / a for a, b in zip(old, new)]),
                                               new_torch=ratio([b / a for a, b in zip(torch, new)]),
                                               new_mps=ratio([b / a for a, b in zip(mps, new)]))
        summary.append(result)
    return dict(passed=True, complete_outputs=288, checked_elements=sum(m * n for m, n, k in SHAPES) * 36, paired_rounds=48, summary=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("selection", "replay"))
    args = parser.parse_args()
    directory = HERE / args.phase
    source = directory / "results.json"
    report = json.loads(source.read_text())
    analyze = selection if args.phase == "selection" else replay
    result = analyze(report, directory)
    mutations = {"missing_row": lambda r: r["results"].pop(),
                 "duplicate_row": lambda r: r["results"].__setitem__(1, r["results"][0]),
                 "changed_binary": lambda r: r["metadata"].update(artifacts_unchanged=False)}
    if args.phase == "replay":
        mutations.update(partial_output=lambda r: r["results"][0]["native"].update(output_elements=1),
                         wrong_divisor=lambda r: r["results"][0]["native"]["device_timing"]["control"].update(repetitions=0),
                         wrong_work=lambda r: r["results"][0]["native"]["execution_plans"][0].update(matrix_issues=0),
                         wrong_loader=lambda r: r["results"][0].update(loader="wrong"),
                         wrong_order=lambda r: r["results"][0].update(implementation_order=["native"]))
    for name, mutate in mutations.items():
        bad = copy.deepcopy(report)
        mutate(bad)
        try:
            analyze(bad, directory)
        except (ValueError, KeyError, ZeroDivisionError):
            continue
        raise ValueError("auditor accepted mutation: " + name)
    result.update(rejected_mutations=list(mutations), input_sha256=digest(source), timing_auditor_sha256=digest(TIMING_AUDIT))
    (directory / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "summary"}), flush=True)


if __name__ == "__main__":
    main()
