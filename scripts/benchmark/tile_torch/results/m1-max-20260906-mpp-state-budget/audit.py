#!/usr/bin/env python3
"""Independent raw-timing, coverage, admission and frozen-plan audit."""
import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
SHAPES = ((1024, 1024, 1024), (4096, 4096, 4096), (1025, 1025, 1024), (4096, 4096, 11008),
          (257, 769, 113), (2049, 4097, 1025), (4097, 4097, 4096), (8192, 8192, 8192))
BLOCKS = ((128, 32, 4096), (128, 64, 4096), (64, 128, 4096), (64, 64, 4096))


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def median(values):
    require(values and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values), "invalid samples")
    return statistics.median(values)


def ratio(values):
    return dict(median=median(values), minimum=min(values), maximum=max(values), faster_rounds=sum(v < 1 for v in values), ratios=values)


def realization(row, variant, directory):
    m, n, k = (row["case"][a] for a in ("m", "n", "k"))
    bm, bn, bk = row["block"]
    native = row["native"]
    threads = native["planner_threads"]
    require(row["valid"] is True and row["backend"] == "metal" and row["case"]["operation"] == "gemm", "invalid output/case")
    require(tuple(row["block"]) in BLOCKS and threads in (64, 128, 256), "unregistered schedule")
    require(native["pipeline_window"] == native["copy_batch"] == 1 and native["output_elements"] == m * n and
            native["metal_mpp"] is True and native["forward_readonly_tile_loads"] is True and
            native["elide_independent_subgroup_barriers"] is False and native["mpp_intrinsics"] > 0, "wrong realization")
    plan, = native["execution_plans"]
    mapping, = plan["matrices"]
    rm, rn = mapping["atom_rows"], mapping["atom_columns"]
    gm, gn = mapping["subgroups_m"], mapping["subgroups_n"]
    require(gm * gn * 32 == threads and gm * rm * 8 == bm and gn * rn * 8 == bn and (rm % 2 == 0 or rn % 2 == 0), "invalid subgroup coverage/descriptor")
    require(mapping["persistent_accumulator"] is True and mapping["direct_accumulator_store"] is True and
            plan["shared_memory_bytes"] == 0 and plan["optimized"] is True and plan["threads"] == threads and
            plan["fragment_scalars_per_lane"] == 2 * rm * rn <= 64, "resource mismatch")
    old_admits = 2 * (rm * rn + rm + rn) <= 64
    require(variant != "reference" or old_admits, "old planner admitted an impossible state budget")
    sha = row["native_source_sha256"]
    source = directory / "sources" / (sha + ".metal")
    require(source.is_file() and digest(source) == sha, "missing/changed source")
    code = source.read_text()
    require("mpp::tensor_ops::matmul2d<" in code and "threadgroup float" not in code, "wrong emitted resource path")
    require(("mode::multiply_accumulate" in code) == (k > bk), "wrong recurrence mode")
    return dict(block=row["block"], threads=threads, subgroup=[gm, gn], local=[8 * rm, 8 * rn],
                logical_scalars=2 * rm * rn, newly_admitted=not old_admits, source=sha)


def metrics(row, samples):
    result = {}
    for provider in ("native", "torch", "system"):
        values = row[provider]
        proof = values["correctness"]
        require(proof["atol"] == proof["rtol"] == 1e-4 and proof["max_abs_error"] == 0, "wrong correctness receipt")
        if provider == "torch":
            require(values["output_policy"] == "preallocated_out", "Torch allocation mismatch")
        if provider == "system":
            require(values["implementation"] == "mps_matrix_multiplication" and values["dtype"] == "float32" and values["alpha"] == 1 and values["beta"] == 0, "wrong system baseline")
        control = values["device_timing"]["control"]
        require(control["method"] == "metal_command_buffer_timestamps_v1" and control["encoder_instrumentation"] is False, "wrong GPU timer")
        require(type(control["repetitions"]) is int and control["repetitions"] > 0, "wrong repetitions")
        result[provider] = {}
        for phase, divisor in (("throughput", control["repetitions"]), ("latency", 1)):
            raw = control[phase]
            require(len(raw) == samples and all(x["command_buffers"] > 0 for x in raw), "missing GPU samples")
            derived = [x["command_buffer_ns"] / (1000 * divisor) for x in raw]
            saved = control["command_buffer_" + phase + "_us"]
            require(len(saved) == samples and all(math.isclose(a, b, rel_tol=1e-10) for a, b in zip(derived, saved)), "GPU denominator mismatch")
            gpu, host = median(derived), median(values[phase + "_us"])
            require(len(values[phase + "_us"]) == samples and math.isclose(gpu, control["command_buffer_" + phase + "_us_p50"], rel_tol=1e-10) and
                    math.isclose(host, values[phase + "_us_p50"], rel_tol=1e-10), "wrong p50")
            result[provider].update({"gpu_" + phase: gpu, "e2e_" + phase: host})
    return result


def audit_replay(report, directory):
    meta = report["metadata"]
    require(report["passed"] is True and meta["phase"] == "replay" and meta["rounds"] == 6 and meta["samples"] == 9 and
            meta["sample_ms"] == 30 and meta["warmup_ms"] == 100, "incomplete replay or wrong protocol")
    require(meta["artifacts_unchanged"] is True and meta["artifacts_before"] == meta["artifacts_after"], "changed compiler")
    require(len(meta["builds"]) == 2 and all(b["exit_code"] == 0 for b in meta["builds"]), "missing full build")
    expected = {}
    for variant, plans in meta["frozen_plans"].items():
        for plan in plans:
            shape = tuple(plan["case"][a] for a in ("m", "n", "k"))
            require(shape in SHAPES and (shape, variant) not in expected, "invalid frozen plan")
            expected[shape, variant] = plan
    shapes = [tuple(p["case"][a] for a in ("m", "n", "k")) for p in meta["frozen_plans"]["reference"]]
    require(len(shapes) == 8 and set(shapes) == set(SHAPES) and len(expected) == 16 and len(report["results"]) == 96, "missing or duplicated case")
    records, mappings = {}, {}
    for row in report["results"]:
        shape = tuple(row["case"][a] for a in ("m", "n", "k"))
        variant, round_index = row["variant"], row["round"]
        key = shape, round_index, variant
        require((shape, variant) in expected and type(round_index) is int and 0 <= round_index < 6 and key not in records, "duplicate/invalid row")
        require(row["loader"] == meta["variants"][variant]["loader"] and row["native_command"][0] == meta["variants"][variant]["binary"] and
                Path(row["loader"].split(":")[0]) == Path(row["native_command"][0]).parent, "wrong compiler loader")
        plan = expected[shape, variant]
        require(row["block"] == plan["gemm_block"] and row["native"]["planner_threads"] == plan["group_threads"], "retuned frozen plan")
        mapping = realization(row, variant, directory)
        require((shape, variant) not in mappings or mappings[shape, variant] == mapping, "unstable realization")
        mappings[shape, variant] = mapping
        records[key] = dict(data=metrics(row, 9), order=tuple(row["implementation_order"]))
    summary = []
    for i, shape in enumerate(shapes):
        for variant in ("reference", "candidate"):
            require({records[shape, r, variant]["order"] for r in range(6)} == set(itertools.permutations(("native", "torch", "system"))), "unbalanced framework order")
        for r in range(6):
            visits = [row["variant"] for row in report["results"] if row["round"] == r and tuple(row["case"][a] for a in ("m", "n", "k")) == shape]
            require(visits == (["reference", "candidate"] if (r + i) % 2 == 0 else ["candidate", "reference"]), "unbalanced variant order")
        result = dict(shape=shape, heldout=shape in SHAPES[4:], mappings={v: mappings[shape, v] for v in ("reference", "candidate")}, metrics={})
        for metric in ("gpu_throughput", "e2e_throughput", "gpu_latency", "e2e_latency"):
            before = [records[shape, r, "reference"]["data"]["native"][metric] for r in range(6)]
            after = [records[shape, r, "candidate"]["data"]["native"][metric] for r in range(6)]
            torch = [records[shape, r, "candidate"]["data"]["torch"][metric] for r in range(6)]
            mps = [records[shape, r, "candidate"]["data"]["system"][metric] for r in range(6)]
            result["metrics"][metric] = dict(reference_us=median(before), candidate_us=median(after), torch_us=median(torch), mps_us=median(mps),
                                               new_old=ratio([b / a for a, b in zip(before, after)]),
                                               new_torch=ratio([b / a for a, b in zip(torch, after)]), new_mps=ratio([b / a for a, b in zip(mps, after)]))
        summary.append(result)
    return dict(passed=True, complete_outputs=288, checked_elements=sum(m * n for m, n, k in shapes) * 36,
                pairs=48, unique_sources=len({m["source"] for m in mappings.values()}), summary=summary)


def audit_search(report, directory):
    meta = report["metadata"]
    require(report["passed"] is True and meta["phase"] in ("search", "heldout"), "incomplete search")
    require(meta["artifacts_unchanged"] is True and meta["artifacts_before"] == meta["artifacts_after"], "changed search compiler")
    shapes = SHAPES[:4] if meta["phase"] == "search" else SHAPES[4:]
    require(len(report["results"]) == 8, "wrong search cohort")
    summaries, seen, sources = [], set(), {}
    outputs, elements, attempts, valid_count = 0, 0, 0, 0
    for record in report["results"]:
        shape, variant = tuple(record["shape"]), record["variant"]
        require(shape in shapes and variant in ("reference", "candidate") and (shape, variant) not in seen and record["passed"], "invalid search record")
        seen.add((shape, variant))
        path = directory / record["report"]
        row, = json.loads(path.read_text())["results"]
        require(tuple(row["case"][a] for a in ("m", "n", "k")) == shape, "search case mismatch")
        tuning = row["tuning"]
        trials = tuning["trials"]
        require(len(trials) == 12 and {(tuple(t["block"]), t["group_threads"]) for t in trials} == set(itertools.product(BLOCKS, (64, 128, 256))), "altered candidate set")
        legal = []
        for index, trial in enumerate(trials):
            attempts += 1
            # With these power-of-two rectangles, the old family admits only
            # <=32 output scalars/lane; the new family also admits exactly64.
            output_state = trial["block"][0] * trial["block"][1] // trial["group_threads"]
            expected = output_state <= (32 if variant == "reference" else 64)
            require(trial["valid"] == expected, "unexpected admission or numerical failure")
            if not expected:
                require("native benchmark failed (2)" in trial["error"], "unexpected rejection path")
                continue
            valid_count += 1
            measured = trial["measurement"]
            require(measured["case"] == row["case"] and measured["block"] == trial["block"] and
                    measured["native"]["planner_threads"] == trial["group_threads"] and
                    measured["native_command"][0] == meta["variants"][variant]["binary"], "trial identity mismatch")
            mapping = realization(measured, variant, path.parent)
            data = metrics(measured, 5)
            gpu = data["native"]["gpu_throughput"]
            cost = sum(p["normalized_kernel_cost"] for p in measured["native"]["execution_plans"])
            require(math.isclose(gpu, trial["selection_score"], rel_tol=1e-10) and math.isclose(cost, trial["model_cost"], rel_tol=1e-10), "wrong search score")
            legal.append((index, gpu, cost, mapping))
            sources[shape, tuple(trial["block"]), trial["group_threads"], variant] = mapping["source"]
            outputs += 3
            elements += shape[0] * shape[1] * 3
        selected = min(legal, key=lambda item: item[1])
        model = min(legal, key=lambda item: item[2])
        require(tuning["selected_trial"] == selected[0] and tuning["model_selected_trial"] == model[0], "wrong selected candidate")
        regret = model[1] / selected[1] - 1
        require(math.isclose(regret, tuning["model_regret"], abs_tol=1e-10), "wrong model regret")
        final = realization(row, variant, path.parent)
        data = metrics(row, 5)
        require(final == selected[3], "fresh winner changed realization")
        outputs += 3
        elements += shape[0] * shape[1] * 3
        summaries.append(dict(shape=shape, variant=variant, valid_candidates=len(legal), selected=final,
                              model=model[3], observed_model_regret=regret, fresh_metrics=data,
                              selection_to_fresh_gpu_ratio=data["native"]["gpu_throughput"] / selected[1]))
    common = 0
    for (shape, block, threads, variant), sha in sources.items():
        if variant == "reference":
            require(sources[shape, block, threads, "candidate"] == sha, "common candidate source changed")
            common += 1
    return dict(passed=True, use="candidate admission and numerical validation; uncontrolled search timings are not accepted performance gains",
                attempted_candidates=attempts, valid_candidates=valid_count, complete_outputs=outputs, checked_elements=elements,
                unchanged_common_candidates=common, summary=summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=HERE / "replay")
    parser.add_argument("--search", action="store_true")
    args = parser.parse_args()
    report = json.loads((args.directory / "results.json").read_text())
    if args.search:
        result = audit_search(report, args.directory)
        result["input_sha256"] = digest(args.directory / "results.json")
        (args.directory / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "summary"}), flush=True)
        return
    result = audit_replay(report, args.directory)
    mutations = {
        "missing_row": lambda r: r["results"].pop(),
        "duplicate": lambda r: r["results"].__setitem__(1, r["results"][0]),
        "partial_output": lambda r: r["results"][0]["native"].update(output_elements=1),
        "wrong_divisor": lambda r: r["results"][0]["native"]["device_timing"]["control"].update(repetitions=1),
        "wrong_loader": lambda r: r["results"][0].update(loader="/wrong"),
        "state_count": lambda r: r["results"][0]["native"]["execution_plans"][0].update(fragment_scalars_per_lane=0),
        "overlap": lambda r: r["results"][0]["native"]["execution_plans"][0]["matrices"][0].update(subgroups_m=0),
        "retuned_plan": lambda r: r["results"][0].update(block=[8, 8, 16]),
        "unbalanced_order": lambda r: r["results"][0].update(implementation_order=["native"]),
    }
    for name, mutate in mutations.items():
        bad = copy.deepcopy(report)
        mutate(bad)
        try:
            audit_replay(bad, args.directory)
        except (ValueError, KeyError, ZeroDivisionError):
            continue
        raise ValueError("auditor accepted mutation: " + name)
    result["rejected_mutations"] = list(mutations)
    result["input_sha256"] = digest(args.directory / "results.json")
    (HERE / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "summary"}), flush=True)


if __name__ == "__main__":
    main()
