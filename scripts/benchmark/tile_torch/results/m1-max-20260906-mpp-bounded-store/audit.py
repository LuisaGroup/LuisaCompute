#!/usr/bin/env python3
"""Independently check replay coverage, raw metrics, realization and controls."""
import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
SHAPES = [(129, 257, 61), (1025, 1025, 1024), (2049, 4097, 1025),
          (4097, 4097, 4096), (1024, 1024, 1024), (4096, 4096, 4096)]


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def median(values):
    require(len(values) > 0 and all(type(x) in (float, int) and math.isfinite(x) and x > 0 for x in values), "invalid timings")
    return statistics.median(values)


def distribution(values):
    return dict(median=median(values), minimum=min(values), maximum=max(values),
                faster_rounds=sum(x < 1 for x in values), ratios=values)


def audit(report, directory):
    meta = report["metadata"]
    require(report["passed"] is True and meta["phase"] == "replay", "incomplete or nonfinal report")
    require((meta["rounds"], meta["samples"], meta["sample_ms"], meta["warmup_ms"], meta["threads"]) == (6, 9, 30, 100, 8), "protocol mismatch")
    require(meta["artifacts_unchanged"] is True and meta["artifacts_before"] == meta["artifacts_after"], "changed artifacts")
    require(len(meta["builds"]) == 2 and all(x["exit_code"] == 0 for x in meta["builds"]), "full build gate missing")
    for variant, config in meta["native_variants"].items():
        first = config["loader"].split(":")[0]
        require(Path(first) == Path(config["binary"]).parent, "wrong library precedence")
        for name in ("benchmark_tile_tirx", "libluisa-tile.dylib", "libluisa-tile-bridge-tirx.dylib"):
            require(str(Path(first) / name) in meta["artifacts_before"], "unfingerprinted compiler variant")
    require(len(report["results"]) == 72, "missing or duplicated row")
    records, sources = {}, {}
    outputs, elements = 0, 0
    for row in report["results"]:
        case = row["case"]
        shape = tuple(case[k] for k in ("m", "n", "k"))
        key = shape, row["round"], row["variant"]
        require(shape in SHAPES and 0 <= row["round"] < 6 and row["variant"] in ("reference", "candidate") and key not in records, "invalid case identity")
        require(row["valid"] is True and row["backend"] == "metal" and case["operation"] == "gemm", "wrong operation or invalid output")
        require(row["block"] == [128, 32, 4096], "retuned geometry")
        require(row["loader"] == meta["native_variants"][row["variant"]]["loader"], "wrong loader")
        require(row["native_command"][0] == meta["native_variants"][row["variant"]]["binary"], "wrong executable")
        native = row["native"]
        require(native["planner_threads"] == 128 and native["copy_batch"] == 1 and native["pipeline_window"] == 1 and
                native["metal_mpp"] is True and native["forward_readonly_tile_loads"] is True and
                native["elide_independent_subgroup_barriers"] is False and native["simdgroup_intrinsics"] == 0 and
                native["mpp_intrinsics"] > 0 and native["output_elements"] == shape[0] * shape[1], "wrong realization/output")
        plan, = native["execution_plans"]
        ragged = shape in SHAPES[:4]
        direct = not ragged or row["variant"] == "candidate"
        require(plan["threads"] == 128 and plan["cost_basis"] == "metal_mpp_memory_v2" and plan["optimized"] is True and
                plan["shared_memory_bytes"] == (0 if direct else 16384), "resource or cost-family mismatch")
        require(plan["matrices"] == [dict(subgroups_m=4, subgroups_n=1, atom_rows=4, atom_columns=4,
                                           persistent_accumulator=True, direct_accumulator_store=direct)], "changed subgroup geometry")
        sha = row["native_source_sha256"]
        source_file = directory / "sources" / (sha + ".metal")
        require(source_file.is_file() and digest(source_file) == sha, "missing or changed generated source")
        code = source_file.read_text()
        require(("mpp_store_rows" in code) == (ragged and direct), "bounded store not realized")
        require(("threadgroup float" in code) == (not direct), "unexpected shared allocation")
        require(("mode::multiply_accumulate" in code) == (not direct), "wrong accumulator mode")
        identity = shape, row["variant"]
        require(identity not in sources or sources[identity] == sha, "unstable generated source")
        sources[identity] = sha
        data = {}
        for provider in ("native", "torch", "system"):
            path = row[provider]
            proof = path["correctness"]
            require(proof["atol"] == proof["rtol"] == 1e-4 and proof["max_abs_error"] == 0, "numeric receipt mismatch")
            if provider == "torch":
                require(path["output_policy"] == "preallocated_out", "unmatched output allocation")
            if provider == "system":
                require(path["implementation"] == "mps_matrix_multiplication" and path["dtype"] == "float32" and
                        path["alpha"] == 1 and path["beta"] == 0 and path["transpose_left"] is False and path["transpose_right"] is False,
                        "wrong MPS baseline")
            control = path["device_timing"]["control"]
            require(control["method"] == "metal_command_buffer_timestamps_v1" and control["encoder_instrumentation"] is False and
                    type(control["repetitions"]) is int and control["repetitions"] > 0, "invalid GPU control")
            metrics = {}
            for phase, divisor in (("throughput", control["repetitions"]), ("latency", 1)):
                raw = control[phase]
                require(len(raw) == 9 and all(s["command_buffers"] > 0 for s in raw), "GPU coverage mismatch")
                values = [sample["command_buffer_ns"] / (1000 * divisor) for sample in raw]
                saved = control["command_buffer_" + phase + "_us"]
                require(len(saved) == 9 and all(math.isclose(a, b, rel_tol=1e-10) for a, b in zip(values, saved)), "GPU denominator mismatch")
                value = median(values)
                require(math.isclose(value, control["command_buffer_" + phase + "_us_p50"], rel_tol=1e-10), "wrong GPU p50")
                host = path[phase + "_us"]
                require(len(host) == 9, "E2E coverage mismatch")
                host_value = median(host)
                require(math.isclose(host_value, path[phase + "_us_p50"], rel_tol=1e-10), "wrong E2E p50")
                metrics["gpu_" + phase] = value
                metrics["e2e_" + phase] = host_value
            data[provider] = metrics
            outputs += 1
            elements += shape[0] * shape[1]
        records[key] = dict(data=data, order=tuple(row["implementation_order"]))
    summary = []
    for shape in SHAPES:
        for variant in ("reference", "candidate"):
            orders = {records[shape, r, variant]["order"] for r in range(6)}
            require(orders == set(itertools.permutations(("native", "torch", "system"))), "unbalanced framework order")
        # Independently verify the rotated case order and alternating A/B.
        index = SHAPES.index(shape)
        for r in range(6):
            visits = [row["variant"] for row in report["results"] if row["round"] == r and tuple(row["case"][k] for k in ("m", "n", "k")) == shape]
            require(visits == (["reference", "candidate"] if (r + index) % 2 == 0 else ["candidate", "reference"]), "unbalanced variant order")
        if shape in SHAPES[4:]:
            require(sources[shape, "reference"] == sources[shape, "candidate"], "aligned source control changed")
        result = dict(shape=shape, aligned_control=shape in SHAPES[4:], metrics={})
        for metric in ("gpu_throughput", "e2e_throughput", "gpu_latency", "e2e_latency"):
            before = [records[shape, r, "reference"]["data"]["native"][metric] for r in range(6)]
            after = [records[shape, r, "candidate"]["data"]["native"][metric] for r in range(6)]
            torch = [records[shape, r, "candidate"]["data"]["torch"][metric] for r in range(6)]
            mps = [records[shape, r, "candidate"]["data"]["system"][metric] for r in range(6)]
            result["metrics"][metric] = dict(reference_us=median(before), candidate_us=median(after), torch_us=median(torch), mps_us=median(mps),
                                               new_old=distribution([b / a for a, b in zip(before, after)]),
                                               new_torch=distribution([b / t for b, t in zip(after, torch)]),
                                               new_mps=distribution([b / s for b, s in zip(after, mps)]))
        summary.append(result)
    return dict(passed=True, complete_outputs=outputs, checked_elements=elements, pairs=36, unique_sources=len(set(sources.values())), summary=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=HERE / "frozen-replay")
    parser.add_argument("--current-artifacts", action="store_true")
    args = parser.parse_args()
    report_path = args.directory / "results.json"
    report = json.loads(report_path.read_text())
    result = audit(report, args.directory)
    if args.current_artifacts:
        for path, sha in report["metadata"]["artifacts_after"].items():
            require(digest(path) == sha, "current artifact changed: " + path)
    adversarial = {}
    mutations = {
        "missing_row": lambda r: r["results"].pop(),
        "duplicate_row": lambda r: r["results"].__setitem__(1, r["results"][0]),
        "failed_row": lambda r: r["results"][0].update(valid=False),
        "partial_output": lambda r: r["results"][0]["native"].update(output_elements=1),
        "wrong_denominator": lambda r: r["results"][0]["native"]["device_timing"]["control"].update(repetitions=1),
        "changed_artifact": lambda r: r["metadata"].update(artifacts_unchanged=False),
        "missing_source": lambda r: r["results"][0].update(native_source_sha256="0" * 64),
        "wrong_loader": lambda r: r["results"][0].update(loader="/wrong/compiler"),
        "extra_staging": lambda r: r["results"][1]["native"]["execution_plans"][0].update(shared_memory_bytes=16384),
        "unbalanced_order": lambda r: r["results"][0].update(implementation_order=["native"]),
    }
    for name, mutate in mutations.items():
        bad = copy.deepcopy(report)
        mutate(bad)
        try:
            audit(bad, args.directory)
        except (ValueError, KeyError):
            adversarial[name] = "rejected"
        else:
            raise ValueError("audit missed adversarial mutation: " + name)
    result.update(adversarial_checks=adversarial, current_artifacts_checked=args.current_artifacts,
                  source_report_sha256=digest(report_path), audit_script_sha256=digest(__file__))
    (HERE / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
