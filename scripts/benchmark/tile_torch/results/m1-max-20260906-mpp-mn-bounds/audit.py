#!/usr/bin/env python3
"""Independent receipt/source/timing audit; no benchmark-validator imports."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
SHAPES = [(129, 257, 61), (1025, 1025, 1024), (2049, 4097, 1025),
          (4097, 4097, 4096), (1024, 1024, 1024)]
ORDERS = ["old-forward", "new-forward", "new-reverse", "old-reverse"]
BLOCKS = [16, 1024, 4096]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def metric(values):
    require(len(values) == 5 and all(type(x) in (float, int) and math.isfinite(x) and x > 0 for x in values),
            "invalid timing samples")
    return statistics.median(values)


def audit_report(report, directory, name):
    reverse, old = name.endswith("reverse"), name.startswith("old")
    shapes = SHAPES[::-1] if reverse else SHAPES
    blocks = BLOCKS[::-1] if reverse else BLOCKS
    metadata = report["metadata"]
    require((metadata["samples"], metadata["sample_ms"], metadata["warmup_ms"]) == (5, 20, 100), "wrong sampling")
    require(metadata["matrix_realization"] == "mpp-views" and metadata["group_threads"] == 128 and
            metadata["pipeline_window"] == 1 and metadata["copy_batch"] == 1, "wrong candidate family")
    require([tuple(row["case"][k] for k in ("m", "n", "k")) for row in report["results"]] == shapes,
            "missing, duplicated or reordered shape")
    measurements, rejections, outputs, elements = [], [], 0, 0
    for ordinal, row in enumerate(report["results"]):
        shape = shapes[ordinal]
        ragged = shape != SHAPES[-1]
        require(row["valid"] is True, "invalid fresh output")
        tuning = row["tuning"]
        trials = tuning["trials"]
        shift = ordinal % 3
        require([t["block"][2] for t in trials] == blocks[shift:] + blocks[:shift], "wrong candidate order")
        accepted = []
        for index, trial in enumerate(trials):
            should_reject = old and ragged and trial["block"][2] != 16
            require(trial["valid"] is not should_reject, "unexpected admission result")
            if should_reject:
                require("native benchmark failed (2)" in trial["error"] and
                        "no legal Metal MPP group plan" in trial["error"], "wrong rejection cause")
                rejections.append(dict(shape=shape, block=trial["block"], error=trial["error"]))
            else:
                accepted.append((index, trial))
        winner = min(accepted, key=lambda item: item[1]["selection_score"])[0]
        require(tuning["selected_trial"] == winner and row["block"] == trials[winner]["block"], "selection mismatch")
        for trial, measured, phase in [(t, t["measurement"], "trial") for _, t in accepted] + [(None, row, "fresh")]:
            require(measured["valid"] is True, "invalid trial output")
            block = measured["block"]
            require(block[:2] == [128, 32] and block[2] in BLOCKS, "shape substitution")
            source = measured["native_source_sha256"]
            source_file = directory / "sources" / (source + ".metal")
            require(digest(source_file) == source, "source hash mismatch")
            code = source_file.read_text()
            require(("mpp_actual_m" in code) == (not old and ragged), "wrong M/N realization")
            plan, = measured["native"]["execution_plans"]
            expected_shared = (26624 if old else 16384) if ragged else 0
            require(plan["metal_mpp"] and plan["threads"] == 128 and
                    plan["shared_memory_bytes"] == expected_shared, "unexpected physical resources")
            require(plan["matrices"] == [dict(subgroups_m=4, subgroups_n=1, atom_rows=4, atom_columns=4,
                                               persistent_accumulator=True, direct_accumulator_store=not ragged)],
                    "matrix distribution/carry confound")
            item = dict(shape=shape, block=block, phase=phase, source_sha256=source, paths={})
            for path in ("native", "torch", "system"):
                data = measured[path]
                proof = data["correctness"]
                require(proof["atol"] == proof["rtol"] == 1e-4 and proof["max_abs_error"] == 0,
                        "full FP64 numeric receipt mismatch")
                if path == "native":
                    require(data["output_elements"] == shape[0] * shape[1] and data["mpp_intrinsics"] > 0 and
                            data["simdgroup_intrinsics"] == 0, "wrong output/realization")
                if path == "system":
                    require(data["dtype"] == "float32" and data["alpha"] == 1 and data["beta"] == 0 and
                            data["transpose_left"] is False and data["transpose_right"] is False, "wrong MPS operation")
                control = data["device_timing"]["control"]
                require(control["encoder_instrumentation"] is False and
                        control["method"] == "metal_command_buffer_timestamps_v1", "instrumented GPU ranking")
                gpu = {}
                for key, divisor in (("throughput", control["repetitions"]), ("latency", 1)):
                    values = [sample["command_buffer_ns"] / (1000 * divisor) for sample in control[key]]
                    saved = control["command_buffer_" + key + "_us"]
                    require(len(values) == len(saved) and all(math.isclose(a, b, rel_tol=1e-10) for a, b in zip(values, saved)),
                            "GPU timing conversion mismatch")
                    gpu[key] = metric(values)
                item["paths"][path] = dict(gpu_batch_us=gpu["throughput"], gpu_single_us=gpu["latency"],
                                            e2e_batch_us=metric(data["throughput_us"]), e2e_single_us=metric(data["latency_us"]))
                outputs += 1
                elements += shape[0] * shape[1]
            if trial is not None:
                require(math.isclose(trial["selection_score"], item["paths"]["native"]["gpu_batch_us"], rel_tol=1e-10),
                        "selection metric mismatch")
            measurements.append(item)
    return dict(measurements=measurements, rejections=rejections, outputs=outputs, elements=elements)


def audit_session(directory, current=False):
    execution = json.loads((directory / "execution.json").read_text())
    require(execution["before"] == execution["after"] and execution["artifacts_unchanged"] is True,
            "artifacts changed during measurements")
    require([r["name"] for r in execution["runs"]] == ORDERS and
            all(r["returncode"] == 0 for r in execution["runs"]), "incomplete ABBA replay")
    if current:
        for path, sha in execution["after"].items():
            require(digest(path) == sha, "current artifact changed: " + path)
    reports = {name: json.loads((directory / name / "results.json").read_text()) for name in ORDERS}
    results = {name: audit_report(report, directory / name, name) for name, report in reports.items()}
    identities = {}
    for name, result in results.items():
        for item in result["measurements"]:
            key = (name.split("-")[0], tuple(item["shape"]), item["block"][2])
            require(key not in identities or identities[key] == item["source_sha256"], "unstable generated source")
            identities[key] = item["source_sha256"]
    for block in BLOCKS:
        require(identities["old", SHAPES[-1], block] == identities["new", SHAPES[-1], block],
                "aligned source control changed")
    def lookup(name, shape, phase, block=None):
        matches = [item for item in results[name]["measurements"] if tuple(item["shape"]) == shape and
                   item["phase"] == phase and (block is None or item["block"][2] == block)]
        require(len(matches) == 1, "ambiguous comparison key")
        return matches[0]
    comparisons = []
    for shape in SHAPES:
        for order in ("forward", "reverse"):
            old, new = [lookup(version + "-" + order, shape, "fresh") for version in ("old", "new")]
            old_fixed, new_fixed = [lookup(version + "-" + order, shape, "trial", 16) for version in ("old", "new")]
            comparisons.append(dict(shape=shape, order=order, old_bk=old["block"][2], new_bk=new["block"][2],
                old=old["paths"], new=new["paths"],
                new_over_old={key: new["paths"]["native"][key] / old["paths"]["native"][key] for key in new["paths"]["native"]},
                same_bk16_new_over_old={key: new_fixed["paths"]["native"][key] / old_fixed["paths"]["native"][key] for key in new_fixed["paths"]["native"]},
                new_over_mps={key: new["paths"]["native"][key] / new["paths"]["system"][key] for key in new["paths"]["native"]},
                new_over_torch={key: new["paths"]["native"][key] / new["paths"]["torch"][key] for key in new["paths"]["native"]}))
    return dict(runs=results, comparisons=comparisons, artifacts=len(execution["after"]),
                validated_outputs=sum(r["outputs"] for r in results.values()),
                checked_elements=sum(r["elements"] for r in results.values()),
                rejected_requests=sum(len(r["rejections"]) for r in results.values()),
                current_artifacts_checked=current)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", nargs="+", default=["replay", "cooperative-replay", "shuffle-replay"])
    parser.add_argument("--current-artifacts", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    result = {name: audit_session(ROOT / name, args.current_artifacts and i == len(args.sessions) - 1)
              for i, name in enumerate(args.sessions)}
    if args.self_test:
        name = args.sessions[-1]
        good = json.loads((ROOT / name / "new-forward/results.json").read_text())
        for kind in range(5):
            bad = copy.deepcopy(good)
            if kind == 0:
                bad["results"].pop()
            elif kind == 1:
                bad["results"][0]["native"]["correctness"]["max_abs_error"] = 1
            elif kind == 2:
                bad["results"][0]["native"]["device_timing"]["control"]["throughput"][0]["command_buffer_ns"] = float("nan")
            elif kind == 3:
                bad["results"][0]["native"]["execution_plans"][0]["shared_memory_bytes"] += 4
            else:
                bad["results"][0]["tuning"]["trials"].reverse()
            try:
                audit_report(bad, ROOT / name / "new-forward", "new-forward")
            except ValueError:
                continue
            raise ValueError("corrupted evidence accepted")
        result["negative_probes_rejected"] = 5
    (ROOT / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: {key: value for key, value in data.items() if key not in ("runs", "comparisons")}
                      for name, data in result.items() if isinstance(data, dict)}, indent=2))


if __name__ == "__main__":
    main()
