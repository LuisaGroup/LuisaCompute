#!/usr/bin/env python3
"""Independent stdlib audit; does not import benchmark statistics/validators."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
SHAPES = {(1024, 1024, 1537), (4096, 4096, 4096), (4096, 4096, 11008)}
BLOCKS = {128, 512, 1024, 4096}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def metric(values, samples=5):
    require(len(values) == samples and all(type(x) in (int, float) and math.isfinite(x) and x > 0
                                         for x in values), "invalid timing samples")
    return statistics.median(values)


def audit_report(report, directory):
    metadata = report["metadata"]
    require(metadata["samples"] == 5 and metadata["sample_ms"] == 20 and metadata["warmup_ms"] == 100,
            "timing protocol mismatch")
    require(metadata["matrix_realization"] == "mpp-views" and metadata["group_threads"] == 128,
            "realization mismatch")
    rows = report["results"]
    require(len(rows) == 3 and {tuple(r["case"][k] for k in ("m", "n", "k")) for r in rows} == SHAPES,
            "missing or duplicate shape")
    results, outputs, elements = [], 0, 0
    for row in rows:
        shape = tuple(row["case"][k] for k in ("m", "n", "k"))
        trials = row["tuning"]["trials"]
        require(len(trials) == 4 and {t["block"][2] for t in trials} == BLOCKS, "candidate coverage mismatch")
        measurements = [(t, t["measurement"], "trial") for t in trials] + [(None, row, "fresh")]
        for trial, measurement, phase in measurements:
            require(measurement["valid"] is True and (trial is None or trial["valid"] is True), "invalid output")
            block = measurement["block"]
            require(block[:2] == [128, 32] and block[2] in BLOCKS, "block mismatch")
            source = measurement["native_source_sha256"]
            require(digest(directory / "sources" / (source + ".metal")) == source, "source fingerprint mismatch")
            plan, = measurement["native"]["execution_plans"]
            require(plan["shared_memory_bytes"] == 0 and plan["independent_subgroups"] is True and
                    plan["threads"] == 128 and plan["metal_mpp"] is True, "group plan confound")
            require(plan["matrices"] == [{"subgroups_m": 4, "subgroups_n": 1, "atom_rows": 4,
                                         "atom_columns": 4, "persistent_accumulator": True,
                                         "direct_accumulator_store": True}], "matrix plan confound")
            item = dict(shape=list(shape), block=block, phase=phase, source_sha256=source, paths={})
            for path in ("native", "torch", "system"):
                data = measurement[path]
                proof = data["correctness"]
                require(proof["atol"] == proof["rtol"] == 1e-4 and proof["max_abs_error"] == 0,
                        "numeric receipt mismatch")
                if path == "native":
                    require(data["output_elements"] == shape[0] * shape[1], "output count mismatch")
                control = data["device_timing"]["control"]
                require(control["encoder_instrumentation"] is False and
                        control["method"] == "metal_command_buffer_timestamps_v1", "instrumented GPU control")
                item["paths"][path] = dict(
                    gpu_batch_us=metric(control["command_buffer_throughput_us"]),
                    gpu_single_us=metric(control["command_buffer_latency_us"]),
                    e2e_batch_us=metric(data["throughput_us"]), e2e_single_us=metric(data["latency_us"]))
                outputs += 1
                elements += shape[0] * shape[1]
            results.append(item)
    return dict(outputs=outputs, elements=elements, measurements=results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-artifacts", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    source = {name: json.loads((ROOT / name / "results.json").read_text()) for name in ("forward", "reverse")}
    result = {name: audit_report(data, ROOT / name) for name, data in source.items()}
    identities = {}
    for order in result.values():
        for item in order["measurements"]:
            key = tuple(item["shape"]) + (item["block"][2],)
            require(key not in identities or identities[key] == item["source_sha256"], "unstable generated source")
            identities[key] = item["source_sha256"]
    execution = json.loads((ROOT / "execution.json").read_text())
    require(len(execution["runs"]) == 2 and all(r["returncode"] == 0 for r in execution["runs"]), "incomplete runs")
    require(execution["compiler_artifacts_unchanged"] is True and
            execution["compiler_artifacts_before"] == execution["compiler_artifacts_after"], "changed compiler")
    if args.current_artifacts:
        paths = dict(execution["compiler_artifacts_before"])
        for report in source.values():
            meta = report["metadata"]
            paths[meta["native_binary"]] = meta["native_sha256"]
            paths[meta["metal_device_timing"]["library"]] = meta["metal_device_timing"]["sha256"]
            paths[meta["system_baseline"]["binary"]] = meta["system_baseline"]["sha256"]
            paths.update({str(Path(meta["native_binary"]).parent / name): sha
                          for name, sha in meta["adjacent_tile_library_sha256"].items()})
        for path, sha in paths.items():
            require(digest(path) == sha, f"current artifact changed: {path}")
        result["current_artifacts_verified"] = len(paths)
    if args.self_test:
        def failed(data):
            try:
                audit_report(data, ROOT / "forward")
            except (ValueError, KeyError):
                return
            raise ValueError("corruption accepted")
        for mode in range(5):
            bad = copy.deepcopy(source["forward"])
            trial = bad["results"][0]["tuning"]["trials"][0]
            if mode == 0:
                bad["results"].pop()
            elif mode == 1:
                trial["measurement"]["valid"] = False
            elif mode == 2:
                trial["measurement"]["native"]["throughput_us"][0] = float("nan")
            elif mode == 3:
                trial["measurement"]["native"]["execution_plans"][0]["matrices"][0]["subgroups_n"] = 2
            else:
                trial["measurement"]["native"]["device_timing"]["control"]["encoder_instrumentation"] = True
            failed(bad)
        result["negative_probes_rejected"] = 5
    result["validated_outputs"] = sum(result[name]["outputs"] for name in source)
    result["validated_elements"] = sum(result[name]["elements"] for name in source)
    (ROOT / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, dict)}))


if __name__ == "__main__":
    main()
