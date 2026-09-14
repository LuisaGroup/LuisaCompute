#!/usr/bin/env python3
"""Audit complete receipts, exact pair ordering and the bounded permutation."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
SHAPES = [(1024, 1024, 1537), (4096, 4096, 4096), (8192, 8192, 8192),
          (4096, 4096, 11008), (2049, 4097, 1025)]
WALKS = dict(legacy=(0, 1), linear=(1, 1), stripe4=(4, 1), stripe8=(8, 1),
             rectangle2x8=(2, 8), rectangle4x16=(4, 16), rectangle8x32=(8, 32))
METRICS = ("throughput_us", "latency_us", "gpu_throughput_us", "gpu_latency_us")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def audit(data, paths, correctness=False):
    require(data["artifacts_unchanged"] is True and data["artifacts_before"] == data["artifacts_after"],
            "changed executable")
    rows = data["rows"]
    require(len(rows) == (48 if correctness else 50), "row count mismatch")
    elements, table, keys, source_ids = 0, [], set(), {}
    expected_order = []
    if not correctness:
        for r in range(2):
            for shape in SHAPES if r == 0 else SHAPES[::-1]:
                offset = SHAPES.index(shape) % len(paths)
                order = paths[offset:] + paths[:offset]
                if r:
                    order = order[::-1]
                expected_order.extend((r, shape, path, order) for path in order)
    for i, row in enumerate(rows):
        shape, path = tuple(row["shape"]), row["path"]
        require(row["valid"] is True and row["returncode"] == 0, "invalid complete output")
        if not correctness:
            require((row["round"], shape, path, tuple(row["order"])) == expected_order[i], "pair order mismatch")
        identity = (tuple(row["config"] or ()), shape, path, None if correctness else row["round"])
        require(identity not in keys, "duplicate output")
        keys.add(identity)
        receipt, measurement = row["correctness"], row["measurement"]
        require(receipt["checked_elements"] == shape[0] * shape[1] and receipt["max_abs_error"] == 0 and
                receipt["atol"] == receipt["rtol"] == 1e-4, "numeric receipt mismatch")
        require(tuple(measurement[k] for k in ("m", "n", "k")) == shape and measurement["backend"] == "metal",
                "workload mismatch")
        if path != "mps":
            require(measurement["precision"] == "fp32" and measurement["relaxed_precision"] is False and
                    measurement["fast_math"] is False and measurement["static_reduction"] is False and
                    measurement["cooperative_output"] is True and measurement["inline_tensors"] is True,
                    "MPP precision/storage mismatch")
            require((measurement["walk_rows"], measurement.get("walk_columns", 1)) == WALKS[path], "wrong walk")
            require(measurement["static_threadgroup_bytes"] == 0 and measurement["thread_execution_width"] == 32,
                    "unexpected resource realization")
            sha = row["source_sha256"]
            file = ROOT / "sources" / (sha + ".metal")
            require(digest(file) == sha, "source fingerprint mismatch")
            source = file.read_text()
            for name, value in (("ROWS_M", shape[0]), ("COLUMNS_N", shape[1]), ("REDUCTION_K", shape[2]),
                                ("WALK_ROWS", WALKS[path][0])):
                require(f"#define {name} {value}\n" in source, "source configuration mismatch")
            key = identity[:-1]
            require(key not in source_ids or source_ids[key] == sha, "source changed between rounds")
            source_ids[key] = sha
        else:
            require(measurement["dtype"] == "float32" and measurement["alpha"] == 1 and
                    measurement["beta"] == 0 and measurement["transpose_left"] is False and
                    measurement["transpose_right"] is False and measurement["api_variant"] == "MPSKernelOptionsNone",
                    "MPS operation mismatch")
        medians = {}
        for metric in METRICS:
            values = measurement[metric]
            require(len(values) == (1 if correctness else 5) and
                    all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values), "invalid time")
            medians[metric] = statistics.median(values)
        elements += receipt["checked_elements"]
        table.append(dict(shape=list(shape), path=path, round=row.get("round"), medians=medians))
    ratios = []
    if not correctness:
        lookup = {(tuple(row["shape"]), row["round"], row["path"]): row["medians"] for row in table}
        for shape in SHAPES:
            for path in paths:
                if path == "mps":
                    continue
                item = dict(shape=list(shape), path=path)
                for base in ("linear", "mps"):
                    item["over_" + base] = {metric: [lookup[shape, r, path][metric] / lookup[shape, r, base][metric]
                                                         for r in range(2)] for metric in METRICS}
                ratios.append(item)
    return dict(outputs=len(rows), elements=elements, medians=table, ratios=ratios,
                independent_source_identities=len(source_ids))


def permutation_checks():
    cases = 0
    for rows in (1, 2, 3, 7, 8, 9, 17, 33):
        for columns in (1, 2, 3, 7, 16, 17, 33, 129):
            for height in (1, 2, 4, 8, 64, 2**31 - 1):
                if height * columns > 2**32 - 1:
                    continue
                for requested_width in (1, 2, 4, 8, 16, 32, 256, 2**31 - 1):
                    width = min(requested_width, columns)
                    mapped = set()
                    for index in range(rows * columns):
                        first_row = index // (height * columns) * height
                        actual_height = min(height, rows - first_row)
                        local = index % (height * columns)
                        first_column = local // (actual_height * width) * width
                        actual_width = min(width, columns - first_column)
                        inside = local % (actual_height * width)
                        row = first_row + inside // actual_width
                        column = first_column + inside % actual_width
                        require(0 <= row < rows and 0 <= column < columns and (row, column) not in mapped,
                                "permutation is not in-bounds/injective")
                        mapped.add((row, column))
                    require(len(mapped) == rows * columns, "permutation misses programs")
                    cases += 1
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--current-artifacts", action="store_true")
    parser.add_argument("--artifact-dir", type=Path, help="frozen pre-format standalone binaries")
    args = parser.parse_args()
    stripe_paths = ("legacy", "linear", "stripe4", "stripe8", "mps")
    rectangle_paths = ("linear", "rectangle2x8", "rectangle4x16", "rectangle8x32", "mps")
    reports = {name: json.loads((ROOT / name).read_text()) for name in (
        "results.json", "results-rectangles.json", "correctness.json",
        "correctness-unsigned.json", "correctness-rectangles.json")}
    result = {name: audit(data, rectangle_paths if "rectangles" in name else stripe_paths, name.startswith("correctness"))
              for name, data in reports.items()}
    if args.current_artifacts:
        require(reports["correctness-rectangles.json"]["artifacts_before"] ==
                reports["results-rectangles.json"]["artifacts_before"], "correctness/performance artifact mismatch")
        for path, sha in reports["results-rectangles.json"]["artifacts_before"].items():
            resolved = args.artifact_dir / Path(path).name if args.artifact_dir else Path(path)
            require(digest(resolved) == sha, f"executable changed: {resolved}")
        result["current_artifacts_verified"] = 2
        result["verified_artifact_directory"] = str(args.artifact_dir) if args.artifact_dir else None
    if args.self_test:
        for mode in range(5):
            bad = copy.deepcopy(reports["results.json"])
            if mode == 0:
                bad["rows"].pop()
            elif mode == 1:
                bad["rows"][0]["round"] = 1
            elif mode == 2:
                bad["rows"][0]["correctness"]["checked_elements"] -= 1
            elif mode == 3:
                bad["rows"][0]["measurement"]["gpu_throughput_us"][0] = float("nan")
            else:
                bad["rows"][0]["measurement"]["walk_rows"] = 8
            try:
                audit(bad, stripe_paths)
            except ValueError:
                continue
            raise ValueError("corruption accepted")
        result["negative_probes_rejected"] = 5
        result["bijective_rectangular_grids_checked"] = permutation_checks()
    result["validated_outputs"] = sum(result[name]["outputs"] for name in reports)
    result["validated_elements"] = sum(result[name]["elements"] for name in reports)
    (ROOT / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, dict)}))


if __name__ == "__main__":
    main()
