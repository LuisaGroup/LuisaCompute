#!/usr/bin/env python3
"""Serial Windows actual-Luisa Tile / torch.compile workload matrix.

This script never builds, installs packages, or changes device clocks/power.
Use --list-cases for a host-only inventory. Every execution requires a new
output directory, an explicit affinity mask, and a completed-build marker.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fnmatch
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ROWS = {"rmsnorm", "layernorm", "softmax", "masked_softmax", "rope", "swiglu",
        "gelu_residual", "reduce_sum", "reduce_max", "argmax", "scan", "scan_ordered"}
OPERATIONS = ROWS | {"gemm", "gemv", "bmm", "embedding", "attention", "attention_tensorcore", "sort", "topk"}
HIDDEN = 0x08000000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now().astimezone().isoformat()


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def save(path, record):
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # Windows readers/antivirus may briefly deny atomic replacement. Retry
    # only that operation; preserve both the original and pending JSON on error.
    deadline = time.monotonic() + 2.0
    while True:
        try:
            temporary.replace(path)
            return
        except PermissionError:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise
            time.sleep(min(0.05, remaining))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON: {value}")))


def case(operation, dimensions, tile, pattern="random", suffix=""):
    name = operation + "-" + "x".join(map(str, dimensions)) + "-" + pattern
    return dict(id=name + suffix, operation=operation, dimensions=list(dimensions),
                tile=list(tile), precision="fp32", pattern=pattern)


def builtin_cases(suite):
    cases = [
        case("rmsnorm", (17, 65), (1, 128, 1)),
        case("layernorm", (4, 257), (1, 512, 1), "adversarial"),
        case("softmax", (16, 127), (1, 128, 1)),
        case("masked_softmax", (4, 65), (1, 128, 1), "adversarial"),
        case("rope", (17, 130), (1, 128, 1)),
        case("swiglu", (17, 129), (1, 256, 1)),
        case("gelu_residual", (4, 257), (1, 512, 1), "adversarial"),
        case("reduce_sum", (4, 257), (1, 512, 1), "cancellation"),
        case("reduce_max", (17, 129), (1, 256, 1), "adversarial"),
        case("scan", (4, 65), (1, 128, 1), "cancellation"),
        case("sort", (2, 33, 33), (1, 64, 1), "adversarial"),
        case("topk", (4, 65, 7), (1, 128, 1), "adversarial"),
        case("gemm", (31, 37, 19), (16, 16, 8)),
        case("gemm", (31, 37, 19), (16, 16, 8), "cancellation"),
        case("gemv", (37, 1, 129), (16, 1, 8)),
        case("attention", (1, 4, 2, 3, 17, 16, 16), (4, 8, 1)),
        case("attention", (1, 4, 1, 1, 65, 32, 32), (1, 16, 1)),
    ]
    if suite == "broad":
        for op in sorted(ROWS - {"scan", "scan_ordered"}):
            for rows in (1, 4, 16, 128, 1024):
                for width in (512, 1024, 2048, 4096, 8192):
                    logical = width // 2 if op == "rope" else width
                    cases.append(case(op, (rows, width), (1, logical, 1)))
        for rows in (1, 4, 16):
            for width in (32, 65, 128, 257, 512):
                cases.append(case("scan", (rows, width), (1, 1 << (width - 1).bit_length(), 1), "cancellation"))
                padded = 1 << (width - 1).bit_length()
                cases.append(case("sort", (rows, width, width), (1, padded, 1), "adversarial"))
                cases.append(case("topk", (rows, width, min(16, width)), (1, padded, 1), "adversarial"))
        for dims, tile in [((128, 128, 128), (16, 16, 8)), ((512, 512, 512), (32, 32, 1)),
                           ((127, 257, 65), (16, 16, 8)), ((4, 8192, 512), (4, 32, 16)),
                           ((128, 512, 257), (16, 16, 8))]:
            cases.append(case("gemm", dims, tile))
        for rows in (128, 1024, 8192):
            for depth in (512, 2048, 8192):
                if rows * depth <= 2**24:
                    cases.append(case("gemv", (rows, 1, depth), (16, 1, 8)))
        for queries, keys in ((1, 512), (1, 2048), (1, 8192), (16, 512), (128, 128), (128, 512)):
            cases.append(case("attention", (1, 4, 1, queries, keys, 64, 64), (min(queries, 8), 32, 1)))
    # The broad suite includes the smoke fixtures; retain an overlapping case once.
    return list({row["id"]: row for row in cases}.values())


def validate_case(row, default_seed):
    require(isinstance(row, dict) and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,159}", row.get("id", "")), "invalid case id")
    require(row.get("operation") in OPERATIONS, f"unknown operation in {row['id']}")
    op = row["operation"]
    require(type(row.get("fast_math", False)) is bool, "fast_math must be boolean")
    ranking = row.get("ranking_algorithm", "full_sort_prefix")
    require(ranking in {"full_sort_prefix", "packed_fp32", "repeated_extrema", "chunked_bitonic_c256", "chunked_bitonic_c512"}, "unknown ranking algorithm")
    require("ranking_algorithm" not in row or op in {"sort", "topk"}, "ranking_algorithm is only applicable to ranking")
    require(ranking != "repeated_extrema" or op == "topk", "repeated_extrema requires topk")
    require(not ranking.startswith("chunked_bitonic_") or op == "sort", "chunked bitonic requires sort")
    dims, tile = row.get("dimensions"), row.get("tile")
    require(isinstance(dims, list) and len(dims) == (2 if op in ROWS else 7 if op in {"attention", "attention_tensorcore"} else 4 if op == "bmm" else 3), "wrong dimension count")
    require(isinstance(tile, list) and len(tile) == 3 and
            all(type(v) is int and 0 < v <= 65536 for v in dims + tile), "invalid dimensions/tile")
    work_limit = 2**34 if op in {"gemm", "gemv", "bmm"} else 2**31 if op in {"attention", "attention_tensorcore"} else 2**28
    require(op == "embedding" or math.prod(dims) <= work_limit, "case exceeds native fixture work bound")
    require(row.get("precision", "fp32") in {"fp32", "fp16", "bf16"}, "unsupported precision")
    require(op != "attention_tensorcore" or row.get("precision", "fp32") in {"fp16", "bf16"}, "attention_tensorcore requires FP16/BF16")
    require(row.get("pattern", "random") in {"random", "cancellation", "adversarial"}, "invalid input pattern")
    seed = row.get("seed", default_seed)
    require(type(seed) is int and 0 <= seed < 2**64, "seed must be uint64")
    if op in ROWS:
        require(math.prod(dims) <= 2**24, "row input exceeds fixture allocation bound")
        blockable = op in {"scan", "reduce_sum", "reduce_max"}
        require((tile[0] == 1 or (blockable and tile[0] in {2, 4, 8})) and
                tile[2] == 1 and tile[1] <= 16384, "invalid row schedule")
        require(op != "rope" or dims[1] % 2 == 0, "RoPE width must be even")
        require(op in {"swiglu", "gelu_residual", "rope"} or tile[1] >= dims[1], "row tile does not cover the logical width")
        require(op != "scan_ordered" or tile[1] <= 1024, "ordered reference scan is bounded to tile width 1024")
    elif op == "embedding":
        vocabulary, width, tokens = dims
        require(max(vocabulary * width, tokens * width) <= 2**24, "embedding tensors exceed allocation bound")
        require(tile[0] in {1, 4, 8} and tile[2] == 1 and tile[1] <= 16384, "invalid embedding schedule")
        require(row.get("pattern", "random") in {"random", "adversarial"}, "embedding requires random/adversarial pattern")
    elif op == "bmm":
        b, m, n, k = dims
        require(max(b * m * k, b * k * n, b * m * n) <= 2**24, "BMM input/output exceeds fixture allocation bound")
        require(tile[0] <= 128 and tile[1] <= 128 and tile[2] <= 256 and
                all(v & (v - 1) == 0 for v in tile), "invalid BMM schedule")
        require((m + tile[0] - 1) // tile[0] <= 65535 and
                (n + tile[1] - 1) // tile[1] <= 65535, "BMM launch grid exceeds CUDA limits")
    elif op in {"sort", "topk"}:
        r, n, k = dims
        require(n <= 16384 and k <= n and r * n <= 2**24 and tile[0] == tile[2] == 1 and
                n <= tile[1] <= 16384 and tile[1] & (tile[1] - 1) == 0 and
                (op != "sort" or k == n), "ranking exceeds padded-network shape/allocation limits")
        if ranking.startswith("chunked_bitonic_"):
            chunk = 256 if ranking == "chunked_bitonic_c256" else 512
            require(tile[1] == 1 << (n - 1).bit_length() and tile[1] >= chunk,
                    "chunked sort requires exact bit_ceil(N) tile width >= selected chunk")
    elif op in {"attention", "attention_tensorcore"}:
        b, h, kh, q, k, d, dv = dims
        require(max(b * h * q * d, b * kh * k * d, b * kh * k * dv, b * h * q * dv) <= 2**24,
                "attention input/output exceeds fixture allocation bound")
        require(h % kh == 0 and k >= q and tile[0] <= 128 and tile[1] <= 256 and tile[2] == 1 and
                b * h * q * k * (d + dv) <= 2**28, "attention exceeds bounded host oracle or schedule")
    else:
        m, n, k = dims
        require(max(m * k, k * n, m * n) <= 2**24, "GEMM input/output exceeds fixture allocation bound")
        require(tile[0] <= 128 and tile[1] <= 128 and tile[2] <= (1024 if op == "gemv" else 256) and
                (op != "gemv" or dims[1] == 1), "invalid GEMM/GEMV schedule")
    return {**row, "seed": seed, "precision": row.get("precision", "fp32"), "pattern": row.get("pattern", "random"),
            **({"ranking_algorithm": ranking} if op in {"sort", "topk"} else {})}


def selected_cases(args):
    rows = builtin_cases(args.suite)
    if args.cases:
        candidate = Path(args.cases)
        if candidate.is_file():
            packet = read_json(candidate)
            require(packet.get("schema") == 1 and isinstance(packet.get("cases"), list), "--cases JSON requires schema=1 and cases list")
            rows = packet["cases"]
        else:
            names = args.cases.split(",")
            by_name = {row["id"]: row for row in rows}
            require(len(names) == len(set(names)) and all(name in by_name for name in names), "--cases contains duplicate/unknown ids")
            rows = [by_name[name] for name in names]
    rows = [validate_case(row, args.seed) for row in rows]
    if args.precisions:
        precisions = args.precisions.split(",")
        require(len(set(precisions)) == len(precisions) and set(precisions) <= {"fp32", "fp16", "bf16"}, "invalid/duplicate precisions")
        rows = [{**row, "precision": precision, "id": row["id"] + ("" if precision == "fp32" else "-" + precision)}
                for row in rows for precision in precisions]
    require(len({row["id"] for row in rows}) == len(rows), "duplicate case ids")
    if args.case_filter:
        rows = [row for row in rows if any(fnmatch.fnmatchcase(row["id"], pattern) for pattern in args.case_filter)]
    require(rows, "case selection is empty")
    return rows


def tensor_receipts(manifest_path, expected_case):
    manifest = read_json(manifest_path)
    for key in ("operation", "dimensions", "tile", "precision", "seed", "pattern"):
        require(manifest.get(key) == expected_case[key], f"manifest {key} disagrees with requested case")
    require(manifest.get("fast_math", False) is expected_case.get("fast_math", False), "manifest fast_math mismatch")
    require(manifest.get("schema") == 1 and manifest.get("endianness") == "little", "unsupported manifest ABI")
    if expected_case["operation"] in {"sort", "topk"}:
        ranking = expected_case.get("ranking_algorithm", "full_sort_prefix")
        require(manifest.get("ranking_algorithm", "full_sort_prefix") == ranking, "manifest ranking algorithm mismatch")
        algorithm = ("stable_packed_fp32_full_sort_prefix" if ranking == "packed_fp32" else
                     "stable_chunked_bitonic_whole_tile_merge" if ranking.startswith("chunked_bitonic_") else
                     "stable_repeated_extrema" if ranking == "repeated_extrema" else
                     "padded_bitonic_full_sort" if expected_case["operation"] == "sort" else "padded_bitonic_full_sort_prefix")
        require(manifest.get("algorithm") == algorithm, "manifest realized ranking algorithm mismatch")
    if expected_case["operation"] in {"swiglu", "gelu_residual", "rope"}:
        from cuda_torch_baseline import pointwise_schedule_algorithm
        algorithm = pointwise_schedule_algorithm(expected_case["operation"], expected_case["dimensions"], expected_case["tile"], manifest.get("backend"))
        require(manifest.get("algorithm") == algorithm, "pointwise realized algorithm mismatch")
    if expected_case["operation"] == "argmax":
        require(manifest.get("algorithm") == "stable_first_index_argmax", "argmax algorithm mismatch")
        require(manifest.get("indices", {}).get("storage_dtype") == "int64", "argmax requires int64 indices")
    sizes = {"float32": 4, "float16": 2, "bfloat16": 2, "int64": 8}
    storage = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}[expected_case["precision"]]
    if expected_case["operation"] == "embedding":
        vocabulary, width, tokens = expected_case["dimensions"]
        algorithm = "uniform_int64_row_gather" if expected_case["tile"][0] == 1 else "serial_grouped_uniform_int64_row_gather"
        require(manifest.get("algorithm") == algorithm, "embedding algorithm mismatch")
        require([entry.get("name") for entry in manifest["inputs"]] == ["input0", "input1"] and
                [entry.get("storage_dtype") for entry in manifest["inputs"]] == [storage, "int64"] and
                [entry.get("shape") for entry in manifest["inputs"]] == [[vocabulary, width], [tokens]] and
                manifest["output"].get("shape") == [tokens, width], "embedding mixed-input ABI mismatch")
    else:
        require(all(entry["storage_dtype"] == storage for entry in manifest["inputs"]), "input storage disagrees with requested precision")
    require(manifest["output"]["storage_dtype"] == storage, "output storage disagrees with requested precision")
    entries = [(entry["path"], entry["shape"], sizes[entry["storage_dtype"]]) for entry in manifest["inputs"]]
    output_shape = manifest["output"]["shape"]
    entries += [(manifest["expected"]["path"], output_shape, 8), (manifest["expected"]["bound_path"], output_shape, 8)]
    for key in ("strict_bound_path", "probability_rounding_bound_path"):
        if key in manifest["expected"]:
            entries.append((manifest["expected"][key], output_shape, 8))
    if "indices" in manifest:
        entries.append((manifest["indices"]["expected_path"], manifest["indices"]["shape"], 8))
    hashes = {}
    root = manifest_path.parent.resolve()
    for name, shape, element_bytes in entries:
        file = (root / name).resolve(strict=True)
        require(file.is_relative_to(root), "manifest tensor is outside its packet")
        require(file.stat().st_size == math.prod(shape) * element_bytes, f"wrong tensor byte count: {file}")
        hashes[name] = digest(file)
    return manifest, hashes


def alignment_receipts(result, requested):
    stages = result.get("pipeline_stages", [{"realization": result.get("realization", "")}])
    receipts = result.get("native_alignment")
    if receipts is None and not requested:
        # Preserve read-only validation of pre-specialization historical runs.
        require(all("aligned16-requested" not in stage.get("realization", "") for stage in stages), "missing alignment receipts")
        return []
    require(isinstance(receipts, list) and len(receipts) == len(stages), "missing per-stage alignment receipts")
    for index, (receipt, stage) in enumerate(zip(receipts, stages)):
        require(receipt.get("stage") == index and receipt.get("requested") is requested, "alignment request/stage mismatch")
        mask, residues = receipt.get("eligible_buffer_mask"), receipt.get("final_argument_mod16")
        require(type(mask) is int and mask >= 0 and isinstance(residues, list) and len(residues) <= 31 and
                all(type(value) is int and 0 <= value < 16 for value in residues) and mask >> len(residues) == 0,
                "invalid alignment pointer/mask receipt")
        realization = stage.get("realization", "")
        require(("aligned16-requested" in realization) is requested, "alignment request/realization mismatch")
        if requested:
            require(f"aligned16-buffer-mask={mask};" in realization and
                    (("host-selected-dual-entry-aligned16-v1" in realization) == (mask != 0)) and
                    (("aligned16-ineligible" in realization) == (mask == 0)), "alignment eligibility mismatch")
        else:
            require(mask == 0, "default run contains alignment specialization")
        aligned = bool(mask) and all(value == 0 for slot, value in enumerate(residues) if mask & (1 << slot))
        streaming = result.get("native_streaming", [])
        selected_streaming = len(streaming) == len(stages) and streaming[index].get("available") and streaming[index].get("static_ranges_disjoint")
        partition = result.get("native_program_partition", [])
        selected_partition = len(partition) == len(stages) and partition[index].get("available") and partition[index].get("static_ranges_disjoint")
        cub = result.get("native_cub_scan", [])
        selected_cub = (len(cub) == len(stages) and cub[index].get("available") and
                        cub[index].get("static_ranges_disjoint") and cub[index].get("final_pointers_aligned16"))
        require(receipt.get("expected_selected_entry") == ("luisa_tile_cub_scan" if selected_cub else "luisa_tile_partition" if selected_partition else "luisa_tile_stream_scan" if selected_streaming else "luisa_tile_aligned16" if aligned else "luisa_tile_main"),
                "alignment selected entry mismatch")
    return receipts


def worker_warps_receipts(result, requested):
    require(type(requested) is int and requested in (0, 4, 8), "invalid native worker-warps request")
    stages = result.get("pipeline_stages", [{"realization": result.get("realization", "")}])
    require(isinstance(stages, list) and stages, "missing worker-warps stage realizations")
    receipts = []
    for index, stage in enumerate(stages):
        realization = stage.get("realization", "")
        require(isinstance(realization, str), "invalid worker-warps realization")
        markers = re.findall(r"(?:^|[;\s])worker-warps-hint=([0-9]+)(?=$|[;\s])", realization)
        require(markers == ([str(requested)] if requested else []) and
                realization.count("worker-warps-hint") == (1 if requested else 0),
                "worker-warps request/realization mismatch")
        receipts.append(dict(stage=index, requested=requested, reported_hint=int(markers[0]) if markers else None))
    return receipts


def validate_native_collective_cost(requested, aligned16=False, worker_warps=0,
                                    scan_chunk=0, independent_axis=0, streaming_scan=0):
    require(type(requested) is bool, "native collective-cost request must be boolean")
    require(not requested or not any((aligned16, worker_warps, scan_chunk, independent_axis, streaming_scan)),
            "native collective-cost is mutually exclusive with aligned16, worker-warps and structural/streaming experiments")


def collective_cost_receipts(result, requested=False):
    require(type(requested) is bool, "native collective-cost request must be boolean")
    stages = result.get("pipeline_stages", [{"realization": result.get("realization", "")}])
    require(isinstance(stages, list) and stages, "missing collective-cost stage realizations")
    receipts = []
    required = {"profile", "workers", "status", "reason"}
    for index, stage in enumerate(stages):
        require(isinstance(stage, dict) and isinstance(stage.get("realization", ""), str),
                "invalid collective-cost realization")
        realization = stage.get("realization", "")
        entries = re.findall(r"(?:^|[;\s])collective-cost-([a-z-]+)=([^;\s]+)(?=$|[;\s])", realization)
        require(len(entries) == realization.count("collective-cost-"), "malformed collective-cost marker")
        if not requested:
            require(not entries, "unrequested collective-cost realization")
            continue
        markers = dict(entries)
        require(len(markers) == len(entries) and required <= markers.keys() <= required | {"log-score"},
                "missing, duplicate or unknown collective-cost marker")
        require(markers["profile"] == "sm89-24-cuda134-v3", "unexpected collective-cost profile")
        require(markers["workers"] in ("0", "8") and markers["status"] in ("selected", "default", "ineligible"),
                "invalid collective-cost workers/status")
        workers, status, reason = int(markers["workers"]), markers["status"], markers["reason"]
        require((status == "selected") == (workers == 8), "collective-cost selection/worker mismatch")
        require(re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", reason) is not None, "invalid collective-cost reason")
        score = None
        if "log-score" in markers:
            raw = markers["log-score"]
            require(re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", raw) is not None,
                    "invalid collective-cost log-score")
            score = float(raw)
            require(math.isfinite(score), "nonfinite collective-cost log-score")
        require(status != "ineligible" or score is None, "ineligible collective-cost stage has a model score")
        require(not (status == "selected" or reason == "predicted-default") or score is not None,
                "model collective-cost decision is missing its score")
        require(reason != "prefix" or (status == "default" and score is None), "invalid collective-cost prefix fallback")
        if score is not None:
            selected = score < -0.05129329438755058  # Frozen v3 profile: strict log(0.95) threshold.
            require(status == ("selected" if selected else "default") and
                    reason == ("predicted-saving" if selected else "predicted-default"),
                    "collective-cost score/decision/reason mismatch")
        else:
            require(reason not in ("predicted-saving", "predicted-default"), "collective-cost prediction is missing its score")
        receipts.append(dict(stage=index, requested=True, profile=markers["profile"], workers=workers,
                             status=status, reason=reason, log_score=score, nondefault_hint_selected=workers == 8))
    return receipts


def collective_cost_worker_receipts(result, cost_receipts):
    stages = result.get("pipeline_stages", [{"realization": result.get("realization", "")}])
    require(len(stages) == len(cost_receipts), "collective-cost/worker stage count mismatch")
    receipts = []
    for index, (stage, decision) in enumerate(zip(stages, cost_receipts)):
        receipt = worker_warps_receipts(dict(realization=stage.get("realization", "")), decision["workers"])[0]
        receipt.update(stage=index, request_source="collective_cost_profile")
        receipts.append(receipt)
    return receipts



def validate_native_structure(scan_chunk, independent_axis, streaming_scan=0):
    require(type(scan_chunk) is int and scan_chunk in (0, 1024, 2048), "invalid native scan-chunk request")
    require(type(independent_axis) is int and independent_axis in (0, 1, 2, 4), "invalid native independent-axis request")
    require(type(streaming_scan) is int and streaming_scan in (0, 1024, 2048), "invalid native streaming-scan request")
    require(sum(bool(value) for value in (scan_chunk, independent_axis, streaming_scan)) <= 1,
            "native scan-chunk, independent-axis and streaming-scan are mutually exclusive")


def structural_receipts(result, scan_chunk=0, independent_axis=0):
    validate_native_structure(scan_chunk, independent_axis)
    stages = result.get("pipeline_stages", [{"realization": result.get("realization", "")}])
    require(isinstance(stages, list) and stages, "missing native structural stage realizations")
    receipts = []
    for index, stage in enumerate(stages):
        require(isinstance(stage, dict) and isinstance(stage.get("realization", ""), str), "invalid native structural realization")
        realization = stage.get("realization", "")
        markers = {}
        for name, requested, is_count in (("scan-chunk", scan_chunk, False), ("chunked-scans", scan_chunk, True),
                                          ("independent-axis-extent", independent_axis, False),
                                          ("partitioned-collectives", independent_axis, True)):
            values = re.findall(r"(?:^|[;\s])" + re.escape(name) + r"=([0-9]+)(?=$|[;\s])", realization)
            tokens = re.findall(r"(?:^|[;\s])" + re.escape(name) + r"(?=$|[=;\s])", realization)
            require(len(values) == (1 if requested else 0) and len(tokens) == (1 if requested else 0),
                    "native structural request/realization mismatch: " + name)
            value = int(values[0]) if values else None
            require(not requested or (value > 0 if is_count else values == [str(requested)]),
                    "native structural request/realization mismatch: " + name)
            markers[name] = value
        receipts.append(dict(stage=index, scan_chunk_requested=scan_chunk, independent_axis_requested=independent_axis,
                             scan_chunk=markers["scan-chunk"], chunked_scans=markers["chunked-scans"] or 0,
                             independent_axis_extent=markers["independent-axis-extent"],
                             partitioned_collectives=markers["partitioned-collectives"] or 0))
    return receipts


def streaming_scan_receipts(result, requested=0):
    validate_native_structure(0, 0, requested)
    stages = result.get("pipeline_stages", [{"realization": result.get("realization", "")}])
    receipts = result.get("native_streaming")
    if receipts is None and not requested:
        require(all("streaming-scan" not in stage.get("realization", "") for stage in stages), "missing streaming receipts")
        return []
    require(isinstance(receipts, list) and len(receipts) == len(stages), "missing per-stage streaming receipts")
    for index, (receipt, stage) in enumerate(zip(receipts, stages)):
        realization = stage.get("realization", "")
        require(receipt.get("stage") == index and receipt.get("chunk_requested") == requested, "streaming request/stage mismatch")
        markers = re.findall(r"(?:^|[;\s])streaming-scan-chunk=([0-9]+)(?=$|[;\s])", realization)
        require(markers == ([str(requested)] if requested else []), "streaming chunk metadata mismatch")
        available, disjoint = receipt.get("available"), receipt.get("static_ranges_disjoint")
        require(type(available) is bool and type(disjoint) is bool and
                available == ("; streaming-scan-available;" in realization), "invalid streaming availability")
        require(not disjoint or available, "streaming disjoint receipt without available candidate")
        if available:
            require(requested != 0, "default run contains streaming specialization")
            for name in ("input_slot", "output_slot", "input_bytes", "output_bytes"):
                value = receipt.get(name)
                token = "streaming-scan-" + name.replace('_', '-')
                values = re.findall(r"(?:^|[;\s])" + re.escape(token) + r"=([0-9]+)(?=$|[;\s])", realization)
                require(type(value) is int and value >= (1 if name.endswith("bytes") else 0) and values == [str(value)],
                        "streaming static view metadata mismatch")
            require(receipt["input_slot"] != receipt["output_slot"], "streaming slots alias statically")
        else:
            require(all(receipt.get(name) == 0 for name in ("input_slot", "output_slot", "input_bytes", "output_bytes")),
                    "unavailable streaming candidate has view metadata")
        selected = available and disjoint
        require((receipt.get("expected_selected_entry") == "luisa_tile_stream_scan") == selected,
                "streaming expected host selection mismatch")
        if requested:
            # Calibration must never time the generic fallback and label it a
            # streamed candidate; runtime fallback remains legal outside this run.
            require(selected, "streaming calibration did not select the available disjoint candidate")
    return receipts


def validate_native_program_rows(rows, aligned16=False, worker_warps=0, scan_chunk=0,
                                 independent_axis=0, streaming_scan=0, collective_cost=False):
    require(type(rows) is int and rows in (0, 1, 2, 4), "invalid native program-rows request")
    require(not rows or not any((aligned16, worker_warps, scan_chunk, independent_axis, streaming_scan, collective_cost)),
            "native program-rows is mutually exclusive with other schedule experiments")


def validate_native_partition_cost(requested, aligned16=False, worker_warps=0, scan_chunk=0,
                                   independent_axis=0, streaming_scan=0, collective_cost=False, program_rows=0):
    require(type(requested) is bool, "invalid native partition-cost request")
    require(not requested or not any((aligned16, worker_warps, scan_chunk, independent_axis,
                                      streaming_scan, collective_cost, program_rows)),
            "native partition-cost is mutually exclusive with other schedule experiments")


def partition_cost_receipts(result, requested=False):
    validate_native_partition_cost(requested)
    stages = result.get("pipeline_stages", [{"realization": result.get("realization", "")}])
    receipts = []
    required = {"requested", "profile", "fit", "status", "reason", "original-rows", "selected-rows"}
    for index, stage in enumerate(stages):
        realization = stage.get("realization", "")
        entries = re.findall(r"(?:^|[;\s])partition-cost-([a-z-]+)=([^;\s]+)(?=$|[;\s])", realization)
        require(len(entries) == realization.count("partition-cost-"), "malformed partition-cost marker")
        if not requested:
            require(not entries, "unrequested partition-cost realization")
            continue
        fields = dict(entries)
        require(len(fields) == len(entries) and required <= fields.keys() <= required | {"original-score", "selected-score"},
                "missing, duplicate or unknown partition-cost marker")
        require(fields["requested"] == "1" and fields["profile"] == "sm89-24-cuda134-partition-linear-v1" and
                fields["fit"] == "63e0677c8554b46b707aa9f1fca3f29f223c653b1ec35fd57b5534b79595b6c2",
                "unexpected partition-cost request/profile/fit")
        status, reason = fields["status"], fields["reason"]
        require(status in ("selected", "retained", "ineligible") and
                re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", reason), "invalid partition-cost status/reason")
        require(all(re.fullmatch(r"[0-9]+", fields[k]) for k in ("original-rows", "selected-rows")),
                "invalid partition-cost row geometry")
        original, selected = int(fields["original-rows"]), int(fields["selected-rows"])
        scores = {}
        for name in ("original-score", "selected-score"):
            if name in fields:
                require(re.fullmatch(r"[+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", fields[name]),
                        "invalid partition-cost score")
                scores[name] = float(fields[name])
                require(math.isfinite(scores[name]) and scores[name] > 0, "nonpositive/nonfinite partition-cost score")
        require(len(scores) in (0, 2), "partial partition-cost score pair")
        if status == "ineligible":
            require(not scores and original == selected and original in (0, 4, 8), "ineligible partition-cost has a scored candidate")
        else:
            require(len(scores) == 2 and original in (4, 8), "partition-cost decision lacks scores or original geometry")
            if status == "selected":
                require(selected in (1, 2) and selected < original and original % selected == 0 and
                        scores["selected-score"] < .95 * scores["original-score"] and reason == "predicted-saving",
                        "partition-cost selected candidate disagrees with frozen threshold")
            else:
                require(selected == original and scores["selected-score"] == scores["original-score"] and
                        reason in ("predicted-original", "candidate-unavailable"), "partition-cost retained decision mismatch")
        receipts.append(dict(stage=index, requested=True, profile=fields["profile"], fit=fields["fit"], status=status,
                             reason=reason, original_rows=original, selected_rows=selected,
                             original_score=scores.get("original-score"), selected_score=scores.get("selected-score")))
    return receipts


def program_partition_receipts(result, requested=0, cost_decisions=None):
    validate_native_program_rows(requested)
    require(not cost_decisions or requested == 0, "partition-cost conflicts with explicit program rows")
    stages = result.get("pipeline_stages", [{"realization": result.get("realization", "")}])
    receipts = result.get("native_program_partition")
    if receipts is None and not requested and not cost_decisions:
        require(all("program-partition-" not in stage.get("realization", "") for stage in stages), "missing program partition receipts")
        return []
    require(isinstance(receipts, list) and len(receipts) == len(stages), "missing per-stage program partition receipts")
    require(cost_decisions is None or len(cost_decisions) == len(stages), "partition-cost/launch stage count mismatch")
    for index, (receipt, stage) in enumerate(zip(receipts, stages)):
        effective_rows = requested
        if cost_decisions is not None:
            decision = cost_decisions[index]
            require(decision["stage"] == index, "partition-cost launch stage mismatch")
            effective_rows = decision["selected_rows"] if decision["status"] == "selected" else 0
        realization = stage.get("realization", "")
        require(receipt.get("stage") == index and type(receipt.get("rows_requested")) is int and
                receipt["rows_requested"] == effective_rows, "program partition request/stage mismatch")
        def values(name):
            return re.findall(r"(?:^|[;\s])program-partition-" + name + r"=([0-9]+)(?=$|[;\s])", realization)
        require(values("rows") == ([str(effective_rows)] if effective_rows else []), "program partition rows metadata mismatch")
        available, disjoint = receipt.get("available"), receipt.get("static_ranges_disjoint")
        require(type(available) is bool and type(disjoint) is bool and
                available == ("; program-partition-available;" in realization), "invalid program partition availability")
        require(not disjoint or available, "program partition disjoint receipt without candidate")
        fields = ("input_slot", "output_slot", "input_bytes", "output_bytes", "original_rows", "grid_x", "original_grid_x")
        for field in fields:
            value = receipt.get(field)
            if available:
                require(type(value) is int and value >= (0 if field.endswith("slot") else 1) and
                        values(field.replace("_", "-")) == [str(value)], "program partition static metadata mismatch")
            else:
                require(type(value) is int and value == 0 and not values(field.replace("_", "-")), "unavailable partition contains plan metadata")
        if available:
            require(effective_rows > 0 and receipt["input_slot"] != receipt["output_slot"] and
                    receipt["original_rows"] > effective_rows and receipt["original_rows"] % effective_rows == 0 and
                    receipt["grid_x"] >= receipt["original_grid_x"], "invalid program partition geometry")
            if cost_decisions is not None:
                require(receipt["original_rows"] == decision["original_rows"], "partition-cost original geometry mismatch")
        original_grid, selected_grid = receipt.get("original_grid"), receipt.get("expected_selected_grid")
        require(isinstance(original_grid, list) and len(original_grid) == 3 and
                all(type(x) is int and x > 0 for x in original_grid), "missing original partition launch grid")
        require(isinstance(selected_grid, list) and len(selected_grid) == 3 and
                all(type(x) is int and x > 0 for x in selected_grid), "missing selected partition launch grid")
        selected = available and disjoint
        if available:
            require(original_grid == [receipt["original_grid_x"], 1, 1], "program partition original grid mismatch")
        require(selected_grid == ([receipt["grid_x"], 1, 1] if selected else original_grid),
                "program partition selected grid mismatch")
        require((receipt.get("expected_selected_entry") == "luisa_tile_partition") == selected,
                "program partition expected entry mismatch")
        if effective_rows:
            require(selected, "program partition calibration did not select an available disjoint candidate")
    return receipts


def validate_native_cub_scan(threads, *other_experiments):
    require(type(threads) is int and threads in (0, 128, 256, 512, 1024), "invalid native CUB scan thread count")
    require(not threads or not any(other_experiments), "native CUB scan is mutually exclusive with other schedule experiments")


def cub_scan_receipts(result, requested=0, cost_records=None):
    validate_native_cub_scan(requested)
    stages = result.get("pipeline_stages", [{"realization": result.get("realization", "")}])
    require(isinstance(stages, list) and stages, "missing CUB scan stage realizations")
    records = result.get("native_cub_scan")
    if records is None and not requested and cost_records is None:
        require(all("cub-scan" not in stage.get("realization", "") for stage in stages), "missing CUB scan receipts")
        return []
    require(isinstance(records, list) and len(records) == len(stages), "missing per-stage CUB scan receipts")
    checked = []
    if cost_records is not None:
        require(requested == 0 and len(cost_records) == len(stages), "CUB cost final stage count mismatch")
    for index, (record, stage) in enumerate(zip(records, stages)):
        requested = cost_records[index]["selected_threads"] if cost_records is not None else requested
        explicit = bool(requested) or cost_records is not None
        text = stage.get("realization", "")
        require(isinstance(text, str) and record.get("stage") == index and
                type(record.get("threads_requested")) is int and record["threads_requested"] == requested,
                "CUB scan request/stage mismatch")
        def numeric(name, expected):
            marker = "cub-scan-" + name
            values = re.findall(r"(?:^|[;\s])" + re.escape(marker) + r"=([0-9]+)(?=$|[;\s])", text)
            require(values == ([str(expected)] if expected is not None else []) and
                    text.count(marker) == (expected is not None), "CUB scan metadata mismatch: " + name)
        numeric("threads", requested if explicit else None)
        numeric("chunk", requested * 8 if explicit else None)
        compile_keys = re.findall(r"(?:^|[;\s])cub-scan-compile-key=([0-9a-f]{16})(?=$|[;\s])", text)
        require(len(compile_keys) == explicit and text.count("cub-scan-compile-key") == explicit,
                "CUB scan compile-key metadata mismatch")
        available = record.get("available")
        disjoint, aligned = record.get("static_ranges_disjoint"), record.get("final_pointers_aligned16")
        require(all(type(value) is bool for value in (available, disjoint, aligned)), "invalid CUB scan guard booleans")
        require(text.count("cub-scan-requested") == explicit and
                text.count("cub-scan-available;") == available and
                text.count("cub-scan-unavailable;") == (explicit and not available),
                "CUB scan availability marker mismatch")
        require(not available or requested != 0, "unrequested CUB scan candidate")
        require(available or not (disjoint or aligned), "unavailable CUB scan has true runtime guards")
        fields = ("input_slot", "output_slot", "input_bytes", "output_bytes")
        if available:
            require(result.get("operation") == "scan" and result.get("precision") in ("fp16", "bf16") and
                    result.get("fast_math", False) is False, "CUB scan fixture semantic mismatch")
            rows, columns = result["dimensions"]
            require(result["tile"] == [1, columns, 1] and columns % (requested * 8) == 0,
                    "CUB scan shape/recipe mismatch")
            expected = dict(input_slot=0, output_slot=3, input_bytes=rows * columns * 2, output_bytes=rows * columns * 2)
            for name in fields:
                require(type(record.get(name)) is int and record[name] == expected[name], "CUB scan static range/ABI mismatch")
                numeric(name.replace("_", "-"), expected[name])
            numeric("grid-x", rows)
            numeric("block-x", requested)
            numeric("alignment-mask", 9)
            alignment = result.get("native_alignment", [])
            require(len(alignment) == len(stages), "CUB scan final-pointer evidence is absent")
            residues = alignment[index].get("final_argument_mod16", [])
            require(len(residues) == 4 and all(type(value) is int and 0 <= value < 16 for value in residues),
                    "CUB scan final-pointer residue count mismatch")
            require(aligned == (residues[0] == residues[3] == 0), "CUB scan final-pointer alignment mismatch")
        else:
            require(all(type(record.get(name)) is int and record[name] == 0 for name in fields),
                    "unavailable CUB scan has static ranges")
            for name in (*fields, "grid_x", "block_x", "alignment_mask"):
                numeric(name.replace("_", "-"), None)
        selected = available and disjoint and aligned
        if explicit:
            require(record.get("expected_selected_entry") == ("luisa_tile_cub_scan" if selected else "luisa_tile_main"),
                    "CUB scan selected entry mismatch")
            require(record.get("expected_selected_block") == [requested if selected else 1, 1, 1],
                    "CUB scan selected block mismatch")
        else:
            require(record.get("expected_selected_entry") != "luisa_tile_cub_scan", "default run selects CUB scan")
        if available:
            require(record.get("expected_selected_grid") == [result["dimensions"][0], 1, 1], "CUB scan selected grid mismatch")
        if cost_records is not None:
            cost = cost_records[index]
            if requested:
                winner = next(item for item in cost["candidates"] if item["threads"] == requested)
                require(available and compile_keys == [winner["compile_key"]] and
                        "; cub-scan-resource-scope=installed-entry" in text, "CUB cost final winner mismatch")
            else:
                require(not available and compile_keys == ["0000000000000000"] and
                        "; cub-scan-resource-scope=not-queried" in text and "cub-scan-source-file=" not in text,
                        "CUB cost retained original borrows a candidate identity")
        checked.append(dict(record, selected=selected, compile_key=compile_keys[0] if compile_keys else None,
                            evidence="predicted shared host selector from final pointers; not a Driver trace"))
    return checked


# Frozen profile parity checks, not a fitter or a new scheduling policy. These
# constants identify the same binary64 coefficients as cuda_tile_scan_cost.h.
CUB_SCAN_COST_PROFILE = "sm89-24-cuda134-scan-nnls-v1"
CUB_SCAN_COST_FIT = "67127e819f80a395aeecae55cd999c8e3f8b1bcf894fdf0672c58611cca87f66"
CUB_SCAN_COST_PROFILE_SHA256 = "d38f317cb43b1e646fcdb959e30ea89675b79722508d973ee037609db24d9186"
CUB_SCAN_COST_TILE = (0.7284119403590213, 0.0, 0.00026641302734655293)
CUB_SCAN_COST_CUB = (0.6027021529393383, 12.227593513162688, 0.027854484240835357, 0.022923736019806327)


def validate_native_cub_scan_cost(requested, *other_experiments):
    require(type(requested) is bool, "native CUB scan cost request must be boolean")
    require(not requested or not any(other_experiments), "native CUB scan cost is mutually exclusive with other schedule experiments")


def cub_scan_cost_receipts(result, requested=False):
    validate_native_cub_scan_cost(requested)
    stages = result.get("pipeline_stages", [{"realization": result.get("realization", "")}])
    require(isinstance(stages, list) and stages, "missing CUB cost stages")
    checked = []
    for stage_index, stage in enumerate(stages):
        text = stage.get("realization", "")
        require(isinstance(text, str), "invalid CUB cost realization")
        if not requested:
            require("cub-scan-cost-" not in text, "unrequested CUB cost metadata")
            continue
        facts = {}
        for token in text.split(";"):
            token = token.strip()
            if token.startswith("cub-scan-cost-"):
                key, separator, value = token.partition("=")
                key = key.removeprefix("cub-scan-cost-")
                require(separator and key not in facts and "\n" not in value and "\r" not in value,
                        "malformed or duplicate CUB cost metadata")
                facts[key] = value

        def value(key):
            require(key in facts, "missing CUB cost metadata: " + key)
            return facts[key]

        def integer(key):
            raw = value(key)
            require(re.fullmatch(r"0|[1-9][0-9]*", raw) is not None, "invalid CUB cost integer: " + key)
            number = int(raw)
            require(number <= 2**64 - 1, "CUB cost integer overflow: " + key)
            return number

        def number(key):
            try:
                number = float(value(key))
            except (ValueError, OverflowError) as error:
                raise ValueError("invalid CUB cost number: " + key) from error
            require(math.isfinite(number) and number >= 0, "invalid CUB cost number: " + key)
            return number

        def vector(key, expected):
            try:
                actual = [float(item) for item in value(key).split(",")]
            except (ValueError, OverflowError) as error:
                raise ValueError("invalid CUB cost features: " + key) from error
            require(len(actual) == len(expected) and all(math.isfinite(a) and math.isclose(a, b, rel_tol=1e-13, abs_tol=1e-13)
                                                        for a, b in zip(actual, expected)), "CUB cost feature mismatch: " + key)
            return actual

        def score(key, expected):
            actual = number(key)
            require(actual > 0 and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), "CUB cost score mismatch: " + key)
            return actual

        def resource(key):
            return None if value(key) == "unknown" else integer(key)

        require(integer("requested") == 1 and value("profile") == CUB_SCAN_COST_PROFILE and
                value("fit") == CUB_SCAN_COST_FIT and value("profile-sha256") == CUB_SCAN_COST_PROFILE_SHA256,
                "CUB cost frozen profile mismatch")
        require(integer("declared-count") == 4, "CUB cost candidate count mismatch")
        search_count, call_count = integer("search-count"), integer("compiler-call-count")
        require(0 <= call_count <= search_count <= 4, "invalid CUB cost search/call counts")
        search_ms = number("search-ms")
        status, reason, selected = value("status"), value("reason"), integer("selected-threads")
        require(status in ("selected", "retained", "ineligible") and selected in (0, 128, 256, 512, 1024),
                "invalid CUB cost status/selection")
        require(value("device-query-ok") in ("true", "false"), "invalid CUB cost device query flag")
        device = {key: integer(key) for key in ("sm", "processors", "warp", "resident-threads", "driver-api", "toolkit", "nvrtc")}
        target = dict(zip(device, (89, 24, 32, 1536, 13040, 13040, 130400)))
        has_score = "original-score" in facts
        require((status != "ineligible") == has_score, "CUB cost original score/status mismatch")
        require(value("original-entry") == "luisa_tile_main" and re.fullmatch(r"[0-9a-f]{16}", value("original-source-key")),
                "CUB cost original entry/key mismatch")
        # Preserve original attributes as diagnostics. Tile block=(1,1,1) is
        # an ABI convention and never supplies a physical worker estimate.
        original_resources = {key: resource("original-" + key) for key in ("registers", "static-shared-bytes", "local-bytes", "max-threads")}
        original_resources["status"] = value("original-resource-status")
        original_features = original_score = None
        rows = width = 0
        if has_score:
            require(value("device-query-ok") == "true" and device == target, "CUB cost device profile mismatch")
            require(result.get("operation") == "scan" and result.get("precision") in ("fp16", "bf16") and
                    result.get("fast_math", False) is False, "CUB cost scored an ineligible fixture")
            rows, width = result["dimensions"]
            require(all(type(x) is int and 0 < x <= 2**31 - 1 for x in (rows, width)) and
                    result.get("tile") == [1, width, 1] and width & (width - 1) == 0 and rows * width * 4 <= 2**64 - 1,
                    "CUB cost original geometry mismatch")
            original_features = vector("original-features", (1, rows * width * 4 / (24 * 2**20), ((rows + 23) // 24) * width))
            original_score = score("original-score", sum(a * b for a, b in zip(CUB_SCAN_COST_TILE, original_features)))
        else:
            require(not any(key in facts for key in ("selected-score", "original-features")) and selected == 0 and search_count == call_count == 0,
                    "ineligible CUB cost performed a search or exposed scores")
            require(reason in ("device-query", "target-profile", "fast-math", "other-experiment", "analysis-or-layout", "original-model-facts", "invalid-cub-profile"),
                    "invalid CUB cost gate reason")
            if reason == "device-query":
                require(value("device-query-ok") == "false", "CUB cost device-query reason mismatch")
            if reason == "target-profile":
                require(value("device-query-ok") == "true" and device != target, "CUB cost target reason mismatch")
            if reason == "fast-math":
                require(result.get("fast_math") is True, "CUB cost fast-math reason mismatch")
        candidates, actual_calls = [], 0
        best_threads, best_score = 0, original_score
        for threads in (128, 256, 512, 1024):
            prefix = f"t{threads}-"
            compile_status, load_status, entry_status = (value(prefix + key) for key in ("compile", "load", "entry"))
            disposition, cleanup, scope = (value(prefix + key) for key in ("disposition", "cleanup", "query-scope"))
            attempted = compile_status != "not-attempted"
            require(compile_status in ("not-attempted", "ok", "failed") and
                    value(prefix + "compile-key-known") == ("true" if attempted else "false"), "CUB cost compilation identity mismatch")
            for key in ("compile-key", "source-key"):
                require(re.fullmatch(r"[0-9a-f]{16}", value(prefix + key)), "invalid CUB cost 64-bit key")
            compile_ms = number(prefix + "compile-ms")
            if not attempted:
                require(value(prefix + "compile-key") == value(prefix + "source-key") == "0000000000000000" and compile_ms == 0 and
                        load_status == entry_status == "not-attempted", "unattempted CUB cost candidate borrows a compilation")
            actual_calls += attempted
            source_file = facts.get(prefix + "source-file")
            require(bool(source_file) == attempted, "CUB cost attempted source receipt missing or unattempted source present")
            require(scope in ("loaded-candidate", "not-queried"), "invalid CUB cost query scope")
            require((scope == "loaded-candidate") == (entry_status == "ok") and
                    (entry_status == "not-attempted" or load_status == "ok") and
                    (load_status == "not-attempted" or compile_status == "ok"), "CUB cost compile/load/query chain mismatch")
            require(load_status in ("not-attempted", "ok") or re.fullmatch(r"cuda-[0-9]+", load_status), "invalid CUB cost load status")
            require(entry_status in ("not-attempted", "ok") or re.fullmatch(r"cuda-[0-9]+", entry_status), "invalid CUB cost entry status")
            require(value(prefix + "resource-entry") == "luisa_tile_cub_scan", "CUB cost candidate entry mismatch")
            installed = disposition == "installed"
            require(installed == (threads == selected and selected != 0), "CUB cost installed candidate mismatch")
            if installed:
                require(cleanup == "shader-owned", "installed CUB cost candidate is not shader-owned")
            elif load_status == "ok":
                require(cleanup == "ok" or re.fullmatch(r"(?:cuda-[0-9]+-retry-)+ok", cleanup), "CUB cost candidate cleanup incomplete")
            else:
                require(cleanup == "not-needed", "unloaded CUB cost cleanup mismatch")
            resources = {key: resource(prefix + key) for key in ("registers", "static-shared-bytes", "local-bytes", "max-threads")}
            resources.update(status=value(prefix + "resource-status"), capacity=resource(prefix + "resident-cta-capacity"),
                             capacity_status=value(prefix + "capacity-status"), capacity_threads=integer(prefix + "capacity-threads"),
                             dynamic_shared_bytes=integer(prefix + "capacity-dynamic-shared-bytes"))
            require(resources["capacity_threads"] == threads and resources["dynamic_shared_bytes"] == 0, "CUB cost capacity launch mismatch")
            features = candidate_score = None
            candidate_reason = value(prefix + "reason")
            if prefix + "score" in facts:
                require(has_score and scope == "loaded-candidate" and compile_status == load_status == entry_status == "ok" and
                        resources["status"] == resources["capacity_status"] == "ok" and all(resources[key] is not None for key in
                        ("registers", "static-shared-bytes", "local-bytes", "max-threads", "capacity")) and
                        resources["local-bytes"] == 0 and resources["max-threads"] >= threads and resources["capacity"] > 0 and
                        width % (threads * 8) == 0 and candidate_reason == "scored", "CUB cost scored unknown or inadmissible resources")
                capacity = resources["capacity"]
                groups = (rows + 24 * capacity - 1) // (24 * capacity)
                chunks = width // (8 * threads)
                require(24 * capacity <= 2**64 - 1 and groups * chunks * threads <= 2**64 - 1, "CUB cost feature arithmetic overflow")
                features = vector(prefix + "features", (1, original_features[1], groups * chunks * 8, groups * chunks * ((threads + 31) // 32)))
                candidate_score = score(prefix + "score", sum(a * b for a, b in zip(CUB_SCAN_COST_CUB, features)))
                if candidate_score < best_score:
                    best_threads, best_score = threads, candidate_score
            else:
                require(prefix + "features" not in facts and candidate_reason != "scored" and not installed,
                        "CUB cost candidate missing score")
            candidates.append(dict(threads=threads, compile_status=compile_status, load_status=load_status, entry_status=entry_status,
                disposition=disposition, cleanup=cleanup, query_scope=scope, reason=candidate_reason, diagnostic=value(prefix + "diagnostic"),
                compile_key_known=attempted, compile_key=value(prefix + "compile-key"), source_key=value(prefix + "source-key"),
                compile_ms=compile_ms, source_file=source_file, resources=resources, features=features, score=candidate_score,
                installed=installed, queried_from_live_entry=scope == "loaded-candidate"))
        require(actual_calls == call_count, "CUB cost compiler call count mismatch")
        if not has_score:
            require(all(candidate["disposition"] == "profile-ineligible" for candidate in candidates),
                    "ineligible CUB cost candidate disposition mismatch")
        predicted = best_threads if has_score and best_threads and best_score / original_score < .95 else 0
        if has_score:
            require(search_count == 4 or reason == "candidate-cleanup-failed", "CUB cost search was incomplete")
            if selected:
                require(status == "selected" and reason == "predicted-saving-installed" and selected == predicted,
                        "CUB cost score/choice mismatch")
                score("selected-score", best_score)
            else:
                require(status == "retained" and reason in ("predicted-original", "candidate-install-failed", "candidate-cleanup-failed"),
                        "CUB cost retained reason mismatch")
                require(reason != "predicted-original" or predicted == 0, "CUB cost ignored predicted improvement")
                require(reason != "candidate-install-failed" or predicted != 0, "CUB cost impossible installation failure")
                score("selected-score", original_score)
        checked.append(dict(stage=stage_index, profile=CUB_SCAN_COST_PROFILE, fit=CUB_SCAN_COST_FIT,
            profile_file_sha256=CUB_SCAN_COST_PROFILE_SHA256, status=status, reason=reason, selected_threads=selected,
            predicted_threads=predicted, device=device, device_query_ok=value("device-query-ok") == "true",
            original_resources=original_resources, original_features=original_features, original_score=original_score,
            selected_score=number("selected-score") if has_score else None, search_count=search_count,
            compiler_call_count=call_count, search_ms=search_ms, candidates=candidates, metadata=facts,
            compiler_call_evidence="CUDACompiler::compile calls may hit the PTX LRU; not NVRTC cache-miss counts",
            selection_evidence="installed winner before per-invocation guards; not a Driver trace"))
    return checked


def cub_scan_cost_sources(directory, records, final_sources):
    receipts = {}
    for stage in records:
        for candidate in stage["candidates"]:
            if not candidate["source_file"]:
                continue
            name = f"cub-cost-source-stage{stage['stage']}-t{candidate['threads']}.cu"
            source = directory / name
            require(source.is_file() and source.resolve().is_relative_to(directory.resolve()) and source.stat().st_size > 0,
                    "CUB cost candidate source receipt is absent or escapes its packet")
            receipt = dict(sha256=digest(source), bytes=source.stat().st_size, stage=stage["stage"], threads=candidate["threads"],
                           compile_key=candidate["compile_key"], source_key=candidate["source_key"], installed=candidate["installed"])
            if candidate["installed"]:
                final = final_sources.get(f"cub-source-stage{stage['stage']}.cu")
                require(final and final["sha256"] == receipt["sha256"] and final["bytes"] == receipt["bytes"] and
                        final["compile_key"] == receipt["compile_key"], "CUB cost winner source/key differs from final candidate")
            receipts[name] = receipt
    return receipts


def route_environment(environment, route, native_aligned16=False, native_worker_warps=0,
                      native_scan_chunk=0, native_independent_axis=0, native_streaming_scan=0,
                      native_collective_cost=False, native_program_rows=0, native_partition_cost=False,
                      native_cub_scan_threads=0, native_cub_scan_cost=False):
    # Never inherit experimental specialization into a control or another
    # route. Only an explicit native request may set the exact opt-in value.
    result = dict(environment)
    result.pop("LUISA_CUDA_TILE_IR_ALIGNED16", None)
    result.pop("LUISA_CUDA_TILE_IR", None)
    result.pop("LUISA_CUDA_TILE_WORKER_WARPS", None)
    result.pop("LUISA_CUDA_TILE_SCAN_CHUNK", None)
    result.pop("LUISA_CUDA_TILE_INDEPENDENT_AXIS", None)
    result.pop("LUISA_CUDA_TILE_STREAMING_SCAN", None)
    result.pop("LUISA_CUDA_TILE_COLLECTIVE_COST", None)
    result.pop("LUISA_CUDA_TILE_PROGRAM_ROWS", None)
    result.pop("LUISA_CUDA_TILE_PARTITION_COST", None)
    result.pop("LUISA_CUDA_TILE_CUB_SCAN", None)
    result.pop("LUISA_CUDA_TILE_CUB_SCAN_COST", None)
    validate_native_cub_scan_cost(native_cub_scan_cost, native_aligned16, native_worker_warps, native_scan_chunk,
        native_independent_axis, native_streaming_scan, native_collective_cost, native_program_rows,
        native_partition_cost, native_cub_scan_threads)
    validate_native_cub_scan(native_cub_scan_threads, native_aligned16, native_worker_warps, native_scan_chunk,
                             native_independent_axis, native_streaming_scan, native_collective_cost,
                             native_program_rows, native_partition_cost)
    validate_native_partition_cost(native_partition_cost, native_aligned16, native_worker_warps,
                                   native_scan_chunk, native_independent_axis, native_streaming_scan,
                                   native_collective_cost, native_program_rows)
    validate_native_program_rows(native_program_rows, native_aligned16, native_worker_warps,
                                 native_scan_chunk, native_independent_axis, native_streaming_scan, native_collective_cost)
    validate_native_collective_cost(native_collective_cost, native_aligned16, native_worker_warps,
                                    native_scan_chunk, native_independent_axis, native_streaming_scan)
    validate_native_structure(native_scan_chunk, native_independent_axis, native_streaming_scan)
    require(type(native_worker_warps) is int and native_worker_warps in (0, 4, 8), "invalid native worker-warps request")
    if route == "native":
        result["LUISA_CUDA_TILE_IR"] = "1"
        if native_aligned16:
            result["LUISA_CUDA_TILE_IR_ALIGNED16"] = "1"
        if native_worker_warps:
            result["LUISA_CUDA_TILE_WORKER_WARPS"] = str(native_worker_warps)
        if native_scan_chunk:
            result["LUISA_CUDA_TILE_SCAN_CHUNK"] = str(native_scan_chunk)
        if native_independent_axis:
            result["LUISA_CUDA_TILE_INDEPENDENT_AXIS"] = str(native_independent_axis)
        if native_streaming_scan:
            result["LUISA_CUDA_TILE_STREAMING_SCAN"] = str(native_streaming_scan)
        if native_collective_cost:
            result["LUISA_CUDA_TILE_COLLECTIVE_COST"] = "1"
        if native_program_rows:
            result["LUISA_CUDA_TILE_PROGRAM_ROWS"] = str(native_program_rows)
        if native_partition_cost:
            result["LUISA_CUDA_TILE_PARTITION_COST"] = "1"
        if native_cub_scan_cost:
            result["LUISA_CUDA_TILE_CUB_SCAN_COST"] = "1"
            result["LUISA_DUMP_SOURCE"] = "1"
        if native_cub_scan_threads:
            result["LUISA_CUDA_TILE_CUB_SCAN"] = str(native_cub_scan_threads)
            result["LUISA_DUMP_SOURCE"] = "1"
    return result


def child(cpu, command, work, environment, mask, timeout):
    work.mkdir(exist_ok=False)
    record = dict(command=list(map(str, command)), cwd=str(work), started=now(), status="running")
    save(work / "process.json", record)
    process = job = None
    start = time.monotonic()
    try:
        with (work / "stdout.log").open("xb") as stdout, (work / "stderr.log").open("xb") as stderr:
            job = cpu.OwnedJob()
            process = subprocess.Popen(record["command"], cwd=work, env=environment, stdout=stdout, stderr=stderr,
                                       creationflags=HIDDEN | cpu.CREATE_SUSPENDED)
            job.attach(process)
            record.update(pid=process.pid, affinity_before=cpu.affinity(int(process._handle)),
                          affinity_actual=cpu.set_owned_affinity(int(process._handle), mask),
                          owned_descendants_cleanup="kill-on-close Windows job, attached before resume")
            record["primary_thread_id"] = cpu.resume_owned_primary_thread(process)
            save(work / "process.json", record)
            record["returncode"] = process.wait(timeout=timeout)
        record["status"] = "exited"
    except subprocess.TimeoutExpired:
        record.update(status="timeout", timeout_seconds=timeout)
    except BaseException as error:
        record.update(status="failed", error=str(error), traceback=traceback.format_exc())
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        if job is not None:
            try:
                job.close()
            except OSError as error:
                record.update(status="failed", cleanup_error=str(error))
        if process is not None:
            try:
                if process.poll() is None:
                    process.kill()
                record["returncode"] = process.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired) as error:
                record.update(status="failed", process_cleanup_error=str(error))
        record.update(finished=now(), elapsed_seconds=time.monotonic() - start)
        save(work / "process.json", record)
    return record


def native_result(process, path, row, args, route):
    if process["status"] != "exited":
        return dict(status="failed", process=process, reason="native child did not complete")
    result = read_json(path)
    require(result.get("schema") == 1, "native result schema mismatch")
    for key in ("operation", "dimensions", "tile", "precision", "seed", "pattern"):
        require(result.get(key) == row[key], f"native result {key} mismatch")
    if row["operation"] in {"sort", "topk"}:
        require(result.get("ranking_algorithm", "full_sort_prefix") == row.get("ranking_algorithm", "full_sort_prefix"),
                "native result ranking algorithm mismatch")
    require(result.get("fast_math", False) is row.get("fast_math", False), "native result fast_math mismatch")
    expected_backend, expected_lowering = ("simd", "native") if route == "simd" else ("cuda", route)
    require(result.get("backend") == expected_backend and result.get("lowering") == expected_lowering, "native route mismatch")
    if result["status"] == "compiler_failure":
        require(process["returncode"] == 1 and result.get("reason"), "invalid compiler failure status")
        return dict(status="failed", failure_kind="compiler_failure", process=process, result=result, result_path=str(path))
    if result["status"] == "unsupported":
        require(process["returncode"] == 3 and result.get("reason"), "invalid unsupported status")
        # Older binaries classified all rejected shaders as unsupported. Preserve
        # their raw packet but normalize actual tool-process failures as failures.
        if result["reason"].startswith(("CUDA Tile IR NVRTC failed (", "CUDA Tile IR tileiras failed (",
                                       "CUDA Tile IR NVRTC could not start:", "CUDA Tile IR tileiras could not start:")):
            return dict(status="failed", failure_kind="compiler_failure", process=process, result=result, result_path=str(path))
    else:
        require(result["status"] == "passed" and process["returncode"] == 0, "native execution or correctness failed")
        check = result["correctness"]
        require(check["errors"] == 0 and check["inputs_unchanged"] and check["guards_unchanged"] and check["all_outputs_finite"], "native correctness gate failed")
        require(result["samples"] == args.samples and len(result["host_wall_us"]) == args.samples, "native sample count mismatch")
        if route != "simd":
            require(len(result["cuda_event_stream_span_us"]) == args.samples, "native CUDA event samples missing")
            if args.graph_batch:
                require(result["graph_batch"] == args.graph_batch and len(result["graph_event_stream_span_us_per_op"]) == args.samples,
                        "native graph contract mismatch")
    if result["status"] == "passed" and route == "native":
        alignment_receipts(result, getattr(args, "native_aligned16", False))
    worker_receipts = []
    structure_receipts = []
    streaming_receipts = []
    cost_receipts = []
    partition_receipts = []
    partition_cost_records = []
    cub_receipts = []
    cub_cost_records = []
    if result["status"] == "passed":
        cub_requested = getattr(args, "native_cub_scan_threads", 0) if route == "native" else 0
        validate_native_cub_scan(cub_requested, getattr(args, "native_aligned16", False),
            getattr(args, "native_worker_warps", 0), getattr(args, "native_scan_chunk", 0),
            getattr(args, "native_independent_axis", 0), getattr(args, "native_streaming_scan", 0),
            getattr(args, "native_collective_cost", False), getattr(args, "native_program_rows", 0),
            getattr(args, "native_partition_cost", False))
        cub_cost_requested = getattr(args, "native_cub_scan_cost", False) if route == "native" else False
        validate_native_cub_scan_cost(cub_cost_requested, getattr(args, "native_aligned16", False),
            getattr(args, "native_worker_warps", 0), getattr(args, "native_scan_chunk", 0),
            getattr(args, "native_independent_axis", 0), getattr(args, "native_streaming_scan", 0),
            getattr(args, "native_collective_cost", False), getattr(args, "native_program_rows", 0),
            getattr(args, "native_partition_cost", False), cub_requested)
        cub_cost_records = cub_scan_cost_receipts(result, cub_cost_requested)
        cub_receipts = cub_scan_receipts(result, cub_requested, cub_cost_records if cub_cost_requested else None)
        partition_cost_requested = getattr(args, "native_partition_cost", False) if route == "native" else False
        validate_native_partition_cost(partition_cost_requested, getattr(args, "native_aligned16", False),
            getattr(args, "native_worker_warps", 0), getattr(args, "native_scan_chunk", 0),
            getattr(args, "native_independent_axis", 0), getattr(args, "native_streaming_scan", 0),
            getattr(args, "native_collective_cost", False), getattr(args, "native_program_rows", 0))
        partition_cost_records = partition_cost_receipts(result, partition_cost_requested)
        program_rows = getattr(args, "native_program_rows", 0) if route == "native" else 0
        validate_native_program_rows(program_rows, getattr(args, "native_aligned16", False),
            getattr(args, "native_worker_warps", 0), getattr(args, "native_scan_chunk", 0),
            getattr(args, "native_independent_axis", 0), getattr(args, "native_streaming_scan", 0),
            getattr(args, "native_collective_cost", False))
        partition_receipts = program_partition_receipts(result, program_rows,
                                                       partition_cost_records if partition_cost_requested else None)
        cost_requested = getattr(args, "native_collective_cost", False) if route == "native" else False
        validate_native_collective_cost(cost_requested, getattr(args, "native_aligned16", False),
            getattr(args, "native_worker_warps", 0), getattr(args, "native_scan_chunk", 0),
            getattr(args, "native_independent_axis", 0), getattr(args, "native_streaming_scan", 0))
        cost_receipts = collective_cost_receipts(result, cost_requested)
        worker_receipts = (collective_cost_worker_receipts(result, cost_receipts) if cost_requested else
                           worker_warps_receipts(result, getattr(args, "native_worker_warps", 0) if route == "native" else 0))
        structure_receipts = structural_receipts(result,
            getattr(args, "native_scan_chunk", 0) if route == "native" else 0,
            getattr(args, "native_independent_axis", 0) if route == "native" else 0)
        streaming_receipts = streaming_scan_receipts(result, getattr(args, "native_streaming_scan", 0) if route == "native" else 0)
    artifacts = {}
    if result["status"] == "passed" and row.get("ranking_algorithm", "").startswith("chunked_bitonic_"):
        chunk = 256 if row["ranking_algorithm"] == "chunked_bitonic_c256" else 512
        count = 1 + int(math.log2(row["tile"][1] // chunk))
        pipeline = result.get("pipeline", {})
        require(pipeline.get("kind") == "chunked_bitonic_whole_tile_merge" and
                pipeline.get("chunk") == chunk and pipeline.get("stages_per_operation") == count and
                pipeline.get("stage_widths") == [chunk << i for i in range(count)] and
                pipeline.get("scratch_elements_per_plane") == (0 if count == 1 else row["dimensions"][0] * row["tile"][1]) and
                pipeline.get("scratch_slots") == (0 if count == 1 else 2),
                "native pipeline identity/stage count mismatch")
        stages = result.get("pipeline_stages", [])
        require(len(stages) == count, "pipeline per-stage receipts missing")
        expected_graph_batch = 0 if route == "simd" else args.graph_batch
        require(result.get("graph_stage_dispatches") == count * expected_graph_batch,
                "pipeline graph must include every stage of every operation")
        replays = result.get("graph_replays_per_sample", 0)
        require(type(replays) is int and replays >= 0, "invalid pipeline graph replay count")
        operations = expected_graph_batch * replays
        require(result.get("graph_operations_per_sample") == operations and
                result.get("graph_stage_dispatches_per_sample") == count * operations,
                "pipeline sample must count every stage of every calibrated replay")
        if expected_graph_batch:
            require(result.get("graph_protocol") == "adaptive_replay_span_v2" and
                    1 <= replays <= min(65536, 10_000_000 // expected_graph_batch),
                    "pipeline graph protocol/replay cap mismatch")
            calibration = result.get("graph_calibration", [])
            require(1 <= len(calibration) <= 4 and calibration[-1].get("replays") == replays,
                    "pipeline must use the final actually calibrated replay count")
            for total_key, per_op_key in (("graph_event_span_ms", "graph_event_stream_span_us_per_op"),
                                           ("graph_host_span_ms", "graph_host_wall_us_per_op")):
                total, per_op = result.get(total_key, []), result.get(per_op_key, [])
                require(len(total) == len(per_op) == args.samples and
                        all(math.isfinite(a) and a > 0 and math.isfinite(b) and b > 0 and
                            math.isclose(a * 1000 / operations, b, rel_tol=1e-12, abs_tol=1e-12)
                            for a, b in zip(total, per_op)),
                        "pipeline graph normalization must divide by logical operations, never stages")
        else:
            require(replays == 0, "graph-disabled pipeline has a nonzero replay count")
        root = path.parent.resolve()
        for index, stage in enumerate(stages):
            require(stage.get("index") == index and stage.get("realization") and stage.get("compile_ms", -1) >= 0,
                    "invalid per-stage receipt")
            expected_source = "source.txt" if index == 0 else f"source-stage{index}.txt"
            require(stage.get("source") == expected_source, "unexpected/duplicate pipeline stage source")
            source = (root / stage["source"]).resolve(strict=True)
            require(source.is_relative_to(root), "pipeline source escapes export packet")
            artifacts[stage["source"]] = digest(source)
        require(math.isclose(sum(stage["compile_ms"] for stage in stages), result.get("compile_ms", -1),
                             rel_tol=1e-12, abs_tol=1e-9), "pipeline compile_ms must include every stage")
    generated_sources = dict(artifacts)
    cub_sources = {}
    for record in cub_receipts:
        if record["available"]:
            source = path.parent / f"cub-source-stage{record['stage']}.cu"
            require(source.is_file() and source.resolve().is_relative_to(path.parent.resolve()), "CUB candidate source receipt is absent or escapes its packet")
            cub_sources[source.name] = dict(sha256=digest(source), bytes=source.stat().st_size,
                                           compile_key=record["compile_key"], stage=record["stage"])
    cub_cost_sources = cub_scan_cost_sources(path.parent, cub_cost_records, cub_sources)
    if result["status"] == "passed" and route == "native" and not result.get("pipeline"):
        # Historical packets may lack exported source; keep read-only validation
        # compatible. A new measurement below requires this receipt to exist.
        source = path.parent / "source.txt"
        if source.is_file():
            require(source.resolve().is_relative_to(path.parent.resolve()), "native source escapes export packet")
            generated_sources["source.txt"] = digest(source)
    return dict(status=result["status"], process=process, result=result, result_path=str(path), pipeline_sources=artifacts,
                generated_sources=generated_sources,
                native_worker_warps=worker_receipts, native_structure=structure_receipts, native_streaming=streaming_receipts,
                native_collective_cost=cost_receipts, native_program_partition=partition_receipts,
                native_partition_cost=partition_cost_records, native_cub_scan=cub_receipts,
                cub_generated_sources=cub_sources, native_cub_scan_cost=cub_cost_records,
                cub_cost_generated_sources=cub_cost_sources)


def torch_result(process, path, row, args):
    if process["status"] != "exited":
        return dict(status="failed", process=process, reason="Torch child did not complete")
    result = read_json(path)
    require(process["returncode"] == 0 and result["status"] == "passed", "Torch compilation or correctness failed")
    require(result["manifest"]["precision"] == row["precision"] and result["manifest"]["dimensions"] == row["dimensions"], "Torch fixture mismatch")
    require(result["compile_requested_mode"] == args.torch_mode and result["compiler_evidence"]["fullgraph"], "missing full-graph Inductor evidence")
    require(len(result["compiled_stream"]["synchronized_host_wall_us"]["samples"]) == args.samples, "Torch sample count mismatch")
    for key in ("compiled_correctness_before", "compiled_correctness_after"):
        require(result[key]["failed_elements"] == 0, "Torch oracle failed")
    if args.graph_batch:
        graph = result["compiled_graph"]
        require(graph["batch"] == args.graph_batch and len(graph["event_us_per_operation"]["samples"]) == args.samples, "Torch graph contract mismatch")
    return dict(status="passed", process=process, result=result, result_path=str(path))


def summarize(records):
    summary = []
    for row in records:
        item = dict(case=row["case"]["id"], status=row["status"], measurements={})
        for name, run in row.get("runs", {}).items():
            measurement = dict(status=run["status"])
            result = run.get("result", {})
            if run["status"] == "passed":
                if name == "torch":
                    measurement.update(cold_compile_first_call_ms=result["cold_compile_first_call_ms"],
                                       host_wall_p50_us=result["compiled_stream"]["synchronized_host_wall_us"]["median"],
                                       allocation_policy=result["allocation_policy"], ranking_contract=result["ranking_contract"])
                    if "compiled_graph" in result:
                        measurement["graph_event_stream_span_p50_us_per_op"] = result["compiled_graph"]["event_us_per_operation"]["median"]
                        measurement["graph_host_wall_p50_us_per_op"] = result["compiled_graph"]["host_wall_us_per_operation"]["median"]
                else:
                    measurement.update(compile_ms=result["compile_ms"], cold_call_ms=result["cold_ms"],
                                       host_wall_p50_us=result["host_wall_p50_us"], realization=result["realization"],
                                       allocation_policy=("preallocated input/output/scratch; graph uses scratch/output hazard DAG; independent stages may overlap across complete calls"
                                                          if result.get("pipeline") else
                                                          "preallocated fixed output; graph dispatches share WAW dependency"))
                    if result["cuda_event_stream_span_us"]:
                        measurement["event_stream_span_p50_us"] = statistics.median(result["cuda_event_stream_span_us"])
                    if result["graph_event_stream_span_us_per_op"]:
                        measurement["graph_event_stream_span_p50_us_per_op"] = statistics.median(result["graph_event_stream_span_us_per_op"])
                        measurement["graph_host_wall_p50_us_per_op"] = statistics.median(result["graph_host_wall_us_per_op"])
            else:
                measurement["reason"] = result.get("reason", run.get("reason", run.get("error")))
            item["measurements"][name] = measurement
        summary.append(item)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("smoke", "broad"), default="smoke")
    parser.add_argument("--cases", help="comma-separated built-in ids, or schema-1 JSON containing a cases array")
    parser.add_argument("--case-filter", action="append", default=[], help="case-id glob; repeated filters are ORed")
    parser.add_argument("--list-cases", action="store_true", help="host-only JSON inventory; does not initialize Windows/GPU tools")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--build-marker", type=Path, help="defaults to BUILD/logs/full-build-success.json")
    parser.add_argument("--torch-python", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--affinity-mask", type=lambda value: int(value, 0), help="explicit Windows group-0 process affinity mask")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--routes", default="native,tirx", help="comma-separated native,tirx,simd; CUDA runs remain serial")
    parser.add_argument("--native-aligned16", action="store_true", help="opt in to host-selected 16-byte aligned native Tile entries; other routes stay unchanged")
    parser.add_argument("--native-worker-warps", type=int, choices=(0, 4, 8), default=0,
                        help="explicit experimental native Tile worker hint; 0 preserves the default, other routes never inherit it")
    parser.add_argument("--native-scan-chunk", type=int, choices=(0, 1024, 2048), default=0,
                        help="explicit native pure scan chunking; 0 preserves default; incompatible with nonzero independent-axis")
    parser.add_argument("--native-independent-axis", type=int, choices=(0, 1, 2, 4), default=0,
                        help="explicit native independent-axis partition extent; 0 preserves default; incompatible with nonzero scan-chunk")
    parser.add_argument("--native-collective-cost", action="store_true",
                        help="experimental native collective cost profile; mutually exclusive with all other native scheduling experiments")
    parser.add_argument("--native-only", action="store_true", help="calibration only: run the native route, omit Torch entirely and report no Torch comparison")
    parser.add_argument("--native-streaming-scan", type=int, choices=(0, 1024, 2048), default=0,
                        help="native streaming prefix chunk; requires a proved disjoint available candidate")
    parser.add_argument("--native-program-rows", type=int, choices=(0, 1, 2, 4), default=0,
                        help="independent rows per native reduction program; requires a proved disjoint candidate")
    parser.add_argument("--native-partition-cost", action="store_true",
                        help="frozen experimental independent-program cost model; excludes other schedule experiments")
    parser.add_argument("--native-cub-scan-cost", action="store_true",
                        help="opt in to the frozen CUB scan cost profile; compile/query up to four candidates")
    parser.add_argument("--native-cub-scan-threads", type=int, choices=(0, 128, 256, 512, 1024), default=0,
                        help="explicit guarded CUB scan realization, eight items per physical CUDA thread; 0 preserves default")
    parser.add_argument("--torch-mode", choices=("default", "max-autotune"), default="max-autotune")
    parser.add_argument("--ranking-contract", choices=("standard", "stable"), default="standard")
    parser.add_argument("--eager", action="store_true", help="also retain the explicitly secondary eager Torch measurement")
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--precisions", help="expand each selected case into fp32,fp16,bf16 variants; omitted uses case JSON precision or FP32 built-ins")
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--sample-ms", type=int, default=100)
    parser.add_argument("--warmup-ms", type=int, default=500)
    parser.add_argument("--graph-batch", type=int, default=32)
    parser.add_argument("--native-timeout", type=float, default=300)
    parser.add_argument("--torch-timeout", type=float, default=900)
    parser.add_argument("--path-prefix", type=Path, action="append", default=[], help="additional CUDA/TVMx DLL directory; repeatable")
    parser.add_argument("--telemetry-ms", type=int, default=1000)
    args = parser.parse_args(argv)
    validate_native_cub_scan_cost(args.native_cub_scan_cost, args.native_aligned16, args.native_worker_warps,
        args.native_scan_chunk, args.native_independent_axis, args.native_streaming_scan,
        args.native_collective_cost, args.native_program_rows, args.native_partition_cost, args.native_cub_scan_threads)
    validate_native_cub_scan(args.native_cub_scan_threads, args.native_aligned16, args.native_worker_warps,
                             args.native_scan_chunk, args.native_independent_axis, args.native_streaming_scan,
                             args.native_collective_cost, args.native_program_rows, args.native_partition_cost)
    validate_native_partition_cost(args.native_partition_cost, args.native_aligned16, args.native_worker_warps,
                                    args.native_scan_chunk, args.native_independent_axis, args.native_streaming_scan,
                                    args.native_collective_cost, args.native_program_rows)
    validate_native_collective_cost(args.native_collective_cost, args.native_aligned16, args.native_worker_warps,
                                    args.native_scan_chunk, args.native_independent_axis, args.native_streaming_scan)
    validate_native_structure(args.native_scan_chunk, args.native_independent_axis, args.native_streaming_scan)
    validate_native_program_rows(args.native_program_rows, args.native_aligned16, args.native_worker_warps,
                                 args.native_scan_chunk, args.native_independent_axis, args.native_streaming_scan,
                                 args.native_collective_cost)
    require(not args.native_only or args.routes == "native", "--native-only requires --routes native")
    rows = selected_cases(args)
    if args.list_cases:
        print(json.dumps(dict(schema=1, suite=args.suite, cases=rows), indent=2))
        return 0
    require(os.name == "nt", "this owned-job/affinity runner currently requires Windows")
    require(args.build_dir and args.output and args.affinity_mask and (args.native_only or args.torch_python),
            "execution requires build-dir, output, affinity-mask, and torch-python unless native-only")
    require(1 <= args.threads <= 64 and 1 <= args.samples <= 99 and args.samples % 2 == 1 and
            1 <= args.sample_ms <= 5000 and 1 <= args.warmup_ms <= 30000 and 0 <= args.graph_batch <= 65536 and
            100 <= args.telemetry_ms <= 60000 and args.affinity_mask > 0 and
            all(math.isfinite(v) and v > 0 for v in (args.native_timeout, args.torch_timeout)), "invalid execution bounds")
    routes = args.routes.split(",")
    require(routes and len(set(routes)) == len(routes) and set(routes) <= {"native", "tirx", "simd"}, "invalid/duplicate routes")
    build = args.build_dir.resolve(strict=True)
    executable = (build / "bin/benchmark_tile_workloads.exe").resolve(strict=True)
    python = (args.torch_python or Path(sys.executable)).resolve(strict=True)
    baseline = HERE / "cuda_torch_baseline.py"
    marker = (args.build_marker or build / "logs/full-build-success.json").resolve(strict=True)
    marker_data = read_json(marker)
    require(isinstance(marker_data, dict), "build marker must be a JSON object")
    marker_time = datetime.fromisoformat(marker_data["completed"]).timestamp() if "completed" in marker_data else marker.stat().st_mtime
    require(executable.stat().st_mtime <= marker_time + 2.0, "benchmark executable is newer than completed build marker")
    import windows_affinity as cpu
    topology = cpu.topology()
    require(args.affinity_mask <= int(topology["caller_affinity"]["system_mask"], 16) and
            args.affinity_mask & int(topology["caller_affinity"]["system_mask"], 16) == args.affinity_mask, "mask is outside the active group-0 system mask")
    selected = [item for item in topology["records"] if item["group"] == 0 and args.affinity_mask & (1 << item["logical_processor"])]
    require(len(selected) == args.affinity_mask.bit_count() and len({item["core_index"] for item in selected}) >= args.threads,
            "affinity must select documented CPU sets and at least threads distinct physical cores")
    environment = os.environ.copy()
    removed = {}
    for key in list(environment):
        if key.startswith("LUISA_TILE_BENCH_") or key in {"LUISA_CUDA_TILE_IR", "LUISA_CUDA_TILE_IR_ALIGNED16", "LUISA_CUDA_TILE_WORKER_WARPS", "LUISA_CUDA_TILE_SCAN_CHUNK", "LUISA_CUDA_TILE_INDEPENDENT_AXIS", "LUISA_CUDA_TILE_STREAMING_SCAN", "LUISA_CUDA_TILE_COLLECTIVE_COST", "LUISA_CUDA_TILE_PROGRAM_ROWS", "LUISA_CUDA_TILE_PARTITION_COST", "LUISA_CUDA_TILE_CUB_SCAN", "LUISA_CUDA_TILE_CUB_SCAN_COST", "LUISA_DUMP_SOURCE", "LUISA_DUMP_SPV", "TVM_COMPILE_FORCE_FALLBACK", "LUISA_CUDA_TILE_FORCE_UNSUPPORTED_PTX", "LUISA_SIMD_ROOT_AXIS_TILES"}:
            removed[key] = environment.pop(key)
    overrides = {"LUISA_SIMD_WORKER_COUNT": str(args.threads), "LUISA_SIMD_WARP_WIDTH": "8", "OPENBLAS_NUM_THREADS": str(args.threads),
                 "OMP_NUM_THREADS": str(args.threads), "MKL_NUM_THREADS": str(args.threads), "GOTO_NUM_THREADS": str(args.threads),
                 "TORCHINDUCTOR_COMPILE_THREADS": "1", "NVIDIA_TF32_OVERRIDE": "0"}
    environment.update(overrides)
    paths = [build / "bin"] + [path.resolve(strict=True) for path in args.path_prefix]
    environment["PATH"] = os.pathsep.join(map(str, paths)) + os.pathsep + environment.get("PATH", "")
    smi = shutil.which("nvidia-smi", path=environment["PATH"])
    require(smi is not None, "nvidia-smi is required for telemetry")
    sources = [ROOT / "src/tests/benchmark/benchmark_tile_workloads.cpp", ROOT / "src/tests/common/tile_workload_test_utils.h",
               ROOT / "src/tests/common/tile_llm_test_utils.h", ROOT / "src/tests/common/tile_rank_test_utils.h",
               ROOT / "src/tests/common/tile_selection_test_utils.h", ROOT / "src/tests/common/tile_argmax_test_utils.h", ROOT / "src/tests/common/tile_embedding_test_utils.h", ROOT / "src/tests/common/tile_sort_pipeline_test_utils.h",
               ROOT / "include/luisa/tile/algorithms.h", ROOT / "include/luisa/tile/value.h", ROOT / "include/luisa/tile/dsl.h",
               ROOT / "include/luisa/tile/collective_plan.h", ROOT / "src/tile/collective_plan.cpp",
               ROOT / "include/luisa/tile/collective_prefix.h", ROOT / "src/tile/collective_prefix.cpp",
               ROOT / "include/luisa/tile/collective_cost.h", ROOT / "src/tile/collective_cost.cpp",
               ROOT / "include/luisa/tile/collective_partition.h", ROOT / "src/tile/collective_partition.cpp",
               ROOT / "src/backends/cuda/tile/cuda_tile_partition_codegen.h",
               ROOT / "src/backends/cuda/tile/cuda_tile_partition_cost.h",
               ROOT / "src/backends/cuda/tile/cuda_tile_collective_cost.h"]
    files = [executable, python, baseline, Path(__file__).resolve(), HERE / "windows_affinity.py", marker, build / "CMakeCache.txt"] + sources
    files += list((build / "bin").glob("luisa*.dll"))
    files += [path for name in ("luisa_cuda_tile_compiler.exe", "luisa_nvrtc.exe") if (path := build / "bin" / name).is_file()]
    files += list((ROOT / "src/backends/cuda/tile").glob("cuda_tile*.cpp"))
    files += [ROOT / name for name in ("src/backends/cuda/tile/cuda_tile_codegen.h", "src/backends/cuda/cuda_shader_tile.h",
                                       "src/backends/cuda/tile/cuda_tile_streaming_scan.h", "src/backends/cuda/tile/cuda_tile_streaming_guard.h",
                                       "src/backends/cuda/tile/cuda_tile_cub_scan.h", "src/backends/cuda/tile/cuda_tile_scan_cost.h",
                                       "src/backends/cuda/cuda_shader_tile.cpp", "src/backends/cuda/extensions/cuda_graph_ext.cpp")]
    for path in args.path_prefix:
        files += list(path.resolve().glob("*tvm*.dll"))
    identities = {str(path): dict(sha256=digest(path), bytes=path.stat().st_size) for path in files}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    record = dict(schema=1, status="running", started=now(), suite=args.suite, options={key: str(value) if isinstance(value, Path) else
                  [str(x) for x in value] if key == "path_prefix" else value for key, value in vars(args).items()},
                  topology=topology, selected_processors=selected, environment_overrides=overrides, environment_removed=removed,
                  build_marker=marker_data, files=identities, planned_cases=rows, cases=[],
                  torch_requested=not args.native_only,
                  methodology=("Native-only calibration; Torch is not invoked and no Torch comparison is produced. " if args.native_only else
                               "Serial actual Tile compile and fullgraph torch.compile; every route uses hash-identical exported inputs/oracle. ") +
                              "Cold phases separate. Host wall, event stream spans and graph replay spans never pooled. Native fixed output and optional scratch/output hazard DAG versus functional Torch allocation and standard ranking tie differences retained; no pure-kernel or matched-allocation claim.")

    def checkpoint():
        record["summary"] = summarize(record["cases"])
        save(output / "results.json", record)

    checkpoint()
    telemetry_job = telemetry_process = None
    try:
        with (output / "gpu-telemetry.csv").open("xb") as stdout, (output / "gpu-telemetry.stderr.log").open("xb") as stderr:
            command = [smi, "--id=0", "--query-gpu=timestamp,name,uuid,driver_version,pstate,clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw,utilization.gpu", "--format=csv", "-lms", str(args.telemetry_ms)]
            telemetry_job = cpu.OwnedJob()
            telemetry_process = subprocess.Popen(command, env=environment, stdout=stdout, stderr=stderr, creationflags=HIDDEN | cpu.CREATE_SUSPENDED)
            telemetry_job.attach(telemetry_process)
            affinity = cpu.set_owned_affinity(int(telemetry_process._handle), args.affinity_mask)
            cpu.resume_owned_primary_thread(telemetry_process)
            record["telemetry"] = dict(command=command, pid=telemetry_process.pid, affinity_actual=affinity)
            for definition in rows:
                require(telemetry_process.poll() is None, "GPU telemetry exited unexpectedly")
                directory = output / definition["id"]
                directory.mkdir(exist_ok=False)
                item = dict(case=definition, status="running", started=now(), runs={})
                record["cases"].append(item)
                checkpoint()
                canonical = canonical_hashes = None
                for route in routes:
                    work = directory / route
                    export = work / "artifacts"
                    backend, lowering = ("simd", "native") if route == "simd" else ("cuda", route)
                    command = [executable, backend, lowering, definition["operation"], definition["precision"],
                               ",".join(map(str, definition["dimensions"])), ",".join(map(str, definition["tile"])),
                               definition["seed"], definition["pattern"], args.samples, args.sample_ms, args.warmup_ms, export]
                    if definition.get("fast_math", False):
                        command += ["--fast-math", "1"]
                    if definition.get("ranking_algorithm", "full_sort_prefix") != "full_sort_prefix":
                        command += ["--ranking-algorithm", definition["ranking_algorithm"]]
                    if args.graph_batch and route != "simd":
                        command += ["--graph-batch", args.graph_batch]
                    child_environment = route_environment(environment, route, args.native_aligned16, args.native_worker_warps,
                                                          args.native_scan_chunk, args.native_independent_axis, args.native_streaming_scan,
                                                          args.native_collective_cost, args.native_program_rows, args.native_partition_cost, args.native_cub_scan_threads, args.native_cub_scan_cost)
                    process = child(cpu, command, work, child_environment, args.affinity_mask, args.native_timeout)
                    try:
                        item["runs"][route] = native_result(process, export / "results.json", definition, args, route)
                        if route == "native" and item["runs"][route]["status"] == "passed":
                            require(item["runs"][route].get("generated_sources"), "new native measurement requires frozen generated source receipts")
                    except Exception as error:
                        item["runs"][route] = dict(status="failed", process=process, error=str(error), traceback=traceback.format_exc())
                    item["runs"][route]["native_aligned16_requested"] = route == "native" and args.native_aligned16
                    item["runs"][route]["native_worker_warps_requested"] = args.native_worker_warps if route == "native" else 0
                    item["runs"][route]["native_scan_chunk_requested"] = args.native_scan_chunk if route == "native" else 0
                    item["runs"][route]["native_independent_axis_requested"] = args.native_independent_axis if route == "native" else 0
                    item["runs"][route]["native_streaming_scan_requested"] = args.native_streaming_scan if route == "native" else 0
                    item["runs"][route]["native_collective_cost_requested"] = route == "native" and args.native_collective_cost
                    item["runs"][route]["native_program_rows_requested"] = args.native_program_rows if route == "native" else 0
                    item["runs"][route]["native_partition_cost_requested"] = route == "native" and args.native_partition_cost
                    item["runs"][route]["native_cub_scan_threads_requested"] = args.native_cub_scan_threads if route == "native" else 0
                    item["runs"][route]["native_cub_scan_cost_requested"] = route == "native" and args.native_cub_scan_cost
                    item["runs"][route]["environment_overrides"] = {
                        key: child_environment[key] for key in ("LUISA_CUDA_TILE_IR", "LUISA_CUDA_TILE_IR_ALIGNED16", "LUISA_CUDA_TILE_WORKER_WARPS",
                                                               "LUISA_CUDA_TILE_SCAN_CHUNK", "LUISA_CUDA_TILE_INDEPENDENT_AXIS", "LUISA_CUDA_TILE_STREAMING_SCAN",
                                                               "LUISA_CUDA_TILE_COLLECTIVE_COST", "LUISA_CUDA_TILE_PROGRAM_ROWS",
                                                               "LUISA_CUDA_TILE_PARTITION_COST", "LUISA_CUDA_TILE_CUB_SCAN", "LUISA_CUDA_TILE_CUB_SCAN_COST", "LUISA_DUMP_SOURCE") if key in child_environment}
                    manifest = export / "manifest.json"
                    if manifest.is_file():
                        try:
                            _, hashes = tensor_receipts(manifest, definition)
                            if canonical is None:
                                canonical, canonical_hashes = manifest, hashes
                                item["canonical_manifest"] = str(canonical)
                                item["tensor_sha256"] = canonical_hashes
                            else:
                                require(hashes == canonical_hashes, "routes produced different inputs/oracle for the same case")
                            item["runs"][route]["fixture_sha256"] = hashes
                        except Exception as error:
                            item["runs"][route].update(status="failed", fixture_error=str(error))
                    checkpoint()
                    save(directory / "case.json", item)
                if args.native_only:
                    item["runs"]["torch"] = dict(status="not_requested", requested=False,
                        reason="--native-only calibration: no Torch child was run", native_worker_warps_requested=0,
                        native_scan_chunk_requested=0, native_independent_axis_requested=0, native_collective_cost_requested=False,
                        native_partition_cost_requested=False, native_program_rows_requested=0, native_cub_scan_threads_requested=0, native_cub_scan_cost_requested=False,
                        environment_overrides={})
                    item["comparison_status"] = "absent_native_only_calibration"
                elif canonical is None:
                    item["runs"]["torch"] = dict(status="failed", reason="no complete exported input/oracle packet; native failures retained")
                else:
                    work = directory / "torch"
                    torch_output = work / "artifacts"
                    command = [python, baseline, "--manifest", canonical, "--output", torch_output,
                               "--precision", definition["precision"], "--mode", args.torch_mode, "--ranking-contract", args.ranking_contract,
                               "--samples", args.samples, "--sample-ms", args.sample_ms, "--warmup-ms", args.warmup_ms,
                               "--graph-batch", args.graph_batch, "--threads", args.threads]
                    if args.eager:
                        command.append("--eager")
                    child_environment = route_environment(environment, "torch")
                    process = child(cpu, command, work, child_environment, args.affinity_mask, args.torch_timeout)
                    try:
                        item["runs"]["torch"] = torch_result(process, torch_output / "result.json", definition, args)
                        _, hashes = tensor_receipts(canonical, definition)
                        require(hashes == canonical_hashes, "exported fixture changed during Torch execution")
                    except Exception as error:
                        item["runs"]["torch"] = dict(status="failed", process=process, error=str(error), traceback=traceback.format_exc())
                    item["runs"]["torch"]["native_worker_warps_requested"] = 0
                    item["runs"]["torch"]["native_collective_cost_requested"] = False
                    item["runs"]["torch"]["native_partition_cost_requested"] = False
                    item["runs"]["torch"]["native_cub_scan_threads_requested"] = 0
                    item["runs"]["torch"]["native_cub_scan_cost_requested"] = False
                    item["runs"]["torch"]["native_program_rows_requested"] = 0
                    item["runs"]["torch"]["native_scan_chunk_requested"] = 0
                    item["runs"]["torch"]["native_independent_axis_requested"] = 0
                    item["runs"]["torch"]["environment_overrides"] = {}
                statuses = [run["status"] for run in item["runs"].values()]
                item.update(status="failed" if "failed" in statuses else "unsupported" if "unsupported" in statuses else "passed", finished=now())
                save(directory / "case.json", item)
                checkpoint()
                print(f"{definition['id']}: {item['status']}", flush=True)
            changed = [name for name, receipt in identities.items() if digest(name) != receipt["sha256"]]
            require(not changed, f"build/runtime/script inputs changed during measurement: {changed}")
            require(telemetry_process.poll() is None, "GPU telemetry exited during measurement")
            states = [item["status"] for item in record["cases"]]
            record["status"] = "failed" if "failed" in states else "completed_with_unsupported" if "unsupported" in states else "passed"
    except BaseException as error:
        record.update(status="failed", error=str(error), traceback=traceback.format_exc())
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        if telemetry_job is not None:
            try:
                telemetry_job.close()
            except OSError as error:
                record.update(status="failed", telemetry_cleanup_error=str(error))
        if telemetry_process is not None:
            try:
                if telemetry_process.poll() is None:
                    telemetry_process.kill()
                telemetry_process.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired) as error:
                record.update(status="failed", telemetry_process_cleanup_error=str(error))
        record["finished"] = now()
        checkpoint()
    return {"passed": 0, "completed_with_unsupported": 3}.get(record["status"], 1)


if __name__ == "__main__":
    sys.exit(main())
