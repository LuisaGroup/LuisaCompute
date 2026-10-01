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
        "gelu_residual", "reduce_sum", "reduce_max", "scan", "scan_ordered"}
OPERATIONS = ROWS | {"gemm", "gemv", "bmm", "attention", "attention_tensorcore", "sort", "topk"}
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
    temporary.replace(path)


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
    require(ranking in {"full_sort_prefix", "repeated_extrema", "chunked_bitonic_c256", "chunked_bitonic_c512"}, "unknown ranking algorithm")
    require("ranking_algorithm" not in row or op in {"sort", "topk"}, "ranking_algorithm is only applicable to ranking")
    require(ranking != "repeated_extrema" or op == "topk", "repeated_extrema requires topk")
    require(not ranking.startswith("chunked_bitonic_") or op == "sort", "chunked bitonic requires sort")
    dims, tile = row.get("dimensions"), row.get("tile")
    require(isinstance(dims, list) and len(dims) == (2 if op in ROWS else 7 if op in {"attention", "attention_tensorcore"} else 4 if op == "bmm" else 3), "wrong dimension count")
    require(isinstance(tile, list) and len(tile) == 3 and
            all(type(v) is int and 0 < v <= 65536 for v in dims + tile), "invalid dimensions/tile")
    work_limit = 2**34 if op in {"gemm", "gemv", "bmm"} else 2**31 if op in {"attention", "attention_tensorcore"} else 2**28
    require(math.prod(dims) <= work_limit, "case exceeds native fixture work bound")
    require(row.get("precision", "fp32") in {"fp32", "fp16", "bf16"}, "unsupported precision")
    require(op != "attention_tensorcore" or row.get("precision", "fp32") in {"fp16", "bf16"}, "attention_tensorcore requires FP16/BF16")
    require(row.get("pattern", "random") in {"random", "cancellation", "adversarial"}, "invalid input pattern")
    seed = row.get("seed", default_seed)
    require(type(seed) is int and 0 <= seed < 2**64, "seed must be uint64")
    if op in ROWS:
        require(math.prod(dims) <= 2**24, "row input exceeds fixture allocation bound")
        require(tile[0] == tile[2] == 1 and tile[1] <= 16384, "invalid row schedule")
        require(op != "rope" or dims[1] % 2 == 0, "RoPE width must be even")
        require(tile[1] >= (dims[1] // 2 if op == "rope" else dims[1]), "row tile does not cover the logical width")
        require(op != "scan_ordered" or tile[1] <= 1024, "ordered reference scan is bounded to tile width 1024")
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
        algorithm = ("stable_chunked_bitonic_whole_tile_merge" if ranking.startswith("chunked_bitonic_") else
                     "stable_repeated_extrema" if ranking == "repeated_extrema" else
                     "padded_bitonic_full_sort" if expected_case["operation"] == "sort" else "padded_bitonic_full_sort_prefix")
        require(manifest.get("algorithm") == algorithm, "manifest realized ranking algorithm mismatch")
    sizes = {"float32": 4, "float16": 2, "bfloat16": 2}
    storage = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}[expected_case["precision"]]
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
    return dict(status=result["status"], process=process, result=result, result_path=str(path), pipeline_sources=artifacts)


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
    rows = selected_cases(args)
    if args.list_cases:
        print(json.dumps(dict(schema=1, suite=args.suite, cases=rows), indent=2))
        return 0
    require(os.name == "nt", "this owned-job/affinity runner currently requires Windows")
    require(args.build_dir and args.torch_python and args.output and args.affinity_mask, "execution requires build-dir, torch-python, output, affinity-mask")
    require(1 <= args.threads <= 64 and 1 <= args.samples <= 99 and args.samples % 2 == 1 and
            1 <= args.sample_ms <= 5000 and 1 <= args.warmup_ms <= 30000 and 0 <= args.graph_batch <= 65536 and
            100 <= args.telemetry_ms <= 60000 and args.affinity_mask > 0 and
            all(math.isfinite(v) and v > 0 for v in (args.native_timeout, args.torch_timeout)), "invalid execution bounds")
    routes = args.routes.split(",")
    require(routes and len(set(routes)) == len(routes) and set(routes) <= {"native", "tirx", "simd"}, "invalid/duplicate routes")
    build = args.build_dir.resolve(strict=True)
    executable = (build / "bin/benchmark_tile_workloads.exe").resolve(strict=True)
    python = args.torch_python.resolve(strict=True)
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
        if key.startswith("LUISA_TILE_BENCH_") or key in {"LUISA_CUDA_TILE_IR", "LUISA_DUMP_SOURCE", "LUISA_DUMP_SPV", "TVM_COMPILE_FORCE_FALLBACK", "LUISA_CUDA_TILE_FORCE_UNSUPPORTED_PTX", "LUISA_SIMD_ROOT_AXIS_TILES"}:
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
               ROOT / "src/tests/common/tile_selection_test_utils.h", ROOT / "src/tests/common/tile_sort_pipeline_test_utils.h",
               ROOT / "include/luisa/tile/algorithms.h", ROOT / "include/luisa/tile/value.h", ROOT / "include/luisa/tile/dsl.h"]
    files = [executable, python, baseline, Path(__file__).resolve(), HERE / "windows_affinity.py", marker, build / "CMakeCache.txt"] + sources
    files += list((build / "bin").glob("luisa*.dll"))
    files += [path for name in ("luisa_cuda_tile_compiler.exe", "luisa_nvrtc.exe") if (path := build / "bin" / name).is_file()]
    files += list((ROOT / "src/backends/cuda/tile").glob("cuda_tile*.cpp"))
    for path in args.path_prefix:
        files += list(path.resolve().glob("*tvm*.dll"))
    identities = {str(path): dict(sha256=digest(path), bytes=path.stat().st_size) for path in files}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    record = dict(schema=1, status="running", started=now(), suite=args.suite, options={key: str(value) if isinstance(value, Path) else
                  [str(x) for x in value] if key == "path_prefix" else value for key, value in vars(args).items()},
                  topology=topology, selected_processors=selected, environment_overrides=overrides, environment_removed=removed,
                  build_marker=marker_data, files=identities, planned_cases=rows, cases=[],
                  methodology="Serial actual Tile compile and fullgraph torch.compile; every route uses hash-identical exported inputs/oracle. Cold phases separate. Host wall, event stream spans and graph replay spans never pooled. Native fixed output and optional scratch/output hazard DAG versus functional Torch allocation and standard ranking tie differences retained; no pure-kernel or matched-allocation claim.")

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
                    child_environment = {**environment, **({"LUISA_CUDA_TILE_IR": "1"} if route == "native" else {})}
                    process = child(cpu, command, work, child_environment, args.affinity_mask, args.native_timeout)
                    try:
                        item["runs"][route] = native_result(process, export / "results.json", definition, args, route)
                    except Exception as error:
                        item["runs"][route] = dict(status="failed", process=process, error=str(error), traceback=traceback.format_exc())
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
                if canonical is None:
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
                    process = child(cpu, command, work, environment, args.affinity_mask, args.torch_timeout)
                    try:
                        item["runs"]["torch"] = torch_result(process, torch_output / "result.json", definition, args)
                        _, hashes = tensor_receipts(canonical, definition)
                        require(hashes == canonical_hashes, "exported fixture changed during Torch execution")
                    except Exception as error:
                        item["runs"]["torch"] = dict(status="failed", process=process, error=str(error), traceback=traceback.format_exc())
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
