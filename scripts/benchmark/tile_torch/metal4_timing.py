#!/usr/bin/env python3
"""Serial, fixed-batch Metal4 timing evidence; run under external caffeinate -diu.

Precise counters measure instrumented dispatch intervals, not zero-overhead
kernel time. Paired feedback-only controls measure command-buffer GPU intervals.
No tuning, planner modification, cross-framework comparison, or winner selection.
All subprocess output and failed rows are retained; tensor exports remain in the
reported temporary directory rather than bloating the result archive.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import tempfile
import time


OPERATIONS = {"rmsnorm", "layernorm", "masked_softmax", "swiglu", "gelu_residual", "rope"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_case(text):
    try:
        op, shape = text.split(":")
        rows, width = map(int, shape.split(","))
        require(op in OPERATIONS and 0 < rows <= 65536 and 0 < width <= 65536,
                "unknown operation or invalid dimensions")
        require(rows * width <= 2**26 and (op != "rope" or width % 2 == 0), "invalid tensor extent")
        return op, (rows, width)
    except ValueError:
        raise argparse.ArgumentTypeError("expected supported row operation:rows,width within benchmark limits") from None


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def receipts(paths):
    return {str(path): {"bytes": path.stat().st_size, "sha256": digest(path)} if path.is_file() else None
            for path in paths}


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def finite(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (not positive or value > 0)


def validate_sample(sample, expected, counters, frequency):
    require(sample["error"] == "" and sample["overflow"] is False, "sample error or overflow")
    require(sample["dispatch_timestamps_enabled"] is counters, "wrong timestamp mode")
    require(integer(sample["timestamp_frequency_hz"]) and
            sample["timestamp_frequency_hz"] == (frequency if counters else 0), "wrong sample frequency")
    buffers, dispatches = sample["command_buffers"], sample["dispatches"]
    require(isinstance(buffers, list) and buffers and len(dispatches) == expected, "wrong record counts")
    by_buffer = {}
    feedback_ns = 0.0
    for buffer in buffers:
        ordinal, count = buffer["ordinal"], buffer["dispatch_count"]
        require(integer(ordinal) and ordinal not in by_buffer and integer(count), "invalid command-buffer identity/count")
        require(buffer["contains_non_dispatch_work"] is False, "unexpected non-dispatch GPU work")
        require(type(buffer["valid"]) is bool, "missing command-buffer validity")
        begin, end = buffer["gpu_begin_seconds"], buffer["gpu_end_seconds"]
        require(finite(begin) and finite(end), "non-finite command-buffer timestamps")
        if count:
            require(buffer["valid"] is True and 0 < begin < end, "invalid nonempty command-buffer interval")
            feedback_ns += (end - begin) * 1e9
        elif buffer["valid"]:
            require(0 < begin < end, "invalid empty command-buffer interval")
        host = [buffer[key] for key in ("host_commit_begin_ns", "host_feedback_begin_ns",
                                        "host_callbacks_end_ns", "host_completion_publish_ns")]
        require(all(integer(value, 1) for value in host) and host == sorted(host), "invalid host boundary ordering")
        require(integer(buffer["host_commit_return_ns"], host[0]), "invalid commit-return observation")
        by_buffer[ordinal] = [count, 0]
    seen, elapsed = set(), []
    for dispatch in dispatches:
        ordinal, owner = dispatch["ordinal"], dispatch["command_buffer_ordinal"]
        require(integer(ordinal) and ordinal not in seen and owner in by_buffer, "invalid dispatch identity/association")
        seen.add(ordinal)
        by_buffer[owner][1] += 1
        for key in ("dispatch_size", "block_size"):
            require(len(dispatch[key]) == 3 and all(integer(v, 1) for v in dispatch[key]), "invalid dispatch geometry")
        require(isinstance(dispatch["shader_checksum"], str) and dispatch["shader_checksum"].isdigit(), "invalid shader checksum")
        begin, end, ns = dispatch["begin_ticks"], dispatch["end_ticks"], dispatch["elapsed_ns"]
        require(integer(begin) and integer(end) and finite(ns), "invalid timestamp values")
        if counters:
            require(dispatch["valid"] is True and 0 < begin < end and ns > 0, "invalid dispatch interval")
            require(math.isclose(ns, (end - begin) * 1e9 / frequency, rel_tol=1e-12, abs_tol=1e-6),
                    "dispatch timestamp conversion mismatch")
            elapsed.append(ns)
        else:
            require(dispatch["valid"] is False and begin == end == ns == 0, "control unexpectedly contains counters")
    require(sum(count for count, _ in by_buffer.values()) == expected and
            all(count == actual for count, actual in by_buffer.values()), "command-buffer dispatch accounting mismatch")
    return statistics.median(elapsed) if counters else feedback_ns / expected


def validate(payload, op, shape, samples, repetitions):
    for key, expected in {"implementation": "tile_xir_metal4", "backend": "metal4", "operation": op,
                          "dimensions": list(shape), "precision": "fp32", "fast_math": False,
                          "relaxed_precision": False, "runtime": "luisa", "repetitions": repetitions,
                          "repetition_policy": "fixed", "timing": "synchronized_host_wall",
                          "batch_policy": "one_runtime_command_list_per_batch"}.items():
        require(payload[key] == expected, f"unexpected benchmark field {key}")
    correctness = payload["correctness"]
    require(correctness["checks"] == 2 and correctness["elements_per_check"] == math.prod(shape) and
            correctness["guard_elements_per_check"] == 34 and finite(correctness["max_abs_error"]),
            "missing complete benchmark oracle/guard checks")
    timing, expected_repetitions = payload["device_timing"], min(repetitions, 64)
    require(timing["method"] == "metal4_precise_dispatch_timestamps_v1" and
            timing["scope"] == "instrumented_dispatch_intervals" and
            timing["host_samples_instrumented"] is False and timing["zero_overhead_kernel_time"] is False and
            timing["repetitions"] == expected_repetitions, "unexpected device timing protocol")
    frequency = timing["capabilities"]["timestamp_frequency_hz"]
    require(timing["capabilities"]["timestamp_heap"] is True and integer(frequency, 1), "timestamp capability unavailable")
    control = timing["control"]
    require(control["method"] == "metal4_commit_feedback_v1" and control["scope"] == "command_buffer_gpu_intervals" and
            control["encoder_instrumentation"] is False and control["repetitions"] == expected_repetitions,
            "unexpected feedback control protocol")
    metrics, sample_ids, dispatch_identities = {}, set(), set()
    for phase, count in (("throughput", expected_repetitions), ("latency", 1)):
        for label, records, counters in (("instrumented_dispatch_ns", timing[phase], True),
                                        ("feedback_only_command_buffer_ns_per_dispatch", control[phase], False)):
            require(len(records) == samples, "wrong device sample count")
            values = []
            for record in records:
                sample_id = record["sample_id"]
                require(integer(sample_id, 1) and sample_id not in sample_ids, "duplicate/invalid sample ID")
                sample_ids.add(sample_id)
                values.append(validate_sample(record, count, counters, frequency))
                dispatch_identities.update((dispatch["shader_checksum"], tuple(dispatch["dispatch_size"]),
                                            tuple(dispatch["block_size"])) for dispatch in record["dispatches"])
            metrics[f"{phase}_{label}"] = {"samples": values, "median": statistics.median(values),
                                            "sample_statistic": "median_dispatch_interval" if counters else
                                            "sum_nonempty_command_buffer_intervals_divided_by_dispatch_count"}
        wall = payload[f"{phase}_us"]
        require(len(wall) == samples and all(finite(value, True) for value in wall), "invalid host timing samples")
        metrics[f"{phase}_host_wall_us_per_dispatch"] = {"samples": wall, "median": statistics.median(wall)}
    require(len(dispatch_identities) == 1, "shader checksum or dispatch/block geometry differs across sampled phases")
    return metrics


def capture(command, environment, timeout, stdout_path, stderr_path):
    """A timeout kills this fresh process group, including any compiler children."""
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        try:
            process = subprocess.Popen(command, env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
        except OSError as error:
            return {"status": "Error", "exit_code": None, "error": str(error)}
        try:
            code = process.wait(timeout=timeout)
            return {"status": "OK" if code == 0 else "Error", "exit_code": code}
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            return {"status": "Interrupted" if isinstance(error, KeyboardInterrupt) else "Timeout",
                    "exit_code": process.returncode, "error": type(error).__name__}


def write_report(path, report):
    temporary = path.with_suffix(".pending.json")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new, nonexistent results directory")
    parser.add_argument("--case", type=parse_case, action="append")
    parser.add_argument("--local-lanes", type=int, nargs="+", default=[1, 32, 0])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if not (1 <= args.rounds <= 1000 and 1 <= args.samples <= 101 and 1 <= args.repetitions <= 100000 and
            finite(args.timeout, True) and all(0 <= value <= 0xffffffff for value in args.local_lanes)):
        parser.error("invalid rounds, samples, repetitions, timeout, or local lanes")
    args.case = args.case or [parse_case(f"{op}:128,1024") for op in ("rmsnorm", "masked_softmax", "swiglu")]
    args.binary = args.binary.resolve()
    args.output = args.output.resolve()
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        parser.error("--binary must name an executable file")
    backends = sorted(path.resolve() for path in args.binary.parent.iterdir()
                      if path.is_file() and path.name in {"luisa-backend-metal4.dylib", "libluisa-backend-metal4.dylib",
                                                        "luisa-backend-metal4.so", "libluisa-backend-metal4.so",
                                                        "luisa-backend-metal4.dll"})
    if len(backends) != 1:
        parser.error("expected exactly one adjacent Metal4 backend library for its SHA256 receipt")
    try:
        args.output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("--output must not already exist")
    tensor_root = Path(tempfile.mkdtemp(prefix="luisa-metal4-timing-tensors-"))
    removed = {key: value for key, value in os.environ.items()
               if key.startswith(("LUISA_TILE_BENCH", "LUISA_METAL"))}
    environment = {key: value for key, value in os.environ.items() if key not in removed}
    declared = {"LUISA_TILE_BENCH_METAL4_TIMING": "1", "LUISA_TILE_BENCH_FIXED_REPETITIONS": str(args.repetitions),
                "LUISA_TILE_BENCH_XIR_BACKEND": "metal4"}
    dependencies = [args.binary.parent / name for name in
                    ("libluisa-tile-bridge-xir.dylib", "libluisa-tile.dylib", "libluisa-xir.dylib",
                     "libluisa-runtime.dylib", "libluisa-core.dylib")]
    artifact_paths = [args.binary, *backends, Path(__file__).resolve(),
                      *(path.resolve() for path in dependencies if path.is_file())]
    artifacts = receipts(artifact_paths)
    report = {"schema": "metal4_serial_timing_v1", "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "platform": platform.platform(), "temporary_tensor_root": str(tensor_root), "tensor_exports_retained": True,
              "artifacts_before": artifacts, "removed_environment": removed, "declared_environment": declared,
              "inherited_diagnostic_environment": {key: environment.get(key) for key in
                                                   ("MTL_DEBUG_LAYER", "MTL_SHADER_VALIDATION", "METAL_DEVICE_WRAPPER_TYPE",
                                                    "LUISA_ENABLE_VALIDATION", "DYLD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES")},
              "protocol": {"serial": True, "reverse_variant_order_on_odd_rounds": True, "rounds": args.rounds,
                           "samples": args.samples, "host_repetitions": args.repetitions,
                           "device_repetitions": min(args.repetitions, 64), "target_ms": 20, "warmup_ms": 10,
                           "timeout_seconds": args.timeout, "zero_overhead_kernel_time": False,
                           "external_caffeinate_required": True, "external_caffeinate_verified": False}, "results": []}
    for round_index in range(args.rounds):
        variants = args.local_lanes if round_index % 2 == 0 else list(reversed(args.local_lanes))
        for op, shape in args.case:
            for lanes in variants:
                report["results"].append({"round": round_index, "operation": op, "dimensions": list(shape),
                                          "requested_local_lanes": lanes, "status": "NotRun"})
    report_path = args.output / "results.json"
    write_report(report_path, report)
    matching_inputs = {}
    for index, row in enumerate(report["results"]):
        op, shape, lanes = row["operation"], row["dimensions"], row["requested_local_lanes"]
        stem = f"{index:04d}-r{row['round']}-{op}-{shape[0]}x{shape[1]}-lanes{lanes}"
        output = tensor_root / f"{stem}.f32"
        stdout, stderr = args.output / f"{stem}.stdout.json", args.output / f"{stem}.stderr.log"
        command = [str(args.binary), "llm", op, ",".join(map(str, shape)), "1", "1", str(args.samples), "20", "10", str(output)]
        overrides = {**declared, "LUISA_TILE_BENCH_XIR_LOCAL_LANES": str(lanes)}
        row.update(command=command, environment=overrides, stdout=stdout.name, stderr=stderr.name,
                   started_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        started = time.monotonic()
        row.update(capture(command, {**environment, **overrides}, args.timeout, stdout, stderr))
        row["process_wall_seconds"] = time.monotonic() - started
        exports = [output, *(Path(str(output) + f".input{i}.f32") for i in range(3))]
        row["tensor_receipts"] = receipts(exports)
        try:
            require(row["status"] == "OK", row.get("error", f"process exited with {row['exit_code']}"))
            payload = json.loads(stdout.read_text())
            row["metrics"] = validate(payload, op, shape, args.samples, args.repetitions)
            require(all(row["tensor_receipts"].values()), "missing tensor export")
            require(output.stat().st_size == math.prod(shape) * 4, "wrong output tensor size")
            require(len(payload["input_shapes"]) == 3, "wrong input shape count")
            for path, dimensions in zip(exports[1:], payload["input_shapes"]):
                require(dimensions and all(integer(value, 1) for value in dimensions) and
                        path.stat().st_size == math.prod(dimensions) * 4, "wrong input tensor size")
            hashes = [row["tensor_receipts"][str(path)]["sha256"] for path in exports[1:]]
            key = op, tuple(shape)
            require(matching_inputs.setdefault(key, hashes) == hashes, "inputs differ between variants/rounds")
            require(receipts(artifact_paths) == artifacts, "benchmark artifacts changed during run")
            row["realization"] = payload["realization"]
            row["correctness"] = payload["correctness"]
        except (ValueError, KeyError, TypeError, OSError, OverflowError) as error:
            if row["status"] == "OK":
                row["status"] = "Error"
            row["error"] = str(error)
            row.pop("metrics", None)
        write_report(report_path, report)
        print(f"{stem}: {row['status']}", flush=True)
        if row["status"] == "Interrupted":
            break
    report["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    report["artifacts_after"] = receipts(artifact_paths)
    report["artifacts_unchanged"] = report["artifacts_after"] == artifacts
    write_report(report_path, report)
    return 0 if report["artifacts_unchanged"] and all(row["status"] == "OK" for row in report["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
