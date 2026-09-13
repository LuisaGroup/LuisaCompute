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

# These helpers import only the standard library; NumPy is loaded lazily for
# the complete oracle, and no Torch/device/build work happens at import time.
# Share the dimensions, causal mask and driver diagnostic contracts rather
# than maintaining a second, subtly different attention implementation.
from compare_llm import gpu_failure_diagnostics, parse_case, reference, shapes_for, validate_output


def require(condition, message):
    if not condition:
        raise ValueError(message)


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


def validate_local_lanes(realization, requested):
    require(isinstance(realization, str) and integer(requested) and requested <= 0xffffffff,
            "invalid local-lane request or realization")
    fields = [field.partition("=") for field in realization.split(";")
              if field.partition("=")[0].strip() == "local_lanes"]
    require(len(fields) == 1, "realization must contain exactly one local_lanes field")
    _, separator, text = fields[0]
    text = text.strip()
    require(separator == "=" and text.isascii() and text.isdecimal(), "invalid realized local_lanes")
    actual = int(text)
    require(0 < actual <= 0xffffffff, "invalid realized local_lanes")
    require(requested == 0 or actual == requested, "realized local_lanes denies explicit request")
    return actual


def valid_block_size(value):
    # This is the uint32 + SIMD-group alignment contract, not a guessed
    # device limit. MetalTileTargetInfo and the created pipeline check the
    # actual device/PSO maximum for an explicit request.
    return integer(value) and value <= 0xffffffff and value % 32 == 0


def validate_block_size(payload, requested):
    require(valid_block_size(requested), "invalid requested block size")
    timing = payload["device_timing"]
    blocks = set()
    for owner in (timing, timing["control"]):
        for phase in ("throughput", "latency"):
            for sample in owner[phase]:
                for dispatch in sample["dispatches"]:
                    block = dispatch["block_size"]
                    require(isinstance(block, list) and len(block) == 3 and
                            all(integer(value, 1) for value in block), "invalid captured block size")
                    blocks.add(tuple(block))
    require(len(blocks) == 1, "captured block sizes differ between sampled phases")
    actual, y, z = next(iter(blocks))
    require(y == z == 1 and actual > 0 and valid_block_size(actual), "expected a 1D SIMD-aligned block")
    require(requested == 0 or actual == requested, "captured block size denies explicit request")
    realization = payload["realization"]
    require(isinstance(realization, str), "missing block realization")
    fields = [field.strip().removesuffix(" threads/group") for field in realization.split(";")
              if field.strip().endswith(" threads/group")]
    require(len(fields) == 1 and fields[0].isascii() and fields[0].isdecimal() and int(fields[0]) == actual,
            "realization threads/group differs from captured block size")
    return actual


def validate_sample(sample, expected, counters, frequency):
    require(integer(expected, 1) and (not counters or integer(frequency, 1)), "invalid timing denominator")
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


def validate(payload, op, shape, samples, repetitions, block=(1, 1)):
    inputs, output = shapes_for(op, shape)
    require(integer(samples, 1) and integer(repetitions, 1), "invalid sample/repetition request")
    dispatch_size = payload["dispatch"]
    require(isinstance(dispatch_size, list) and len(dispatch_size) == 3 and
            all(integer(value, 1) for value in dispatch_size), "invalid benchmark dispatch geometry")
    for key, expected in {"implementation": "tile_xir_metal4", "backend": "metal4", "operation": op,
                          "dimensions": list(shape), "precision": "fp32", "fast_math": False,
                          "relaxed_precision": False, "runtime": "luisa", "repetitions": repetitions,
                          "repetition_policy": "fixed", "timing": "synchronized_host_wall",
                          "batch_policy": "one_runtime_command_list_per_batch",
                          "attention_block": list(block), "input_shapes": [list(s) for s in inputs],
                          "output_shape": list(output)}.items():
        require(type(payload[key]) is type(expected) and payload[key] == expected, f"unexpected benchmark field {key}")
    for dimensions in [payload["dimensions"], payload["attention_block"], payload["output_shape"], *payload["input_shapes"]]:
        require(all(integer(value, 1) for value in dimensions), "noninteger tensor/block dimensions")
    for key in ("attention_qk", "attention_pv"):
        # Current attention captures must explicitly acknowledge their source
        # modes. Preserve old row artifacts that predate these metadata fields.
        if op == "attention" or key in payload:
            require(payload.get(key) == ("mma" if op == "attention" else "not_applicable"),
                    f"unexpected benchmark field {key}")
    correctness = payload["correctness"]
    require(integer(correctness["checks"]) and correctness["checks"] == 2 and
            integer(correctness["elements_per_check"]) and correctness["elements_per_check"] == math.prod(output) and
            integer(correctness["guard_elements_per_check"]) and correctness["guard_elements_per_check"] == 34 and
            correctness["atol"] == correctness["rtol"] == 5e-5 and
            finite(correctness["max_abs_error"]) and correctness["max_abs_error"] >= 0,
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
                require(all(dispatch["dispatch_size"] == dispatch_size for dispatch in record["dispatches"]),
                        "sample dispatch geometry differs from benchmark dispatch")
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


def validate_exports(output, op, shape):
    """Validate every FP32 value against an independently recomputed FP64 oracle.

    The producer's .expected.f64 is intentionally not used as the reference.
    Seven attention dimensions are a problem description, not an output shape.
    """
    import numpy as np
    inputs, output_shape = shapes_for(op, shape)
    paths = [*(Path(str(output) + f".input{i}.f32") for i in range(3)), output]
    arrays = []
    for path, dimensions in zip(paths, [*inputs, output_shape]):
        require(path.is_file() and path.stat().st_size == math.prod(dimensions) * 4,
                f"wrong tensor export size: {path.name}")
        array = np.fromfile(path, dtype=np.float32).reshape(dimensions)
        require(bool(np.isfinite(array).all()), f"non-finite tensor export: {path.name}")
        arrays.append(array)
    expected = reference(op, shape, arrays[:3])
    return validate_output(arrays[3], expected)


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


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new, nonexistent results directory")
    parser.add_argument("--case", type=parse_case, action="append")
    parser.add_argument("--attention-block", type=int, nargs=2, default=(16, 32), metavar=("BQ", "BK"),
                        help="attention tile shape; row kernels always use 1,1")
    parser.add_argument("--local-lanes", type=int, nargs="+", default=[1, 32, 0])
    parser.add_argument("--block-size", type=int, default=0,
                        help="0 keeps automatic grouping; explicit 1D threads/group must be a multiple of 32; device limits are checked by the backend")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args(argv)
    if not (1 <= args.rounds <= 1000 and 1 <= args.samples <= 101 and 1 <= args.repetitions <= 100000 and
            finite(args.timeout, True) and all(0 <= value <= 0xffffffff for value in args.local_lanes)):
        parser.error("invalid rounds, samples, repetitions, timeout, or local lanes")
    if not (1 <= args.attention_block[0] <= 128 and 1 <= args.attention_block[1] <= 256):
        parser.error("invalid attention block")
    if not valid_block_size(args.block_size):
        parser.error("--block-size must be 0 or a uint32 multiple of 32; the backend checks device limits")
    args.case = args.case or [parse_case(f"{op}:128,1024") for op in ("rmsnorm", "masked_softmax", "swiglu")]
    return args


def main():
    args = parse_arguments()
    parser = argparse.ArgumentParser(description=__doc__)
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
    if args.block_size != 0:
        declared["LUISA_TILE_BENCH_XIR_BLOCK_SIZE"] = str(args.block_size)
    dependencies = [args.binary.parent / name for name in
                    ("libluisa-tile-bridge-xir.dylib", "libluisa-tile.dylib", "libluisa-xir.dylib",
                     "libluisa-runtime.dylib", "libluisa-core.dylib")]
    helpers = [Path(__file__).resolve().with_name(name) for name in ("compare_llm.py", "repeat.py", "run.py")]
    artifact_paths = [args.binary, *backends, Path(__file__).resolve(), *helpers,
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
                           "attention_block": list(args.attention_block), "attention_qk": "mma", "attention_pv": "mma",
                           "requested_block_size": args.block_size,
                           "external_caffeinate_required": True, "external_caffeinate_verified": False},
              "cohort_valid": False, "gpu_diagnostics_valid": True, "results": []}
    for round_index in range(args.rounds):
        variants = args.local_lanes if round_index % 2 == 0 else list(reversed(args.local_lanes))
        for op, shape in args.case:
            for lanes in variants:
                report["results"].append({"round": round_index, "operation": op, "dimensions": list(shape),
                                          "requested_local_lanes": lanes, "requested_block_size": args.block_size,
                                          "status": "NotRun", "valid": False})
    report_path = args.output / "results.json"
    write_report(report_path, report)
    matching_inputs = {}
    for index, row in enumerate(report["results"]):
        op, shape, lanes = row["operation"], row["dimensions"], row["requested_local_lanes"]
        block = args.attention_block if op == "attention" else (1, 1)
        stem = f"{index:04d}-r{row['round']}-{op}-{'x'.join(map(str, shape))}-lanes{lanes}"
        output = tensor_root / f"{stem}.f32"
        stdout, stderr = args.output / f"{stem}.stdout.json", args.output / f"{stem}.stderr.log"
        command = [str(args.binary), "llm", op, ",".join(map(str, shape)), *map(str, block), str(args.samples), "20", "10", str(output)]
        overrides = {**declared, "LUISA_TILE_BENCH_XIR_LOCAL_LANES": str(lanes)}
        row.update(command=command, environment=overrides, stdout=stdout.name, stderr=stderr.name,
                   started_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        started = time.monotonic()
        row.update(capture(command, {**environment, **overrides}, args.timeout, stdout, stderr))
        row["process_wall_seconds"] = time.monotonic() - started
        row["gpu_failure_diagnostics"] = gpu_failure_diagnostics(stdout.read_bytes(), stderr.read_bytes())
        gpu_stop = row["status"] in {"Timeout", "Interrupted"} or bool(row["gpu_failure_diagnostics"])
        exports = [output, *(Path(str(output) + f".input{i}.f32") for i in range(3))]
        row["tensor_receipts"] = receipts(exports)
        try:
            require(not row["gpu_failure_diagnostics"], "GPU failure diagnostic; entire cohort invalid")
            require(row["status"] == "OK", row.get("error", f"process exited with {row['exit_code']}"))
            payload = json.loads(stdout.read_text())
            row["actual_local_lanes"] = validate_local_lanes(payload["realization"], lanes)
            row["actual_block_size"] = validate_block_size(payload, args.block_size)
            row["metrics"] = validate(payload, op, shape, args.samples, args.repetitions, block)
            require(all(row["tensor_receipts"].values()), "missing tensor export")
            row["independent_correctness"] = validate_exports(output, op, shape)
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
        if gpu_stop:
            report["gpu_diagnostics_valid"] = False
            report["stop_reason"] = "GPU failure or interrupted/timed-out capture; queue health is not established"
            for remaining in report["results"][index + 1:]:
                remaining["error"] = "not launched: " + report["stop_reason"]
        write_report(report_path, report)
        print(f"{stem}: {row['status']}", flush=True)
        if gpu_stop:
            break
    report["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    report["artifacts_after"] = receipts(artifact_paths)
    report["artifacts_unchanged"] = report["artifacts_after"] == artifacts
    report["cohort_valid"] = report["artifacts_unchanged"] and report["gpu_diagnostics_valid"] and all(row["status"] == "OK" for row in report["results"])
    for row in report["results"]:
        row["valid"] = report["cohort_valid"]
    write_report(report_path, report)
    return 0 if report["cohort_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
