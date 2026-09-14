#!/usr/bin/env python3
"""Top-K/sort screening with independent fixtures and two typed outputs.

Complete the relevant full build before invoking a native route. This driver
never builds. Native failures do not suppress independently generated Torch
baselines. Host-wall measurements include Runtime/framework dispatch; they are
not pure kernel timings. GPU intervals, when available, remain separate.
"""
from __future__ import annotations

import argparse
from array import array
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time


NATIVE_ROUTES = {"xir-simd": "simd", "xir-metal4": "metal4", "tirx-metal": "metal"}
DEFAULT_CASES = ("topk:1,31,1", "topk:17,65,8", "topk:17,65,64",
                 "topk:128,257,1", "topk:128,257,16", "topk:17,1025,1024",
                 "sort:1,31,31", "sort:17,65,65", "sort:128,257,257", "sort:17,1025,1025")


@dataclass(frozen=True)
class Case:
    operation: str
    rows: int
    columns: int
    k: int
    direction: str = "descending"

    def __post_init__(self):
        require(self.operation in {"topk", "sort"}, "unknown operation")
        require(all(type(value) is int for value in (self.rows, self.columns, self.k)), "integer dimensions required")
        require(0 < self.rows <= 65536 and 0 < self.columns <= 65536 and
                self.rows * self.columns <= 2**24, "shape limit")
        require(1 <= self.k <= self.columns and (self.operation != "sort" or self.k == self.columns), "invalid K")
        require(self.direction in {"ascending", "descending"}, "invalid direction")

    @property
    def key(self):
        return f"{self.operation}-{self.rows}x{self.columns}x{self.k}-{self.direction}"

    @property
    def dimensions(self):
        return [self.rows, self.columns, self.k]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_case(text):
    try:
        operation, numbers = text.split(":")
        rows, columns, k = (int(value) for value in numbers.split(","))
        return Case(operation, rows, columns, k)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("expected topk:R,N,K or sort:R,N,N; 1<=K<=N, finite bounded shape") from None


def parse_native(text):
    try:
        route, path = text.split("=", 1)
        require(route in NATIVE_ROUTES and bool(path), "invalid native route")
        return route, str(Path(path).expanduser().resolve())
    except ValueError:
        raise argparse.ArgumentTypeError("expected xir-simd=PATH, xir-metal4=PATH or tirx-metal=PATH") from None


def fixture(case):
    # Integer arithmetic and dyadic floats match the C++ fixture exactly.
    return [(((column * 37 + row * 17) % 31) - 15) * 0.25
            for row in range(case.rows) for column in range(case.columns)]


def typed_bytes(values, code):
    require(code in {"f", "q"}, "unsupported tensor dtype")
    result = array(code, values)
    require(result.itemsize == (4 if code == "f" else 8), "unsupported host typed-array ABI")
    if sys.byteorder != "little":
        result.byteswap()
    return result.tobytes()


def read_typed(path, code, count):
    require(code in {"f", "q"}, "unsupported tensor dtype")
    require(type(count) is int and count >= 0, "invalid tensor element count")
    data = Path(path).read_bytes()
    require(len(data) == count * (4 if code == "f" else 8), "typed output byte count mismatch")
    result = array(code)
    result.frombytes(data)
    if sys.byteorder != "little":
        result.byteswap()
    return result.tolist()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_output(case, inputs, values, indices, *, stable):
    require(type(stable) is bool, "stable policy must be boolean")
    require(len(inputs) == case.rows * case.columns and len(values) == len(indices) == case.rows * case.k,
            "ranking tensor shape mismatch")
    require(all(type(value) in (float, int) and math.isfinite(value) for value in inputs), "nonfinite input")
    descending = case.direction == "descending"
    require(case.direction in {"ascending", "descending"}, "invalid direction")
    for row in range(case.rows):
        source = inputs[row * case.columns:(row + 1) * case.columns]
        output = values[row * case.k:(row + 1) * case.k]
        selected = indices[row * case.k:(row + 1) * case.k]
        require(all(type(index) is int and 0 <= index < case.columns for index in selected), "invalid ranking index")
        require(len(set(selected)) == case.k, "duplicate ranking index")
        require(all(type(value) in (float, int) and math.isfinite(value) for value in output), "nonfinite ranking output")
        require(all(value == source[index] for value, index in zip(output, selected)), "value/index correspondence mismatch")
        expected_indices = sorted(range(case.columns), key=lambda index: ((-source[index] if descending else source[index]), index))[:case.k]
        expected_values = [source[index] for index in expected_indices]
        # Matching sorted values plus unique source indices proves the exact
        # top-K multiset/threshold; arbitrary indices among threshold ties are
        # permitted only for a baseline without the stable-tie contract.
        require(output == expected_values, "incorrect order or top-K threshold/multiplicity")
        if stable:
            require(selected == expected_indices, "stable tie/index order mismatch")
    return dict(elements=case.rows * case.k, exact_values=True, unique_indices=True,
                in_range_indices=True, value_index_correspondence=True, topk_threshold=True,
                stable_ties=stable, max_abs_error=0.0)


def metal4_metrics(device, samples, repetitions):
    """Validate precise dispatch probes and uninstrumented feedback controls."""
    from metal4_timing import integer, validate_sample
    count = min(repetitions, 64)
    require(device.get("scope") == "instrumented_dispatch_intervals" and
            device.get("host_samples_instrumented") is False and
            device.get("zero_overhead_kernel_time") is False and
            type(device.get("repetitions")) is int and device["repetitions"] == count,
            "invalid Metal4 timing protocol")
    capabilities = device["capabilities"]
    frequency = capabilities["timestamp_frequency_hz"]
    require(capabilities["timestamp_heap"] is True and integer(frequency, 1), "invalid Metal4 timestamp capability")
    control = device["control"]
    require(control.get("method") == "metal4_commit_feedback_v1" and
            control.get("scope") == "command_buffer_gpu_intervals" and
            control.get("encoder_instrumentation") is False and
            type(control.get("repetitions")) is int and control["repetitions"] == count,
            "invalid Metal4 control protocol")
    metrics, sample_ids, identities = {}, set(), set()
    for phase, expected, label in (("throughput", count, "batch"), ("latency", 1, "single")):
        for container, counters in ((device, True), (control, False)):
            records = container[phase]
            require(isinstance(records, list) and len(records) == samples, "invalid Metal4 sample count")
            values = []
            for record in records:
                sample_id = record["sample_id"]
                require(integer(sample_id, 1) and sample_id not in sample_ids, "duplicate Metal4 sample identity")
                sample_ids.add(sample_id)
                values.append(validate_sample(record, expected, counters, frequency) / 1000)
                identities.update((dispatch["shader_checksum"], tuple(dispatch["dispatch_size"]), tuple(dispatch["block_size"]))
                                  for dispatch in record["dispatches"])
            prefix = "gpu_instrumented_dispatch" if counters else "gpu_cb"
            metrics[f"{prefix}_{label}_us"] = statistics.median(values)
    require(len(identities) == 1, "Metal4 shader or launch geometry changed within visit")
    return metrics


def timing_metrics(measurement, samples):
    require(isinstance(measurement, dict) and type(samples) is int and samples > 0, "invalid timing record")
    require(measurement.get("timing") == "synchronized_host_wall", "invalid host timing scope")
    repetitions = measurement.get("repetitions")
    require(type(repetitions) is int and 1 <= repetitions <= 100000, "invalid repetition count")
    metrics = {}
    for key in ("throughput_us", "latency_us"):
        values = measurement.get(key)
        require(isinstance(values, list) and len(values) == samples, "missing host timing samples")
        require(all(type(value) in (float, int) and math.isfinite(value) and value > 0 for value in values), "invalid host timing sample")
        metrics["e2e_batch_us" if key == "throughput_us" else "e2e_single_us"] = statistics.median(values)
    if "device_timing" in measurement:
        device = measurement["device_timing"]
        require(isinstance(device, dict), "invalid GPU timing record")
        if device.get("method") == "metal4_precise_dispatch_timestamps_v1":
            metrics.update(metal4_metrics(device, samples, repetitions))
            return metrics
        # Reuse the established legacy Metal interval validator, not a Python
        # wall clock relabeled as a GPU clock. Unknown schemas fail closed.
        from run import summarize_device_timing
        summarize_device_timing(device, samples)
        require(device["repetitions"] == min(repetitions, 64), "GPU/host repetition count mismatch")
        metrics["gpu_instrumented_compute_batch_us"] = device["compute_throughput_us_p50"]
        metrics["gpu_instrumented_compute_single_us"] = device["compute_latency_us_p50"]
        if "control" in device:
            control = device["control"]
            metrics["gpu_cb_batch_us"] = control["command_buffer_throughput_us_p50"]
            metrics["gpu_cb_single_us"] = control["command_buffer_latency_us_p50"]
    return metrics


def native_command(binary, case, args, prefix):
    return [str(binary), "rank", case.operation, str(case.rows), str(case.columns), str(case.k),
            case.direction, str(args.samples), str(args.sample_ms), str(args.warmup_ms), str(prefix)]


def check_native_metadata(measurement, case):
    require(isinstance(measurement, dict), "native measurement must be an object")
    expected = dict(status="passed", operation=case.operation, dimensions=case.dimensions,
                    direction=case.direction, precision="fp32", index_dtype="int64",
                    input_shape=[case.rows, case.columns], output_shape=[case.rows, case.k], stable_ties=True)
    for key, value in expected.items():
        require(type(measurement.get(key)) is type(value) and measurement[key] == value, "native metadata mismatch: " + key)
        if isinstance(value, list):
            require(all(type(item) is int for item in measurement[key]), "invalid native shape type")
    require(isinstance(measurement.get("algorithm"), str) and bool(measurement["algorithm"]), "missing native algorithm metadata")
    correctness = measurement.get("correctness")
    require(isinstance(correctness, dict), "missing native correctness metadata")
    expected_correctness = dict(checks=2, values_per_check=case.rows * case.k,
                                indices_per_check=case.rows * case.k, input_elements_per_check=case.rows * case.columns,
                                guard_elements_per_check=102, values_bitwise_equal=True, indices_exact=True,
                                input_immutable=True, all_guards_intact=True)
    for key, value in expected_correctness.items():
        require(type(correctness.get(key)) is type(value) and correctness[key] == value,
                "native correctness metadata mismatch: " + key)


def native_visit(binary, route, case, inputs, args, directory, environment):
    require(route in NATIVE_ROUTES, "unknown native route")
    prefix = directory / "output"
    command = native_command(binary, case, args, prefix)
    local_env = dict(environment)
    if route.startswith("xir-"):
        local_env["LUISA_TILE_BENCH_XIR_BACKEND"] = NATIVE_ROUTES[route]
    if args.metal_device_timing and route == "tirx-metal":
        local_env["LUISA_TILE_BENCH_METAL_TIMING"] = str(args.metal_device_timing)
    if args.metal4_device_timing and route == "xir-metal4":
        local_env["LUISA_TILE_BENCH_METAL4_TIMING"] = "1"
    source = directory / "kernel.source"
    local_env["LUISA_TILE_BENCH_DUMP_SOURCE"] = str(source)
    recorded_env = {key: value for key, value in local_env.items()
                    if key.startswith("LUISA_") or key in {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"}}
    (directory / "command.json").write_text(json.dumps(dict(argv=command, environment=recorded_env), indent=2) + "\n")
    # Reuse process-group cleanup so a timed-out compiler child cannot keep
    # running and contaminate later baseline measurements.
    from metal4_timing import capture
    completed = capture(command, local_env, args.timeout, directory / "stdout.log", directory / "stderr.log")
    (directory / "process.json").write_text(json.dumps(completed, indent=2, allow_nan=False) + "\n")
    if completed["status"] == "Interrupted":
        raise KeyboardInterrupt
    exported_input = Path(str(prefix) + ".input.f32")
    if exported_input.exists():
        require(exported_input.read_bytes() == typed_bytes(inputs, "f"), "native fixture differs from independent baseline")
    require(completed["status"] == "OK", f"native {completed['status']}, exit {completed.get('exit_code')}, "
            f"error={completed.get('error')}; see process.json/stdout/stderr")
    measurement = json.loads((directory / "stdout.log").read_text())
    check_native_metadata(measurement, case)
    require(measurement.get("backend") == NATIVE_ROUTES[route] and
            measurement.get("implementation") == "tile_" + route.replace("-", "_"), "native lowering route mismatch")
    requested_interval = (bool(args.metal_device_timing) and route == "tirx-metal") or (args.metal4_device_timing and route == "xir-metal4")
    require(not requested_interval or "device_timing" in measurement, "requested native device intervals were not reported")
    if "device_timing" in measurement:
        expected_method = {"tirx-metal": "metal_compute_pass_timestamps_v1", "xir-metal4": "metal4_precise_dispatch_timestamps_v1"}.get(route)
        require(expected_method is not None and isinstance(measurement["device_timing"], dict) and
                measurement["device_timing"].get("method") == expected_method, "device timing method does not match native route")
    for key, suffix in (("input_path", ".input.f32"), ("values_path", ".values.f32"), ("indices_path", ".indices.i64")):
        require(Path(measurement[key]).resolve() == Path(str(prefix) + suffix).resolve(), "native output path mismatch")
    require(exported_input.exists(), "missing native fixture export")
    values = read_typed(Path(measurement["values_path"]), "f", case.rows * case.k)
    indices = read_typed(Path(measurement["indices_path"]), "q", case.rows * case.k)
    check = validate_output(case, inputs, values, indices, stable=True)
    return dict(measurement=measurement, correctness=check, metrics=timing_metrics(measurement, args.samples),
                device_interval=dict(status="OK" if "device_timing" in measurement else "NotRun",
                                     reason=None if "device_timing" in measurement else "native route did not report device intervals"),
                output_sha256=dict(values=digest(Path(measurement["values_path"])), indices=digest(Path(measurement["indices_path"]))),
                source_sha256=digest(source) if source.exists() else None)


class NotAvailable(RuntimeError):
    pass


def torch_visit(device, case, inputs, args, directory):
    # Import only when the requested baseline is actually visited. A broken
    # native compile/export cannot prevent this independent input construction.
    require(device in {"cpu", "mps"}, "unknown Torch device")
    # An MPS result must not silently include CPU fallback, even when the
    # parent shell enabled that mode for unrelated work. main sets this before
    # the lazy import; direct calls reject an already-enabled environment.
    require(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0", "MPS CPU fallback must be disabled")
    import torch
    from run import time_torch, time_metal_device
    if device == "mps" and not torch.backends.mps.is_available():
        raise NotAvailable("Torch MPS unavailable")
    torch.set_num_threads(args.threads)
    sync = torch.mps.synchronize if device == "mps" else lambda: None
    guard = 17
    value_sentinel, index_sentinel = 1234567.0, -(2**62)
    input_storage = torch.full((len(inputs) + 2 * guard,), value_sentinel, device=device, dtype=torch.float32)
    x = input_storage[guard:-guard].view(case.rows, case.columns)
    x.copy_(torch.tensor(inputs, dtype=torch.float32, device=device).view_as(x))
    output_size = case.rows * case.k
    value_storage = torch.full((output_size + 2 * guard,), value_sentinel, device=device, dtype=torch.float32)
    index_storage = torch.full((output_size + 2 * guard,), index_sentinel, device=device, dtype=torch.int64)
    values = value_storage[guard:-guard].view(case.rows, case.k)
    indices = index_storage[guard:-guard].view(case.rows, case.k)
    values.fill_(float("nan"))
    indices.fill_(-1)

    def invoke():
        if case.operation == "topk":
            torch.topk(x, case.k, dim=-1, largest=case.direction == "descending", sorted=True, out=(values, indices))
        else:
            torch.sort(x, dim=-1, descending=case.direction == "descending", stable=True, out=(values, indices))

    def validate():
        sync()
        actual_values = values.flatten().cpu().tolist()
        actual_indices = indices.flatten().cpu().tolist()
        check = validate_output(case, inputs, actual_values, actual_indices, stable=case.operation == "sort")
        require(typed_bytes(x.flatten().cpu().tolist(), "f") == typed_bytes(inputs, "f"), "Torch changed input")
        for storage, sentinel in ((input_storage, value_sentinel), (value_storage, value_sentinel), (index_storage, index_sentinel)):
            require(all(value == sentinel for value in storage[:guard].cpu().tolist() + storage[-guard:].cpu().tolist()), "Torch allocation guard corruption")
        return check, actual_values, actual_indices

    invoke()
    before, _, _ = validate()
    measurement = time_torch(invoke, sync, args)
    measurement.update(timing="synchronized_host_wall", host_scope="Python plus eager Torch dispatch through completion",
                       expression="preallocated_topk_sorted_out" if case.operation == "topk" else "preallocated_stable_sort_out",
                       torch_version=torch.__version__, torch_git_version=torch.version.git_version,
                       device=device, precision="fp32", index_dtype="int64", cpu_threads_requested=args.threads,
                       cpu_threads_actual=torch.get_num_threads(), stable_ties=case.operation == "sort")
    if args.metal_device_timing and device == "mps":
        measurement["device_timing"] = time_metal_device(invoke, sync, args, measurement["repetitions"])
    after, actual_values, actual_indices = validate()
    value_file, index_file = directory / "output.values.f32", directory / "output.indices.i64"
    value_file.write_bytes(typed_bytes(actual_values, "f"))
    index_file.write_bytes(typed_bytes(actual_indices, "q"))
    return dict(measurement=measurement,
                device_interval=dict(status="OK" if "device_timing" in measurement else "NotRun",
                                     reason=None if "device_timing" in measurement else "CPU baseline or no Metal timing helper requested"),
                correctness=dict(before=before, after=after, checks=2,
                guards_per_check=6 * guard, inputs_unchanged=True), metrics=timing_metrics(measurement, args.samples),
                output_sha256=dict(values=digest(value_file), indices=digest(index_file)))


def make_plan(cases, routes, rounds):
    require(cases and routes and type(rounds) is int and rounds > 0, "empty experiment")
    require(len({case.key for case in cases}) == len(cases), "duplicate case")
    require(len(set(routes)) == len(routes), "duplicate route")
    rows = []
    for iteration in range(rounds):
        start = iteration % len(routes)
        order = routes[start:] + routes[:start]
        if (iteration // len(routes)) % 2:
            order = list(reversed(order))
        shift = iteration % len(cases)
        for case in cases[shift:] + cases[:shift]:
            for route in order:
                rows.append(dict(case=case.key, operation=case.operation, dimensions=case.dimensions, direction=case.direction,
                                 route=route, round=iteration, order=order, status="NotRun", reason="not yet visited"))
    return rows


def execute_rows(rows, invoke, persist):
    for row in rows:
        try:
            row.update(invoke(row))
            row.update(status="OK", reason=None)
        except NotAvailable as error:
            row.update(status="NotRun", reason=str(error))
        except KeyboardInterrupt:
            row.update(status="Error", reason="interrupted during this visit")
            persist()
            return False
        except Exception as error:
            row.update(status="Error", reason=f"{type(error).__name__}: {error}")
        persist()
    return True


def make_summary(rows, rounds):
    summary = []
    for key in sorted({(row["case"], row["route"]) for row in rows}):
        selected = [row for row in rows if (row["case"], row["route"]) == key]
        complete = len(selected) == rounds and {row["round"] for row in selected} == set(range(rounds)) and all(row["status"] == "OK" for row in selected)
        item = dict(case=key[0], route=key[1], counts=dict(Counter(row["status"] for row in selected)), complete=complete)
        if complete:
            metrics = set.intersection(*(set(row["metrics"]) for row in selected))
            item["median_us"] = {metric: statistics.median(row["metrics"][metric] for row in selected) for metric in sorted(metrics)}
        summary.append(item)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", action="append", type=parse_native, default=[])
    parser.add_argument("--torch", action="append", choices=("cpu", "mps"), default=[])
    parser.add_argument("--case", action="append", type=parse_case)
    parser.add_argument("--direction", choices=("ascending", "descending", "both"), default="both")
    parser.add_argument("--rounds", type=int, default=0, help="0 uses 2*route count position/precedence-balanced rounds")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--sample-ms", type=float, default=20)
    parser.add_argument("--warmup-ms", type=float, default=100)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--metal-device-timing", type=Path)
    parser.add_argument("--metal4-device-timing", action="store_true", help="request precise Metal4 dispatch probes and feedback-only controls")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    routes = [name for name, _ in args.native] + ["torch-" + device for device in args.torch]
    if not routes or len(set(routes)) != len(routes):
        parser.error("request at least one route, without duplicates")
    if (args.rounds < 0 or not 1 <= args.samples <= 101 or args.threads <= 0 or
            not all(math.isfinite(v) and v > 0 for v in (args.sample_ms, args.warmup_ms, args.timeout)) or
            args.sample_ms > 10000 or args.warmup_ms > 60000):
        parser.error("invalid measurement bounds")
    rounds = args.rounds or 2 * len(routes)
    cases = args.case or [parse_case(text) for text in DEFAULT_CASES]
    directions = ("ascending", "descending") if args.direction == "both" else (args.direction,)
    cases = [Case(case.operation, case.rows, case.columns, case.k, direction) for case in cases for direction in directions]
    try:
        rows = make_plan(cases, routes, rounds)
    except ValueError as error:
        parser.error(str(error))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    args.metal_device_timing = args.metal_device_timing.resolve() if args.metal_device_timing else None
    environment = {key: value for key, value in os.environ.items() if not key.startswith(("LUISA_", "DYLD_"))}
    fixed_env = {key: str(args.threads) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "LUISA_SIMD_WORKER_COUNT")}
    fixed_env["LUISA_SIMD_WARP_WIDTH"] = "8"
    fixed_env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    environment.update(fixed_env)
    for key, value in fixed_env.items():
        os.environ[key] = value
    native = dict(args.native)
    runner_files = [Path(__file__).resolve(), Path(__file__).resolve().with_name("run.py"),
                    Path(__file__).resolve().with_name("metal4_timing.py")]
    source_hashes = {str(path): digest(path) for path in runner_files}
    artifact_paths = set(native.values())
    if args.metal_device_timing:
        artifact_paths.add(str(args.metal_device_timing))
    binaries = {path: digest(Path(path)) for path in artifact_paths if Path(path).is_file()}
    report = dict(format="tile-ranking-v1", status="running", started_unix=time.time(), platform=platform.platform(),
                  routes=routes, rounds=rounds, samples=args.samples, sample_ms=args.sample_ms, warmup_ms=args.warmup_ms,
                  requested_threads=args.threads, fixed_environment=fixed_env, source_sha256=source_hashes,
                  metal4_device_timing_requested=args.metal4_device_timing,
                  metal_timing_helper=str(args.metal_device_timing) if args.metal_device_timing else None,
                  binary_sha256=binaries, selection="fixed cases, no timing-based winner selection",
                  identity_coverage="runner sources, requested native executables and optional Metal timing helper only; no full dynamic-library/build closure attestation",
                  timing_scope="Warm synchronized E2E, not pure kernel time; optional GPU intervals are separately sampled and instrumented/control-labeled",
                  fixture="float(((column*37 + row*17)%31)-15)*0.25; finite negative values and repeated exact ties",
                  comparison="Tile stable ties; Torch sort stable; Torch topk may choose different tied indices but must satisfy sorted values, uniqueness, range, source correspondence and exact threshold multiplicity",
                  position_precedence_balanced=rounds % (2 * len(routes)) == 0, results=rows)

    def persist():
        report["summary"] = make_summary(rows, rounds)
        temporary = output / "results.json.partial"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "results.json")

    persist()
    by_key = {case.key: case for case in cases}
    arrays = {case.key: fixture(case) for case in cases}
    for case in cases:
        path = output / (case.key + ".input.f32")
        path.write_bytes(typed_bytes(arrays[case.key], "f"))
    report["input_sha256"] = {case.key: digest(output / (case.key + ".input.f32")) for case in cases}

    def invoke(row):
        case = by_key[row["case"]]
        directory = output / f"{case.key}-r{row['round']}-{row['route']}"
        directory.mkdir()
        if row["route"].startswith("torch-"):
            return torch_visit(row["route"].removeprefix("torch-"), case, arrays[case.key], args, directory)
        return native_visit(native[row["route"]], row["route"], case, arrays[case.key], args, directory, environment)

    completed = execute_rows(rows, invoke, persist)
    report["source_unchanged"] = all(digest(Path(path)) == expected for path, expected in source_hashes.items())
    report["binary_unchanged"] = all(digest(Path(path)) == expected for path, expected in binaries.items())
    report["status"] = "completed" if completed else "interrupted"
    report["finished_unix"] = time.time()
    report["counts"] = dict(Counter(row["status"] for row in rows))
    persist()
    print(json.dumps(dict(report=str(output / "results.json"), status=report["status"], counts=report["counts"])))
    return 0 if completed and all(row["status"] == "OK" for row in rows) and report["source_unchanged"] and report["binary_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
