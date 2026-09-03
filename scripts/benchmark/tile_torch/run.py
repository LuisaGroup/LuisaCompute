#!/usr/bin/env python3
"""Compare native TileIR/TVMx with eager PyTorch on matched CPU/Metal inputs.

The native executable must already be built. This script neither builds the
project nor installs dependencies. Timing excludes allocation and transfer,
but includes each framework's host dispatch/binding overhead. It is not a GPU
event timer and must not be presented as pure hardware kernel time.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable


@dataclasses.dataclass(frozen=True)
class Case:
    operation: str
    m: int
    n: int
    k: int = 1

    @property
    def name(self) -> str:
        shape = f"{self.m}x{self.n}"
        return f"{self.operation}_{shape}x{self.k}" if self.operation == "gemm" else f"{self.operation}_{shape}"


def make_cases(operations: list[str], quick: bool = False) -> list[Case]:
    gemm = [(32, 32, 32), (128, 128, 128), (512, 512, 512), (1024, 1024, 1024),
            (256, 1024, 128), (1024, 128, 256), (127, 193, 61), (513, 257, 129)]
    elementwise = [(1, 127), (17, 257), (128, 1024), (4096, 256)]
    reduction = [(1, 127), (17, 257), (128, 1024), (64, 4096)]
    if quick:
        gemm = [gemm[0], gemm[1], gemm[-2]]
        elementwise = elementwise[:2]
        reduction = reduction[:2]
    cases: list[Case] = []
    for operation in operations:
        if operation == "gemm":
            cases.extend(Case(operation, *shape) for shape in gemm)
        elif operation in ("add", "sum", "softmax"):
            cases.extend(Case(operation, *shape) for shape in (elementwise if operation == "add" else reduction))
        else:
            raise ValueError(f"unknown operation {operation!r}")
    return cases


def percentile(values: list[float], fraction: float) -> float:
    if not values or not 0 <= fraction <= 1 or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("percentile requires finite nonnegative samples and a fraction in [0, 1]")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(result: dict[str, Any]) -> None:
    for metric in ("throughput_us", "latency_us"):
        result[metric + "_p50"] = percentile(result[metric], 0.5)
        result[metric + "_p90"] = percentile(result[metric], 0.9)


def time_batch(invoke: Callable[[], Any], synchronize: Callable[[], None], repetitions: int) -> float:
    synchronize()
    start = time.perf_counter_ns()
    for _ in range(repetitions):
        invoke()
    synchronize()
    return (time.perf_counter_ns() - start) / 1e6


def time_torch(invoke: Callable[[], Any], synchronize: Callable[[], None], args: argparse.Namespace) -> dict[str, Any]:
    cold = time_batch(invoke, synchronize, 1)
    start = time.perf_counter_ns()
    while (time.perf_counter_ns() - start) / 1e6 < args.warmup_ms:
        time_batch(invoke, synchronize, 8)
    warmup = (time.perf_counter_ns() - start) / 1e6
    repetitions = 1
    for _ in range(8):
        elapsed = time_batch(invoke, synchronize, repetitions)
        if elapsed >= args.sample_ms * 0.8 or repetitions == 100000:
            break
        repetitions = min(100000, max(repetitions + 1, int(repetitions * args.sample_ms / max(elapsed, 1e-6))))
    throughput = [1000 * time_batch(invoke, synchronize, repetitions) / repetitions for _ in range(args.samples)]
    latency = [1000 * time_batch(invoke, synchronize, 1) for _ in range(args.samples)]
    result = dict(cold_call_ms=cold, warmup_ms=warmup, repetitions=repetitions,
                  throughput_us=throughput, latency_us=latency)
    summarize(result)
    return result


def tolerance(operation: str) -> tuple[float, float]:
    if operation == "add":
        return 0.0, 0.0
    if operation == "gemm":
        return 1e-4, 1e-4
    if operation == "sum":
        return 1e-5, 1e-5
    return 2e-6, 2e-5


def validate(torch: Any, actual: Any, expected: Any, operation: str) -> dict[str, float]:
    if actual.shape != expected.shape or not torch.isfinite(actual).all().item():
        raise AssertionError("output shape mismatch or non-finite result")
    absolute, relative = tolerance(operation)
    difference = (actual.double() - expected).abs()
    bound = absolute + relative * expected.abs()
    if (difference > bound).any().item():
        raise AssertionError(f"{operation}: max absolute error {difference.max().item():.8g} exceeds atol={absolute}, rtol={relative}")
    return {"max_abs_error": difference.max().item(), "atol": absolute, "rtol": relative}


def block_shape(case: Case, gemm_block: tuple[int, int, int]) -> tuple[int, int, int]:
    if case.operation == "gemm":
        return gemm_block
    return (1, 256 if case.operation == "add" else case.n, 1)


def parse_gemm_block(text: str) -> tuple[int, int, int]:
    block = tuple(int(x) for x in text.split(","))
    if len(block) != 3 or min(block) <= 0:
        raise ValueError("GEMM blocks must contain three positive dimensions")
    return block


def tuning_candidates(block: tuple[int, int, int], window: int, blocks: str | None,
                      windows: str | None) -> list[tuple[tuple[int, int, int], int]]:
    if blocks is None and windows is None:
        return []
    shapes = [parse_gemm_block(value) for value in blocks.split(";")] if blocks is not None else [block]
    stages = [int(value) for value in windows.split(",")] if windows is not None else [window]
    if any(value not in (1, 2) for value in stages):
        raise ValueError("tuning pipeline windows must be 1 or 2")
    return list(dict.fromkeys((shape, stage) for shape in shapes for stage in stages))


def validate_native_metadata(native: dict[str, Any], case: Case, backend: str, execution_scope: str,
                             pipeline_window: int = 2, cooperative_matrix: bool = False,
                             gemm_block: tuple[int, int, int] = (8, 8, 16), vectorize: bool = True,
                             auto_vectorize: bool = False, group_threads: int = 0) -> None:
    if native.get("backend") != backend or native.get("operation") != case.operation:
        raise RuntimeError("native backend/operation metadata does not match the request")
    if native.get("execution_scope") != execution_scope:
        raise RuntimeError("native execution-scope metadata does not match the request")
    if native.get("pipeline_window") != pipeline_window:
        raise RuntimeError("native pipeline-window metadata does not match the request")
    if case.operation == "gemm" and native.get("mma_operations") != 1:
        raise RuntimeError("GEMM must contain one semantic TileIR MMA, not a scalar-memory substitute")
    if native.get("cooperative_matrix") is not cooperative_matrix:
        raise RuntimeError("native cooperative-matrix metadata does not match the request")
    if native.get("vectorize") is not vectorize:
        raise RuntimeError("native vectorization metadata does not match the request")
    if native.get("auto_vectorize") is not auto_vectorize:
        raise RuntimeError("native automatic-vectorization metadata does not match the request")
    requested_threads = native.get("planner_threads", 0)
    if type(requested_threads) is not int or requested_threads != group_threads:
        raise RuntimeError("native group-thread constraint does not match the request")
    if group_threads:
        plans = native.get("execution_plans")
        if not isinstance(plans, list) or not plans or any(p.get("threads") != group_threads for p in plans):
            raise RuntimeError("native realized group threads do not match the exact constraint")
    calls = native.get("matrix_intrinsics")
    if type(calls) is not int or calls < 0:
        raise RuntimeError("native matrix-intrinsic count must be a nonnegative integer")
    eligible = (cooperative_matrix and backend == "metal" and execution_scope == "group"
                and case.operation == "gemm" and all(size % 8 == 0 for size in gemm_block)
                and (group_threads == 0 or group_threads >= 32 and group_threads % 32 == 0))
    if bool(calls) != eligible:
        raise RuntimeError("generated matrix-intrinsic calls do not match the benchmark's eligible path")


def run_case(torch: Any, np: Any, args: argparse.Namespace, case: Case, backend: str, ordinal: int) -> dict[str, Any]:
    def inputs(rows: int, columns: int, seed: int) -> Any:
        indices = torch.arange(rows * columns, dtype=torch.int64)
        return (((indices * seed + 17) % 127 - 63).float() / 64).reshape(rows, columns)

    a_host = inputs(case.m, case.k if case.operation == "gemm" else case.n, 5)
    b_host = inputs(case.k if case.operation == "gemm" else case.m, case.n, 11) if case.operation in ("gemm", "add") else None
    if case.operation == "gemm":
        reference = a_host.double() @ b_host.double()
    elif case.operation == "add":
        reference = a_host.double() + b_host.double()
    elif case.operation == "sum":
        reference = a_host.double().sum(dim=1)
    else:
        reference = a_host.double().softmax(dim=1)

    result: dict[str, Any] = {"case": dataclasses.asdict(case), "name": case.name, "backend": backend,
                             "block": block_shape(case, args.gemm_block), "timing_order": "native_first" if ordinal % 2 == 0 else "torch_first"}

    def run_native() -> None:
        with tempfile.TemporaryDirectory(prefix="luisa-tile-benchmark-") as temporary:
            output = Path(temporary) / "output.f32"
            command = [str(args.native), backend, case.operation, str(case.m), str(case.n), str(case.k),
                       *(str(x) for x in result["block"]), str(args.samples), str(args.sample_ms), str(args.warmup_ms), str(output),
                       args.execution_scope, str(args.pipeline_window), "matrix" if args.cooperative_matrix else "scalar",
                       "auto-vectorize" if args.auto_vectorize else "no-vectorize" if args.no_vectorize else "vectorize"]
            group_threads = getattr(args, "group_threads", 0)
            # Preserve compatibility with frozen binaries predating this
            # optional constraint. They receive no new positional argument.
            if group_threads:
                command.append(str(group_threads))
            process = subprocess.run(command, capture_output=True, text=True, check=False, timeout=args.timeout)
            if process.returncode:
                raise RuntimeError(f"native benchmark failed ({process.returncode}):\n{process.stderr}\n{process.stdout}")
            lines = [line for line in process.stdout.splitlines() if line.startswith("{")]
            if len(lines) != 1:
                raise RuntimeError("native executable did not emit exactly one JSON result")
            native = json.loads(lines[0])
            validate_native_metadata(native, case, backend, args.execution_scope, args.pipeline_window,
                                     args.cooperative_matrix, args.gemm_block, not args.no_vectorize, args.auto_vectorize, group_threads)
            array = np.fromfile(output, dtype="<f4")
            if array.size != reference.numel():
                raise RuntimeError("native output byte count is incorrect")
            actual = torch.from_numpy(array).reshape(reference.shape)
            native["correctness"] = validate(torch, actual, reference, case.operation)
            summarize(native)
            result["native"] = native

    def run_pytorch() -> None:
        device = "mps" if backend == "metal" else "cpu"
        synchronize = torch.mps.synchronize if device == "mps" else lambda: None
        synchronize()
        start = time.perf_counter_ns()
        a = a_host.to(device)
        b = b_host.to(device) if b_host is not None else None
        out = torch.empty(reference.shape, dtype=torch.float32, device=device)
        synchronize()
        allocation_upload_ms = (time.perf_counter_ns() - start) / 1e6
        if case.operation == "gemm":
            invoke = lambda: torch.mm(a, b, out=out)
        elif case.operation == "add":
            invoke = lambda: torch.add(a, b, out=out)
        elif case.operation == "sum":
            invoke = lambda: torch.sum(a, dim=1, out=out)
        else:
            invoke = lambda: torch.softmax(a, dim=1, out=out)
        measured = time_torch(invoke, synchronize, args)
        start = time.perf_counter_ns()
        actual = out.cpu()
        synchronize()
        measured["download_ms"] = (time.perf_counter_ns() - start) / 1e6
        measured["allocation_upload_ms"] = allocation_upload_ms
        measured["device"] = str(out.device)
        measured["correctness"] = validate(torch, actual, reference, case.operation)
        result["torch"] = measured

    with torch.inference_mode():
        if ordinal % 2 == 0:
            run_native()
            run_pytorch()
        else:
            run_pytorch()
            run_native()
    result["slowdown"] = result["native"]["throughput_us_p50"] / result["torch"]["throughput_us_p50"]
    result["valid"] = True
    return result


def run_tuned_case(torch: Any, np: Any, args: argparse.Namespace, case: Case,
                   backend: str, ordinal: int) -> dict[str, Any]:
    # Each candidate is ordinary host configuration: recapture and native JIT
    # happen again in run_case. No symbolic super-kernel or capture-once graph.
    candidates = args.tuning_candidates
    shift = ordinal % len(candidates)
    candidates = candidates[shift:] + candidates[:shift]
    trials: list[dict[str, Any]] = []
    start = time.perf_counter_ns()
    for index, (block, window) in enumerate(candidates):
        trial: dict[str, Any] = {"block": block, "pipeline_window": window}
        candidate_args = argparse.Namespace(**vars(args))
        candidate_args.gemm_block, candidate_args.pipeline_window = block, window
        print(f"  JIT trial {index + 1}/{len(candidates)}: block={block}, window={window}", flush=True)
        try:
            measured = run_case(torch, np, candidate_args, case, backend, ordinal * len(candidates) + index)
            score = measured["native"]["throughput_us_p50"]
            if not measured.get("valid") or not math.isfinite(score) or score <= 0:
                raise RuntimeError("candidate lacks a valid positive timing")
            trial.update(valid=True, measurement=measured)
            print(f"    validated; native {score:.3f} us", flush=True)
        except Exception as error:
            trial.update(valid=False, error=str(error))
            print(f"    rejected: {str(error).splitlines()[0]}", flush=True)
        trials.append(trial)
    tuning: dict[str, Any] = {
        "selection_wall_ms": (time.perf_counter_ns() - start) / 1e6,
        "selection_metric": "native_throughput_us_p50",
        "reported_measurement": "fresh post-selection recapture/JIT and timing, not the search minimum",
        "trials": trials,
    }
    valid = [index for index, trial in enumerate(trials) if trial["valid"]]
    failed = {"case": dataclasses.asdict(case), "name": case.name, "backend": backend, "valid": False}
    if not valid:
        return failed | {"error": "no numerically valid JIT candidate", "tuning": tuning}
    selected = min(valid, key=lambda index: trials[index]["measurement"]["native"]["throughput_us_p50"])
    tuning["selected_trial"] = selected
    winner = trials[selected]
    candidate_args = argparse.Namespace(**vars(args))
    candidate_args.gemm_block = winner["block"]
    candidate_args.pipeline_window = winner["pipeline_window"]
    print(f"  Selected block={winner['block']}, window={winner['pipeline_window']}; fresh validation/timing", flush=True)
    try:
        # Keep the fresh comparison's framework order alternating across
        # shapes even when the candidate count is even.
        result = run_case(torch, np, candidate_args, case, backend, ordinal)
    except Exception as error:
        result = failed | {"error": f"selected candidate failed revalidation: {error}"}
    result["tuning"] = tuning
    return result


def write_report(report: dict[str, Any], directory: Path) -> None:
    (directory / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    metadata = report["metadata"]
    lines = ["# TileIR/TVMx vs PyTorch", "", f"Generated: {metadata['timestamp']}", "",
             f"Hardware: {metadata['cpu']}; {metadata['platform']}. PyTorch {metadata['torch_version']}; FP32; {metadata['threads']} CPU threads.", "",
             f"Native root execution request: `{metadata.get('execution_scope', 'auto')}`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.", "",
             f"Native TIRx vectorization: `{metadata.get('vectorize', 'unrecorded')}`; experimental automatic CPU packing: `{metadata.get('auto_vectorize', 'unrecorded')}`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.", "",
             "Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).", "",
             f"Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `{metadata.get('cooperative_matrix', False)}`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Base pipeline window: `{metadata.get('pipeline_window', 'unspecified')}`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.", "",
             "Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.", "",
             "| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["results"]:
        if not row.get("valid"):
            lines.append(f"| {row['backend']} | {row['name']} | FAILED | | | | | | | | |")
            continue
        native, pytorch = row["native"], row["torch"]
        lines.append(f"| {row['backend']} | {row['name']} | {'×'.join(map(str, row['block']))} / {native['pipeline_window']} | {native.get('matrix_intrinsics', 'unrecorded')} | {native['throughput_us_p50']:.3f} | {pytorch['throughput_us_p50']:.3f} | {native['throughput_us_p90']:.3f} | {pytorch['throughput_us_p90']:.3f} | {row['slowdown']:.2f}× | {native['latency_us_p50']:.3f} | {pytorch['latency_us_p50']:.3f} |")
    lines.extend(["", "## Setup and cold-call phases", "", "Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.", "",
                  "| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in report["results"]:
        if row.get("valid"):
            a, b = row["native"], row["torch"]
            lines.append(f"| {row['backend']} / {row['name']} | {a['capture_ms']:.3f} | {a['compile_ms']:.3f} | {a['allocation_upload_ms']:.3f} | {b['allocation_upload_ms']:.3f} | {a['cold_call_ms']:.3f} | {b['cold_call_ms']:.3f} | {a['download_ms']:.3f} | {b['download_ms']:.3f} |")
    tuned = [row for row in report["results"] if "tuning" in row]
    if tuned:
        lines.extend(["", "## JIT search", "",
                      "All candidates are recaptured, compiled, and checked against the same FP64 oracle. Invalid candidates are retained in JSON but cannot win. Candidate order rotates across cases. Tables above use a fresh post-selection run, not the search minimum; a revalidation failure remains a failure. This is not a confidence interval or an exhaustive search.", "",
                      "Selection wall time below includes JIT, validation, native/PyTorch measurements, and process overhead; it is excluded from warm timings. Full candidate settings, rejected cases, and raw trial samples are in results.json.", "",
                      "| Device / case | Valid / attempted candidates | Selection wall ms |", "|---|---:|---:|"])
        for row in tuned:
            tuning = row["tuning"]
            trials = tuning["trials"]
            lines.append(f"| {row['backend']} / {row['name']} | {sum(trial['valid'] for trial in trials)} / {len(trials)} | {tuning['selection_wall_ms']:.3f} |")
    lines.extend(["", "Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).", ""])
    (directory / "results.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True, help="already-built benchmark_tile_tirx executable")
    parser.add_argument("--output", type=Path, required=True, help="new directory for JSON and Markdown results")
    parser.add_argument("--backends", default="cpu,metal")
    parser.add_argument("--operations", default="gemm,add,sum,softmax")
    parser.add_argument("--gemm-block", default="8,8,16")
    parser.add_argument("--tune-gemm-blocks", help="opt-in JIT search, e.g. '8,8,16;16,32,32;32,32,32'; final timing is a fresh run")
    parser.add_argument("--tune-pipeline-windows", help="opt-in JIT windows, e.g. '1,2'; combined with tuning blocks")
    parser.add_argument("--execution-scope", choices=("auto", "worker", "group"), default="auto")
    parser.add_argument("--pipeline-window", type=int, choices=(1, 2), default=2,
                        help="GEMM scheduling window: 1 is ordered, 2 permits software prefetching")
    parser.add_argument("--cooperative-matrix", action="store_true",
                        help="assert native FP32 matrix capability (Metal requires Apple GPU family 7+); default off")
    vectorization = parser.add_mutually_exclusive_group()
    vectorization.add_argument("--no-vectorize", action="store_true", help="disable TIRx vectorization")
    vectorization.add_argument("--auto-vectorize", action="store_true", help="opt in to experimental CPU independent-element SIMD packing; default off")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--group-threads", type=int, default=0,
                        help="exact Metal group worker count; 0 lets the compiler planner choose (not CPU threads)")
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--sample-ms", type=int, default=20)
    parser.add_argument("--warmup-ms", type=int, default=150)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--quick", action="store_true", help="smoke run; omits the large shape cases")
    args = parser.parse_args()
    args.native = args.native.resolve(strict=True)
    args.output = args.output.resolve()
    try:
        args.gemm_block = parse_gemm_block(args.gemm_block)
        args.tuning_candidates = tuning_candidates(args.gemm_block, args.pipeline_window,
                                                  args.tune_gemm_blocks, args.tune_pipeline_windows)
    except ValueError as error:
        parser.error(str(error))
    if min(args.threads, args.samples, args.sample_ms, args.warmup_ms) <= 0:
        parser.error("block dimensions, thread count, and timing parameters must be positive")
    backends = args.backends.split(",")
    if any(backend not in ("cpu", "metal") for backend in backends):
        parser.error("backends must be cpu and/or metal")
    if not 0 <= args.group_threads <= 0xffffffff or (args.group_threads and (backends != ["metal"] or args.execution_scope != "group")):
        parser.error("group threads must be uint32; an explicit count requires only Metal group execution")
    for key in ("TVM_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = str(args.threads)
    # Set the environment before either framework initializes a thread pool.
    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if "metal" in backends and not torch.backends.mps.is_available():
        parser.error("PyTorch MPS is unavailable; CPU fallback is not allowed")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.run(["git", "diff", "--quiet"], cwd=root, check=False).returncode != 0
    cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip() if sys.platform == "darwin" else platform.processor()
    report: dict[str, Any] = {"metadata": {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), "cpu": cpu,
        "platform": platform.platform(), "python": sys.version, "torch_version": torch.__version__,
        "torch_config": torch.__config__.show(), "threads": torch.get_num_threads(),
        "thread_environment": {key: os.environ[key] for key in ("TVM_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        "git_revision": revision, "worktree_dirty": dirty,
        "native_binary": str(args.native), "native_sha256": hashlib.sha256(args.native.read_bytes()).hexdigest(),
        # The bridge is dynamically linked: an unchanged executable hash alone
        # cannot identify its implementation. This is not a full loader trace.
        "adjacent_tile_library_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(args.native.parent.glob("*luisa-tile*"))
            if path.is_file() and path.suffix in (".dylib", ".so", ".dll")
        },
        "samples": args.samples, "sample_ms": args.sample_ms, "warmup_ms": args.warmup_ms,
        "execution_scope": args.execution_scope,
        "pipeline_window": args.pipeline_window,
        "cooperative_matrix": args.cooperative_matrix,
        "vectorize": not args.no_vectorize,
        "auto_vectorize": args.auto_vectorize,
        "group_threads": args.group_threads,
        "gemm_tuning_candidates": [{"block": block, "pipeline_window": window} for block, window in args.tuning_candidates],
        "quick": args.quick, "timing": "synchronized device-resident host wall time including dispatch",
    }, "results": []}
    cases = make_cases(args.operations.split(","), args.quick)
    failed = False
    for backend in backends:
        for case in cases:
            print(f"{backend:5s} {case.name} ...", flush=True)
            try:
                measure = run_tuned_case if case.operation == "gemm" and args.tuning_candidates else run_case
                row = measure(torch, np, args, case, backend, len(report["results"]))
            except Exception as error:
                row = {"name": case.name, "backend": backend, "case": dataclasses.asdict(case), "valid": False, "error": str(error)}
            if row.get("valid"):
                print(f"  validated; native {row['native']['throughput_us_p50']:.3f} us, torch {row['torch']['throughput_us_p50']:.3f} us, ratio {row['slowdown']:.2f}x", flush=True)
            else:
                failed = True
                print(f"  FAILED: {row['error']}", file=sys.stderr, flush=True)
            report["results"].append(row)
            write_report(report, args.output)
    print(f"Results: {args.output / 'results.md'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
