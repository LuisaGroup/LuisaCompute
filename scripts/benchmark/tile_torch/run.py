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
                       *(str(x) for x in result["block"]), str(args.samples), str(args.sample_ms), str(args.warmup_ms), str(output)]
            process = subprocess.run(command, capture_output=True, text=True, check=False, timeout=args.timeout)
            if process.returncode:
                raise RuntimeError(f"native benchmark failed ({process.returncode}):\n{process.stderr}\n{process.stdout}")
            lines = [line for line in process.stdout.splitlines() if line.startswith("{")]
            if len(lines) != 1:
                raise RuntimeError("native executable did not emit exactly one JSON result")
            native = json.loads(lines[0])
            if native["backend"] != backend or native["operation"] != case.operation:
                raise RuntimeError("native backend/operation metadata does not match the request")
            if case.operation == "gemm" and native["mma_operations"] != 1:
                raise RuntimeError("GEMM must contain one semantic TileIR MMA, not a scalar-memory substitute")
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


def write_report(report: dict[str, Any], directory: Path) -> None:
    (directory / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    metadata = report["metadata"]
    lines = ["# TileIR/TVMx vs PyTorch", "", f"Generated: {metadata['timestamp']}", "",
             f"Hardware: {metadata['cpu']}; {metadata['platform']}. PyTorch {metadata['torch_version']}; FP32; {metadata['threads']} CPU threads.", "",
             "Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).", "",
             "Native GEMM retains an MMA in TileIR, but the current TVMx schedule lowers it to loops: no tensor-core claim. Pipeline stage markers currently run serially. Sort is not included in this performance comparison.", "",
             "Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.", "",
             "| Device | Operator / M×N[×K] | Block | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["results"]:
        if not row.get("valid"):
            lines.append(f"| {row['backend']} | {row['name']} | FAILED | | | | | | | |")
            continue
        native, pytorch = row["native"], row["torch"]
        lines.append(f"| {row['backend']} | {row['name']} | {'×'.join(map(str, row['block']))} | {native['throughput_us_p50']:.3f} | {pytorch['throughput_us_p50']:.3f} | {native['throughput_us_p90']:.3f} | {pytorch['throughput_us_p90']:.3f} | {row['slowdown']:.2f}× | {native['latency_us_p50']:.3f} | {pytorch['latency_us_p50']:.3f} |")
    lines.extend(["", "## Setup and cold-call phases", "", "Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.", "",
                  "| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in report["results"]:
        if row.get("valid"):
            a, b = row["native"], row["torch"]
            lines.append(f"| {row['backend']} / {row['name']} | {a['capture_ms']:.3f} | {a['compile_ms']:.3f} | {a['allocation_upload_ms']:.3f} | {b['allocation_upload_ms']:.3f} | {a['cold_call_ms']:.3f} | {b['cold_call_ms']:.3f} | {a['download_ms']:.3f} | {b['download_ms']:.3f} |")
    lines.extend(["", "Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).", ""])
    (directory / "results.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True, help="already-built benchmark_tile_tirx executable")
    parser.add_argument("--output", type=Path, required=True, help="new directory for JSON and Markdown results")
    parser.add_argument("--backends", default="cpu,metal")
    parser.add_argument("--operations", default="gemm,add,sum,softmax")
    parser.add_argument("--gemm-block", default="8,8,16")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--sample-ms", type=int, default=20)
    parser.add_argument("--warmup-ms", type=int, default=150)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--quick", action="store_true", help="smoke run; omits the large shape cases")
    args = parser.parse_args()
    args.native = args.native.resolve(strict=True)
    args.output = args.output.resolve()
    args.gemm_block = tuple(int(x) for x in args.gemm_block.split(","))
    if len(args.gemm_block) != 3 or min(args.gemm_block) <= 0 or min(args.threads, args.samples, args.sample_ms, args.warmup_ms) <= 0:
        parser.error("block dimensions, thread count, and timing parameters must be positive")
    backends = args.backends.split(",")
    if any(backend not in ("cpu", "metal") for backend in backends):
        parser.error("backends must be cpu and/or metal")
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
        "samples": args.samples, "sample_ms": args.sample_ms, "warmup_ms": args.warmup_ms,
        "quick": args.quick, "timing": "synchronized device-resident host wall time including dispatch",
    }, "results": []}
    cases = make_cases(args.operations.split(","), args.quick)
    failed = False
    for backend in backends:
        for case in cases:
            print(f"{backend:5s} {case.name} ...", flush=True)
            try:
                row = run_case(torch, np, args, case, backend, len(report["results"]))
                print(f"  validated; native {row['native']['throughput_us_p50']:.3f} us, torch {row['torch']['throughput_us_p50']:.3f} us, ratio {row['slowdown']:.2f}x", flush=True)
            except Exception as error:
                failed = True
                row = {"name": case.name, "backend": backend, "case": dataclasses.asdict(case), "valid": False, "error": str(error)}
                print(f"  FAILED: {error}", file=sys.stderr, flush=True)
            report["results"].append(row)
            write_report(report, args.output)
    print(f"Results: {args.output / 'results.md'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
