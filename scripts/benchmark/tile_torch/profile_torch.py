#!/usr/bin/env python3
"""A single, correctness-checked eager GEMM workload for an external profiler.

The warm interval is deliberately long enough to sample. Its wall time is
profiling metadata, not a replacement for the uninstrumented benchmark.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import time

from run import validate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "metal"), required=True)
    parser.add_argument("--shape", default="1024,1024,1024", help="M,N,K")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--mps-path", choices=("default", "metal"), default="default")
    parser.add_argument("--signposts", action="store_true")
    parser.add_argument("--capture-dir", type=Path, help="new directory for a warmed Metal capture")
    args = parser.parse_args()
    try:
        m, n, k = (int(v) for v in args.shape.split(","))
    except ValueError:
        parser.error("shape must be M,N,K")
    if not math.isfinite(args.seconds) or min(m, n, k, args.seconds, args.threads, args.batch) <= 0:
        parser.error("shape, duration, threads, and batch must be positive")
    if args.backend != "metal" and (args.signposts or args.capture_dir or args.mps_path != "default"):
        parser.error("MPS profiling options require --backend metal")
    if args.capture_dir and args.capture_dir.exists():
        parser.error("Metal capture path already exists")
    for key in ("TVM_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = str(args.threads)
    # Both variants are explicit; an inherited preference must not turn the
    # default-path experiment into the fallback-path experiment accidentally.
    os.environ.pop("PYTORCH_MPS_PREFER_METAL", None)
    if args.mps_path == "metal":
        os.environ["PYTORCH_MPS_PREFER_METAL"] = "1"
    if args.capture_dir:
        os.environ["MTL_CAPTURE_ENABLED"] = "1"

    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.backend == "metal" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable; CPU fallback is not allowed")
    device = "mps" if args.backend == "metal" else "cpu"
    synchronize = torch.mps.synchronize if device == "mps" else lambda: None

    def inputs(rows: int, columns: int, seed: int):
        indices = torch.arange(rows * columns, dtype=torch.int64)
        return (((indices * seed + 17) % 127 - 63).float() / 64).reshape(rows, columns)

    with torch.inference_mode():
        a_host, b_host = inputs(m, k, 5), inputs(k, n, 11)
        reference = a_host.double() @ b_host.double()
        a, b = a_host.to(device), b_host.to(device)
        out = torch.empty((m, n), dtype=torch.float32, device=device)
        for _ in range(8):
            torch.mm(a, b, out=out)
        synchronize()
        validate(torch, out.cpu(), reference, "gemm")
        captures = []
        if args.capture_dir:
            directory = args.capture_dir.resolve()
            directory.mkdir(parents=True, exist_ok=False)
            # PyTorch prefixes a capture counter and appends .gputrace; its
            # argument is a basename, despite looking like a filesystem path.
            with contextlib.chdir(directory):
                with torch.mps.profiler.metal_capture("gemm"):
                    torch.mm(a, b, out=out)
                    synchronize()
            captures = [str(p) for p in sorted(directory.glob("*.gputrace"))]
            if not captures:
                raise RuntimeError("Metal capture completed without producing a .gputrace")
        profiling = torch.mps.profiler.profile() if args.signposts else contextlib.nullcontext()
        started = time.perf_counter_ns()
        count = 0
        with profiling:
            while (time.perf_counter_ns() - started) / 1e9 < args.seconds:
                for _ in range(args.batch):
                    torch.mm(a, b, out=out)
                synchronize()
                count += args.batch
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        correctness = validate(torch, out.cpu(), reference, "gemm")
    print(json.dumps({
        "backend": args.backend, "shape_mnk": [m, n, k], "threads": args.threads,
        "torch_version": torch.__version__, "torch_git_version": torch.version.git_version,
        "torch_config": torch.__config__.show(), "mps_path": args.mps_path,
        "signposts": args.signposts, "iterations": count, "profile_wall_ms": elapsed_ms,
        "metal_captures": captures,
        "correctness": correctness,
        "measurement": "profiled workload, not an uninstrumented performance result",
    }, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
