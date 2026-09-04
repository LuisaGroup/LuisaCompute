#!/usr/bin/env python3
"""Replay frozen GEMM schedules against eager PyTorch and direct BLAS/MPS.

Six rounds balance all three implementation orders for every shape. No tuning
or selection by timing takes place; failed rows are retained and fail the run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

from repeat import load_plan
from run import Case, percentile, run_case


def order_for_round(keys: list[Any], round_index: int):
    shift = round_index % len(keys)
    for key in keys[shift:] + keys[:shift]:
        yield key, (keys.index(key) + round_index) % 6


def write_report(report: dict[str, Any], output: Path) -> None:
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# TileIR/TVMx, PyTorch, and direct BLAS/MPS GEMM", "",
             "Compact row-major FP32 C=A×B, alpha=1, beta=0, no transpose or prepacking. All three full outputs use the same deterministic inputs and FP64 oracle (atol=rtol=1e-4).", "",
             "Native schedules are frozen from the supplied reports; their historical timing scores are not used. Fresh JIT, setup and upload are excluded. Warm times include host API/encoding/submission overhead and synchronization; these are not GPU hardware-event times. MPSMatrixMultiplication uses private buffers, MPSKernelOptionsNone, and one command buffer per batch; PyTorch uses its default eager MPS path, not a claim of the same internal MPS kernel.", "",
             "CPU uses Accelerate's classic LP64 cblas_sgemm. Thread environment is recorded, not a measurement of actual library worker counts. The runner clears optional PyTorch MPS fast-math, prefer-Metal and fallback overrides before importing PyTorch.", "",
             "Every shape gets all six implementation orders, and case order rotates. Values are medians of per-round p50s in microseconds. Ratios are paired round medians [min, max], not confidence intervals. Native/system >1 means Tile is slower. No failing or slow round is removed.", "",
             "| Backend / shape M×N×K | Valid rounds | Tile µs | PyTorch µs | BLAS/MPS µs | Tile / BLAS or MPS [range] |",
             "|---|---:|---:|---:|---:|---:|"]
    keys = list(dict.fromkeys((r["backend"], r["name"]) for r in report["results"]))
    for backend, name in keys:
        rows = [r for r in report["results"] if (r["backend"], r["name"]) == (backend, name)]
        valid = [r for r in rows if r.get("valid")]
        count = f"{len(valid)}/{report['metadata']['rounds']}"
        if len(valid) != report["metadata"]["rounds"]:
            lines.append(f"| {backend} / {name} | {count} | INCOMPLETE | — | — | — |")
            continue
        times = {implementation: percentile([r[implementation]["throughput_us_p50"] for r in valid], 0.5)
                 for implementation in ("native", "torch", "system")}
        ratios = [r["system_slowdown"] for r in valid]
        lines.append(f"| {backend} / {name} | {count} | {times['native']:.3f} | {times['torch']:.3f} | {times['system']:.3f} | "
                     f"{percentile(ratios, 0.5):.3f}× [{min(ratios):.3f}, {max(ratios):.3f}] |")
    failures = sum(not r.get("valid") for r in report["results"])
    lines.extend(["", f"Failed measurements: {failures}. Binary stability check: {report['metadata'].get('artifacts_unchanged', 'pending')}.", "",
                  "Raw timing/latency samples, cold phases, all correctness errors, schedules, device/API identities, compiler versions and binary hashes: [results.json](results.json).", ""])
    (output / "results.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="append", type=Path, required=True, help="frozen run.py report; repeat to combine CPU/Metal")
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--system-baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--sample-ms", type=int, default=40)
    parser.add_argument("--warmup-ms", type=int, default=200)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if args.rounds <= 0 or args.rounds % 6 or min(args.samples, args.sample_ms, args.warmup_ms, args.threads, args.timeout) <= 0:
        parser.error("use a positive multiple of six rounds and positive timing/thread settings")
    args.native = args.native.resolve(strict=True)
    args.system_baseline = args.system_baseline.resolve(strict=True)
    plan = {}
    try:
        for path in args.plan:
            added = load_plan(path, {"gemm"})
            if any(backend not in ("cpu", "metal") for backend, _ in added):
                raise ValueError("only CPU/Metal GEMM is supported")
            if plan.keys() & added.keys():
                raise ValueError("plans contain overlapping cases")
            plan.update(added)
    except (KeyError, ValueError) as error:
        parser.error(str(error))
    for key in ("TVM_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = str(args.threads)
    overrides = ("PYTORCH_MPS_FAST_MATH", "PYTORCH_MPS_PREFER_METAL", "PYTORCH_ENABLE_MPS_FALLBACK")
    prior_overrides = {key: os.environ.pop(key, None) for key in overrides}
    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    keys = sorted(plan, key=lambda key: (key[1], key[0]))
    if any(backend == "metal" for backend, _ in keys) and not torch.backends.mps.is_available():
        parser.error("MPS unavailable; CPU fallback is not allowed")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    artifacts = [args.native, args.system_baseline, *[p for p in sorted(args.native.parent.glob("*luisa-tile*"))
                 if p.is_file() and p.suffix in (".dylib", ".so", ".dll")]]
    hashes = {str(path): digest(path) for path in artifacts}
    report = {"metadata": {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), "platform": platform.platform(),
        "torch_version": torch.__version__, "torch_config": torch.__config__.show(),
        "torch_git_version": torch.version.git_version, "threads": torch.get_num_threads(),
        "thread_environment": {key: os.environ[key] for key in ("TVM_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        "removed_mps_overrides": prior_overrides,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "worktree_dirty": subprocess.run(["git", "diff", "--quiet"], cwd=root).returncode != 0,
        "artifacts_sha256": hashes,
        "plan_sources": [{"path": str(path.resolve()), "sha256": digest(path)} for path in args.plan],
        "rounds": args.rounds, "samples": args.samples, "sample_ms": args.sample_ms, "warmup_ms": args.warmup_ms,
        "timing": "synchronized device-resident host wall including dispatch, not GPU-event time",
    }, "results": []}
    failed = False
    for round_index in range(args.rounds):
        for (backend, name), ordinal in order_for_round(keys, round_index):
            config = plan[backend, name]
            case = Case(**config["case"])
            invocation = argparse.Namespace(**(vars(args) | config))
            print(f"round {round_index + 1}/{args.rounds} {backend} {name} ...", flush=True)
            try:
                row = run_case(torch, np, invocation, case, backend, ordinal)
            except Exception as error:
                row = {"backend": backend, "name": name, "case": config["case"], "valid": False, "error": str(error)}
            row["round"] = round_index
            if row.get("valid"):
                print("  validated; " + ", ".join(f"{impl} {row[impl]['throughput_us_p50']:.3f} us" for impl in ("native", "torch", "system")), flush=True)
            else:
                failed = True
                print(f"  FAILED: {row['error']}", flush=True)
            report["results"].append(row)
            write_report(report, args.output)
    report["metadata"]["artifacts_unchanged"] = all(path.is_file() and digest(path) == hashes[str(path)] for path in artifacts)
    write_report(report, args.output)
    print(f"Results: {args.output / 'results.md'}", flush=True)
    return int(failed or not report["metadata"]["artifacts_unchanged"])


if __name__ == "__main__":
    raise SystemExit(main())
