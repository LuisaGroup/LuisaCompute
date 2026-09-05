#!/usr/bin/env python3
"""Balanced XIR/SIMD planner pilot against a fixed execution map and eager Torch.

This is not the TIRx benchmark. Both XIR paths use the same Tile program.
By default compare planned/canonical maps; --baseline instead compares two
frozen compilers with the same planned map. No timing-based tuning.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess

from repeat import artifact_hashes
from run import summarize, time_torch


def positive_triple(text: str) -> tuple[int, int, int]:
    try:
        values = tuple(int(v) for v in text.split(","))
        if len(values) != 3 or min(values) <= 0:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError("expected three positive integers separated by commas") from None
    return values


def check_metadata(row: dict, shape: tuple[int, int, int], policy: str, samples: int,
                   block: tuple[int, int, int] = (1, 1, 8)) -> None:
    expected = {"implementation": "tile_xir_simd", "backend": "cpu", "precision": "fp32",
                "fast_math": False, "relaxed_precision": False, "block": list(block),
                "m": shape[0], "n": shape[1], "k": shape[2], "planner_policy": policy,
                "timing": "synchronized_host_wall", "batch_policy": "one_runtime_command_list_per_batch"}
    if any(type(row.get(k)) is not type(v) or row[k] != v for k, v in expected.items()):
        raise ValueError("XIR benchmark metadata mismatch")
    realization = row.get("realization", "")
    if "XIR SSA" not in realization or "uncalibrated cost" not in realization or "W8" not in realization:
        raise ValueError("missing XIR realization/cost/packet identity")
    if policy == "canonical" and ("64 workers/block" not in realization or "root order [0,1]" not in realization):
        raise ValueError("fixed plan was not honored")
    if type(row.get("repetitions")) is not int or not 1 <= row["repetitions"] <= 100000:
        raise ValueError("invalid repetition count")
    for metric in ("throughput_us", "latency_us"):
        values = row.get(metric)
        if not isinstance(values, list) or len(values) != samples or any(not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("invalid timing samples")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--baseline", type=Path, help="frozen prior executable and its adjacent backend libraries")
    parser.add_argument("--shape", type=positive_triple, action="append", help="repeatable M,N,K; defaults to the original three cases")
    parser.add_argument("--block", type=positive_triple, default=(1, 1, 8), help="fixed Tile M,N,K for both XIR paths")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compiler-artifact", action="append", type=Path, default=[])
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--sample-ms", type=int, default=20)
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.rounds <= 0 or args.rounds % 6 or min(args.samples, args.sample_ms, args.warmup_ms, args.threads) <= 0:
        parser.error("positive settings and a multiple of six rounds required")
    args.native = args.native.resolve(strict=True)
    if args.baseline:
        args.baseline = args.baseline.resolve(strict=True)
        if args.baseline == args.native:
            parser.error("baseline and candidate must be different executable paths")
    shapes = args.shape or [(32, 32, 32), (128, 128, 128), (127, 193, 61)]
    if len(set(shapes)) != len(shapes) or any(max(shape) > 16384 for shape in shapes) or max(args.block) > 64:
        parser.error("duplicate/oversized shapes or oversized Tile block")
    binaries = [args.native] + ([args.baseline] if args.baseline else [])
    control = "baseline" if args.baseline else "canonical"
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    for key in ("TVM_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "LUISA_SIMD_WORKER_COUNT"):
        os.environ[key] = str(args.threads)
    removed = {}
    for key in list(os.environ):
        if key.startswith("LUISA_SIMD_") and key != "LUISA_SIMD_WORKER_COUNT" or key in ("LUISA_ENABLE_VALIDATION", "DYLD_PRINT_LIBRARIES", "LUISA_TILE_BENCH_DUMP_SOURCE"):
            removed[key] = os.environ.pop(key)
    os.environ["LUISA_SIMD_WARP_WIDTH"] = "8"
    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    hashes = artifact_hashes(binaries, args.compiler_artifact)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    root = Path(__file__).resolve().parents[3]
    report = {"metadata": {"timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), "platform": platform.platform(),
                            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                            "worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root)),
                            "torch_version": torch.__version__, "torch_git_version": torch.version.git_version,
                            "torch_config": torch.__config__.show(), "numpy_version": np.__version__,
                            "rounds": args.rounds, "samples": args.samples, "sample_ms": args.sample_ms, "warmup_ms": args.warmup_ms,
                            "requested_threads": args.threads, "removed_environment": removed,
                            "shapes": shapes, "block": args.block, "control": control,
                            "artifacts_sha256": hashes, "script_sha256": digest(Path(__file__)),
                            "timing": "warm synchronized host wall including each path's dispatch; no JIT/allocation/upload"}, "results": []}
    orders = list(itertools.permutations(("planned", control, "torch")))
    failed = False
    for round_index in range(args.rounds):
        shift = round_index % len(shapes)
        for shape in shapes[shift:] + shapes[:shift]:
            m, n, k = shape
            values = lambda count, seed: (((np.arange(count, dtype=np.int64) * seed + 17) % 127 - 63) / 64).astype(np.float32)
            a = values(m * k, 5).reshape(m, k)
            b = values(k * n, 11).reshape(k, n)
            expected = a.astype(np.float64) @ b.astype(np.float64)
            ta, tb, tc = torch.from_numpy(a), torch.from_numpy(b), torch.empty((m, n), dtype=torch.float32)
            order = orders[(round_index + shapes.index(shape)) % 6]
            for policy in order:
                stem = f"r{round_index}-{m}x{n}x{k}-{policy}"
                row = {"round": round_index, "shape": list(shape), "policy": policy, "order": list(order), "valid": False}
                output = args.output / f"{stem}.f32"
                try:
                    if policy == "torch":
                        result = time_torch(lambda: torch.mm(ta, tb, out=tc), lambda: None, args)
                        actual = tc.numpy().copy()
                        actual.tofile(output)
                    else:
                        source = args.output / f"{stem}.ll"
                        binary = args.baseline if policy == "baseline" else args.native
                        requested_policy = "planned" if policy == "baseline" else policy
                        command = [str(binary), "fp32", str(m), str(n), str(k), str(args.samples), str(args.sample_ms),
                                   str(args.warmup_ms), str(output), *map(str, args.block), requested_policy]
                        if policy == "canonical":
                            command.append("64")
                        row["command"] = command
                        completed = subprocess.run(command, env=os.environ | {"LUISA_TILE_BENCH_DUMP_SOURCE": str(source)},
                                                   capture_output=True, text=True, timeout=180)
                        (args.output / f"{stem}.log").write_text(completed.stdout + completed.stderr)
                        completed.check_returncode()
                        result = json.loads(completed.stdout)
                        check_metadata(result, shape, requested_policy, args.samples, args.block)
                        summarize(result)
                        row["source"] = source.name
                        row["source_sha256"] = digest(source)
                        actual = np.fromfile(output, dtype=np.float32).reshape(m, n)
                    difference = np.abs(actual.astype(np.float64) - expected)
                    if not np.isfinite(actual).all() or not np.all(difference <= 1e-4 + 1e-4 * np.abs(expected)):
                        raise ValueError("full FP64 oracle mismatch")
                    row.update(valid=True, measurement=result, output=output.name, output_sha256=digest(output),
                               checked_elements=m * n, max_abs_error=float(difference.max()), atol=1e-4, rtol=1e-4)
                except Exception as error:
                    row["error"] = str(error)
                    failed = True
                report["results"].append(row)
                (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print(stem, "PASS" if row["valid"] else row["error"], flush=True)
    after = artifact_hashes(binaries, args.compiler_artifact)
    report["metadata"]["artifacts_unchanged"] = hashes == after
    failed |= hashes != after
    report["summary"] = []
    for shape in shapes:
        rows = [r for r in report["results"] if r["shape"] == list(shape)]
        if not all(r["valid"] for r in rows):
            continue
        times = {p: statistics.median([r["measurement"]["throughput_us_p50"] for r in rows if r["policy"] == p]) for p in orders[0]}
        ratios = []
        for index in range(args.rounds):
            paired = {r["policy"]: r["measurement"]["throughput_us_p50"] for r in rows if r["round"] == index}
            ratios.append(paired["planned"] / paired[control])
        report["summary"].append({"shape": list(shape), "median_us": times,
                                  "control": control,
                                  "paired_planned_over_fixed": {"median": statistics.median(ratios), "min": min(ratios), "max": max(ratios)},
                                  "planned_slower_rounds": sum(v > 1 for v in ratios)})
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
