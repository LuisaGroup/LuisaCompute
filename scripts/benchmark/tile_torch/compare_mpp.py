#!/usr/bin/env python3
"""Search native MPP schedules, then replay frozen choices against direct MPS.

No builds, dependency installation, or timing-based selection in replay mode.
FP32 only: alpha=1, beta=0, compact row-major, no relaxed precision or fast math.
Every subprocess output is compared in full against a NumPy FP64 oracle.
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
import subprocess
import tempfile
from typing import Any

from run import percentile


DEFAULT_CONFIG = (64, 64, 4, 1, 0, 1, 4, 1)
DEFAULT_SHAPES = [(32, 32, 32), (128, 128, 128), (512, 512, 512), (1024, 1024, 1024),
                  (256, 1024, 128), (1024, 128, 256), (127, 193, 61), (513, 257, 129)]
METRICS = ("throughput_us", "latency_us", "gpu_throughput_us", "gpu_latency_us")


def parse_shape(text: str) -> tuple[int, int, int]:
    values = tuple(int(x) for x in text.replace("x", ",").split(","))
    if len(values) != 3 or min(values) <= 0 or max(values) > 2**31 - 1:
        raise ValueError("shape must be positive int32 M,N,K")
    return values


def parse_config(text: str) -> tuple[int, ...]:
    values = tuple(int(x) for x in text.split(","))
    if len(values) == 6:
        values += (values[2], 1)
    if len(values) != 8 or min(values[:3] + values[6:]) <= 0 or any(x not in (0, 1) for x in values[3:6]):
        raise ValueError("config must be tile_m,tile_n,simdgroups,cooperative,static_k,inline[,group_simdgroups,cohort_rows]")
    if values[0] % 8 or values[1] % 8 or (values[0] % 16 and values[1] % 16):
        raise ValueError("MPP tile extents must be multiples of 8, with at least one a multiple of 16")
    cohorts = values[6] if values[2] == 1 else 1
    if (values[2] != 1 and values[6] != values[2]) or cohorts % values[7]:
        raise ValueError("MPP execution scope must be one subgroup or the whole group; cohort rows must divide independent groups")
    return values


def oracle(np: Any, shape: tuple[int, int, int]) -> Any:
    m, n, k = shape
    def values(rows, columns, seed):
        return (((np.arange(rows * columns, dtype=np.int64) * seed + 17) % 127 - 63) / 64.0).reshape(rows, columns)
    return values(m, k, 5) @ values(k, n, 11)


def validate_output(np: Any, actual: Any, expected: Any) -> dict[str, float]:
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError("output shape mismatch or non-finite value")
    error = np.abs(actual.astype(np.float64) - expected)
    if np.any(error > 1e-4 + 1e-4 * np.abs(expected)):
        raise ValueError(f"full FP64 oracle failed: max absolute error {error.max()}")
    return {"max_abs_error": float(error.max()), "atol": 1e-4, "rtol": 1e-4,
            "checked_elements": int(actual.size)}


def validate_metadata(result: dict[str, Any], shape: tuple[int, int, int],
                      config: tuple[int, ...] | None, samples: int) -> None:
    if tuple(result.get(key) for key in ("m", "n", "k")) != shape or result.get("backend") != "metal":
        raise ValueError("binary returned a different workload/backend")
    if config is None:
        if (result.get("implementation") != "mps_matrix_multiplication" or
                result.get("api_variant") != "MPSKernelOptionsNone" or result.get("dtype") != "float32" or
                result.get("alpha") != 1 or result.get("beta") != 0 or
                result.get("transpose_left") is not False or result.get("transpose_right") is not False):
            raise ValueError("MPS precision/operation mismatch")
    else:
        tm, tn, sg, coop, static, inline, group, cohort_rows = config
        if (result.get("implementation") != "mpp_tensor_ops_matmul2d" or result.get("precision") != "fp32" or
                result.get("relaxed_precision") is not False or result.get("fast_math") is not False or
                result.get("block") != [tm, tn] or result.get("execution_simdgroups") != sg or
                result.get("cooperative_output") != bool(coop) or result.get("static_reduction") != bool(static) or
                result.get("inline_tensors") != bool(inline) or result.get("group_simdgroups") != group or
                result.get("cohort_rows") != cohort_rows):
            raise ValueError("MPP precision/schedule mismatch")
    for metric in METRICS:
        values = result.get(metric, [])
        if len(values) != samples or any(not math.isfinite(x) or x <= 0 for x in values):
            raise ValueError(f"invalid {metric} samples")


def measure(args: argparse.Namespace, np: Any, shape: tuple[int, int, int],
            config: tuple[int, ...] | None, expected: Any) -> dict[str, Any]:
    row = {"shape": list(shape), "config": config, "valid": False}
    try:
        with tempfile.TemporaryDirectory(prefix="mpp-output-") as folder:
            output = Path(folder) / "output.f32"
            binary = args.mps if config is None else args.mpp
            command = [str(binary), "metal" if config is None else "fp32", *map(str, shape),
                       str(args.samples), str(args.sample_ms), str(args.warmup_ms), str(output)]
            if config is not None:
                tm, tn, sg, coop, static, inline, group, cohort_rows = config
                command += list(map(str, (tm, tn, sg, coop, 0, static, inline, group, cohort_rows)))
            row["command"] = command
            process = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
            row["stderr"] = process.stderr
            if process.returncode != 0:
                raise RuntimeError(f"exit {process.returncode}: {process.stderr[-6000:]}")
            result = json.loads(process.stdout)
            validate_metadata(result, shape, config, args.samples)
            actual = np.fromfile(output, dtype="<f4").reshape(shape[:2])
            row["correctness"] = validate_output(np, actual, expected)
            for metric in METRICS:
                result[metric + "_p50"] = percentile(result[metric], 0.5)
            row.update(valid=True, measurement=result)
    except Exception as error:
        row["error"] = str(error)
    return row


def pick_winners(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    winners = {}
    shapes = dict.fromkeys(tuple(row["shape"]) for row in rows)
    for shape in shapes:
        valid = [row for row in rows if tuple(row["shape"]) == shape and row["valid"] and row["config"] is not None]
        if valid:
            # Use GPU batch time to screen the shader schedule, not differences
            # in host encoder overhead. This minimum is never a replay result.
            best = min(valid, key=lambda row: row["measurement"]["gpu_throughput_us_p50"])
            winners["x".join(map(str, shape))] = list(best["config"])
    return winners


def save(report: dict[str, Any], output: Path) -> None:
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# Native MPP versus MPS", "",
             "FP32 C=A×B, no transpose/prepacking, private compact row-major buffers, alpha=1, beta=0. "
             "MPP relaxed precision and compiler fast math are disabled. Complete outputs use the same FP64 oracle (atol=rtol=1e-4).", "",
             "Host wall time includes encoding, submission and synchronization. GPU time is the command-buffer/batch interval, "
             "including dispatch and synchronization costs; it is not an isolated instruction latency. Setup, uploads and validation are untimed.", ""]
    if report["metadata"]["mode"] == "search":
        lines += ["Exploratory search only. Selected minima require a separate frozen, counterbalanced replay.", "",
                  "| M×N×K | Config (M,N,op-SG,coop,static-K,inline,group-SG,cohort-M) | Valid | Host µs | GPU µs |", "|---|---|---|---:|---:|"]
        for row in report["results"]:
            name = "x".join(map(str, row["shape"]))
            config = "MPS" if row["config"] is None else ",".join(map(str, row["config"]))
            if row["valid"]:
                m = row["measurement"]
                lines.append(f"| {name} | {config} | yes | {m['throughput_us_p50']:.3f} | {m['gpu_throughput_us_p50']:.3f} |")
            else:
                lines.append(f"| {name} | {config} | FAILED (see JSON) | — | — |")
    else:
        lines += ["Frozen replay: each shape visits all six orders of MPS/default-MPP/selected-MPP. "
                  "Values are medians of per-round medians; ratios are paired round medians [min,max], not confidence intervals. "
                  "Ratio >1 means selected MPP is slower. No failed/slow round is removed.", "",
                  "| M×N×K | Rounds | MPS GPU µs | Default MPP GPU µs | Selected MPP GPU µs | MPP/MPS GPU | Host MPP/MPS |", "|---|---:|---:|---:|---:|---:|---:|"]
        for shape in report["metadata"]["shapes"]:
            rows = [r for r in report["results"] if r["shape"] == shape]
            complete = [r for r in rows if r.get("valid")]
            name = "x".join(map(str, shape))
            if len(complete) != report["metadata"]["rounds"]:
                lines.append(f"| {name} | {len(complete)}/{report['metadata']['rounds']} | INCOMPLETE | — | — | — | — |")
                continue
            times = [percentile([r[k]["measurement"]["gpu_throughput_us_p50"] for r in complete], 0.5)
                     for k in ("mps", "default", "selected")]
            ratios = [r["selected"]["measurement"]["gpu_throughput_us_p50"] / r["mps"]["measurement"]["gpu_throughput_us_p50"] for r in complete]
            host = [r["selected"]["measurement"]["throughput_us_p50"] / r["mps"]["measurement"]["throughput_us_p50"] for r in complete]
            lines.append(f"| {name} | {len(complete)} | {times[0]:.3f} | {times[1]:.3f} | {times[2]:.3f} | "
                         f"{percentile(ratios, .5):.3f} [{min(ratios):.3f},{max(ratios):.3f}] | {percentile(host, .5):.3f} |")
    lines += ["", f"Artifacts unchanged: {report['metadata'].get('artifacts_unchanged', 'pending')}.", "",
              "Raw samples, errors, configurations and artifact hashes: [results.json](results.json).", ""]
    (output / "results.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpp", type=Path, required=True)
    parser.add_argument("--mps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", action="append", default=[])
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--plan", type=Path, help="replay selected configs from a completed search report")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--sample-ms", type=int, default=30)
    parser.add_argument("--warmup-ms", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if min(args.rounds, args.samples, args.sample_ms, args.warmup_ms, args.timeout) <= 0 or args.rounds % 6:
        parser.error("positive timing settings and rounds divisible by six are required")
    if args.plan and args.config:
        parser.error("replay uses only frozen plan configs")
    try:
        shapes = [parse_shape(x) for x in args.shape] or DEFAULT_SHAPES
        configs = [parse_config(x) for x in args.config] or [DEFAULT_CONFIG]
        selected = {}
        if args.plan:
            source = json.loads(args.plan.read_text())
            if source["metadata"]["mode"] != "search" or source["metadata"].get("artifacts_unchanged") is not True:
                raise ValueError("plan must come from a completed, artifact-stable search")
            selected = {parse_shape(k): parse_config(",".join(map(str, v))) for k, v in source["selected"].items()}
            if not args.shape:
                shapes = list(selected)
            if any(shape not in selected for shape in shapes):
                raise ValueError("frozen plan is missing a requested shape")
        args.mpp = args.mpp.resolve(strict=True)
        args.mps = args.mps.resolve(strict=True)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
    import numpy as np
    args.output.mkdir(parents=True, exist_ok=False)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    hashes = {str(p): digest(p) for p in (args.mpp, args.mps)}
    if args.plan and source["metadata"].get("mpp_sha256") != hashes[str(args.mpp)]:
        parser.error("MPP executable differs from the frozen search artifact")
    root = Path(__file__).resolve().parents[3]
    report = {"metadata": {
        "mode": "replay" if args.plan else "search", "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "platform": platform.platform(), "numpy_version": np.__version__, "artifacts_sha256": hashes,
        "mpp_sha256": hashes[str(args.mpp)], "mps_sha256": hashes[str(args.mps)],
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "worktree_dirty": subprocess.run(["git", "diff", "--quiet"], cwd=root).returncode != 0,
        "shapes": [list(s) for s in shapes], "rounds": args.rounds if args.plan else 1,
        "samples": args.samples, "sample_ms": args.sample_ms, "warmup_ms": args.warmup_ms,
        "plan_sha256": digest(args.plan) if args.plan else None,
        "default_config": DEFAULT_CONFIG,
    }, "results": [], "selected": {"x".join(map(str, k)): list(v) for k, v in selected.items()}}
    failed = False
    # Validation/oracle work is outside every native timed process.
    references = {shape: oracle(np, shape) for shape in shapes}
    if not args.plan:
        for index, shape in enumerate(shapes):
            order = configs[index % len(configs):] + configs[:index % len(configs)]
            for config in [None, *order]:
                print(f"search {shape} {config or 'MPS'} ...", flush=True)
                row = measure(args, np, shape, config, references[shape])
                report["results"].append(row)
                failed |= not row["valid"]
                print(f"  GPU {row['measurement']['gpu_throughput_us_p50']:.3f} us; validated" if row["valid"] else f"  FAILED: {row['error']}", flush=True)
                report["selected"] = pick_winners(report["results"])
                save(report, args.output)
    else:
        orders = list(itertools.permutations(("mps", "default", "selected")))
        for round_index in range(args.rounds):
            shift = round_index % len(shapes)
            for shape in shapes[shift:] + shapes[:shift]:
                order = orders[(round_index + shapes.index(shape)) % len(orders)]
                row = {"shape": list(shape), "round": round_index, "order": order, "valid": True}
                print(f"round {round_index + 1}/{args.rounds} {shape} {order} ...", flush=True)
                for label in order:
                    config = {"mps": None, "default": DEFAULT_CONFIG, "selected": selected[shape]}[label]
                    row[label] = measure(args, np, shape, config, references[shape])
                    row["valid"] &= row[label]["valid"]
                    if row[label]["valid"]:
                        print(f"  {label}: GPU {row[label]['measurement']['gpu_throughput_us_p50']:.3f} us; validated", flush=True)
                    else:
                        print(f"  {label} FAILED: {row[label]['error']}", flush=True)
                failed |= not row["valid"]
                report["results"].append(row)
                save(report, args.output)
    report["metadata"]["artifacts_unchanged"] = all(digest(Path(p)) == h for p, h in hashes.items())
    save(report, args.output)
    print(f"Results: {args.output / 'results.md'}", flush=True)
    return int(failed or not report["metadata"]["artifacts_unchanged"])


if __name__ == "__main__":
    raise SystemExit(main())
