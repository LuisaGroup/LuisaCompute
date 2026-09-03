#!/usr/bin/env python3
"""Replay two frozen benchmark schedules in counterbalanced, fresh-JIT rounds.

No parameter search takes place here. Original report timings are not used to
score the repeated measurements. Every repetition checks the complete output.
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

from run import Case, percentile, run_case


def load_plan(path: Path, operations: set[str]) -> dict[tuple[str, str], dict[str, Any]]:
    report = json.loads(path.read_text())
    plan = {}
    for row in report["results"]:
        case = Case(**row["case"])
        if case.operation not in operations:
            continue
        if not row.get("valid"):
            raise ValueError(f"cannot replay invalid case {case.name}")
        native = row["native"]
        # Older reports predate the opt-in switch; guessing its historical
        # meaning could silently replay a different implementation policy.
        for flag in ("cooperative_matrix", "vectorize", "auto_vectorize"):
            if type(native.get(flag)) is not bool:
                raise ValueError(f"{case.name} has no explicit {flag} policy")
        key = row["backend"], case.name
        if key in plan:
            raise ValueError(f"duplicate case {key}")
        plan[key] = {
            "case": row["case"], "gemm_block": tuple(row["block"]),
            "execution_scope": native["execution_scope"],
            "pipeline_window": native["pipeline_window"],
            "cooperative_matrix": native["cooperative_matrix"],
            "no_vectorize": not native["vectorize"],
            "auto_vectorize": native["auto_vectorize"],
        }
    if not plan:
        raise ValueError("the report contains no requested cases")
    return plan


def order_for_round(keys: list[Any], round_index: int):
    # Rotate shapes, and balance A/B schedule order independently of the
    # native/PyTorch framework order. All variants get both orders per pair
    # of rounds; a lucky early/cold shape cannot always run first.
    shift = round_index % len(keys)
    for key in keys[shift:] + keys[:shift]:
        index = keys.index(key)
        variants = ("reference", "candidate") if (round_index + index) % 2 == 0 else ("candidate", "reference")
        for variant in variants:
            framework_order = (round_index + index + (variant == "candidate")) % 2
            yield key, variant, framework_order


def write_report(report: dict[str, Any], output: Path) -> None:
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# Frozen-schedule repeat measurements", "",
             "Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.", "",
             "Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.", "",
             "| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |",
             "|---|---:|---:|---:|---:|---:|"]
    keys = list(dict.fromkeys((r["backend"], r["name"]) for r in report["results"]))
    for backend, name in keys:
        rows = [r for r in report["results"] if (r["backend"], r["name"]) == (backend, name)]
        pairs = []
        for round_index in range(report["metadata"]["rounds"]):
            pair = {r["variant"]: r for r in rows if r["round"] == round_index}
            if len(pair) == 2 and all(r.get("valid") for r in pair.values()):
                pairs.append(pair)
        if not pairs:
            lines.append(f"| {backend} / {name} | 0 | — | — | — | — |")
            continue
        reference = [p["reference"]["native"]["throughput_us_p50"] for p in pairs]
        candidate = [p["candidate"]["native"]["throughput_us_p50"] for p in pairs]
        ratios = [a / b for a, b in zip(reference, candidate)]
        torch = [p["candidate"]["torch"]["throughput_us_p50"] for p in pairs]
        median = lambda values: percentile(values, 0.5)
        lines.append(f"| {backend} / {name} | {len(pairs)} | {median(reference):.3f} | {median(candidate):.3f} | "
                     f"{median(ratios):.3f}× [{min(ratios):.3f}, {max(ratios):.3f}] | {median(torch):.3f} |")
    failures = [r for r in report["results"] if not r.get("valid")]
    lines += ["", f"Failed measurements: {len(failures)}. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.", ""]
    (output / "results.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operations", default="gemm")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--sample-ms", type=int, default=40)
    parser.add_argument("--warmup-ms", type=int, default=200)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--candidate-vector-mode", choices=("reported", "vectorize", "no-vectorize", "auto-vectorize"), default="reported")
    args = parser.parse_args()
    if args.rounds < 2 or args.rounds % 2 or min(args.samples, args.sample_ms, args.warmup_ms, args.threads, args.timeout) <= 0:
        parser.error("use an even number of rounds >= 2, and positive timing/thread settings")
    args.native = args.native.resolve()
    if not args.native.is_file():
        parser.error("the native executable must already be built")
    try:
        operations = set(args.operations.split(","))
        if not operations <= {"gemm", "add", "sum", "softmax"}:
            raise ValueError("unknown operation in replay selection")
        plans = {"reference": load_plan(args.reference, operations), "candidate": load_plan(args.candidate, operations)}
        if plans["reference"].keys() != plans["candidate"].keys():
            raise ValueError("reports must contain exactly the same requested backend/cases")
        if args.candidate_vector_mode != "reported":
            for config in plans["candidate"].values():
                config["no_vectorize"] = args.candidate_vector_mode == "no-vectorize"
                config["auto_vectorize"] = args.candidate_vector_mode == "auto-vectorize"
    except (KeyError, ValueError) as error:
        parser.error(str(error))
    for key in ("TVM_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = str(args.threads)
    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    keys = list(plans["reference"])
    if any(backend == "metal" for backend, _ in keys) and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable; CPU fallback is not allowed")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    report = {"metadata": {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), "platform": platform.platform(),
        "torch_version": torch.__version__, "torch_git_version": torch.version.git_version,
        "torch_config": torch.__config__.show(), "threads": torch.get_num_threads(),
        "thread_environment": {key: os.environ[key] for key in ("TVM_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "worktree_dirty": subprocess.run(["git", "diff", "--quiet"], cwd=root).returncode != 0,
        "native_sha256": digest(args.native),
        "adjacent_tile_library_sha256": {p.name: digest(p) for p in sorted(args.native.parent.glob("*luisa-tile*"))
                                        if p.is_file() and p.suffix in (".dylib", ".so", ".dll")},
        "rounds": args.rounds, "samples": args.samples, "sample_ms": args.sample_ms, "warmup_ms": args.warmup_ms,
        "candidate_vector_mode": args.candidate_vector_mode,
        "timing": "synchronized device-resident host wall time including dispatch; no profiler",
        "source_reports": {variant: {"path": str(path), "sha256": digest(path)}
                           for variant, path in (("reference", args.reference), ("candidate", args.candidate))},
        "frozen_plans": {variant: [{"backend": key[0], "name": key[1], **config} for key, config in plan.items()]
                         for variant, plan in plans.items()},
    }, "results": []}
    failed = False
    for round_index in range(args.rounds):
        for (backend, name), variant, order in order_for_round(keys, round_index):
            config = dict(plans[variant][backend, name])
            case = Case(**config.pop("case"))
            run_args = argparse.Namespace(**(vars(args) | config))
            print(f"round {round_index + 1}/{args.rounds}: {backend} {name} {variant}", flush=True)
            try:
                row = run_case(torch, np, run_args, case, backend, order)
                print(f"  validated; native {row['native']['throughput_us_p50']:.3f} us; torch {row['torch']['throughput_us_p50']:.3f} us", flush=True)
            except Exception as error:
                row = {"backend": backend, "name": name, "case": vars(case), "valid": False, "error": str(error)}
                failed = True
                print(f"  FAILED: {error}", file=sys.stderr, flush=True)
            row.update(round=round_index, variant=variant)
            report["results"].append(row)
            write_report(report, args.output)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
