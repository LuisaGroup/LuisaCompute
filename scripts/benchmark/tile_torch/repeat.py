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

from run import Case, percentile, run_case, validate_cpu_target_policy


def artifact_hashes(binaries: list[Path], extra: list[Path]) -> dict[str, str]:
    artifacts = {p.resolve(strict=True) for p in binaries + extra}
    for binary in binaries:
        artifacts.update(p.resolve() for p in binary.parent.iterdir()
                         if p.is_file() and p.suffix in (".dylib", ".so", ".dll"))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(artifacts)}


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
        group_threads = native.get("planner_threads", 0)
        if type(group_threads) is not int or not 0 <= group_threads <= 0xffffffff:
            raise ValueError(f"{case.name} has an invalid group-thread constraint")
        copy_batch = native.get("copy_batch", 1)
        if type(copy_batch) is not int or not 1 <= copy_batch <= 16:
            raise ValueError(f"{case.name} has an invalid copy-batch policy")
        elide = native.get("elide_independent_subgroup_barriers", False)
        if type(elide) is not bool or elide and native.get("forward_readonly_tile_loads") is not True:
            raise ValueError(f"{case.name} has an invalid subgroup-fence policy")
        key = row["backend"], case.name
        cpu_stack = native.get("cpu_stack_bytes", 0)
        if type(cpu_stack) is not int or not 0 <= cpu_stack <= 65536 or (cpu_stack and row["backend"] != "cpu"):
            raise ValueError(f"{case.name} has an invalid CPU stack budget")
        cpu_lanes = native.get("cpu_vector_lanes", 16)
        if type(cpu_lanes) is not int or cpu_lanes not in (16, 32, 64, 128) or (
                cpu_lanes != 16 and (row["backend"] != "cpu" or not native["auto_vectorize"] or not native["vectorize"])):
            raise ValueError(f"{case.name} has an invalid CPU vector-lane policy")
        forwarding = native.get("forward_readonly_tile_loads", False)
        if type(forwarding) is not bool:
            raise ValueError(f"{case.name} has an invalid input-view policy")
        metal_subgroup_reductions = native.get("metal_subgroup_reductions", False)
        if type(metal_subgroup_reductions) is not bool:
            raise ValueError(f"{case.name} has an invalid Metal subgroup-reduction policy")
        packing = native.get("reduction_programs_per_group", 0)
        unroll = native.get("reduction_unroll_factor", 1)
        lanes = native.get("reduction_lane_elements", 1)
        cache_inputs = native.get("cache_reduction_inputs", False)
        if type(cache_inputs) is not bool or (cache_inputs and not metal_subgroup_reductions):
            raise ValueError(f"{case.name} has an invalid reduction input-cache policy")
        if type(lanes) is not int or lanes not in (1, 2, 4, 8) or (lanes != 1 and not metal_subgroup_reductions):
            raise ValueError(f"{case.name} has invalid reduction lane elements")
        if type(unroll) is not int or not 1 <= unroll <= 16 or (unroll != 1 and not metal_subgroup_reductions):
            raise ValueError(f"{case.name} has an invalid reduction unroll factor")
        if type(packing) is not int or not 0 <= packing <= 8 or (packing and not metal_subgroup_reductions):
            raise ValueError(f"{case.name} has an invalid reduction program packing")
        element_grid = native.get("fuse_gpu_elementwise")
        if element_grid is not None and type(element_grid) is not bool:
            raise ValueError(f"{case.name} has an invalid element-grid mapping policy")
        if metal_subgroup_reductions:
            execution_plans = native.get("execution_plans")
            if (row["backend"] != "metal" or case.operation not in ("sum", "softmax", "rmsnorm", "layernorm", "residual_layernorm", "cross_entropy") or
                    native["execution_scope"] != "auto" or native.get("metal_mpp", False) is not False or
                    forwarding is not True or not isinstance(execution_plans, list) or not execution_plans or
                    any(plan.get("optimized") is not True or type(plan.get("threads")) is not int or
                        plan["threads"] < 32 or plan["threads"] % 32 for plan in execution_plans)):
                raise ValueError(f"{case.name} has an unrealized Metal subgroup-reduction policy")
            if packing and any(item.get("reduction_programs_per_group") != packing for item in execution_plans):
                raise ValueError(f"{case.name} has an unrealized reduction program packing")
            if unroll != 1 and any(item.get("reduction_unroll_factor") != unroll for item in execution_plans):
                raise ValueError(f"{case.name} has an unrealized reduction unroll factor")
            if lanes != 1 and any(item.get("reduction_lane_elements") != lanes for item in execution_plans):
                raise ValueError(f"{case.name} has unrealized reduction lane elements")
        cpu_views = row["backend"] == "cpu" and forwarding
        cpu_model = native.get("cpu_target_policy") if row["backend"] == "cpu" else None
        validate_cpu_target_policy(native, cpu_model, row["backend"])
        cpu_matrix = native.get("cpu_matrix_backend", "reference")
        if cpu_matrix not in ("reference", "cblas") or (cpu_matrix == "cblas" and
                (row["backend"] != "cpu" or case.operation != "gemm" or native["execution_scope"] != "auto")):
            raise ValueError(f"{case.name} has an invalid CPU matrix realization")
        cpu_math = native.get("cpu_math_backend", "reference")
        if cpu_math not in ("reference", "accelerate") or (cpu_math == "accelerate" and row["backend"] != "cpu"):
            raise ValueError(f"{case.name} has an invalid CPU array-math realization")
        shared_tiles = native.get("shared_tile_materialization")
        if shared_tiles not in ("preserve", "expensive-only"):
            raise ValueError(f"{case.name} has no explicit shared-Tile materialization policy")
        if cpu_views and native.get("metal_mpp", False) is not False:
            raise ValueError(f"{case.name} combines CPU input views with MPP")
        if key in plan:
            raise ValueError(f"duplicate case {key}")
        plan[key] = {
            "case": row["case"], "gemm_block": tuple(row["block"]),
            "execution_scope": native["execution_scope"],
            "pipeline_window": native["pipeline_window"],
            "cooperative_matrix": native["cooperative_matrix"],
            "no_vectorize": not native["vectorize"],
            "auto_vectorize": native["auto_vectorize"],
            "group_threads": group_threads,
            "copy_batch": copy_batch,
            "cpu_stack_bytes": cpu_stack,
            "cpu_vector_lanes": cpu_lanes,
            "cpu_input_views": cpu_views,
            "input_views": row["backend"] == "metal" and forwarding and not metal_subgroup_reductions and native.get("metal_mpp", False) is False,
            "cpu_model": cpu_model,
            "cpu_matrix_backend": cpu_matrix,
            "cpu_math_backend": cpu_math,
            "shared_tile_materialization": shared_tiles,
            "metal_subgroup_reductions": metal_subgroup_reductions,
            "reduction_programs_per_group": packing,
            "reduction_unroll": unroll,
            "reduction_lane_elements": lanes,
            "cache_reduction_inputs": cache_inputs,
            "element_grid": None if element_grid is None else "auto" if element_grid else "reference",
            "expected_cpu_model": native.get("cpu_model") if row["backend"] == "cpu" else None,
            "elide_independent_subgroup_barriers": elide,
            "matrix_realization": "mpp-views" if forwarding and native.get("metal_mpp", False) else
                                  "mpp" if native.get("metal_mpp") else "simdgroup",
        }
    if not plan:
        raise ValueError("the report contains no requested cases")
    return plan


def override_cpu_vector_lanes(plan: dict[tuple[str, str], dict[str, Any]], budget: int) -> None:
    if type(budget) is not int or budget not in (16, 32, 64, 128) or any(
            backend != "cpu" or config["auto_vectorize"] is not True or config["no_vectorize"] is not False
            for (backend, _), config in plan.items()):
        raise ValueError("CPU vector-lane override requires auto-vectorized CPU plans and 16/32/64/128 lanes")
    for config in plan.values():
        config["cpu_vector_lanes"] = budget


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
    paired_rows = {}
    for backend, name in keys:
        rows = [r for r in report["results"] if (r["backend"], r["name"]) == (backend, name)]
        pairs = []
        for round_index in range(report["metadata"]["rounds"]):
            pair = {r["variant"]: r for r in rows if r["round"] == round_index}
            if len(pair) == 2 and all(r.get("valid") for r in pair.values()):
                pairs.append(pair)
        paired_rows[backend, name] = pairs
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
    if report["metadata"].get("metal_device_timing") or any(
            "device_timing" in row.get("native", {}) for row in report["results"]):
        lines += ["", "## Separately sampled GPU execution", "",
                  "Completed command-buffer GPU timestamps, with no encoder hooks or counter attachments. This includes GPU work/gaps inside each command buffer, not CPU encoding or completion notification, and is not individual-kernel time. Each GPU phase uses its own repetition count. Values are medians of per-round p50 times. Speedups are medians of paired round ratios, with min–max ranges, not confidence intervals. Incomplete GPU control pairs withhold statistics; instrumented compute-pass samples remain diagnostics in JSON.", "",
                  "| Backend / case | GPU pairs | Reference GPU µs/op | Candidate GPU µs/op | Paired GPU speedup [range] | Candidate-run Torch GPU µs/op |",
                  "|---|---:|---:|---:|---:|---:|"]
        latency_lines = ["", "## Single-call GPU versus end-to-end dispatch", "",
                         "These are separate phases; host samples are uninstrumented. Do not subtract their medians to estimate CPU cost. Torch remains the recorded eager operator sequence, not a compiled fused graph.", "",
                         "| Backend / case | Candidate GPU µs | Candidate E2E µs | Candidate-run Torch GPU µs | Candidate-run Torch E2E µs |",
                         "|---|---:|---:|---:|---:|"]
        for key, pairs in paired_rows.items():
            gpu_pairs = [p for p in pairs if all("control" in p[variant][provider].get("device_timing", {})
                                               for variant in ("reference", "candidate") for provider in ("native", "torch"))]
            label = " / ".join(key)
            if len(gpu_pairs) != report["metadata"]["rounds"]:
                lines.append(f"| {label} | {len(gpu_pairs)}/{report['metadata']['rounds']} | INCOMPLETE | — | — | — |")
                latency_lines.append(f"| {label} | INCOMPLETE | — | — | — |")
                continue
            median = lambda values: percentile(values, 0.5)
            reference = [p["reference"]["native"]["device_timing"]["control"]["command_buffer_throughput_us_p50"] for p in gpu_pairs]
            candidate = [p["candidate"]["native"]["device_timing"]["control"]["command_buffer_throughput_us_p50"] for p in gpu_pairs]
            ratios = [a / b for a, b in zip(reference, candidate)]
            torch = [p["candidate"]["torch"]["device_timing"]["control"]["command_buffer_throughput_us_p50"] for p in gpu_pairs]
            lines.append(f"| {label} | {len(gpu_pairs)}/{report['metadata']['rounds']} | {median(reference):.3f} | "
                         f"{median(candidate):.3f} | {median(ratios):.3f}× [{min(ratios):.3f}, {max(ratios):.3f}] | {median(torch):.3f} |")
            single = [median([p["candidate"][provider]["device_timing"]["control"]["command_buffer_latency_us_p50"] for p in gpu_pairs])
                      for provider in ("native", "torch")]
            host = [median([p["candidate"][provider]["latency_us_p50"] for p in gpu_pairs]) for provider in ("native", "torch")]
            latency_lines.append(f"| {label} | {single[0]:.3f} | {host[0]:.3f} | {single[1]:.3f} | {host[1]:.3f} |")
        lines += latency_lines
    failures = [r for r in report["results"] if not r.get("valid")]
    lines += ["", f"Failed measurements: {len(failures)}. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.", ""]
    (output / "results.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--candidate-native", type=Path, help="optional second prebuilt executable for implementation A/B tests")
    parser.add_argument("--compiler-artifact", type=Path, action="append", default=[],
                        help="additional compiler/runtime library to fingerprint before and after timing")
    parser.add_argument("--capture-sources", action="store_true",
                        help="require and archive generated LLVM/Metal sources; both binaries must support source dumping")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metal-device-timing", type=Path,
                        help="prebuilt GPU compute-pass timing library; separate from host-wall replay samples")
    parser.add_argument("--operations", default="gemm")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--sample-ms", type=int, default=40)
    parser.add_argument("--warmup-ms", type=int, default=200)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--candidate-vector-mode", choices=("reported", "vectorize", "no-vectorize", "auto-vectorize"), default="reported")
    parser.add_argument("--candidate-subgroup-fences", choices=("reported", "retain", "elide"), default="reported")
    parser.add_argument("--candidate-cpu-stack-bytes", type=int, help="override only the CPU candidate's storage budget")
    parser.add_argument("--cpu-stack-bytes", type=int, help="hold the CPU storage budget fixed for BOTH variants")
    parser.add_argument("--candidate-cpu-vector-lanes", type=int, choices=(16, 32, 64, 128),
                        help="override only the CPU candidate's logical SIMD-pack budget")
    parser.add_argument("--cpu-vector-lanes", type=int, choices=(16, 32, 64, 128),
                        help="hold the CPU logical SIMD-pack budget fixed for BOTH variants")
    parser.add_argument("--candidate-cpu-input-views", action="store_true", help="enable immutable input views only for CPU candidates")
    parser.add_argument("--candidate-cpu-model", choices=("generic", "native"), help="override only CPU candidate codegen; reference remains frozen")
    args = parser.parse_args()
    if args.rounds < 2 or args.rounds % 2 or min(args.samples, args.sample_ms, args.warmup_ms, args.threads, args.timeout) <= 0:
        parser.error("use an even number of rounds >= 2, and positive timing/thread settings")
    args.native = args.native.resolve()
    args.candidate_native = (args.candidate_native or args.native).resolve()
    if not args.native.is_file() or not args.candidate_native.is_file():
        parser.error("both native executables must already be built")
    try:
        operations = set(args.operations.split(","))
        if not operations <= {"gemm", "add", "gelu_add", "sum", "softmax", "rmsnorm", "layernorm", "residual_layernorm", "cross_entropy"}:
            raise ValueError("unknown operation in replay selection")
        plans = {"reference": load_plan(args.reference, operations), "candidate": load_plan(args.candidate, operations)}
        if plans["reference"].keys() != plans["candidate"].keys():
            raise ValueError("reports must contain exactly the same requested backend/cases")
        if args.metal_device_timing is not None:
            if sys.platform != "darwin" or any(backend != "metal" for backend, _ in plans["reference"]):
                raise ValueError("Metal device timing requires only Metal plans on macOS")
            args.metal_device_timing = args.metal_device_timing.resolve(strict=True)
            args.compiler_artifact.append(args.metal_device_timing)
        if args.cpu_stack_bytes is not None:
            if not 0 <= args.cpu_stack_bytes <= 65536 or any(k[0] != "cpu" for k in plans["reference"]):
                raise ValueError("shared CPU stack override requires CPU plans and a budget in [0,65536]")
            for plan in plans.values():
                for config in plan.values():
                    config["cpu_stack_bytes"] = args.cpu_stack_bytes
        if args.candidate_vector_mode != "reported":
            for config in plans["candidate"].values():
                config["no_vectorize"] = args.candidate_vector_mode == "no-vectorize"
                config["auto_vectorize"] = args.candidate_vector_mode == "auto-vectorize"
        if args.cpu_vector_lanes is not None:
            for plan in plans.values():
                override_cpu_vector_lanes(plan, args.cpu_vector_lanes)
        if args.candidate_subgroup_fences != "reported":
            for config in plans["candidate"].values():
                if config["matrix_realization"] != "mpp-views":
                    raise ValueError("subgroup-fence override requires MPP-view candidates")
                config["elide_independent_subgroup_barriers"] = args.candidate_subgroup_fences == "elide"
        if args.candidate_cpu_stack_bytes is not None:
            if not 0 <= args.candidate_cpu_stack_bytes <= 65536 or any(k[0] != "cpu" for k in plans["candidate"]):
                raise ValueError("CPU stack override requires CPU candidates and a budget in [0,65536]")
            for config in plans["candidate"].values():
                config["cpu_stack_bytes"] = args.candidate_cpu_stack_bytes
        if args.candidate_cpu_vector_lanes is not None:
            override_cpu_vector_lanes(plans["candidate"], args.candidate_cpu_vector_lanes)
        if args.candidate_cpu_input_views:
            for (backend, _), config in plans["candidate"].items():
                if backend != "cpu" or config["matrix_realization"] != "simdgroup":
                    raise ValueError("CPU input-view override requires CPU reference-lowering candidates")
                config["cpu_input_views"] = True
        if args.candidate_cpu_model is not None:
            if any(backend != "cpu" for backend, _ in plans["candidate"]):
                raise ValueError("CPU model override requires CPU candidates")
            for config in plans["candidate"].values():
                config["cpu_model"] = args.candidate_cpu_model
                config["expected_cpu_model"] = None
    except (KeyError, ValueError) as error:
        parser.error(str(error))
    for key in ("TVM_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = str(args.threads)
    removed = {key: os.environ.pop(key, None) for key in (
        "PYTORCH_MPS_FAST_MATH", "PYTORCH_MPS_PREFER_METAL", "PYTORCH_ENABLE_MPS_FALLBACK",
        "LUISA_ENABLE_VALIDATION", "MTL_DEBUG_LAYER", "MTL_SHADER_VALIDATION",
        "DYLD_PRINT_LIBRARIES", "LUISA_TILE_BENCH_DUMP_SOURCE")}
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
    hashes = artifact_hashes([args.native, args.candidate_native], args.compiler_artifact)
    report = {"metadata": {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), "platform": platform.platform(),
        "torch_version": torch.__version__, "torch_git_version": torch.version.git_version,
        "torch_config": torch.__config__.show(), "threads": torch.get_num_threads(),
        "thread_environment": {key: os.environ[key] for key in ("TVM_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root)),
        "removed_environment": removed, "artifacts_sha256": hashes, "capture_sources": args.capture_sources,
        "metal_device_timing": str(args.metal_device_timing) if args.metal_device_timing else None,
        "native_sha256": digest(args.native),
        "adjacent_tile_library_sha256": {p.name: digest(p) for p in sorted(args.native.parent.glob("*luisa-tile*"))
                                        if p.is_file() and p.suffix in (".dylib", ".so", ".dll")},
        "native_variants": {
            variant: {"binary": str(binary), "sha256": digest(binary),
                      "adjacent_tile_library_sha256": {p.name: digest(p) for p in sorted(binary.parent.glob("*luisa-tile*"))
                                                      if p.is_file() and p.suffix in (".dylib", ".so", ".dll")}}
            for variant, binary in (("reference", args.native), ("candidate", args.candidate_native))
        },
        "rounds": args.rounds, "samples": args.samples, "sample_ms": args.sample_ms, "warmup_ms": args.warmup_ms,
        "candidate_vector_mode": args.candidate_vector_mode,
        "candidate_subgroup_fences": args.candidate_subgroup_fences,
        "candidate_cpu_stack_bytes": args.candidate_cpu_stack_bytes,
        "shared_cpu_stack_bytes": args.cpu_stack_bytes,
        "candidate_cpu_vector_lanes": args.candidate_cpu_vector_lanes,
        "shared_cpu_vector_lanes": args.cpu_vector_lanes,
        "candidate_cpu_input_views": args.candidate_cpu_input_views,
        "candidate_cpu_model": args.candidate_cpu_model,
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
            run_args.native = args.native if variant == "reference" else args.candidate_native
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
    report["metadata"]["artifacts_unchanged"] = all(Path(p).is_file() and digest(Path(p)) == h for p, h in hashes.items())
    write_report(report, args.output)
    return int(failed or not report["metadata"]["artifacts_unchanged"])


if __name__ == "__main__":
    raise SystemExit(main())
