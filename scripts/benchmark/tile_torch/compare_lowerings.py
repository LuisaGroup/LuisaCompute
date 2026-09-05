#!/usr/bin/env python3
"""Compare Tile native MPP, Tile TIRx, handwritten MPP, direct MPS, and eager Torch.

Prebuilt artifacts only. No parameter search. Same full FP64 oracle for every
path. Host timing and optional command-buffer GPU controls remain separate.
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
import subprocess
import tempfile
from typing import Any

from compare_mpp import DEFAULT_SHAPES, oracle, parse_config, parse_shape, validate_metadata, validate_output
from repeat import load_plan
from run import (Case, percentile, summarize, summarize_device_timing, time_metal_device, time_torch,
                 validate_native_metadata, validate_tirx_realization, validate_subgroup_policy)

PATHS = ("tile_native_mpp", "tile_tirx", "handwritten_mpp", "mps", "torch")
RUNTIME_PATHS = ("tile_tirx_luisa", "tile_tirx_luisa_fast")
MPP_PATHS = ("tile_tirx_mpp",)
VIEW_PATHS = ("tile_tirx_mpp_views",)
TITLES = dict(zip(PATHS + RUNTIME_PATHS, ("Tile→MPP", "TIRx/TVM", "Hand MPP", "MPS", "Torch", "TIRx/Luisa", "TIRx/Luisa fast")))
TITLES["tile_tirx_mpp"] = "TIRx→MPP/TVM"
TITLES["tile_tirx_mpp_views"] = "TIRx→MPP views/TVM"


def validate_mpp_tirx(result: dict[str, Any], views: bool = False) -> None:
    validate_tirx_realization(result, "mpp-views" if views else "mpp")
    if result.get("metal_mpp") is not True or result.get("simdgroup_intrinsics") != 0:
        raise ValueError("TIRx MPP path must be explicitly realized, without SIMD-group fallback")
    calls = result.get("mpp_intrinsics")
    if type(calls) is not int or calls <= 0 or calls != result.get("matrix_intrinsics"):
        raise ValueError("TIRx MPP must report actual generated MPP call sites")
    plans = result.get("execution_plans")
    if not isinstance(plans, list) or not plans or any(
        p.get("metal_mpp") is not True or p.get("cost_basis") != "metal_mpp_memory_v2" for p in plans
    ):
        raise ValueError("TIRx MPP must identify its explicit MPP memory cost basis")


def implementation_order(round_index: int, case_index: int = 0, paths: tuple[str, ...] = PATHS) -> tuple[str, ...]:
    # Rotations and their reversals balance every position and pair over 2N
    # rounds. Adding a Runtime control must also update the balancing period.
    index = (round_index + case_index) % (2 * len(paths))
    shift = index % len(paths)
    order = paths[shift:] + paths[:shift]
    return order if index < len(paths) else order[::-1]


def validate_tirx_runtime(result: dict[str, Any], shape: tuple[int, ...], tirx: dict[str, Any], fast: bool) -> None:
    expected = dict(backend="metal", runtime="luisa", timing="synchronized_host_wall", fast_math=fast,
                    operation="gemm", output_elements=shape[0] * shape[1], mma_operations=1,
                    execution_scope=tirx["execution_scope"], pipeline_window=tirx["pipeline_window"],
                    cooperative_matrix=tirx["cooperative_matrix"], vectorize=not tirx["no_vectorize"],
                    auto_vectorize=tirx["auto_vectorize"], planner_threads=tirx["group_threads"], copy_batch=tirx["copy_batch"])
    for key, value in expected.items():
        if type(result.get(key)) is not type(value) or result[key] != value:
            raise ValueError(f"TIRx/Luisa metadata mismatch: {key}")
    threads = result.get("realized_threads")
    if type(threads) is not int or threads <= 0 or tirx["group_threads"] and threads != tirx["group_threads"]:
        raise ValueError("TIRx/Luisa realized thread constraint mismatch")
    calls = result.get("matrix_intrinsics")
    if type(calls) is not int or calls < 0:
        raise ValueError("invalid TIRx/Luisa matrix call count")


def validate_runtime_controls(row: dict[str, Any]) -> None:
    # Matching generated source (not merely equal requested planner knobs)
    # makes the Runtime/fast-math experiment an actual same-device-code control.
    baseline = row["tile_tirx"]
    for label in RUNTIME_PATHS:
        if label not in row:
            continue
        control = row[label]
        if not baseline.get("valid") or not control.get("valid"):
            raise ValueError("Runtime control requires both valid TIRx paths")
        if not baseline.get("source_sha256") or baseline["source_sha256"] != control.get("source_sha256"):
            raise ValueError("TIRx Runtime controls generated different device source")
        expected = baseline["measurement"]
        actual = control["measurement"]
        if expected["matrix_intrinsics"] != actual["matrix_intrinsics"]:
            raise ValueError("TIRx Runtime controls have different matrix realizations")
        plans = expected.get("execution_plans", [])
        if plans and any(p["threads"] != actual["realized_threads"] for p in plans):
            raise ValueError("TIRx Runtime controls have different threadgroup widths")


def validate_native(result: dict[str, Any], shape: tuple[int, ...], config: tuple[int, ...], samples: int) -> None:
    tm, tn, sg, coop, static, inline, group, cm = config
    if not coop or static or not inline:
        raise ValueError("native bootstrap requires cooperative output, dynamic K, inline tensors")
    expected = dict(implementation="tile_native_mpp", backend="metal", precision="fp32", fast_math=False,
                    relaxed_precision=False, timing="synchronized_host_wall", m=shape[0], n=shape[1], k=shape[2],
                    block=[tm, tn], execution_simdgroups=sg, group_simdgroups=group, cohort_rows=cm)
    for key, value in expected.items():
        if type(result.get(key)) is not type(value) or result[key] != value:
            raise ValueError(f"native MPP metadata mismatch: {key}")


def validate_times(result: dict[str, Any], samples: int) -> None:
    for metric in ("throughput_us", "latency_us"):
        values = result.get(metric, [])
        if len(values) != samples or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in values):
            raise ValueError(f"invalid {metric} samples")


def gpu_samples(result: dict[str, Any], label: str, phase: str, samples: int) -> list[float]:
    if phase not in ("throughput", "latency"):
        raise ValueError("GPU phase must be throughput or latency")
    if label == "handwritten_mpp":
        # This standalone control already measures completed MTLCommandBuffer
        # intervals without encoder instrumentation. Do not relabel it as a
        # compute-pass counter measurement.
        values = result.get(f"gpu_{phase}_us", [])
    else:
        device = result.get("device_timing", {})
        if "control" not in device:
            raise ValueError("GPU comparison requires an uninstrumented command-buffer control")
        summarize_device_timing(device, samples)
        values = device["control"][f"command_buffer_{phase}_us"]
    if len(values) != samples or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("invalid command-buffer GPU samples")
    return values


def measure(args: argparse.Namespace, np: Any, torch: Any, label: str, shape: tuple[int, ...],
            config: tuple[int, ...], tirx: dict[str, Any], reference: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"valid": False}
    try:
        if label == "torch":
            m, n, k = shape
            def inputs(rows, columns, seed):
                x = torch.arange(rows * columns, dtype=torch.int64)
                return (((x * seed + 17) % 127 - 63).float() / 64).reshape(rows, columns).to("mps")
            a, b = inputs(m, k, 5), inputs(k, n, 11)
            out = torch.full((m, n), float("nan"), dtype=torch.float32, device="mps")
            torch.mps.synchronize()
            with torch.inference_mode():
                result = time_torch(lambda: torch.mm(a, b, out=out), torch.mps.synchronize, args)
                if getattr(args, "metal_device_timing", None) is not None:
                    result["device_timing"] = time_metal_device(
                        lambda: torch.mm(a, b, out=out), torch.mps.synchronize, args, result["repetitions"])
            actual = out.cpu().numpy()
            torch.mps.synchronize()
            result.update(implementation="torch_eager_mm_out", device=str(out.device), dtype=str(out.dtype))
        else:
            with tempfile.TemporaryDirectory(prefix="tile-lowerings-") as folder:
                output = Path(folder) / "output.f32"
                timing = [str(args.samples), str(args.sample_ms), str(args.warmup_ms), str(output)]
                tm, tn, sg, coop, static, inline, group, cm = config
                if label == "tile_tirx" or label in RUNTIME_PATHS + MPP_PATHS + VIEW_PATHS:
                    command = [str(args.tirx), "metal", "gemm", *map(str, shape), *map(str, tirx["gemm_block"]),
                               *timing, tirx["execution_scope"], str(tirx["pipeline_window"]),
                               "mpp-views" if label in VIEW_PATHS else "mpp" if label in MPP_PATHS else "matrix" if tirx["cooperative_matrix"] else "scalar",
                               "auto-vectorize" if tirx["auto_vectorize"] else "no-vectorize" if tirx["no_vectorize"] else "vectorize",
                               str(tirx["group_threads"]) if tirx["group_threads"] else "auto", str(tirx["copy_batch"])]
                    if label in RUNTIME_PATHS:
                        command.append("luisa-fast" if label.endswith("_fast") else "luisa")
                    if tirx.get("elide_independent_subgroup_barriers", False):
                        if label not in VIEW_PATHS:
                            raise ValueError("subgroup-fence elision currently requires the MPP-view path")
                        command.extend(("tvm", "elide-subgroup-fences"))
                elif label == "mps":
                    command = [str(args.mps), "metal", *map(str, shape), *timing]
                elif label == "handwritten_mpp":
                    command = [str(args.mpp), "fp32", *map(str, shape), *timing,
                               *map(str, (tm, tn, sg, coop, 0, static, inline, group, cm))]
                else:
                    command = [str(args.native), "fp32", *map(str, shape), *timing, *map(str, (tm, tn, sg, group, cm))]
                row["command"] = command
                environment = os.environ.copy()
                timing_library = getattr(args, "metal_device_timing", None)
                environment.pop("LUISA_TILE_BENCH_METAL_TIMING", None)
                if timing_library is not None and label != "handwritten_mpp":
                    environment["LUISA_TILE_BENCH_METAL_TIMING"] = str(timing_library)
                source_path = Path(folder) / "device.metal"
                if label == "tile_tirx" or label in RUNTIME_PATHS + MPP_PATHS + VIEW_PATHS:
                    environment["LUISA_TILE_BENCH_DUMP_SOURCE"] = str(source_path)
                process = subprocess.run(command, text=True, capture_output=True, timeout=args.timeout, env=environment)
                row["stderr"] = process.stderr
                if process.returncode:
                    raise RuntimeError(f"exit {process.returncode}: {process.stderr[-6000:]} {process.stdout[-2000:]}")
                lines = [line for line in process.stdout.splitlines() if line.startswith("{")]
                if len(lines) != 1:
                    raise ValueError("expected exactly one JSON result")
                result = json.loads(lines[0])
                if label != "handwritten_mpp" and (("device_timing" in result) != (timing_library is not None)):
                    raise ValueError("requested Metal device timing was not realized")
                if source_path.exists():
                    source = source_path.read_bytes()
                    row["source_sha256"] = hashlib.sha256(source).hexdigest()
                    source_dir = args.output / "sources"
                    source_dir.mkdir(exist_ok=True)
                    destination = source_dir / (row["source_sha256"] + ".metal")
                    if not destination.exists():
                        destination.write_bytes(source)
                if label == "tile_native_mpp":
                    validate_native(result, shape, config, args.samples)
                elif label in ("mps", "handwritten_mpp"):
                    validate_metadata(result, shape, config if label == "handwritten_mpp" else None, args.samples)
                elif label in RUNTIME_PATHS:
                    validate_tirx_runtime(result, shape, tirx, label.endswith("_fast"))
                else:
                    validate_native_metadata(result, Case("gemm", *shape), "metal", tirx["execution_scope"],
                                             tirx["pipeline_window"], tirx["cooperative_matrix"], tirx["gemm_block"],
                                             not tirx["no_vectorize"], tirx["auto_vectorize"], tirx["group_threads"], tirx["copy_batch"])
                    if label in MPP_PATHS + VIEW_PATHS:
                        validate_mpp_tirx(result, label in VIEW_PATHS)
                    else:
                        validate_tirx_realization(result, "simdgroup")
                    validate_subgroup_policy(result, tirx.get("elide_independent_subgroup_barriers", False))
                if output.stat().st_size != reference.size * 4:
                    raise ValueError("incorrect output byte count")
                actual = np.fromfile(output, dtype="<f4").reshape(shape[:2])
        validate_times(result, args.samples)
        correctness = validate_output(np, actual, reference)
        summarize(result)
        if getattr(args, "metal_device_timing", None) is not None:
            for phase in ("throughput", "latency"):
                result[f"gpu_control_{phase}_us_p50"] = percentile(gpu_samples(result, label, phase, args.samples), .5)
        row.update(valid=True, measurement=result, correctness=correctness)
    except Exception as error:
        row["error"] = str(error)
    return row


def save(report: dict[str, Any], folder: Path) -> None:
    (folder / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    paths = tuple(report["metadata"].get("paths", PATHS))
    period = 2 * len(paths)
    lines = ["# Tile lowering comparison", "",
             "FP32 C=A×B; compact row-major, no transpose or prepacking. All complete outputs use identical inputs "
             "and an FP64 oracle (atol=rtol=1e-4). Native and handwritten MPP use the same fixed configuration; TIRx has a separate recorded schedule. "
             "No search or minimum-of-rounds selection occurs here.", "",
             "All table values are synchronized host-wall batched throughput (µs/call), including each runtime's dispatch, "
             "encoding/submission and synchronization overhead. Compilation, allocation and transfers are excluded. "
             "They are NOT GPU kernel times. Raw MPS/MPP GPU intervals are retained only in JSON; no cross-metric ratios are computed. "
             "Torch is its default eager MPS path, not a claim that it uses the direct MPS benchmark's kernel.", "",
             f"{period} rounds balance positions and pairwise precedence. Fewer rounds are explicitly an unbalanced smoke/exploration run. "
             "No failed or slow row is removed. Ratios below are paired round medians, not confidence intervals.", "",
             f"Order balanced: {report['metadata']['balanced']}. Rounds: {report['metadata']['rounds']}.", "",
             "Native/hand MPP and TIRx/Luisa use fast math off; TIRx/Luisa fast explicitly uses it on. "
             "TVMx's current Metal runtime hardcodes fast math on. Optional TIRx Runtime controls must have identical "
             "generated Metal source hashes and matching threadgroup widths. Compiler language/resource options and submission APIs "
             "can still differ; this is not an isolated GPU execution-overhead measurement.", "",
             "Optional TIRx→MPP is the patched TVM Metal code generator using non-owning memory inputs, "
             "not the native MPP emitter. It reuses the frozen TIRx geometry; current MPP reports use the separately "
             "versioned metal_mpp_memory_v2 relative-work model, not an instruction count, measured register use, "
             "or calibrated time prediction.", "",
             "Optional TIRx→MPP views enables proven read-only snapshot forwarding, with a separately frozen schedule. "
             "It is not a same-geometry ablation unless the recorded schedules match; original TIRx and non-forwarding MPP remain controls.", "",
             f"MPP-view subgroup-fence policy override: {report['metadata'].get('tirx_view_subgroup_fences', 'reported')}. "
             "The default is retention; requesting elision still requires a reported whole-group independence proof. "
             "This policy is not assumed profitable.", "",
             "| M×N×K | Valid rounds | " + " | ".join(TITLES[p] for p in paths) + " | Native/MPS | Native/hand MPP |",
             "|---|---:" + "|---:" * (len(paths) + 2) + "|"]
    for shape in report["metadata"]["shapes"]:
        rows = [row for row in report["results"] if row["shape"] == shape]
        valid = [row for row in rows if row["valid"]]
        name = "×".join(map(str, shape))
        count = f"{len(valid)}/{report['metadata']['rounds']}"
        if len(valid) != report["metadata"]["rounds"]:
            lines.append(f"| {name} | {count} | INCOMPLETE " + "| — " * (len(paths) + 1) + "|")
            continue
        times = {p: [r[p]["measurement"]["throughput_us_p50"] for r in valid] for p in paths}
        columns = " | ".join(f"{percentile(times[p], .5):.3f}" for p in paths)
        ratios = [percentile([a / b for a, b in zip(times[PATHS[0]], times[p])], .5) for p in ("mps", "handwritten_mpp")]
        lines.append(f"| {name} | {count} | {columns} | {ratios[0]:.3f} | {ratios[1]:.3f} |")
    if report["metadata"].get("metal_device_timing"):
        lines += ["", "## Separate GPU command-buffer controls", "",
                  "These no-counter GPU intervals include work and gaps inside completed command buffers, not host encoding or waits. "
                  "They are not isolated-kernel timestamps. Handwritten MPP uses its existing direct command-buffer timer; "
                  "other paths use the shared timing helper's uninstrumented control phase. Compute-pass counters remain diagnostic JSON only. "
                  "Do not subtract independently measured GPU/host medians to infer dispatch cost.", "",
                  "| M×N×K | Valid rounds | " + " | ".join(TITLES[p] + " GPU µs" for p in paths) + " |",
                  "|---|---:" + "|---:" * len(paths) + "|"]
        for shape in report["metadata"]["shapes"]:
            rows = [r for r in report["results"] if r["shape"] == shape and r["valid"]]
            name = "×".join(map(str, shape))
            count = f"{len(rows)}/{report['metadata']['rounds']}"
            if len(rows) != report["metadata"]["rounds"]:
                lines.append(f"| {name} | {count} | INCOMPLETE " + "| — " * (len(paths) - 1) + "|")
            else:
                values = [percentile([r[p]["measurement"]["gpu_control_throughput_us_p50"] for r in rows], .5) for p in paths]
                lines.append(f"| {name} | {count} | " + " | ".join(f"{v:.3f}" for v in values) + " |")
    lines += ["", f"Artifacts unchanged: {report['metadata'].get('artifacts_unchanged', 'pending')}.", "",
              "Raw samples, configs, ordering, failed paths and provenance: [results.json](results.json).", ""]
    (folder / "results.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native", "tirx", "mpp", "mps", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--shape", action="append", default=[])
    parser.add_argument("--mpp-config", default="32,32,1,1,0,1,4,4")
    parser.add_argument("--mpp-plan", type=Path, help="reuse selected configs only; never reuse historical times")
    parser.add_argument("--tirx-plan", type=Path, help="frozen run.py report; otherwise use group 32x32x32, window 1")
    parser.add_argument("--tirx-runtime-controls", action="store_true", help="add Luisa Runtime TIRx with fast math off/on; require matching device source")
    parser.add_argument("--tirx-mpp", action="store_true", help="add independent TVM MPP codegen with the same frozen TIRx geometry")
    parser.add_argument("--tirx-view-plan", type=Path, help="add read-only-forwarding TIRx MPP with this separately frozen run.py schedule")
    parser.add_argument("--tirx-view-subgroup-fences", choices=("reported", "retain", "elide"), default="reported",
                        help="explicit synchronization policy override, only for the MPP-view path")
    parser.add_argument("--compiler-artifact", type=Path, action="append", default=[], help="also fingerprint externally linked compiler/runtime libraries")
    parser.add_argument("--metal-device-timing", type=Path, help="also measure separate GPU controls with the prebuilt Metal timing helper")
    parser.add_argument("--rounds", type=int, help="default: twice the number of implementations")
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--sample-ms", type=int, default=30)
    parser.add_argument("--warmup-ms", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.metal_device_timing is not None:
        args.metal_device_timing = args.metal_device_timing.resolve(strict=True)
        args.compiler_artifact.append(args.metal_device_timing)
    if args.tirx_view_subgroup_fences != "reported" and not args.tirx_view_plan:
        parser.error("subgroup-fence override requires --tirx-view-plan")
    paths = PATHS + RUNTIME_PATHS if args.tirx_runtime_controls else PATHS
    if args.tirx_mpp:
        paths += MPP_PATHS
        if not args.compiler_artifact:
            parser.error("--tirx-mpp requires --compiler-artifact entries for the patched TVM libraries")
    if args.tirx_view_plan:
        if not args.tirx_mpp:
            parser.error("--tirx-view-plan requires --tirx-mpp so the non-forwarding control remains present")
        paths += VIEW_PATHS
    if args.rounds is None:
        args.rounds = 2 * len(paths)
    if min(args.rounds, args.samples, args.sample_ms, args.warmup_ms, args.timeout, args.threads) <= 0:
        parser.error("positive timing/thread settings required")
    shapes = list(dict.fromkeys(parse_shape(x) for x in args.shape)) or DEFAULT_SHAPES
    mpp_plan = json.loads(args.mpp_plan.read_text())["selected"] if args.mpp_plan else {}
    tirx_plan = load_plan(args.tirx_plan, {"gemm"}) if args.tirx_plan else {}
    configs = {s: parse_config(",".join(map(str, mpp_plan["x".join(map(str, s))]))) if args.mpp_plan else parse_config(args.mpp_config) for s in shapes}
    if any(not c[3] or c[4] or not c[5] for c in configs.values()):
        parser.error("native bootstrap only supports cooperative output, dynamic K and inline tensors")
    schedules = {s: tirx_plan["metal", Case("gemm", *s).name] if args.tirx_plan else dict(
        gemm_block=(32, 32, 32), execution_scope="group", pipeline_window=1, cooperative_matrix=True,
        no_vectorize=False, auto_vectorize=False, group_threads=128, copy_batch=1) for s in shapes}
    if args.tirx_mpp and any(not c["cooperative_matrix"] or c["execution_scope"] != "group" for c in schedules.values()):
        parser.error("TIRx MPP requires frozen cooperative group schedules")
    view_plan = load_plan(args.tirx_view_plan, {"gemm"}) if args.tirx_view_plan else {}
    view_schedules = {s: view_plan["metal", Case("gemm", *s).name] for s in shapes} if view_plan else {}
    if args.tirx_view_subgroup_fences != "reported":
        for config in view_schedules.values():
            config["elide_independent_subgroup_barriers"] = args.tirx_view_subgroup_fences == "elide"
    if any(not c["cooperative_matrix"] or c["execution_scope"] != "group" or c["matrix_realization"] != "mpp-views" for c in view_schedules.values()):
        parser.error("view plan must explicitly select forwarding MPP with cooperative group schedules")
    for key in ("TVM_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = str(args.threads)
    removed = {key: os.environ.pop(key, None) for key in (
        "PYTORCH_MPS_FAST_MATH", "PYTORCH_MPS_PREFER_METAL", "PYTORCH_ENABLE_MPS_FALLBACK",
        "LUISA_ENABLE_VALIDATION", "MTL_DEBUG_LAYER", "MTL_SHADER_VALIDATION",
        "DYLD_PRINT_LIBRARIES", "LUISA_TILE_BENCH_DUMP_SOURCE", "LUISA_TILE_BENCH_METAL_TIMING")}
    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if not torch.backends.mps.is_available():
        parser.error("MPS unavailable; CPU fallback is not allowed")
    binaries = []
    for label in ("native", "tirx", "mpp", "mps"):
        path = getattr(args, label).resolve(strict=True)
        setattr(args, label, path)
        binaries.append(path)
    artifacts = set(binaries)
    artifacts.update(p.resolve(strict=True) for p in args.compiler_artifact)
    for path in binaries:
        artifacts.update(p.resolve() for p in path.parent.iterdir() if p.is_file() and p.suffix in (".dylib", ".so"))
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    hashes = {str(p): digest(p) for p in sorted(artifacts)}
    root = Path(__file__).resolve().parents[3]
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"metadata": dict(
        timestamp=dt.datetime.now(dt.timezone.utc).isoformat(), platform=platform.platform(),
        torch_version=torch.__version__, torch_config=torch.__config__.show(), torch_git_version=torch.version.git_version,
        numpy_version=np.__version__, removed_environment=removed, threads=args.threads,
        tirx_view_subgroup_fences=args.tirx_view_subgroup_fences,
        git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        worktree_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root)),
        artifacts_sha256=hashes, plan_sha256={str(p): digest(p) for p in (args.mpp_plan, args.tirx_plan, args.tirx_view_plan) if p},
        shapes=[list(s) for s in shapes], paths=paths, rounds=args.rounds, balanced=args.rounds % (2 * len(paths)) == 0,
        samples=args.samples, sample_ms=args.sample_ms, warmup_ms=args.warmup_ms,
        metal_device_timing=str(args.metal_device_timing) if args.metal_device_timing else None,
        timing="synchronized_device_resident_host_wall"), "results": []}
    references = {s: oracle(np, s) for s in shapes}
    for ri in range(args.rounds):
        shift = ri % len(shapes)
        for shape in shapes[shift:] + shapes[:shift]:
            order = implementation_order(ri, shapes.index(shape), paths)
            row = dict(shape=list(shape), round=ri, order=order, config=configs[shape], tirx_schedule=schedules[shape], valid=True)
            if view_schedules:
                row["tirx_view_schedule"] = view_schedules[shape]
            for label in order:
                print(f"round {ri + 1}/{args.rounds} {shape} {label} ...", flush=True)
                schedule = view_schedules[shape] if label in VIEW_PATHS else schedules[shape]
                row[label] = measure(args, np, torch, label, shape, configs[shape], schedule, references[shape])
                row["valid"] &= row[label]["valid"]
                print(f"  {row[label]['measurement']['throughput_us_p50']:.3f} us; validated" if row[label]["valid"] else f"  FAILED: {row[label]['error']}", flush=True)
            if args.tirx_runtime_controls:
                try:
                    validate_runtime_controls(row)
                    row["matching_device_source"] = True
                except ValueError as error:
                    row.update(valid=False, matching_device_source=False, control_error=str(error))
                    print(f"  CONTROL FAILED: {error}", flush=True)
            report["results"].append(row)
            save(report, args.output)
    report["metadata"]["artifacts_unchanged"] = all(p.is_file() and digest(p) == hashes[str(p)] for p in artifacts)
    save(report, args.output)
    return int(not report["metadata"]["artifacts_unchanged"] or any(not r["valid"] for r in report["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
