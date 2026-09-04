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
import re
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
        elif operation in ("add", "sum", "softmax", "rmsnorm"):
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


def mapping_candidates(threads: int, batch: int, thread_list: str | None,
                       batch_list: str | None) -> list[tuple[int, int]]:
    if thread_list is None and batch_list is None:
        return []
    widths = [int(value) for value in thread_list.split(",")] if thread_list is not None else [threads]
    batches = [int(value) for value in batch_list.split(",")] if batch_list is not None else [batch]
    if any(not 0 <= value <= 0xffffffff for value in widths):
        raise ValueError("tuning group threads must be uint32; zero requests the planner's automatic choice")
    if any(not 1 <= value <= 16 for value in batches):
        raise ValueError("tuning copy batches must be in [1,16]")
    return list(dict.fromkeys((width, copies) for width in widths for copies in batches))


def joint_candidates(args: argparse.Namespace) -> list[tuple[tuple[int, int, int], int, int, int]]:
    blocks = args.tuning_candidates or [(args.gemm_block, args.pipeline_window)]
    mappings = getattr(args, "mapping_tuning_candidates", []) or [
        (getattr(args, "group_threads", 0), getattr(args, "copy_batch", 1))]
    budget = getattr(args, "max_tuning_candidates", 256)
    if budget <= 0 or len(blocks) * len(mappings) > budget:
        raise ValueError("joint JIT candidate budget exceeded; constrain the lists or increase --max-tuning-candidates")
    return [(block, window, threads, batch) for block, window in blocks for threads, batch in mappings]


def validate_native_metadata(native: dict[str, Any], case: Case, backend: str, execution_scope: str,
                             pipeline_window: int = 2, cooperative_matrix: bool = False,
                             gemm_block: tuple[int, int, int] = (8, 8, 16), vectorize: bool = True,
                             auto_vectorize: bool = False, group_threads: int = 0,
                             copy_batch: int = 1, metal_subgroup_reductions: bool = False) -> None:
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
    if native.get("metal_subgroup_reductions", False) is not metal_subgroup_reductions:
        raise RuntimeError("native Metal SIMD-group reduction policy does not match the request")
    if metal_subgroup_reductions:
        plans = native.get("execution_plans")
        if not isinstance(plans, list) or not plans or any(
                plan.get("optimized") is not True or type(plan.get("independent_subgroups")) is not bool or
                type(plan.get("threads")) is not int or plan["threads"] < 32 or plan["threads"] % 32
                for plan in plans):
            raise RuntimeError("Metal SIMD-group reduction request was not realized by a valid execution plan")
    requested_threads = native.get("planner_threads", 0)
    if type(requested_threads) is not int or requested_threads != group_threads:
        raise RuntimeError("native group-thread constraint does not match the request")
    if group_threads:
        plans = native.get("execution_plans")
        if not isinstance(plans, list) or not plans or any(p.get("threads") != group_threads for p in plans):
            raise RuntimeError("native realized group threads do not match the exact constraint")
    reported_copy_batch = native.get("copy_batch", 1)
    if type(reported_copy_batch) is not int or reported_copy_batch != copy_batch:
        raise RuntimeError("native copy-batch policy does not match the request")
    if copy_batch != 1:
        plans = native.get("execution_plans")
        if not isinstance(plans, list) or not plans or any(p.get("max_copy_batch") != copy_batch for p in plans):
            raise RuntimeError("native copy-batch plan does not match the requested limit")
    calls = native.get("matrix_intrinsics")
    if type(calls) is not int or calls < 0:
        raise RuntimeError("native matrix-intrinsic count must be a nonnegative integer")
    eligible = (cooperative_matrix and backend == "metal" and execution_scope == "group"
                and case.operation == "gemm" and all(size % 8 == 0 for size in gemm_block)
                and (group_threads == 0 or group_threads >= 32 and group_threads % 32 == 0))
    if bool(calls) != eligible:
        raise RuntimeError("generated matrix-intrinsic calls do not match the benchmark's eligible path")


def validate_tirx_realization(native: dict[str, Any], realization: str, cpu_input_views: bool = False,
                              metal_subgroup_reductions: bool = False) -> None:
    if type(cpu_input_views) is not bool:
        raise ValueError("CPU input-view policy must be boolean")
    if cpu_input_views:
        if realization != "simdgroup" or native.get("backend") != "cpu" or native.get("metal_mpp", False) is not False or native.get("forward_readonly_tile_loads") is not True:
            raise ValueError("CPU input views require explicit LLVM forwarding without MPP")
        return
    if metal_subgroup_reductions:
        if realization != "simdgroup" or native.get("backend") != "metal" or native.get("metal_mpp", False) is not False or native.get("forward_readonly_tile_loads") is not True:
            raise ValueError("Metal SIMD-group reductions require the reference TIRx bridge with proved input views")
        return
    if realization == "simdgroup":
        if native.get("metal_mpp", False) is not False or native.get("forward_readonly_tile_loads", False) is not False:
            raise ValueError("reference TIRx must not silently enable MPP or view forwarding")
        return
    if realization not in ("mpp", "mpp-views"):
        raise ValueError("unknown TIRx matrix realization")
    if native.get("metal_mpp") is not True or native.get("forward_readonly_tile_loads") is not (realization == "mpp-views"):
        raise ValueError("TIRx MPP/view policy mismatch")
    calls = native.get("mpp_intrinsics")
    if type(calls) is not int or calls <= 0 or calls != native.get("matrix_intrinsics") or native.get("simdgroup_intrinsics") != 0:
        raise ValueError("MPP requires actual generated MPP calls without SIMD-group fallback")


def validate_subgroup_policy(native: dict[str, Any], elide: bool) -> None:
    if native.get("elide_independent_subgroup_barriers", False) is not elide:
        raise ValueError("TIRx subgroup-fence policy mismatch")
    if elide:
        plans = native.get("execution_plans")
        if not isinstance(plans, list) or not plans:
            raise ValueError("subgroup-fence elision requires reported proof results")
        for plan in plans:
            if type(plan.get("independent_subgroups")) is not bool:
                raise ValueError("missing subgroup independence proof result")
            counts = [plan.get(key) for key in ("group_barrier_sites_before", "group_barrier_sites_after")]
            if any(type(value) is not int or value < 0 for value in counts) or counts[1] > counts[0]:
                raise ValueError("invalid subgroup barrier-site counts")
            if plan["independent_subgroups"] and plan.get("group_barrier_sites_after") != 0:
                raise ValueError("subgroup-fence elision was not realized")


def implementation_order(ordinal: int, system_baseline: bool = False) -> tuple[str, ...]:
    if not system_baseline:
        return ("native", "torch") if ordinal % 2 == 0 else ("torch", "native")
    # Six orders balance both position and pairwise precedence over six rounds.
    return (("native", "torch", "system"), ("system", "torch", "native"),
            ("torch", "system", "native"), ("native", "system", "torch"),
            ("system", "native", "torch"), ("torch", "native", "system"))[ordinal % 6]


def validate_system_metadata(result: dict[str, Any], case: Case, backend: str, samples: int) -> None:
    expected = dict(backend=backend, operation="gemm", dtype="float32", layout="compact_row_major",
                    m=case.m, n=case.n, k=case.k, alpha=1, beta=0,
                    transpose_left=False, transpose_right=False,
                    row_bytes=[case.k * 4, case.n * 4, case.n * 4],
                    implementation="accelerate_cblas_sgemm" if backend == "cpu" else "mps_matrix_multiplication",
                    api_variant="classic_lp64" if backend == "cpu" else "MPSKernelOptionsNone",
                    storage="host" if backend == "cpu" else "private",
                    batch_policy="synchronous_calls" if backend == "cpu" else "one_command_buffer_per_batch")
    if case.operation != "gemm" or backend not in ("cpu", "metal"):
        raise RuntimeError("system baselines support only CPU/Metal GEMM")
    for key, value in expected.items():
        if type(result.get(key)) is not type(value) or result[key] != value:
            raise RuntimeError(f"system baseline metadata mismatch: {key}")
    if type(result.get("repetitions")) is not int or not 1 <= result["repetitions"] <= 100000:
        raise RuntimeError("system baseline repetition count is invalid")
    for metric in ("throughput_us", "latency_us"):
        values = result.get(metric)
        if not isinstance(values, list) or len(values) != samples or any(
                type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values):
            raise RuntimeError(f"system baseline samples are invalid: {metric}")


def validate_cpu_storage_policy(native: dict[str, Any], requested: int) -> None:
    actual = native.get("cpu_stack_bytes", 0)
    if type(requested) is not int or not 0 <= requested <= 65536 or type(actual) is not int or actual != requested:
        raise ValueError("native CPU stack budget differs from the requested policy")


def validate_cpu_vector_policy(native: dict[str, Any], requested: int) -> None:
    actual = native.get("cpu_vector_lanes", 16)
    if type(requested) is not int or requested not in (16, 32, 64, 128) or type(actual) is not int or actual != requested:
        raise ValueError("native CPU vector lanes differ from the requested policy")


def validate_cpu_target_policy(native: dict[str, Any], requested: str | None, backend: str,
                               source: bytes | None = None, expected_model: str | None = None) -> None:
    if requested not in (None, "generic", "native") or (requested is not None and backend != "cpu"):
        raise ValueError("CPU target policy requires a CPU case and generic/native")
    policy = native.get("cpu_target_policy", "generic")
    model = native.get("cpu_model", "generic")
    if policy != (requested or "generic") or not isinstance(model, str) or not model:
        raise ValueError("native CPU target policy/model differs from the request")
    if requested is not None and not {"cpu_target_policy", "cpu_model"} <= native.keys():
        raise ValueError("explicit CPU target policy requires reported model metadata")
    if (policy == "generic" and model != "generic") or (policy == "native" and model in ("generic", "native")):
        raise ValueError("native CPU model was not resolved, or generic fallback occurred")
    if expected_model is not None and model != expected_model:
        raise ValueError("resolved CPU model differs from the frozen plan")
    if backend == "cpu" and source is not None:
        models = set(re.findall(rb'"target-cpu"="([^"]+)"', source))
        if models != {model.encode()}:
            raise ValueError("generated LLVM CPU model differs from the reported model")


def validate_cpu_matrix_policy(native: dict[str, Any], requested: str, backend: str,
                               operation: str) -> None:
    if requested not in ("reference", "cblas"):
        raise ValueError("CPU matrix realization must be reference or cblas")
    actual = native.get("cpu_matrix_backend", "reference")
    if actual != requested:
        raise ValueError("native CPU matrix realization differs from the request")
    calls = native.get("external_matrix_calls", 0)
    if type(calls) is not int or calls < 0:
        raise ValueError("native external-matrix call count must be a nonnegative integer")
    eligible = requested == "cblas" and backend == "cpu" and operation == "gemm"
    if (requested == "cblas") != eligible:
        raise ValueError("CBLAS realization requires a CPU GEMM")
    if calls != int(eligible):
        raise ValueError("generated external-matrix calls do not match the requested realization")


def validate_cpu_math_policy(native: dict[str, Any], requested: str, backend: str,
                             operation: str) -> None:
    if requested not in ("reference", "accelerate"):
        raise ValueError("CPU array-math realization must be reference or accelerate")
    actual = native.get("cpu_math_backend", "reference")
    if actual != requested:
        raise ValueError("native CPU array-math realization differs from the request")
    calls = native.get("external_vector_math_calls", 0)
    if type(calls) is not int or calls < 0:
        raise ValueError("native external-vector-math call count must be a nonnegative integer")
    if requested == "accelerate" and backend != "cpu":
        raise ValueError("Accelerate array math requires a CPU case")
    eligible = requested == "accelerate" and operation in ("sum", "softmax")
    if bool(calls) != eligible:
        raise ValueError("generated external-vector-math calls do not match the requested realization")


def optional_native_arguments(args: argparse.Namespace) -> list[str]:
    group_threads = getattr(args, "group_threads", 0)
    copy_batch = getattr(args, "copy_batch", 1)
    elide = getattr(args, "elide_independent_subgroup_barriers", False)
    cpu_stack = getattr(args, "cpu_stack_bytes", 0)
    cpu_lanes = getattr(args, "cpu_vector_lanes", 16)
    cpu_views = getattr(args, "cpu_input_views", False)
    cpu_model = getattr(args, "cpu_model", None)
    cpu_matrix = getattr(args, "cpu_matrix_backend", "reference")
    cpu_math = getattr(args, "cpu_math_backend", "reference")
    result = []
    # Do not send new options to frozen executables when no new policy was
    # requested. Later options require explicit padding of every prior slot.
    if group_threads or copy_batch != 1 or elide or cpu_stack or cpu_lanes != 16 or cpu_views or cpu_model is not None or cpu_matrix != "reference" or cpu_math != "reference":
        result.append(str(group_threads) if group_threads else "auto")
    if copy_batch != 1 or elide or cpu_stack or cpu_lanes != 16 or cpu_views or cpu_model is not None or cpu_matrix != "reference" or cpu_math != "reference":
        result.append(str(copy_batch))
    if elide or cpu_stack or cpu_lanes != 16 or cpu_views or cpu_model is not None or cpu_matrix != "reference" or cpu_math != "reference":
        result.extend(("tvm", "elide-subgroup-fences" if elide else "retain-subgroup-fences"))
    if cpu_stack or cpu_lanes != 16 or cpu_views or cpu_model is not None or cpu_matrix != "reference" or cpu_math != "reference":
        result.append(str(cpu_stack))
    if cpu_lanes != 16 or cpu_views or cpu_model is not None or cpu_matrix != "reference" or cpu_math != "reference":
        result.append(str(cpu_lanes))
    if cpu_views or cpu_model is not None or cpu_matrix != "reference" or cpu_math != "reference":
        result.append("forward-input-views" if cpu_views else "retain-input-snapshots")
    if cpu_model is not None or cpu_matrix != "reference" or cpu_math != "reference":
        result.append(cpu_model or "generic")
    if cpu_matrix != "reference" or cpu_math != "reference":
        result.append(cpu_matrix)
    if cpu_math != "reference":
        result.append(cpu_math)
    return result


def run_case(torch: Any, np: Any, args: argparse.Namespace, case: Case, backend: str, ordinal: int) -> dict[str, Any]:
    def inputs(rows: int, columns: int, seed: int) -> Any:
        indices = torch.arange(rows * columns, dtype=torch.int64)
        return (((indices * seed + 17) % 127 - 63).float() / 64).reshape(rows, columns)

    a_host = inputs(case.m, case.k if case.operation == "gemm" else case.n, 5)
    b_rows = case.k if case.operation == "gemm" else 1 if case.operation == "rmsnorm" else case.m
    b_host = inputs(b_rows, case.n, 11) if case.operation in ("gemm", "add", "rmsnorm") else None
    if case.operation == "gemm":
        reference = a_host.double() @ b_host.double()
    elif case.operation == "add":
        reference = a_host.double() + b_host.double()
    elif case.operation == "sum":
        reference = a_host.double().sum(dim=1)
    elif case.operation == "rmsnorm":
        x = a_host.double()
        reference = x * torch.rsqrt((x * x).mean(dim=1, keepdim=True) + 1e-5) * b_host.double()
    else:
        reference = a_host.double().softmax(dim=1)

    result: dict[str, Any] = {"case": dataclasses.asdict(case), "name": case.name, "backend": backend,
                             "block": block_shape(case, args.gemm_block), "timing_order": "native_first" if ordinal % 2 == 0 else "torch_first"}
    system_binary = getattr(args, "system_baseline", None)
    order = implementation_order(ordinal, system_binary is not None and case.operation == "gemm")
    result["implementation_order"] = order
    if len(order) == 3:
        result["timing_order"] = "_then_".join(order)

    def run_native() -> None:
        with tempfile.TemporaryDirectory(prefix="luisa-tile-benchmark-") as temporary:
            output = Path(temporary) / "output.f32"
            realization = getattr(args, "matrix_realization", "simdgroup")
            matrix_mode = ("subgroup-reduce" if getattr(args, "metal_subgroup_reductions", False) else
                           realization if realization != "simdgroup" else
                           "matrix" if args.cooperative_matrix else "scalar")
            command = [str(args.native), backend, case.operation, str(case.m), str(case.n), str(case.k),
                       *(str(x) for x in result["block"]), str(args.samples), str(args.sample_ms), str(args.warmup_ms), str(output),
                       args.execution_scope, str(args.pipeline_window), matrix_mode,
                       "auto-vectorize" if args.auto_vectorize else "no-vectorize" if args.no_vectorize else "vectorize"]
            group_threads = getattr(args, "group_threads", 0)
            copy_batch = getattr(args, "copy_batch", 1)
            elide = getattr(args, "elide_independent_subgroup_barriers", False)
            cpu_stack = getattr(args, "cpu_stack_bytes", 0)
            cpu_lanes = getattr(args, "cpu_vector_lanes", 16)
            cpu_views = getattr(args, "cpu_input_views", False)
            cpu_model = getattr(args, "cpu_model", None)
            cpu_matrix = getattr(args, "cpu_matrix_backend", "reference")
            cpu_math = getattr(args, "cpu_math_backend", "reference")
            command.extend(optional_native_arguments(args))
            environment = os.environ.copy()
            capture_source = getattr(args, "capture_sources", False)
            suffix = ".metal" if backend == "metal" else ".ll"
            source_path = Path(temporary) / ("device" + suffix)
            if capture_source:
                environment["LUISA_TILE_BENCH_DUMP_SOURCE"] = str(source_path)
            process = subprocess.run(command, capture_output=True, text=True, check=False, timeout=args.timeout, env=environment)
            if process.returncode:
                raise RuntimeError(f"native benchmark failed ({process.returncode}):\n{process.stderr}\n{process.stdout}")
            lines = [line for line in process.stdout.splitlines() if line.startswith("{")]
            if len(lines) != 1:
                raise RuntimeError("native executable did not emit exactly one JSON result")
            native = json.loads(lines[0])
            result["native_command"] = command
            result["native_stderr"] = process.stderr
            if capture_source:
                source = source_path.read_bytes()
                result["native_source_sha256"] = hashlib.sha256(source).hexdigest()
                source_dir = args.output / "sources"
                source_dir.mkdir(exist_ok=True)
                destination = source_dir / (result["native_source_sha256"] + suffix)
                if not destination.exists():
                    destination.write_bytes(source)
            validate_tirx_realization(native, realization, cpu_views, getattr(args, "metal_subgroup_reductions", False))
            validate_subgroup_policy(native, elide)
            validate_cpu_storage_policy(native, cpu_stack)
            validate_cpu_vector_policy(native, cpu_lanes)
            validate_cpu_target_policy(native, cpu_model, backend, source if capture_source else None,
                                       getattr(args, "expected_cpu_model", None))
            validate_cpu_matrix_policy(native, cpu_matrix, backend, case.operation)
            validate_cpu_math_policy(native, cpu_math, backend, case.operation)
            validate_native_metadata(native, case, backend, args.execution_scope, args.pipeline_window,
                                     args.cooperative_matrix, args.gemm_block, not args.no_vectorize, args.auto_vectorize, group_threads, copy_batch,
                                     getattr(args, "metal_subgroup_reductions", False))
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
        out = None if case.operation == "rmsnorm" else torch.empty(reference.shape, dtype=torch.float32, device=device)
        synchronize()
        allocation_upload_ms = (time.perf_counter_ns() - start) / 1e6
        if case.operation == "gemm":
            invoke = lambda: torch.mm(a, b, out=out)
        elif case.operation == "add":
            invoke = lambda: torch.add(a, b, out=out)
        elif case.operation == "sum":
            invoke = lambda: torch.sum(a, dim=1, out=out)
        elif case.operation == "rmsnorm":
            invoke = lambda: torch.nn.functional.rms_norm(a, (case.n,), b[0], eps=1e-5)
        else:
            invoke = lambda: torch.softmax(a, dim=1, out=out)
        measured = time_torch(invoke, synchronize, args)
        start = time.perf_counter_ns()
        actual = (invoke() if out is None else out).cpu()
        synchronize()
        measured["download_ms"] = (time.perf_counter_ns() - start) / 1e6
        measured["allocation_upload_ms"] = allocation_upload_ms
        measured["device"] = str(a.device)
        measured["output_policy"] = (
            "framework_return_value" if out is None else "preallocated_out")
        measured["correctness"] = validate(torch, actual, reference, case.operation)
        result["torch"] = measured

    def run_system() -> None:
        with tempfile.TemporaryDirectory(prefix="luisa-tile-system-") as temporary:
            output = Path(temporary) / "output.f32"
            command = [str(system_binary), backend, str(case.m), str(case.n), str(case.k),
                       str(args.samples), str(args.sample_ms), str(args.warmup_ms), str(output)]
            process = subprocess.run(command, capture_output=True, text=True, check=False, timeout=args.timeout)
            if process.returncode:
                raise RuntimeError(f"system baseline failed ({process.returncode}):\n{process.stderr}\n{process.stdout}")
            measured = json.loads(process.stdout)
            validate_system_metadata(measured, case, backend, args.samples)
            if output.stat().st_size != reference.numel() * 4:
                raise RuntimeError("system output byte count is incorrect")
            actual = torch.from_numpy(np.fromfile(output, dtype="<f4")).reshape(reference.shape)
            measured["correctness"] = validate(torch, actual, reference, case.operation)
            summarize(measured)
            result["system"] = measured

    with torch.inference_mode():
        actions = {"native": run_native, "torch": run_pytorch, "system": run_system}
        for implementation in order:
            actions[implementation]()
    result["slowdown"] = result["native"]["throughput_us_p50"] / result["torch"]["throughput_us_p50"]
    if "system" in result:
        result["system_slowdown"] = result["native"]["throughput_us_p50"] / result["system"]["throughput_us_p50"]
    result["valid"] = True
    return result


def run_tuned_case(torch: Any, np: Any, args: argparse.Namespace, case: Case,
                   backend: str, ordinal: int) -> dict[str, Any]:
    # Each candidate is ordinary host configuration: recapture and native JIT
    # happen again in run_case. No symbolic super-kernel or capture-once graph.
    candidates = joint_candidates(args)
    shift = ordinal % len(candidates)
    candidates = candidates[shift:] + candidates[:shift]
    trials: list[dict[str, Any]] = []
    start = time.perf_counter_ns()
    for index, (block, window, threads, batch) in enumerate(candidates):
        trial: dict[str, Any] = {"block": block, "pipeline_window": window,
                                "group_threads": threads, "copy_batch": batch}
        candidate_args = argparse.Namespace(**vars(args))
        candidate_args.gemm_block, candidate_args.pipeline_window = block, window
        candidate_args.group_threads, candidate_args.copy_batch = threads, batch
        print(f"  JIT trial {index + 1}/{len(candidates)}: block={block}, window={window}, "
              f"threads={threads or 'auto'}, copy_batch={batch}", flush=True)
        try:
            measured = run_case(torch, np, candidate_args, case, backend, ordinal * len(candidates) + index)
            score = measured["native"]["throughput_us_p50"]
            if not measured.get("valid") or not math.isfinite(score) or score <= 0:
                raise RuntimeError("candidate lacks a valid positive timing")
            trial.update(valid=True, measurement=measured)
            plans = measured.get("native", {}).get("execution_plans")
            if isinstance(plans, list) and plans:
                model_costs = [plan.get("normalized_kernel_cost") for plan in plans]
                if all(type(cost) in (int, float) and math.isfinite(cost) and cost >= 0 for cost in model_costs):
                    trial["model_cost"] = sum(model_costs)
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
    model_valid = [index for index in valid if "model_cost" in trials[index]]
    if model_valid:
        model_selected = min(model_valid, key=lambda index: trials[index]["model_cost"])
        tuning["model_selection_metric"] = "sum_execution_plan_normalized_kernel_cost"
        tuning["model_selected_trial"] = model_selected
        measured_best = trials[selected]["measurement"]["native"]["throughput_us_p50"]
        measured_model = trials[model_selected]["measurement"]["native"]["throughput_us_p50"]
        tuning["model_regret"] = measured_model / measured_best - 1.0
    winner = trials[selected]
    candidate_args = argparse.Namespace(**vars(args))
    candidate_args.gemm_block = winner["block"]
    candidate_args.pipeline_window = winner["pipeline_window"]
    candidate_args.group_threads = winner["group_threads"]
    candidate_args.copy_batch = winner["copy_batch"]
    print(f"  Selected block={winner['block']}, window={winner['pipeline_window']}, "
          f"threads={winner['group_threads'] or 'auto'}, copy_batch={winner['copy_batch']}; fresh validation/timing", flush=True)
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
             "Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; `functional.rms_norm` has no `out=` overload and its returned-output allocation remains inside warm timing. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).", "",
             f"Native GEMM retains an MMA in TileIR. CPU matrix realization: `{metadata.get('cpu_matrix_backend', 'reference')}`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `{metadata.get('cpu_math_backend', 'reference')}`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared FP32 exp materialization; the DSL and execution hierarchy remain target-independent. Cooperative-matrix capability requested: `{metadata.get('cooperative_matrix', False)}`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `{metadata.get('pipeline_window', 'unspecified')}`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.", "",
             "Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.", "",
             "| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["results"]:
        if not row.get("valid"):
            lines.append(f"| {row['backend']} | {row['name']} | FAILED | | | | | | | | |")
            continue
        native, pytorch = row["native"], row["torch"]
        calls = native.get('matrix_intrinsics', 0) + native.get('external_matrix_calls', 0)
        lines.append(f"| {row['backend']} | {row['name']} | {'×'.join(map(str, row['block']))} / {native['pipeline_window']} | {calls} | {native['throughput_us_p50']:.3f} | {pytorch['throughput_us_p50']:.3f} | {native['throughput_us_p90']:.3f} | {pytorch['throughput_us_p90']:.3f} | {row['slowdown']:.2f}× | {native['latency_us_p50']:.3f} | {pytorch['latency_us_p50']:.3f} |")
    lines.extend(["", "## Setup and cold-call phases", "", "Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.", "",
                  "| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in report["results"]:
        if row.get("valid"):
            a, b = row["native"], row["torch"]
            lines.append(f"| {row['backend']} / {row['name']} | {a['capture_ms']:.3f} | {a['compile_ms']:.3f} | {a['allocation_upload_ms']:.3f} | {b['allocation_upload_ms']:.3f} | {a['cold_call_ms']:.3f} | {b['cold_call_ms']:.3f} | {a['download_ms']:.3f} | {b['download_ms']:.3f} |")
    tuned = [row for row in report["results"] if "tuning" in row]
    if any("system" in row for row in report["results"]):
        lines.extend(["", "## Direct system-library GEMM baselines", "",
                      "Same FP32 inputs, compact row-major strides, alpha=1, beta=0, no transpose or reduced-precision option. CPU uses classic LP64 Accelerate cblas_sgemm; Metal uses MPSMatrixMultiplication (not MPSGraph) with private buffers and one command buffer per timed batch. Timings include API/encoding/submission costs, not setup or uploads. Complete outputs pass the same FP64 oracle. Raw samples and each case's implementation order are recorded in JSON; use compare_system.py for per-case six-order balance.", "",
                      "| Device / case | System implementation | System p50 µs | Native / system | System latency µs |",
                      "|---|---|---:|---:|---:|"])
        for row in report["results"]:
            if row.get("valid") and "system" in row:
                system = row["system"]
                lines.append(f"| {row['backend']} / {row['name']} | {system['implementation']} | {system['throughput_us_p50']:.3f} | {row['system_slowdown']:.3f}× | {system['latency_us_p50']:.3f} |")
    if tuned:
        lines.extend(["", "## JIT search", "",
                      "All candidates are recaptured, compiled, and checked against the same FP64 oracle. Invalid candidates are retained in JSON but cannot win. Candidate order rotates across cases. Tables above use a fresh post-selection run, not the search minimum; a revalidation failure remains a failure. This is not a confidence interval or an exhaustive search.", "",
                      "Selection wall time below includes JIT, validation, native/PyTorch measurements, and process overhead; it is excluded from warm timings. Full candidate settings, rejected cases, and raw trial samples are in results.json.", "",
                      "The model column is diagnostic only: it compares the lowest reported normalized kernel cost with the measured winner inside the same finite candidate set. Model regret is measured(model pick) / measured(best) - 1; it is not a hardware-optimality claim.", "",
                      "| Device / case | Valid / attempted candidates | Model pick / measured pick | Model regret | Selection wall ms |", "|---|---:|---|---:|---:|"])
        for row in tuned:
            tuning = row["tuning"]
            trials = tuning["trials"]
            if "model_selected_trial" in tuning:
                model = trials[tuning["model_selected_trial"]]
                measured = trials[tuning["selected_trial"]]
                choices = (f"{'×'.join(map(str, model['block']))} @ {model['group_threads']}t / "
                           f"{'×'.join(map(str, measured['block']))} @ {measured['group_threads']}t")
                regret = f"{100.0 * tuning['model_regret']:.2f}%"
            else:
                choices, regret = "unavailable", "unavailable"
            lines.append(f"| {row['backend']} / {row['name']} | {sum(trial['valid'] for trial in trials)} / {len(trials)} | {choices} | {regret} | {tuning['selection_wall_ms']:.3f} |")
    lines.extend(["", "Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).", ""])
    (directory / "results.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True, help="already-built benchmark_tile_tirx executable")
    parser.add_argument("--system-baseline", type=Path, help="optional prebuilt benchmark_tile_system; adds direct BLAS/MPS for GEMM")
    parser.add_argument("--output", type=Path, required=True, help="new directory for JSON and Markdown results")
    parser.add_argument("--backends", default="cpu,metal")
    parser.add_argument("--operations", default="gemm,add,sum,softmax")
    parser.add_argument("--gemm-block", default="8,8,16")
    parser.add_argument("--tune-gemm-blocks", help="opt-in JIT search, e.g. '8,8,16;16,32,32;32,32,32'; final timing is a fresh run")
    parser.add_argument("--tune-pipeline-windows", help="opt-in JIT windows, e.g. '1,2'; combined with tuning blocks")
    parser.add_argument("--tune-group-threads", help="opt-in Metal group widths, e.g. '128,256'; 0 requests the automatic planner")
    parser.add_argument("--tune-copy-batches", help="opt-in cooperative-copy batches, e.g. '1,4,8'; combined with other tuning lists")
    parser.add_argument("--max-tuning-candidates", type=int, default=256,
                        help="maximum joint candidates per shape; reject oversized searches without truncation")
    parser.add_argument("--execution-scope", choices=("auto", "worker", "group"), default="auto")
    parser.add_argument("--pipeline-window", type=int, choices=(1, 2), default=2,
                        help="GEMM scheduling window: 1 is ordered, 2 permits software prefetching")
    parser.add_argument("--cooperative-matrix", action="store_true",
                        help="assert native FP32 matrix capability (Metal requires Apple GPU family 7+); default off")
    parser.add_argument("--matrix-realization", choices=("simdgroup", "mpp", "mpp-views"), default="simdgroup",
                        help="independent TIRx realization; MPP options require a patched compiler and Metal group GEMM")
    parser.add_argument("--metal-subgroup-reductions", action="store_true",
                        help="opt in to proved Metal SIMD-group add/max/min reduction mapping and its FP32 tree order")
    parser.add_argument("--elide-independent-subgroup-barriers", action="store_true",
                        help="opt-in synchronization candidate; only proved independent MPP-view groups can elide fences")
    vectorization = parser.add_mutually_exclusive_group()
    vectorization.add_argument("--no-vectorize", action="store_true", help="disable TIRx vectorization")
    vectorization.add_argument("--auto-vectorize", action="store_true", help="opt in to experimental CPU independent-element SIMD packing; default off")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cpu-stack-bytes", type=int, default=0,
                        help="opt-in compiler-local LLVM stack payload budget (0..65536); zero retains workspace allocation")
    parser.add_argument("--cpu-vector-lanes", type=int, choices=(16, 32, 64, 128), default=16,
                        help="logical CPU SIMD-pack budget; >16 enables Cartesian row packing, not a hardware width")
    parser.add_argument("--cpu-input-views", action="store_true", help="opt in to proved immutable LLVM input views, retaining lazy bounds/zero-fill expressions")
    parser.add_argument("--cpu-model", choices=("generic", "native"),
                        help="CPU codegen model; native resolves and validates the host model through C++ LLVM APIs")
    parser.add_argument("--cpu-matrix-backend", choices=("reference", "cblas"), default="reference",
                        help="whole-GEMM CPU realization; cblas requires a proved TileIR contract and registered provider")
    parser.add_argument("--cpu-math-backend", choices=("reference", "accelerate"), default="reference",
                        help="CPU array math; Accelerate consumes proved FP32 reductions/shared exp materializations")
    parser.add_argument("--capture-sources", action="store_true", help="archive LLVM IR or Metal source by SHA256")
    parser.add_argument("--group-threads", type=int, default=0,
                        help="exact Metal group worker count; 0 lets the compiler planner choose (not CPU threads)")
    parser.add_argument("--copy-batch", type=int, default=1,
                        help="maximum in-flight values per Metal cooperative copy; 1 preserves scalar load/store order")
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--sample-ms", type=int, default=20)
    parser.add_argument("--warmup-ms", type=int, default=150)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--quick", action="store_true", help="smoke run; omits the large shape cases")
    args = parser.parse_args()
    if args.elide_independent_subgroup_barriers and args.matrix_realization != "mpp-views":
        parser.error("subgroup-fence elision currently requires mpp-views")
    args.native = args.native.resolve(strict=True)
    if args.system_baseline is not None:
        args.system_baseline = args.system_baseline.resolve(strict=True)
    args.output = args.output.resolve()
    try:
        args.gemm_block = parse_gemm_block(args.gemm_block)
        args.tuning_candidates = tuning_candidates(args.gemm_block, args.pipeline_window,
                                                  args.tune_gemm_blocks, args.tune_pipeline_windows)
        args.mapping_tuning_candidates = mapping_candidates(args.group_threads, args.copy_batch,
                                                            args.tune_group_threads, args.tune_copy_batches)
        candidates = joint_candidates(args)
    except ValueError as error:
        parser.error(str(error))
    if min(args.threads, args.samples, args.sample_ms, args.warmup_ms) <= 0:
        parser.error("block dimensions, thread count, and timing parameters must be positive")
    backends = args.backends.split(",")
    if not 0 <= args.cpu_stack_bytes <= 65536 or (args.cpu_stack_bytes and backends != ["cpu"]):
        parser.error("CPU stack budget must be in [0,65536] and requires only the CPU backend")
    if args.cpu_vector_lanes != 16 and (backends != ["cpu"] or not args.auto_vectorize):
        parser.error("non-default CPU vector lanes require only CPU with auto-vectorization")
    if args.cpu_input_views and backends != ["cpu"]:
        parser.error("CPU input views require only the CPU backend")
    if args.cpu_model is not None and backends != ["cpu"]:
        parser.error("CPU model selection requires only the CPU backend")
    if args.cpu_matrix_backend == "cblas" and (backends != ["cpu"] or args.operations != "gemm" or
                                                args.execution_scope != "auto"):
        parser.error("CBLAS realization requires only CPU GEMM with automatic execution binding")
    if args.cpu_math_backend == "accelerate" and backends != ["cpu"]:
        parser.error("Accelerate array math requires only the CPU backend")
    requested_operations = args.operations.split(",")
    if args.metal_subgroup_reductions and (backends != ["metal"] or args.execution_scope != "auto" or
                                           args.matrix_realization != "simdgroup" or args.cooperative_matrix or
                                           any(operation not in ("sum", "softmax", "rmsnorm") for operation in requested_operations)):
        parser.error("Metal SIMD-group reductions require only automatic Metal sum, softmax, or RMSNorm with the reference TIRx realization")
    if any(backend not in ("cpu", "metal") for backend in backends):
        parser.error("backends must be cpu and/or metal")
    if args.matrix_realization != "simdgroup" and (backends != ["metal"] or args.operations != "gemm" or
                                                  args.execution_scope != "group" or not args.cooperative_matrix):
        parser.error("MPP realizations require only Metal group GEMM with cooperative matrices enabled")
    if not 0 <= args.group_threads <= 0xffffffff or (args.group_threads and
            (backends != ["metal"] or (args.execution_scope != "group" and not args.metal_subgroup_reductions))):
        parser.error("group threads must be uint32; an explicit count requires Metal group execution or subgroup reductions")
    if not 1 <= args.copy_batch <= 16 or (args.copy_batch != 1 and (backends != ["metal"] or args.execution_scope != "group")):
        parser.error("copy batch must be in [1,16]; batching requires only Metal group execution")
    if args.metal_subgroup_reductions and args.tuning_candidates:
        parser.error("GEMM block/pipeline tuning does not apply to Metal SIMD-group reductions")
    if args.mapping_tuning_candidates and (
            backends != ["metal"] or
            (args.execution_scope != "group" and not args.metal_subgroup_reductions)):
        parser.error("mapping tuning requires Metal group execution or Metal SIMD-group reductions")
    if args.metal_subgroup_reductions and any(batch != 1 for _, batch in args.mapping_tuning_candidates):
        parser.error("copy-batch tuning does not apply to Metal SIMD-group reductions")
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
        "system_baseline": {"binary": str(args.system_baseline),
                            "sha256": hashlib.sha256(args.system_baseline.read_bytes()).hexdigest()} if args.system_baseline else None,
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
        "matrix_realization": args.matrix_realization,
        "metal_subgroup_reductions": args.metal_subgroup_reductions,
        "vectorize": not args.no_vectorize,
        "auto_vectorize": args.auto_vectorize,
        "cpu_stack_bytes": args.cpu_stack_bytes, "capture_sources": args.capture_sources,
        "cpu_vector_lanes": args.cpu_vector_lanes,
        "cpu_input_views": args.cpu_input_views,
        "cpu_model": args.cpu_model,
        "cpu_matrix_backend": args.cpu_matrix_backend,
        "cpu_math_backend": args.cpu_math_backend,
        "group_threads": args.group_threads,
        "copy_batch": args.copy_batch,
        "gemm_tuning_candidates": [{"block": block, "pipeline_window": window} for block, window in args.tuning_candidates],
        "joint_tuning_candidates": [dict(block=block, pipeline_window=window, group_threads=width, copy_batch=batch)
                                    for block, window, width, batch in candidates]
                                   if args.tuning_candidates or args.mapping_tuning_candidates else [],
        "max_tuning_candidates": args.max_tuning_candidates,
        "quick": args.quick, "timing": "synchronized device-resident host wall time including dispatch",
    }, "results": []}
    cases = make_cases(args.operations.split(","), args.quick)
    failed = False
    for backend in backends:
        for case in cases:
            print(f"{backend:5s} {case.name} ...", flush=True)
            try:
                tune = ((case.operation == "gemm" and
                         (args.tuning_candidates or args.mapping_tuning_candidates)) or
                        (args.metal_subgroup_reductions and
                         bool(args.mapping_tuning_candidates)))
                measure = run_tuned_case if tune else run_case
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
