#!/usr/bin/env python3
"""CUDA torch.compile baseline for exported benchmark_tile_workloads fixtures.

One process, manifest, precision and compiler mode per invocation. Inputs and
the complete FP64 oracle/bounds come from the native export, not a new fixture.
Compiled and optional eager expressions return normal functional outputs.
--check-inputs validates the packet without importing Torch or initializing CUDA.
GPU execution is explicit; this script never builds or changes system settings.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback


OPERATIONS = {"gemm", "gemv", "bmm", "softmax", "masked_softmax", "rmsnorm",
              "layernorm", "rope", "swiglu", "gelu_residual", "attention", "attention_tensorcore",
              "scan", "scan_ordered", "reduce_sum", "reduce_max", "argmax", "sort", "topk"}
PRECISIONS = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def shape(value):
    require(isinstance(value, list) and value and
            all(type(x) is int and 0 < x <= 65536 for x in value), "invalid tensor shape")
    require(math.prod(value) <= 2**28, "tensor exceeds the benchmark size limit")
    return tuple(value)


def expected_shapes(op, dims):
    if op in {"gemm", "gemv"}:
        m, n, k = dims
        require(op != "gemv" or n == 1, "GEMV requires N=1")
        return [(m, k), (k, n), (1,)], (m, n)
    if op == "bmm":
        b, m, n, k = dims
        return [(b, m, k), (b, k, n), (1,)], (b, m, n)
    if op in {"attention", "attention_tensorcore"}:
        b, h, kh, q, k, d, dv = dims
        return [(b, h, q, d), (b, kh, k, d), (b, kh, k, dv)], (b, h, q, dv)
    if op in {"sort", "topk"}:
        r, n, k = dims
        return [(r, n), (1,), (1,)], (r, k)
    r, n = dims
    aux = (1 if op in {"rmsnorm", "layernorm"} else r, n // 2 if op == "rope" else n)
    return [(r, n), aux, aux], (r, 1 if op in {"reduce_sum", "reduce_max", "argmax"} else n)


def check_semantics(manifest):
    op, semantics = manifest["operation"], manifest.get("semantics", {})
    constraints = {"accumulation": "float32"}
    if op == "rope":
        constraints["rope_pairing"] = "half_split"
    if op in {"softmax", "masked_softmax"}:
        constraints["softmax_axis"] = -1
        constraints["mask"] = "column<=row%width" if op == "masked_softmax" else "none"
    if op in {"scan", "scan_ordered"}:
        constraints["scan"] = "inclusive_sum"
    if op in {"sort", "topk"}:
        constraints.update(descending=True, stable=True, tie_break="original_index_ascending")
    if op in {"attention", "attention_tensorcore"}:
        constraints.update(causal=True, query_positions="last_Q_in_K")
        require(math.isfinite(semantics.get("attention_scale", float("nan"))) and semantics["attention_scale"] > 0,
                "missing/invalid attention scale")
    if op in {"rmsnorm", "layernorm"}:
        constraints["affine"] = True
        require(math.isfinite(semantics.get("epsilon", float("nan"))) and semantics["epsilon"] > 0,
                "missing/invalid normalization epsilon")
    if op == "gelu_residual":
        constraints["gelu_approximation"] = "tanh"
    for name, value in constraints.items():
        require(semantics.get(name) == value, f"unsupported/missing semantic contract: {name}")


def load_packet(path):
    import numpy as np
    path = Path(path).resolve(strict=True)
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    require(manifest.get("schema", manifest.get("schema_version")) == 1, "manifest schema must be 1")
    require(manifest["operation"] in OPERATIONS, "unsupported operation")
    require(manifest["precision"] in PRECISIONS, "unsupported precision")
    require(manifest.get("endianness") == "little", "fixture must declare little-endian storage")
    check_semantics(manifest)
    dimensions = manifest["dimensions"]
    require(isinstance(dimensions, list) and dimensions and
            all(type(x) is int and 0 < x <= 65536 for x in dimensions), "invalid operation dimensions")
    dims = tuple(dimensions)
    op = manifest["operation"]
    require(len(dims) == (7 if op in {"attention", "attention_tensorcore"} else 4 if op == "bmm" else
                         3 if op in {"gemm", "gemv", "sort", "topk"} else 2), "operation dimension arity mismatch")
    if op == "rope":
        require(dims[1] % 2 == 0, "RoPE requires an even width")
    if op in {"attention", "attention_tensorcore"}:
        require(dims[1] % dims[2] == 0 and dims[4] >= dims[3], "invalid attention GQA or causal dimensions")
    if op in {"sort", "topk"}:
        require(dims[2] <= dims[1] and (op != "sort" or dims[2] == dims[1]), "invalid ranking K")
    if op == "attention_tensorcore":
        require(manifest["precision"] in {"fp16", "bf16"}, "attention_tensorcore requires FP16/BF16")
        contract = manifest.get("precision_contract", {})
        require(contract.get("name") == "attention_single_narrow_probability_v1" and
                contract.get("stage") == "unnormalized_probability_before_pv_per_kv_block" and
                contract.get("rounding") == "rne", "missing/unsupported precision contract")
        require(contract.get("u_T") == (2**-11 if manifest["precision"] == "fp16" else 2**-8) and
                contract.get("eta_T") == (2**-24 if manifest["precision"] == "fp16" else 2**-133) and
                contract.get("u32") == 2**-24, "incorrect precision contract constants")
    receipts = {str(path): digest(path)}

    def read(entry, dtype, expected_dtype, key="path", tensor_shape=None):
        require(entry.get("storage_dtype", expected_dtype) == expected_dtype, "unexpected storage dtype")
        file = (path.parent / entry[key]).resolve(strict=True)
        sizes = shape(list(tensor_shape)) if tensor_shape is not None else shape(entry["shape"])
        expected_size = math.prod(sizes) * np.dtype(dtype).itemsize
        require(file.stat().st_size == expected_size, f"tensor byte count mismatch: {file}")
        value = np.fromfile(file, dtype=dtype).reshape(sizes)
        require(np.isfinite(value).all(), f"nonfinite fixture: {file}")
        receipts[str(file)] = digest(file)
        return value

    entries = manifest["inputs"]
    require(isinstance(entries, list) and entries, "missing input tensors")
    require([entry["name"] for entry in entries] == [f"input{i}" for i in range(len(entries))], "input order/name mismatch")
    storage_dtype = PRECISIONS[manifest["precision"]]
    def read_input(entry):
        if storage_dtype == "bfloat16":
            bits = read(entry, "<u2", storage_dtype)
            value = (bits.astype("<u4") << 16).view("<f4")
        else:
            value = read(entry, "<f2" if storage_dtype == "float16" else "<f4", storage_dtype).astype("<f4")
        require(np.isfinite(value).all(), "nonfinite decoded input storage")
        return value
    arrays = [read_input(entry) for entry in entries]
    output_shape = shape(manifest["output"]["shape"])
    input_shapes, required_output_shape = expected_shapes(op, dims)
    require([array.shape for array in arrays] == input_shapes and output_shape == required_output_shape,
            "exported tensor shapes differ from the operation contract")
    require(manifest["output"].get("storage_dtype") == storage_dtype, "output storage must match declared precision")
    expected_entry = manifest.get("expected", {"path": "expected.f64", "bound_path": "per_element_bound.f64", "storage_dtype": "float64"})
    expected = read(expected_entry, "<f8", "float64", tensor_shape=output_shape)
    bounds = read(expected_entry, "<f8", "float64", key="bound_path", tensor_shape=output_shape)
    require((bounds >= 0).all(), "negative oracle bound")
    strict_bounds = probability_bounds = None
    if op == "attention_tensorcore":
        require("strict_bound_path" in expected_entry and "probability_rounding_bound_path" in expected_entry,
                "missing strict/probability oracle bounds")
        strict_bounds = read(expected_entry, "<f8", "float64", key="strict_bound_path", tensor_shape=output_shape)
        probability_bounds = read(expected_entry, "<f8", "float64", key="probability_rounding_bound_path", tensor_shape=output_shape)
        require((strict_bounds >= 0).all() and (probability_bounds >= 0).all(), "negative secondary oracle bound")
        composed = strict_bounds + (1 + contract["u_T"]) * probability_bounds
        require(np.allclose(bounds, composed, rtol=4*np.finfo(np.float64).eps, atol=0), "primary bound composition mismatch")
    indices = None
    if op in {"sort", "topk", "argmax"}:
        require("indices" in manifest, "missing index oracle")
        indices = read(manifest["indices"], "<i8", "int64", key="expected_path")
        require(indices.shape == expected.shape, "value/index oracle shape mismatch")
    return dict(path=path, manifest=manifest, inputs=arrays, expected=expected,
                bounds=bounds, strict_bounds=strict_bounds, probability_bounds=probability_bounds,
                indices=indices, receipts=receipts)


def verify_packet(packet):
    for name, expected in packet["receipts"].items():
        require(digest(name) == expected, f"exported fixture changed: {name}")


def validate_output(packet, values, indices=None, *, ranking_contract="standard"):
    import numpy as np
    expected, bounds = packet["expected"], packet["bounds"]
    require(values.shape == expected.shape and np.isfinite(values).all(), "invalid/nonfinite output")
    difference = np.abs(values.astype(np.float64) - expected)
    failures = difference > bounds
    result = dict(elements=int(values.size), failed_elements=int(failures.sum()),
                  max_abs_error=float(difference.max(initial=0)),
                  max_allowed_bound=float(bounds.max(initial=0)),
                  oracle="complete exported FP64 reference and per-element bound; no tolerance override")
    strict_bounds = packet.get("strict_bounds")
    if strict_bounds is not None:
        ratios = np.divide(difference, strict_bounds, out=np.zeros_like(difference), where=strict_bounds > 0)
        result["strict_correctness"] = dict(primary_acceptance=False,
            failed_elements=int((difference > strict_bounds).sum()), max_error_over_bound=float(ratios.max(initial=0)),
            zero_bound_mismatches=int(((strict_bounds == 0) & (difference != 0)).sum()))
    if failures.any():
        flat = int(np.flatnonzero(failures)[0])
        raise ValueError(f"oracle mismatch: {result}; first index={flat}, "
                         f"actual={values.flat[flat]}, expected={expected.flat[flat]}, bound={bounds.flat[flat]}")
    if packet["indices"] is not None:
        require(indices is not None and indices.shape == packet["indices"].shape and
                np.issubdtype(indices.dtype, np.integer), "missing/invalid index result")
        exact_indices = ranking_contract == "stable" or packet["manifest"]["operation"] == "argmax"
        if exact_indices:
            require(np.array_equal(indices, packet["indices"]), "index mismatch, including stable ties")
        width = packet["manifest"]["dimensions"][1]
        require(((indices >= 0) & (indices < width)).all(), "out-of-range index")
        if packet["manifest"]["operation"] in {"sort", "topk"}:
            require(all(len(set(row.tolist())) == len(row) for row in indices), "duplicate selected index")
            source = packet["rounded_inputs"][0] if "rounded_inputs" in packet else packet["inputs"][0]
            selected = np.take_along_axis(source, indices, axis=-1)
            require(np.array_equal(selected.astype("<f4").view("<u4"), values.astype("<f4").view("<u4")),
                    "value/index bit correspondence mismatch")
        result["exact_indices_and_ties"] = exact_indices
        result["index_contract"] = "exact stable ties" if exact_indices else "valid unique source indices; tied-index permutations allowed"
    return result


def make_program(torch, packet, tensors, ranking_contract="standard"):
    """Return one static expression; masks are prepared outside measured calls."""
    op = packet["manifest"]["operation"]
    dims = packet["manifest"]["dimensions"]
    semantics = packet["manifest"].get("semantics", {})
    eps = float(semantics.get("epsilon", 1.0000000656873453e-5))
    x = tensors[0]
    u = tensors[1] if len(tensors) > 1 else None
    v = tensors[2] if len(tensors) > 2 else None
    mask = None
    scale = None
    is_causal = False
    attention_mask_description = None
    if op == "masked_softmax":
        mask = torch.arange(dims[1], device="cuda")[None, :] <= torch.arange(dims[0], device="cuda")[:, None] % dims[1]
    elif op in {"attention", "attention_tensorcore"}:
        import numpy as np
        b, h, kh, q, k, d, dv = dims
        if q == 1:
            # Last-Q positions: the single query can see every supplied key.
            attention_mask_description = "no mask (single last-position query), is_causal=False"
        elif q == k:
            is_causal = True
            attention_mask_description = "no explicit mask, is_causal=True (square causal attention)"
        else:
            mask = torch.arange(k, device="cuda")[None, :] <= torch.arange(q, device="cuda")[:, None] + k - q
            attention_mask_description = "explicit bottom-right causal mask, is_causal=False"
        scale = float(semantics.get("attention_scale", np.float32(1) / np.sqrt(np.float32(d))))
    stable = ranking_contract == "stable"
    descriptions = {"topk": "stable descending sort then prefix" if stable else "torch.topk, descending sorted values, tie permutations allowed",
                    "sort": f"torch.sort, descending, stable={stable}",
                    "attention": f"functional SDPA, {attention_mask_description}, GQA",
                    "attention_tensorcore": f"default functional SDPA, {attention_mask_description}, GQA; evaluated against both strict and declared single-probability-narrowing contracts",
                    "gemm": "functional mm", "gemv": "functional mm with N=1", "bmm": "functional bmm"}

    def cast(value):
        return value.to(dtype=x.dtype)

    def invoke():
        if op in {"gemm", "gemv"}:
            return torch.mm(x, u)
        elif op == "bmm":
            return torch.bmm(x, u)
        elif op == "softmax":
            return cast(torch.softmax(x.float(), dim=-1))
        elif op == "masked_softmax":
            return cast(torch.softmax(torch.where(mask, x.float(), -float("inf")), dim=-1))
        elif op == "rmsnorm":
            xf = x.float()
            return cast(xf * torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + eps) * u.float())
        elif op == "layernorm":
            return cast(torch.nn.functional.layer_norm(x.float(), (dims[1],), u.float().reshape(-1), v.float().reshape(-1), eps))
        elif op == "rope":
            left, right = x.float().chunk(2, dim=-1)
            return cast(torch.cat((left * u.float() - right * v.float(), left * v.float() + right * u.float()), dim=-1))
        elif op == "swiglu":
            return cast(torch.nn.functional.silu(x.float()) * u.float())
        elif op == "gelu_residual":
            return cast(torch.nn.functional.gelu(x.float(), approximate="tanh") + u.float())
        elif op in {"attention", "attention_tensorcore"}:
            return torch.nn.functional.scaled_dot_product_attention(x, u, v, attn_mask=mask, is_causal=is_causal, scale=scale, enable_gqa=dims[1] != dims[2])
        elif op in {"scan", "scan_ordered"}:
            return cast(torch.cumsum(x, dim=-1, dtype=torch.float32))
        elif op == "reduce_sum":
            return cast(torch.sum(x, dim=-1, keepdim=True, dtype=torch.float32))
        elif op == "reduce_max":
            return torch.amax(x, dim=-1, keepdim=True)
        elif op == "argmax":
            return torch.max(x, dim=-1, keepdim=True)
        elif op == "sort":
            return torch.sort(x, dim=-1, descending=True, stable=stable)
        elif op == "topk":
            if stable:
                values, positions = torch.sort(x, dim=-1, descending=True, stable=True)
                return values[:, :dims[2]], positions[:, :dims[2]]
            return torch.topk(x, dims[2], dim=-1, largest=True, sorted=True)
        else:
            raise ValueError(f"unsupported operation: {op}")
    return invoke, descriptions.get(op, f"functional {op}, FP32 arithmetic/reductions and declared precision output")


def stats(values):
    return dict(samples=values, median=statistics.median(values), minimum=min(values), maximum=max(values))


def package_versions():
    result = {}
    for name in ("torch", "numpy", "triton", "triton-windows"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def time_stream(torch, invoke, samples, sample_ms, warmup_ms):
    start = time.perf_counter()
    warmup_calls = 0
    while (time.perf_counter() - start) * 1000 < warmup_ms or warmup_calls < 3:
        invoke()
        torch.cuda.synchronize()
        warmup_calls += 1
    repetitions = 1
    while True:
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        for _ in range(repetitions):
            invoke()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter_ns() - start) / 1e6
        if elapsed >= sample_ms / 4 or repetitions >= 65536:
            break
        repetitions *= 2
    repetitions = max(1, min(65536, math.ceil(repetitions * sample_ms / max(elapsed, 1e-6))))
    wall = []
    for _ in range(samples):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        for _ in range(repetitions):
            invoke()
        torch.cuda.synchronize()
        wall.append((time.perf_counter_ns() - start) / 1000 / repetitions)
    return dict(repetitions=repetitions, warmup_calls=warmup_calls,
                synchronized_host_wall_us=stats(wall),
                scope="Python/framework dispatch, functional allocations, operator execution and final synchronization")


def time_graph(torch, invoke, batch, samples, warmup_ms):
    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            invoke()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    with torch.cuda.graph(graph, stream=side):
        for _ in range(batch):
            invoke()
    torch.cuda.synchronize()
    capture_ms = (time.perf_counter_ns() - start) / 1e6
    graph.replay()
    torch.cuda.synchronize()
    warmup_start = time.perf_counter_ns()
    warmup_replays = 0
    while (time.perf_counter_ns() - warmup_start) / 1e6 < warmup_ms:
        graph.replay()
        torch.cuda.synchronize()
        warmup_replays += 1
    warmup_actual_ms = (time.perf_counter_ns() - warmup_start) / 1e6
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    event, wall = [], []
    for _ in range(samples):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter_ns() - start) / 1000 / batch)
        event.append(begin.elapsed_time(end) * 1000 / batch)
    return dict(batch=batch, capture_ms=capture_ms, warmup_target_ms=warmup_ms,
                warmup_actual_ms=warmup_actual_ms, warmup_replays=warmup_replays,
                warmup_scope="Synchronized graph replays after capture and the first replay; excluded from capture cost and timing samples",
                event_us_per_operation=stats(event),
                host_wall_us_per_operation=stats(wall),
                scope="Two CUDA events around one graph replay containing N sequential calls on one capture stream; only final output retained; event span divided by N",
                allocation_policy="Normal functional returns, capture-aware memory pool reuse, no forced copy or N-output retention",
                capture_stream=int(side.cuda_stream),
                internal_inductor_cudagraphs=False)


def generated_calls(source):
    """Extract real call names; comments/graph-fragment strings are not calls."""
    def qualified(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = qualified(node.value)
            return f"{prefix}.{node.attr}" if prefix else None
        return None
    return {name for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)
            if (name := qualified(node.func)) is not None}


def compiler_evidence(torch, directory):
    from torch._dynamo.utils import counters
    from torch._inductor import metrics
    counts = {name: dict(values) for name, values in counters.items() if values}
    require(counts.get("stats", {}).get("unique_graphs", 0) >= 1, "no captured Dynamo graph")
    require(not counts.get("graph_break", {}), "unexpected graph break")
    artifacts, cuda_markers, external_calls, triton_calls = [], [], [], []
    for file in sorted(directory.rglob("*")):
        if file.is_file() and file.suffix in {".py", ".cpp", ".cu", ".ptx", ".cubin", ".json"}:
            artifacts.append(dict(path=str(file), sha256=digest(file), bytes=file.stat().st_size))
            if file.suffix == ".py":
                text = file.read_text(encoding="utf-8", errors="replace")
                if "cuda" in text:
                    cuda_markers.append(str(file))
                calls = generated_calls(text)
                if any(call.startswith(("extern_kernels.", "torch.ops.aten.")) for call in calls):
                    external_calls.append(str(file))
                if "async_compile.triton" in calls:
                    triton_calls.append(str(file))
    require(cuda_markers, "missing generated Inductor CUDA wrapper evidence")
    return dict(backend="inductor", fullgraph=True, suppress_errors=False, counters=counts,
                generated_kernel_count=metrics.generated_kernel_count,
                cuda_wrappers=cuda_markers, triton_wrappers=triton_calls,
                external_call_wrappers=external_calls, artifacts=artifacts,
                interpretation="Full-graph Inductor may call cuBLAS/ATen GPU operators; external calls are recorded and are not claimed as generated Triton kernels")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--precision", choices=tuple(PRECISIONS), help="must equal the native manifest precision")
    parser.add_argument("--mode", choices=("default", "max-autotune"), default="default")
    parser.add_argument("--ranking-contract", choices=("standard", "stable"), default="standard")
    parser.add_argument("--eager", action="store_true", help="also measure a separate secondary eager baseline")
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--sample-ms", type=float, default=100.)
    parser.add_argument("--warmup-ms", type=float, default=500.)
    parser.add_argument("--graph-batch", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if not (1 <= args.samples <= 100 and 0 < args.sample_ms <= 5000 and
            0 <= args.warmup_ms <= 30000 and 0 <= args.graph_batch <= 65536 and 1 <= args.threads <= 64):
        parser.error("invalid sample, warmup, graph batch or thread bound")
    packet = load_packet(args.manifest)
    if args.precision and args.precision != packet["manifest"]["precision"]:
        parser.error("precision must match the exported native oracle; cross-precision comparisons are not allowed")
    if args.check_inputs:
        print(json.dumps(dict(status="inputs_verified", manifest=packet["manifest"], sha256=packet["receipts"]), indent=2))
        return 0
    if args.output is None:
        parser.error("--output is required for GPU execution")
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    cache = output_dir / "inductor-cache"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache)
    os.environ["TRITON_CACHE_DIR"] = str(output_dir / "triton-cache")
    # Inductor's multiprocess compiler uses pass_fds, unsupported on Windows.
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1" if os.name == "nt" else str(args.threads)
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    for name in ("TORCH_COMPILE_DISABLE", "TORCHDYNAMO_DISABLE"):
        require(os.environ.get(name, "0") in {"", "0"}, f"{name} must not disable compilation")
    record = dict(status="prepared", started=datetime.now().astimezone().isoformat(),
                  options={name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
                  source=dict(path=str(Path(__file__).resolve()), sha256=digest(__file__)),
                  manifest=packet["manifest"], input_sha256=packet["receipts"],
                  python=sys.version, platform=platform.platform(), executable=sys.executable,
                  packages=package_versions(),
                  environment={key: value for key, value in os.environ.items() if key.startswith(("CUDA", "TORCH", "TRITON", "OMP", "NVIDIA_TF32"))})
    result_path = output_dir / "result.json"

    def save():
        temporary = result_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(result_path)

    save()
    try:
        import numpy as np
        import torch
        from torch._dynamo.utils import counters
        from torch._inductor import metrics
        require(torch.cuda.is_available(), "CUDA PyTorch is unavailable")
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        torch._dynamo.config.suppress_errors = False
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        precision = packet["manifest"]["precision"]
        dtype = getattr(torch, PRECISIONS[precision])
        cpu_inputs = [torch.from_numpy(array.copy()).to(dtype=dtype) for array in packet["inputs"]]
        packet["rounded_inputs"] = [tensor.float().numpy() for tensor in cpu_inputs]
        for original, restored in zip(packet["inputs"], packet["rounded_inputs"]):
            require(np.array_equal(original.view("<u4"), restored.view("<u4")),
                    "Torch conversion changed exported storage bits")
        tensors = [tensor.to(device="cuda") for tensor in cpu_inputs]
        eager, expression = make_program(torch, packet, tensors, args.ranking_contract)
        last = None
        record.update(device=str(torch.cuda.get_device_properties(0)), torch_version=torch.__version__,
                      torch_git_version=torch.version.git_version, torch_cuda_version=torch.version.cuda,
                      torch_config=torch.__config__.show(), expression=expression,
                      numerical_policy=dict(input_storage=PRECISIONS[precision], output_storage=PRECISIONS[precision], compute_precision="float32 accumulation",
                                            matmul_fp32_precision=torch.backends.cuda.matmul.fp32_precision,
                                            fp16_reduced_precision_reduction=False, bf16_reduced_precision_reduction=False),
                      allocation_policy="Functional return; allocations included in stream timing, graph capture pool reused on replay",
                      ranking_contract=args.ranking_contract)
        if packet["manifest"]["operation"] == "attention_tensorcore":
            record["precision_contract_evidence"] = dict(
                acceptance_kind="predeclared_numerical_envelope", backend_precision_contract_verified=False,
                limitation="Automatic SDPA call evidence does not establish one probability narrowing, FP32 intermediates, or absence of FTZ")
        modes = dict(torch._inductor.list_mode_options(args.mode))
        if args.graph_batch:
            modes["triton.cudagraphs"] = False
        record["compile_options"] = modes
        record["compile_requested_mode"] = args.mode
        counters.clear()
        metrics.reset()

        def check():
            torch.cuda.synchronize()
            output, indices = last if isinstance(last, tuple) else (last, None)
            require(output.device.type == "cuda" and output.dtype == dtype, "unexpected output device/precision")
            values = output.detach().float().cpu().numpy()
            positions = indices.cpu().numpy() if indices is not None else None
            validation = validate_output(packet, values, positions, ranking_contract=args.ranking_contract)
            for gpu, cpu in zip(tensors, cpu_inputs):
                require(torch.equal(gpu.cpu(), cpu), "operator mutated input")
            verify_packet(packet)
            return validation

        def save_outputs(label):
            output, indices = last if isinstance(last, tuple) else (last, None)
            files = []
            precision = packet["manifest"]["precision"]
            extension = {"fp32": "f32", "fp16": "f16", "bf16": "bf16"}[precision]
            values_path = output_dir / f"{label}-output.{extension}"
            cpu_output = output.detach().cpu().contiguous()
            if precision == "bf16":
                cpu_output.view(torch.uint16).numpy().astype("<u2").tofile(values_path)
            else:
                cpu_output.numpy().astype("<f2" if precision == "fp16" else "<f4").tofile(values_path)
            files.append(dict(path=str(values_path), sha256=digest(values_path), storage_dtype=PRECISIONS[precision]))
            if indices is not None:
                indices_path = output_dir / f"{label}-indices.i64"
                indices.cpu().numpy().astype("<i8").tofile(indices_path)
                files.append(dict(path=str(indices_path), sha256=digest(indices_path), storage_dtype="int64"))
            return files

        with torch.inference_mode():
            record["phase"] = "compile"
            save()
            start = time.perf_counter_ns()
            compiled = torch.compile(eager, backend="inductor", fullgraph=True, dynamic=False, options=modes)

            def invoke_compiled():
                nonlocal last
                last = None
                last = compiled()

            def invoke_eager():
                nonlocal last
                last = None
                last = eager()

            invoke_compiled()
            torch.cuda.synchronize()
            record["cold_compile_first_call_ms"] = (time.perf_counter_ns() - start) / 1e6
            record["cold_scope"] = "torch.compile creation, first full-graph compile and one execution; device initialization and upload excluded"
            record["phase"] = "correctness_before"
            record["compiler_evidence"] = compiler_evidence(torch, cache)
            record["compiled_outputs_before"] = save_outputs("compiled-first")
            save()
            record["compiled_correctness_before"] = check()
            record["phase"] = "warm_compiled"
            save()
            record["compiled_stream"] = time_stream(torch, invoke_compiled, args.samples, args.sample_ms, args.warmup_ms)
            if args.graph_batch:
                record["compiled_graph"] = time_graph(torch, invoke_compiled, args.graph_batch, args.samples, args.warmup_ms)
            record["compiled_correctness_after"] = check()
            record["compiled_outputs"] = save_outputs("compiled")
            record["compiler_evidence"] = compiler_evidence(torch, cache)
            if args.eager:
                record["phase"] = "warm_eager"
                save()
                invoke_eager()
                record["eager_correctness_before"] = check()
                record["eager_stream"] = time_stream(torch, invoke_eager, args.samples, args.sample_ms, args.warmup_ms)
                if args.graph_batch:
                    record["eager_graph"] = time_graph(torch, invoke_eager, args.graph_batch, args.samples, args.warmup_ms)
                record["eager_correctness_after"] = check()
                record["eager_outputs"] = save_outputs("eager")
        record.update(status="passed", phase="complete")
        return 0
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        record["finished"] = datetime.now().astimezone().isoformat()
        save()
        print(f"Evidence: {result_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
