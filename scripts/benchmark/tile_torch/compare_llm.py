#!/usr/bin/env python3
"""Matched LLM Tile captures versus eager Torch, with complete FP64 oracles.

Native inputs are exported before compilation and reused bit-for-bit by Torch.
Run each backend separately. No measurement is used to choose a schedule.
JSON/source/log evidence is retained; large, reproducible tensor exports are
temporary, with SHA256 receipts. Failed cases remain in the result matrix.
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
import re
import signal
import statistics
import subprocess
import sys
import tempfile

from repeat import artifact_hashes
from run import summarize, time_metal_device, time_torch


OPERATIONS = {"swiglu", "rope", "rmsnorm", "layernorm", "gelu_residual", "masked_softmax", "attention"}

# Deliberately match failure diagnostics, not every occurrence of "error"
# (e.g. max_abs_error in a successful JSON record). This is a conservative
# safety guard, not an exhaustive decoder for every driver/version message.
GPU_FAILURE_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r"\bGPU[\s_-]*Hang[\s_-]*Error\b",
    r"\bGPU\s+(?:Address Fault|Page Fault|Timeout|Internal)\s+Error\b",
    r"\bMTLCommandBuffer(?:ErrorDomain|StatusError)\b",
    r"\bError\s+Domain=(?:MTL|AGX|Metal|MPS)\w*",
    r"\bexecution of the command buffer was aborted\b",
    r"\b(?:command[ _-]?buffer|metal|mps|gpu)\b[^\r\n]{0,120}\b(?:failed|failure|faulted|aborted|timed out|out of memory|illegal (?:memory )?access)\b",
    r"\b(?:command[ _-]?buffer|metal|mps)\b[^\r\n]{0,80}\berror\b(?![\"']?\s*[:=]\s*(?:\"\"|''|null\b|false\b|0(?:[,\s}]|$)))",
    r"\b(?:error(?![\"']?\s*[:=]\s*(?:\"\"|''|null\b|false\b|0(?:[,\s}]|$)))|failure|failed)\b[^\r\n]{0,80}\b(?:command[ _-]?buffer|metal|mps|gpu)\b",
))


def gpu_failure_diagnostics(stdout, stderr):
    matches = []
    for channel, payload in (("stdout", stdout), ("stderr", stderr)):
        text = payload.decode("utf-8", errors="replace") if isinstance(payload, bytes) else payload or ""
        for number, line in enumerate(text.splitlines(), 1):
            for pattern in GPU_FAILURE_PATTERNS:
                match = pattern.search(line)
                if match:
                    begin = max(0, match.start() - 80)
                    matches.append(dict(channel=channel, line=number, pattern=pattern.pattern,
                                        excerpt=line[begin:begin + 360]))
                    break
            if len(matches) == 16:
                return matches
    return matches


def record_process_output(directory, stem, stdout, stderr, row, process):
    raw = lambda value: value if isinstance(value, bytes) else (value or "").encode("utf-8")
    stdout, stderr = raw(stdout), raw(stderr)
    logs = {}
    for channel, payload in (("stdout", stdout), ("stderr", stderr)):
        path = directory / f"{stem}.{channel}.log"
        path.write_bytes(payload)
        logs[channel] = dict(path=path.name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    # Keep the previous combined artifact for existing report tooling.
    (directory / f"{stem}.log").write_bytes(stdout + stderr)
    row.update(process=process, process_logs=logs, gpu_failure_diagnostics=gpu_failure_diagnostics(stdout, stderr))
    (directory / f"{stem}.process.json").write_text(json.dumps(
        dict(process=process, logs=logs, gpu_failure_diagnostics=row["gpu_failure_diagnostics"]), indent=2) + "\n")


def capture_benchmark(command, environment, timeout, directory, stem, row):
    """Capture OS-level output for every arm, including the Torch worker."""
    try:
        process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=True)
    except OSError as error:
        record_process_output(directory, stem, b"", b"", row,
                              dict(exit_code=None, timed_out=False, error=str(error)))
        raise
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
        record_process_output(directory, stem, stdout, stderr, row,
                              dict(exit_code=process.returncode, timed_out=isinstance(error, subprocess.TimeoutExpired),
                                   interrupted=isinstance(error, KeyboardInterrupt), error=type(error).__name__))
        if isinstance(error, KeyboardInterrupt):
            raise
        raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr) from None
    record_process_output(directory, stem, stdout, stderr, row, dict(exit_code=process.returncode, timed_out=False))
    if row["gpu_failure_diagnostics"]:
        first = row["gpu_failure_diagnostics"][0]
        raise RuntimeError(f"GPU failure diagnostic in {first['channel']}: {first['excerpt']}; entire cohort invalid")
    result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    result.check_returncode()
    return result


def parse_case(text):
    try:
        op, numbers = text.split(":")
        shape = tuple(int(v) for v in numbers.split(","))
        if op not in OPERATIONS or len(shape) != (7 if op == "attention" else 2) or min(shape) <= 0 or max(shape) > 65536:
            raise ValueError
        if op == "rope" and shape[1] % 2:
            raise ValueError
        if op == "attention" and (shape[1] % shape[2] or shape[4] < shape[3]):
            raise ValueError
        inputs, output = shapes_for(op, shape)
        if any(math.prod(s) > 2**26 for s in [*inputs, output]):
            raise ValueError
        if op == "attention" and math.prod(shape[:2]) * shape[3] * shape[4] > 2**26:
            raise ValueError
        return op, shape
    except (ValueError, IndexError):
        raise argparse.ArgumentTypeError("expected op:rows,width or attention:B,Hq,Hkv,Q,K,D,Dv within benchmark limits") from None


def shapes_for(op, dims):
    if op == "attention":
        b, h, kh, q, k, d, v = dims
        return [(b, h, q, d), (b, kh, k, d), (b, kh, k, v)], (b, h, q, v)
    m, n = dims
    aux = (1 if op in {"rmsnorm", "layernorm"} else m, n // 2 if op == "rope" else n)
    return [(m, n), aux, aux], (m, n)


def reference(op, dims, arrays):
    import numpy as np
    x, u, v = [a.astype(np.float64) for a in arrays]
    if op == "swiglu":
        return x / (1 + np.exp(-x)) * u
    if op == "rope":
        left, right = np.split(x, 2, axis=-1)
        return np.concatenate((left * u - right * v, left * v + right * u), axis=-1)
    if op in {"rmsnorm", "layernorm"}:
        centered = x - x.mean(axis=-1, keepdims=True) if op == "layernorm" else x
        y = centered / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + float(np.float32(1e-5))) * u
        return y + v if op == "layernorm" else y
    if op == "gelu_residual":
        return .5 * x * (1 + np.tanh(float(np.float32(.7978845608)) * (x + float(np.float32(.044715)) * x**3))) + u
    if op == "masked_softmax":
        m, n = dims
        mask = np.arange(n)[None, :] <= np.arange(m)[:, None] % n
        scores = np.where(mask, x, -1e30)
        e = np.where(mask, np.exp(scores - scores.max(axis=-1, keepdims=True)), 0)
        return e / e.sum(axis=-1, keepdims=True)
    b, h, kh, q, k, d, dv = dims
    u, v = np.repeat(u, h // kh, axis=1), np.repeat(v, h // kh, axis=1)
    scale = float(np.float32(1) / np.sqrt(np.float32(d)))
    score = (x @ u.swapaxes(-1, -2)) * scale
    valid = np.arange(k)[None, :] <= np.arange(q)[:, None] + k - q
    score = np.where(valid, score, -1e30)
    p = np.where(valid, np.exp(score - score.max(axis=-1, keepdims=True)), 0)
    return (p / p.sum(axis=-1, keepdims=True)) @ v


def torch_program(torch, op, dims, arrays, device):
    import numpy as np
    x, u, v = [torch.from_numpy(a).to(device) for a in arrays]
    _, output_shape = shapes_for(op, dims)
    y = torch.empty(output_shape, device=device, dtype=torch.float32)
    if op == "swiglu":
        scratch = torch.empty_like(x)

        def invoke():
            torch.ops.aten.silu.out(x, out=scratch)
            torch.mul(scratch, u, out=y)
        return invoke, lambda: y, "preallocated_aten_silu_out_then_mul_out"
    if op == "rope":
        left, right = x.chunk(2, dim=-1)
        lo, hi = y.chunk(2, dim=-1)
        t0, t1 = torch.empty_like(u), torch.empty_like(u)

        def invoke():
            torch.mul(left, u, out=t0)
            torch.mul(right, v, out=t1)
            torch.sub(t0, t1, out=lo)
            torch.mul(left, v, out=t0)
            torch.mul(right, u, out=t1)
            torch.add(t0, t1, out=hi)
        return invoke, lambda: y, "preallocated_split_half_rope_six_out_ops"
    if op == "attention":
        b, h, kh, q, k, d, dv = dims
        # Queries are the final Q KV positions. is_causal=True alone would
        # use the wrong alignment for decode / unequal Q and K lengths.
        mask = (torch.arange(k)[None, :] <= torch.arange(q)[:, None] + k - q).to(device)
        scale = float(np.float32(1) / np.sqrt(np.float32(d)))

        def invoke():
            nonlocal y
            y = torch.nn.functional.scaled_dot_product_attention(x, u, v, attn_mask=mask, scale=scale, enable_gqa=h != kh)
        return invoke, lambda: y, "functional_sdpa_explicit_bottom_right_causal_mask_output_allocation_included"
    if op == "masked_softmax":
        m, n = dims
        mask = (torch.arange(n)[None, :] <= torch.arange(m)[:, None] % n).to(device)

    def invoke():
        nonlocal y
        if op == "rmsnorm":
            y = x / torch.sqrt((x * x).mean(dim=-1, keepdim=True) + 1e-5) * u
        elif op == "layernorm":
            y = torch.nn.functional.layer_norm(x, (dims[1],), u[0], v[0], 1e-5)
        elif op == "gelu_residual":
            y = torch.nn.functional.gelu(x, approximate="tanh") + u
        elif op == "masked_softmax":
            y = torch.softmax(torch.where(mask, x, -1e30), dim=-1)
        else:
            raise ValueError("unsupported Torch expression")
    return invoke, lambda: y, "functional_eager_intermediate_and_output_allocations_included"


def validate_output(actual, expected):
    import numpy as np
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError("invalid output shape or nonfinite output")
    difference = np.abs(actual.astype(np.float64) - expected)
    if not np.all(difference <= 5e-5 + 5e-5 * np.abs(expected)):
        raise ValueError(f"complete FP64 oracle mismatch: max abs error {difference.max()}")
    return dict(elements=expected.size, max_abs_error=float(difference.max()), atol=5e-5, rtol=5e-5)


def torch_worker(request_path):
    """Internal fresh-process entry: C/C++ Torch fd output is captured by the parent."""
    import numpy as np
    request = json.loads(Path(request_path).read_text())
    if request.get("format") != "tile-llm-torch-worker-v1" or request.get("backend") not in ("cpu", "metal", "metal4"):
        raise ValueError("invalid Torch worker request")
    op, dims = parse_case(request["operation"] + ":" + ",".join(map(str, request["dimensions"])))
    args = argparse.Namespace(**request["timing"])
    if any(type(value) is not int or value <= 0 for value in (args.samples, args.sample_ms, args.warmup_ms, args.threads)):
        raise ValueError("invalid Torch worker timing bounds")
    args.metal_device_timing = Path(request["metal_device_timing"]) if request["metal_device_timing"] else None
    if request["backend"] == "cpu" and args.metal_device_timing:
        raise ValueError("GPU timing cannot be used for CPU")
    input_shapes, output_shape = shapes_for(op, dims)
    if len(request["input_paths"]) != 3:
        raise ValueError("invalid Torch worker input count")
    arrays, hashes = [], []
    for text, shape in zip(request["input_paths"], input_shapes):
        data = Path(text).read_bytes()
        if len(data) != 4 * math.prod(shape):
            raise ValueError("invalid Torch worker input byte count")
        hashes.append(hashlib.sha256(data).hexdigest())
        arrays.append(np.frombuffer(data, dtype=np.float32).reshape(shape).copy())
    if hashes != request["input_sha256"]:
        raise ValueError("Torch worker input identity mismatch")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = "cpu" if request["backend"] == "cpu" else "mps"
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Torch MPS unavailable")
    sync = torch.mps.synchronize if device == "mps" else lambda: None
    expected = reference(op, dims, arrays)
    invoke, read, expression = torch_program(torch, op, dims, arrays, device)
    invoke()
    sync()
    precheck = validate_output(read().cpu().numpy(), expected)
    measurement = time_torch(invoke, sync, args)
    measurement.update(expression=expression, precision="fp32", fast_math_policy="torch_default",
                       pre_timing_correctness=precheck)
    if args.metal_device_timing:
        measurement["device_timing"] = time_metal_device(invoke, sync, args, measurement["repetitions"])
    actual = read().cpu().numpy().copy()
    after = validate_output(actual, expected)
    actual.tofile(request["output_path"])
    result = dict(format="tile-llm-torch-worker-v1", operation=op, dimensions=list(dims), backend=request["backend"],
                  input_sha256=hashes, output_path=request["output_path"], output_shape=list(output_shape),
                  output_sha256=hashlib.sha256(actual.tobytes()).hexdigest(), measurement=measurement,
                  correctness=after, torch_info=dict(version=torch.__version__, git_version=torch.version.git_version,
                                                    config=torch.__config__.show(), mps_cpu_fallback=False))
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0


def run_torch_worker(op, dims, arrays, args, directory, stem, output, row):
    request = dict(format="tile-llm-torch-worker-v1", operation=op, dimensions=list(dims), backend=args.backend,
                   input_paths=[], input_sha256=[], output_path=str(output),
                   timing={key: getattr(args, key) for key in ("samples", "sample_ms", "warmup_ms", "threads")},
                   metal_device_timing=str(args.metal_device_timing) if args.metal_device_timing else None)
    for index, value in enumerate(arrays):
        path = Path(str(output) + f".worker.input{index}.f32")
        value.tofile(path)
        request["input_paths"].append(str(path))
        request["input_sha256"].append(hashlib.sha256(path.read_bytes()).hexdigest())
    request_path = directory / f"{stem}.worker.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n")
    command = [sys.executable, str(Path(__file__).resolve()), "--torch-worker", str(request_path)]
    row["command"] = command
    completed = capture_benchmark(command, dict(os.environ), args.timeout, directory, stem, row)
    result = json.loads(completed.stdout)
    expected = {key: request[key] for key in ("format", "operation", "dimensions", "backend", "input_sha256", "output_path")}
    expected["output_shape"] = list(shapes_for(op, dims)[1])
    if any(type(result.get(key)) is not type(value) or result[key] != value for key, value in expected.items()):
        raise ValueError("Torch worker response metadata mismatch")
    if result["torch_info"].get("mps_cpu_fallback") is not False:
        raise ValueError("Torch worker must not enable CPU fallback")
    if hashlib.sha256(output.read_bytes()).hexdigest() != result["output_sha256"]:
        raise ValueError("Torch worker output identity mismatch")
    measurement = result["measurement"]
    if measurement.get("precision") != "fp32" or not isinstance(measurement.get("expression"), str) or not measurement["expression"]:
        raise ValueError("invalid Torch worker expression/precision metadata")
    for check in (measurement.get("pre_timing_correctness"), result.get("correctness")):
        if (not isinstance(check, dict) or type(check.get("elements")) is not int or
                check["elements"] != math.prod(expected["output_shape"]) or check.get("atol") != 5e-5 or check.get("rtol") != 5e-5 or
                type(check.get("max_abs_error")) not in (int, float) or not math.isfinite(check["max_abs_error"]) or check["max_abs_error"] < 0):
            raise ValueError("incomplete Torch worker pre/post oracle")
    if type(measurement["repetitions"]) is not int or not 1 <= measurement["repetitions"] <= 100000:
        raise ValueError("invalid Torch worker repetition count")
    for key in ("throughput_us", "latency_us"):
        values = measurement[key]
        if not isinstance(values, list) or len(values) != args.samples or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("invalid Torch worker timing samples")
    summarize(measurement)
    return result


def check_metadata(result, backend, op, dims, block, samples, reduction_tree=False, group_threads=0, forward_input_views=False, attention_qk="mma", attention_pv="mma"):
    inputs, output = shapes_for(op, dims)
    fields = dict(implementation={"metal": "tile_tirx_metal", "metal4": "tile_xir_metal4", "cpu": "tile_xir_simd"}[backend], backend=backend,
                  precision="fp32", fast_math=False, relaxed_precision=False, runtime="luisa",
                  timing="synchronized_host_wall", batch_policy="one_runtime_command_list_per_batch",
                  operation=op, dimensions=list(dims), attention_block=list(block),
                  input_shapes=[list(s) for s in inputs], output_shape=list(output))
    # Older default-policy artifacts may omit these fields. An explicit
    # request must be acknowledged, never silently ignored by an old binary.
    for key, value in (("reduction_tree", reduction_tree), ("requested_group_threads", group_threads), ("requested_input_views", forward_input_views)):
        if value or key in result:
            fields[key] = value
    for key, mode in (("attention_qk", attention_qk), ("attention_pv", attention_pv)):
        if mode != "mma" or key in result:
            fields[key] = mode if op == "attention" else "not_applicable"
    if "source_reduction_policy" in result:
        fields["source_reduction_policy"] = "unordered_tree"
    if "reduction_candidate_setting" in result:
        settings = ("not_applicable",) if backend != "metal" else ("enabled",) if reduction_tree else ("automatic", "disabled")
        if result["reduction_candidate_setting"] not in settings:
            raise ValueError("native reduction candidate metadata mismatch")
    if any(type(result.get(k)) is not type(v) or result[k] != v for k, v in fields.items()):
        raise ValueError("native benchmark metadata mismatch")
    check = result["correctness"]
    if check["checks"] != 2 or check["elements_per_check"] != math.prod(output) or check["guard_elements_per_check"] != 34 or check["atol"] != 5e-5 or check["rtol"] != 5e-5:
        raise ValueError("incomplete native oracle or guard coverage")
    if type(result["repetitions"]) is not int or not 1 <= result["repetitions"] <= 100000:
        raise ValueError("invalid repetition count")
    for key in ("throughput_us", "latency_us"):
        if len(result[key]) != samples or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in result[key]):
            raise ValueError("invalid timing sample")
    summarize(result)


def make_summary(rows, rounds):
    result = []
    gpu_failure = any(row.get("gpu_failure_diagnostics") for row in rows)
    paths = ("native", "baseline", "torch") if any(r["path"] == "baseline" for r in rows) else ("native", "torch")
    keys = sorted({(r["operation"], tuple(r["dimensions"])) for r in rows})
    for op, dims in keys:
        selected = [r for r in rows if r["operation"] == op and r["dimensions"] == list(dims)]
        item = dict(operation=op, dimensions=list(dims), complete=not gpu_failure and len(selected) == len(paths) * rounds and all(r["valid"] for r in selected))
        if gpu_failure:
            item["invalid_reason"] = "GPU failure diagnostic invalidates the entire benchmark cohort"
        if item["complete"]:
            for metric in ("throughput_us_p50", "latency_us_p50", "gpu_control_throughput_us_p50", "gpu_control_latency_us_p50"):
                def value(row):
                    m = row["measurement"]
                    return m["device_timing"]["control"][metric.removeprefix("gpu_control_").replace("throughput", "command_buffer_throughput").replace("latency", "command_buffer_latency")] if metric.startswith("gpu_control_") else m[metric]
                if metric.startswith("gpu_control_") and any("device_timing" not in r["measurement"] for r in selected):
                    continue
                ratios, old_ratios = [], []
                for index in range(rounds):
                    pair = {r["path"]: value(r) for r in selected if r["round"] == index}
                    if set(pair) != set(paths):
                        raise ValueError("missing or duplicate pair")
                    ratios.append(pair["native"] / pair["torch"])
                    if "baseline" in pair:
                        old_ratios.append(pair["native"] / pair["baseline"])
                item[metric] = dict(median_us={p: statistics.median(value(r) for r in selected if r["path"] == p) for p in paths},
                                    paired_native_over_torch_median=statistics.median(ratios), min_ratio=min(ratios), max_ratio=max(ratios),
                                    slower_rounds=sum(r > 1 for r in ratios))
                if old_ratios:
                    item[metric]["baseline_comparison"] = dict(paired_native_over_baseline_median=statistics.median(old_ratios),
                                                                 min_ratio=min(old_ratios), max_ratio=max(old_ratios), slower_rounds=sum(r > 1 for r in old_ratios))
        result.append(item)
    return result


def make_visit_plan(cases, rounds, baseline=False):
    rows = []
    for op, dims in cases:
        for index in range(rounds):
            order = list(list(itertools.permutations(("native", "baseline", "torch")))[index % 6]) if baseline else ["native", "torch"] if index % 2 == 0 else ["torch", "native"]
            for path in order:
                rows.append(dict(operation=op, dimensions=list(dims), round=index, path=path, order=order,
                                 valid=False, status="NotRun", error="not yet visited"))
    return rows


def native_operation_environment(environment, operation):
    result = dict(environment)
    if operation != "attention":
        for key in ("LUISA_TILE_BENCH_ATTENTION_QK", "LUISA_TILE_BENCH_ATTENTION_PV"):
            result.pop(key, None)
    return result


def mark_remaining_gpu_visits_not_run(rows, backend):
    if backend != "cpu" and any(row.get("gpu_failure_diagnostics") for row in rows):
        for row in rows:
            if row.get("status") == "NotRun":
                row["error"] = "not launched: GPU failure invalidated the cohort; queue health is not established"
        return True
    return False


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, help="frozen old binary and adjacent ABI-coherent libraries; requires all six orders")
    parser.add_argument("--build-dir", type=Path, required=True, help="complete selected configuration is built before native execution")
    parser.add_argument("--backend", choices=("cpu", "metal", "metal4"), required=True)
    parser.add_argument("--case", type=parse_case, action="append", required=True)
    parser.add_argument("--attention-block", type=int, nargs=2, default=(16, 32))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler-artifact", type=Path, action="append", default=[])
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--sample-ms", type=int, default=30)
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--metal-device-timing", type=Path)
    parser.add_argument("--subgroup-reductions", action="store_true", help="enable the FP32 subgroup candidate family; source reduce already defaults to unordered tree; requires the TIRx llm entry")
    parser.add_argument("--forward-input-views", action="store_true", help="explicitly request immutable input views; the subgroup family already attempts forwarding, so enable this for both controls when isolating collective emission")
    parser.add_argument("--attention-qk", choices=("mma", "reduce"), default="mma", help="benchmark-only QK decomposition probe using existing DSL; not a production planner optimization; requires an acknowledging llm entry")
    parser.add_argument("--attention-pv", choices=("mma", "reduce"), default="mma", help="benchmark-only PV decomposition probe using existing DSL; not a production planner optimization; requires an acknowledging llm entry")
    parser.add_argument("--group-threads", type=int, default=0, help="exact Metal group width; zero uses the planner")
    args = parser.parse_args(argv)
    if args.rounds < 2 or args.rounds % 2 or len(set(args.case)) != len(args.case) or min(args.samples, args.sample_ms, args.warmup_ms, args.threads, args.timeout) <= 0:
        parser.error("unique cases, positive settings and an even number of rounds >= 2 required")
    if not (0 < args.attention_block[0] <= 128 and 0 < args.attention_block[1] <= 256):
        parser.error("invalid attention block")
    if args.backend != "metal" and args.metal_device_timing:
        parser.error("GPU timing cannot be used for CPU")
    if not 0 <= args.group_threads <= 1024 or args.backend != "metal" and (args.subgroup_reductions or args.group_threads or args.forward_input_views):
        parser.error("subgroup candidates/group constraints/input views require Metal; group width must be 0..1024")
    for label, mode in (("QK", args.attention_qk), ("PV", args.attention_pv)):
        if mode != "mma" and not any(op == "attention" for op, _ in args.case):
            parser.error(f"{label} decomposition probe requires an attention case")
    if args.baseline and (args.baseline.resolve() == args.native.resolve() or args.rounds % 6):
        parser.error("distinct frozen baseline and a multiple of six rounds required")
    return args


def configure_probe_environment(args):
    for key in ("TVM_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "LUISA_SIMD_WORKER_COUNT"):
        os.environ[key] = str(args.threads)
    removed = {}
    for key in list(os.environ):
        if key.startswith("LUISA_SIMD_") and key != "LUISA_SIMD_WORKER_COUNT" or key in ("LUISA_ENABLE_VALIDATION", "DYLD_PRINT_LIBRARIES") or key.startswith("LUISA_TILE_BENCH_"):
            removed[key] = os.environ.pop(key)
    os.environ["LUISA_SIMD_WARP_WIDTH"] = "8"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    if args.backend in ("cpu", "metal4"):
        os.environ["LUISA_TILE_BENCH_XIR_BACKEND"] = "simd" if args.backend == "cpu" else "metal4"
    if args.subgroup_reductions:
        os.environ["LUISA_TILE_BENCH_REDUCTION_TREE"] = "1"
    if args.forward_input_views:
        os.environ["LUISA_TILE_BENCH_INPUT_VIEWS"] = "1"
    if args.attention_qk != "mma":
        os.environ["LUISA_TILE_BENCH_ATTENTION_QK"] = args.attention_qk
    if args.attention_pv != "mma":
        os.environ["LUISA_TILE_BENCH_ATTENTION_PV"] = args.attention_pv
    if args.group_threads:
        os.environ["LUISA_TILE_BENCH_GROUP_THREADS"] = str(args.group_threads)
    return removed


def main():
    args = parse_arguments()
    args.native = args.native.resolve(strict=True)
    if args.baseline:
        args.baseline = args.baseline.resolve(strict=True)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    build = subprocess.run(["cmake", "--build", str(args.build_dir.resolve(strict=True)), "--parallel", "8"], capture_output=True, text=True)
    (args.output / "build.log").write_text(build.stdout + build.stderr)
    build.check_returncode()
    removed = configure_probe_environment(args)
    if args.metal_device_timing:
        args.metal_device_timing = args.metal_device_timing.resolve(strict=True)
        os.environ["LUISA_TILE_BENCH_METAL_TIMING"] = str(args.metal_device_timing)
        args.compiler_artifact.append(args.metal_device_timing)
    import numpy as np
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    binaries = [args.native] + ([args.baseline] if args.baseline else [])
    hashes = artifact_hashes(binaries, args.compiler_artifact)
    root = Path(__file__).resolve().parents[3]
    report = dict(metadata=dict(timestamp=dt.datetime.now(dt.timezone.utc).isoformat(), backend=args.backend, platform=platform.platform(),
                                git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                                torch_version=None, torch_git_version=None, torch_config=None,
                                rounds=args.rounds, samples=args.samples, sample_ms=args.sample_ms, warmup_ms=args.warmup_ms,
                                requested_threads=args.threads, removed_environment=removed, artifacts_sha256=hashes,
                                source_sha256={p: digest(root / p) for p in ["src/tests/common/tile_llm_test_utils.h", "src/tests/common/tile_llm_benchmark.h", "scripts/benchmark/tile_torch/compare_llm.py", "scripts/benchmark/tile_torch/run.py"]},
                                comparison="FP32 same exported inputs; warmed E2E excludes compile/upload; GPU control uses uninstrumented command-buffer intervals, not isolated kernel time",
                                baseline=str(args.baseline) if args.baseline else None,
                                reduction_tree=args.subgroup_reductions, requested_group_threads=args.group_threads,
                                requested_input_views=args.forward_input_views,
                                attention_qk=args.attention_qk,
                                attention_pv=args.attention_pv,
                                gpu_diagnostics_valid=True,
                                gpu_diagnostic_guard="native/baseline and isolated Torch workers: captured OS stdout/stderr; known failure signatures invalidate all cohort summaries; not an exhaustive driver diagnostic decoder",
                                torch_execution="fresh subprocess per visit; worker JIT/runtime initialization excluded from warm samples; stdout/stderr captured through process exit; no CPU fallback for MPS",
                                selection="fixed capture; source unordered-tree policy; recorded candidate/group/input-view constraints; no timing-based tuning"),
                  results=make_visit_plan(args.case, args.rounds, args.baseline is not None))
    planned = {(row["operation"], tuple(row["dimensions"]), row["round"], row["path"]): row for row in report["results"]}
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    failed = False
    for op, dims in args.case:
        block = args.attention_block if op == "attention" else (1, 1)
        inputs, output_shape = shapes_for(op, dims)
        case_id = op + "-" + "x".join(map(str, dims))
        arrays = expected = None
        input_hashes = None
        with tempfile.TemporaryDirectory(prefix="luisa-llm-") as temp:
            for index in range(args.rounds):
                order = list(list(itertools.permutations(("native", "baseline", "torch")))[index % 6]) if args.baseline else ["native", "torch"] if index % 2 == 0 else ["torch", "native"]
                for path in order:
                    stem = f"{case_id}-r{index}-{path}"
                    row = planned[op, dims, index, path]
                    if mark_remaining_gpu_visits_not_run(report["results"], args.backend):
                        continue
                    row["status"] = "Running"
                    row.pop("error", None)
                    output = Path(temp) / f"{stem}.f32"
                    try:
                        if path != "torch":
                            source = args.output / (stem + (".metal" if args.backend == "metal" else ".ll"))
                            binary = args.baseline if path == "baseline" else args.native
                            command = [str(binary), "llm", op, ",".join(map(str, dims)), *map(str, block), str(args.samples), str(args.sample_ms), str(args.warmup_ms), str(output)]
                            row["command"] = command
                            environment = native_operation_environment(os.environ, op) | {"LUISA_TILE_BENCH_DUMP_SOURCE": str(source)}
                            if path == "baseline":
                                environment["DYLD_LIBRARY_PATH"] = str(binary.parent) + os.pathsep + environment.get("DYLD_LIBRARY_PATH", "")
                            try:
                                completed = capture_benchmark(command, environment, args.timeout, args.output, stem, row)
                            finally:
                                if source.exists():
                                    row.update(source=source.name, source_sha256=digest(source))
                            measurement = json.loads(completed.stdout)
                            check_metadata(measurement, args.backend, op, dims, block, args.samples, args.subgroup_reductions, args.group_threads, args.forward_input_views, args.attention_qk, args.attention_pv)
                            actual = np.fromfile(output, dtype=np.float32).reshape(output_shape)
                        else:
                            if arrays is None:
                                raise ValueError("native input export unavailable")
                            worker = run_torch_worker(op, dims, arrays, args, args.output, stem, output, row)
                            info = worker["torch_info"]
                            for key, field in (("torch_version", "version"), ("torch_git_version", "git_version"), ("torch_config", "config")):
                                if report["metadata"][key] is not None and report["metadata"][key] != info[field]:
                                    raise ValueError("Torch worker environment changed between visits")
                                report["metadata"][key] = info[field]
                            measurement = worker["measurement"]
                            actual = np.fromfile(output, dtype=np.float32).reshape(output_shape)
                        if path != "torch":
                            # Also populated on failure below, before the paired Torch run.
                            exported = [Path(str(output) + f".input{i}.f32") for i in range(3)]
                            current_hashes = [digest(p) for p in exported]
                            if input_hashes is not None and input_hashes != current_hashes:
                                raise ValueError("native inputs changed between rounds")
                            input_hashes = current_hashes
                            if arrays is None:
                                arrays = [np.fromfile(p, dtype=np.float32).reshape(s) for p, s in zip(exported, inputs)]
                                expected = reference(op, dims, arrays)
                        row.update(valid=True, status="OK", measurement=measurement, correctness=validate_output(actual, expected),
                                   input_sha256=input_hashes, output_sha256=hashlib.sha256(actual.tobytes()).hexdigest())
                    except Exception as error:
                        failed = True
                        row["status"] = "Error"
                        row["error"] = str(error)
                        if path != "torch" and arrays is None:
                            exported = [Path(str(output) + f".input{i}.f32") for i in range(3)]
                            if all(p.exists() for p in exported):
                                arrays = [np.fromfile(p, dtype=np.float32).reshape(s) for p, s in zip(exported, inputs)]
                                input_hashes = [digest(p) for p in exported]
                                expected = reference(op, dims, arrays)
                    mark_remaining_gpu_visits_not_run(report["results"], args.backend)
                    report["metadata"]["gpu_diagnostics_valid"] = not any(r.get("gpu_failure_diagnostics") for r in report["results"])
                    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                    label = ("PASS" if report["metadata"]["gpu_diagnostics_valid"] else "CHECKED, COHORT INVALID") if row["valid"] else row["error"]
                    print(stem, label, flush=True)
    report["metadata"]["artifacts_unchanged"] = hashes == artifact_hashes(binaries, args.compiler_artifact)
    report["metadata"]["cohort_valid"] = not failed and report["metadata"]["artifacts_unchanged"] and report["metadata"]["gpu_diagnostics_valid"]
    report["summary"] = make_summary(report["results"], args.rounds)
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return int(not report["metadata"]["cohort_valid"])


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--torch-worker":
        raise SystemExit(torch_worker(sys.argv[2]))
    raise SystemExit(main())
