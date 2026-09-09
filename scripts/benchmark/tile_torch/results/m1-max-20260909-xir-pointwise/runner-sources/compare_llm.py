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
import statistics
import subprocess
import tempfile

from repeat import artifact_hashes
from run import summarize, time_metal_device, time_torch


OPERATIONS = {"swiglu", "rope", "rmsnorm", "layernorm", "gelu_residual", "masked_softmax", "attention"}


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


def check_metadata(result, backend, op, dims, block, samples, reduction_tree=False, group_threads=0, forward_input_views=False, attention_qk="mma"):
    inputs, output = shapes_for(op, dims)
    fields = dict(implementation="tile_tirx_metal" if backend == "metal" else "tile_xir_simd", backend=backend,
                  precision="fp32", fast_math=False, relaxed_precision=False, runtime="luisa",
                  timing="synchronized_host_wall", batch_policy="one_runtime_command_list_per_batch",
                  operation=op, dimensions=list(dims), attention_block=list(block),
                  input_shapes=[list(s) for s in inputs], output_shape=list(output))
    # Older default-policy artifacts may omit these fields. An explicit
    # request must be acknowledged, never silently ignored by an old binary.
    for key, value in (("reduction_tree", reduction_tree), ("requested_group_threads", group_threads), ("requested_input_views", forward_input_views)):
        if value or key in result:
            fields[key] = value
    if attention_qk != "mma" or "attention_qk" in result:
        fields["attention_qk"] = attention_qk if op == "attention" else "not_applicable"
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
    paths = ("native", "baseline", "torch") if any(r["path"] == "baseline" for r in rows) else ("native", "torch")
    keys = sorted({(r["operation"], tuple(r["dimensions"])) for r in rows})
    for op, dims in keys:
        selected = [r for r in rows if r["operation"] == op and r["dimensions"] == list(dims)]
        item = dict(operation=op, dimensions=list(dims), complete=len(selected) == len(paths) * rounds and all(r["valid"] for r in selected))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, help="frozen old binary and adjacent ABI-coherent libraries; requires all six orders")
    parser.add_argument("--build-dir", type=Path, required=True, help="complete selected configuration is built before native execution")
    parser.add_argument("--backend", choices=("cpu", "metal"), required=True)
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
    parser.add_argument("--attention-qk", choices=("mma", "reduce"), default="mma", help="benchmark-only QK decomposition probe using existing DSL; not a production planner optimization; requires the TIRx llm entry")
    parser.add_argument("--group-threads", type=int, default=0, help="exact Metal group width; zero uses the planner")
    args = parser.parse_args()
    if args.rounds < 2 or args.rounds % 2 or len(set(args.case)) != len(args.case) or min(args.samples, args.sample_ms, args.warmup_ms, args.threads, args.timeout) <= 0:
        parser.error("unique cases, positive settings and an even number of rounds >= 2 required")
    if not (0 < args.attention_block[0] <= 128 and 0 < args.attention_block[1] <= 256):
        parser.error("invalid attention block")
    if args.backend != "metal" and args.metal_device_timing:
        parser.error("GPU timing cannot be used for CPU")
    if not 0 <= args.group_threads <= 1024 or args.backend != "metal" and (args.subgroup_reductions or args.group_threads or args.forward_input_views):
        parser.error("subgroup candidates/group constraints/input views require Metal; group width must be 0..1024")
    if args.attention_qk != "mma" and (args.backend != "metal" or not any(op == "attention" for op, _ in args.case)):
        parser.error("QK decomposition probe requires Metal and an attention case")
    args.native = args.native.resolve(strict=True)
    if args.baseline:
        args.baseline = args.baseline.resolve(strict=True)
        if args.baseline == args.native or args.rounds % 6:
            parser.error("distinct frozen baseline and a multiple of six rounds required")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    build = subprocess.run(["cmake", "--build", str(args.build_dir.resolve(strict=True)), "--parallel", "8"], capture_output=True, text=True)
    (args.output / "build.log").write_text(build.stdout + build.stderr)
    build.check_returncode()
    for key in ("TVM_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "LUISA_SIMD_WORKER_COUNT"):
        os.environ[key] = str(args.threads)
    removed = {}
    for key in list(os.environ):
        if key.startswith("LUISA_SIMD_") and key != "LUISA_SIMD_WORKER_COUNT" or key in ("LUISA_ENABLE_VALIDATION", "DYLD_PRINT_LIBRARIES") or key.startswith("LUISA_TILE_BENCH_"):
            removed[key] = os.environ.pop(key)
    os.environ["LUISA_SIMD_WARP_WIDTH"] = "8"
    if args.subgroup_reductions:
        os.environ["LUISA_TILE_BENCH_REDUCTION_TREE"] = "1"
    if args.forward_input_views:
        os.environ["LUISA_TILE_BENCH_INPUT_VIEWS"] = "1"
    if args.attention_qk != "mma":
        os.environ["LUISA_TILE_BENCH_ATTENTION_QK"] = args.attention_qk
    if args.group_threads:
        os.environ["LUISA_TILE_BENCH_GROUP_THREADS"] = str(args.group_threads)
    if args.metal_device_timing:
        args.metal_device_timing = args.metal_device_timing.resolve(strict=True)
        os.environ["LUISA_TILE_BENCH_METAL_TIMING"] = str(args.metal_device_timing)
        args.compiler_artifact.append(args.metal_device_timing)
    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = "mps" if args.backend == "metal" else "cpu"
    sync = torch.mps.synchronize if device == "mps" else lambda: None
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    binaries = [args.native] + ([args.baseline] if args.baseline else [])
    hashes = artifact_hashes(binaries, args.compiler_artifact)
    root = Path(__file__).resolve().parents[3]
    report = dict(metadata=dict(timestamp=dt.datetime.now(dt.timezone.utc).isoformat(), backend=args.backend, platform=platform.platform(),
                                git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                                torch_version=torch.__version__, torch_git_version=torch.version.git_version, torch_config=torch.__config__.show(),
                                rounds=args.rounds, samples=args.samples, sample_ms=args.sample_ms, warmup_ms=args.warmup_ms,
                                requested_threads=args.threads, removed_environment=removed, artifacts_sha256=hashes,
                                source_sha256={p: digest(root / p) for p in ["src/tests/common/tile_llm_test_utils.h", "src/tests/common/tile_llm_benchmark.h", "scripts/benchmark/tile_torch/compare_llm.py", "scripts/benchmark/tile_torch/run.py"]},
                                comparison="FP32 same exported inputs; warmed E2E excludes compile/upload; GPU control uses uninstrumented command-buffer intervals, not isolated kernel time",
                                baseline=str(args.baseline) if args.baseline else None,
                                reduction_tree=args.subgroup_reductions, requested_group_threads=args.group_threads,
                                requested_input_views=args.forward_input_views,
                                attention_qk=args.attention_qk,
                                selection="fixed capture; source unordered-tree policy; recorded candidate/group/input-view constraints; no timing-based tuning"), results=[])
    failed = False
    for op, dims in args.case:
        block = args.attention_block if op == "attention" else (1, 1)
        inputs, output_shape = shapes_for(op, dims)
        case_id = op + "-" + "x".join(map(str, dims))
        arrays = expected = program = None
        input_hashes = None
        with tempfile.TemporaryDirectory(prefix="luisa-llm-") as temp:
            for index in range(args.rounds):
                order = list(list(itertools.permutations(("native", "baseline", "torch")))[index % 6]) if args.baseline else ["native", "torch"] if index % 2 == 0 else ["torch", "native"]
                for path in order:
                    stem = f"{case_id}-r{index}-{path}"
                    row = dict(operation=op, dimensions=list(dims), round=index, path=path, order=order, valid=False)
                    output = Path(temp) / f"{stem}.f32"
                    try:
                        if path != "torch":
                            source = args.output / (stem + (".metal" if args.backend == "metal" else ".ll"))
                            binary = args.baseline if path == "baseline" else args.native
                            command = [str(binary), "llm", op, ",".join(map(str, dims)), *map(str, block), str(args.samples), str(args.sample_ms), str(args.warmup_ms), str(output)]
                            row["command"] = command
                            environment = os.environ | {"LUISA_TILE_BENCH_DUMP_SOURCE": str(source)}
                            if path == "baseline":
                                environment["DYLD_LIBRARY_PATH"] = str(binary.parent) + os.pathsep + environment.get("DYLD_LIBRARY_PATH", "")
                            try:
                                completed = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=args.timeout)
                            except subprocess.TimeoutExpired as error:
                                (args.output / f"{stem}.log").write_bytes((error.stdout or b"") + (error.stderr or b""))
                                raise
                            (args.output / f"{stem}.log").write_text(completed.stdout + completed.stderr)
                            if source.exists():
                                row.update(source=source.name, source_sha256=digest(source))
                            completed.check_returncode()
                            measurement = json.loads(completed.stdout)
                            check_metadata(measurement, args.backend, op, dims, block, args.samples, args.subgroup_reductions, args.group_threads, args.forward_input_views, args.attention_qk)
                            actual = np.fromfile(output, dtype=np.float32).reshape(output_shape)
                        else:
                            if arrays is None:
                                raise ValueError("native input export unavailable")
                            if program is None:
                                program = torch_program(torch, op, dims, arrays, device)
                            invoke, read, expression = program
                            invoke()
                            sync()
                            precheck = validate_output(read().cpu().numpy(), expected)
                            measurement = time_torch(invoke, sync, args)
                            measurement.update(expression=expression, precision="fp32", fast_math_policy="torch_default", pre_timing_correctness=precheck)
                            if args.metal_device_timing:
                                measurement["device_timing"] = time_metal_device(invoke, sync, args, measurement["repetitions"])
                            actual = read().cpu().numpy().copy()
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
                        row.update(valid=True, measurement=measurement, correctness=validate_output(actual, expected),
                                   input_sha256=input_hashes, output_sha256=hashlib.sha256(actual.tobytes()).hexdigest())
                    except Exception as error:
                        failed = True
                        row["error"] = str(error)
                        if path != "torch" and arrays is None:
                            exported = [Path(str(output) + f".input{i}.f32") for i in range(3)]
                            if all(p.exists() for p in exported):
                                arrays = [np.fromfile(p, dtype=np.float32).reshape(s) for p, s in zip(exported, inputs)]
                                input_hashes = [digest(p) for p in exported]
                                expected = reference(op, dims, arrays)
                    report["results"].append(row)
                    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                    print(stem, "PASS" if row["valid"] else row["error"], flush=True)
    report["metadata"]["artifacts_unchanged"] = hashes == artifact_hashes(binaries, args.compiler_artifact)
    report["summary"] = make_summary(report["results"], args.rounds)
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return int(failed or not report["metadata"]["artifacts_unchanged"])


if __name__ == "__main__":
    raise SystemExit(main())
