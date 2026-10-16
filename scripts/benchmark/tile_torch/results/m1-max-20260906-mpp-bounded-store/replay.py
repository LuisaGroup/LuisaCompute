#!/usr/bin/env python3
"""Frozen bounded-output experiment; independently load old/new compiler stacks."""
import argparse
from contextlib import contextmanager
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[1]
sys.path.insert(0, str(BENCH))
from repeat import artifact_hashes, load_plan, order_for_round
from run import Case, implementation_order, run_case


@contextmanager
def loader_path(value):
    """All measurements are sequential; restore the caller's environment."""
    previous = os.environ.get("DYLD_LIBRARY_PATH")
    os.environ["DYLD_LIBRARY_PATH"] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("DYLD_LIBRARY_PATH", None)
        else:
            os.environ["DYLD_LIBRARY_PATH"] = previous


def schedule(keys, phase):
    if phase == "pilot":
        # old-pilot predates implementation. Complete old/new/new/old, without
        # treating these sequential pilot runs as final paired evidence.
        for round_index, variant in enumerate(("candidate", "candidate", "reference")):
            for index, key in enumerate(keys):
                yield round_index, key, variant, round_index + index
    else:
        for round_index in range(6):
            for key, variant, _ in order_for_round(keys, round_index):
                index = keys.index(key)
                # Each variant gets all six native/Torch/MPS permutations.
                ordinal = round_index + index + (3 if variant == "candidate" else 0)
                yield round_index, key, variant, ordinal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "replay"), required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--build", type=Path, default=Path("/tmp/luisa-tvm-mpp.VaKmzx/luisa-build"))
    parser.add_argument("--tvm-build", type=Path, default=Path("/tmp/luisa-tvm-mpp.VaKmzx/build"))
    parser.add_argument("--baseline", type=Path, default=Path("/tmp/luisa-mpp-store-baseline.u37ixE"))
    args = parser.parse_args()
    if not args.tag or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in args.tag):
        parser.error("use a unique lowercase hyphenated tag")
    output = HERE / args.tag
    output.mkdir(exist_ok=False)
    build, tvm, baseline = args.build.resolve(), args.tvm_build.resolve(), args.baseline.resolve()
    report = {"metadata": {"phase": args.phase, "rounds": 6 if args.phase == "replay" else 3,
                           "started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "builds": []},
              "results": []}
    save = lambda: (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for tree in (tvm, build):
        command = ["cmake", "--build", str(tree), "--parallel", "8"]
        built = subprocess.run(command, capture_output=True, text=True)
        log = "build-" + tree.name + ".log"
        (output / log).write_text(built.stdout + built.stderr)
        report["metadata"]["builds"].append(dict(command=command, exit_code=built.returncode, log=log))
        save()
        if built.returncode:
            print(built.stdout + built.stderr, flush=True)
            return 1
    binaries = {"reference": baseline / "benchmark_tile_tirx", "candidate": build / "bin/benchmark_tile_tirx"}
    loaders = {"reference": f"{baseline}:{build / 'bin'}", "candidate": f"{build / 'bin'}:{tvm / 'lib'}"}
    system = build / "bin/benchmark_tile_system"
    timing = build / "bin/libluisa-benchmark-metal-timing.dylib"
    evidence = HERE / "old-pilot/results.json"
    plans = load_plan(evidence, {"gemm"})
    keys = list(plans)
    expected = {(129, 257, 61), (1025, 1025, 1024), (2049, 4097, 1025),
                (4097, 4097, 4096), (1024, 1024, 1024), (4096, 4096, 4096)}
    if len(keys) != 6 or {(p["case"]["m"], p["case"]["n"], p["case"]["k"]) for p in plans.values()} != expected:
        raise ValueError("the frozen protocol requires the original six shapes")
    for (backend, _), plan in plans.items():
        if (backend != "metal" or plan["gemm_block"] != (128, 32, 4096) or
                plan["group_threads"] != 128 or plan["pipeline_window"] != 1 or plan["copy_batch"] != 1 or
                plan["matrix_realization"] != "mpp-views"):
            raise ValueError("frozen schedule mismatch")
    events = list(schedule(keys, args.phase))
    if args.phase == "replay":
        for key in keys:
            for variant in binaries:
                orders = [implementation_order(o, True) for _, k, v, o in events if k == key and v == variant]
                if len(orders) != 6 or len(set(orders)) != 6:
                    raise ValueError("unbalanced framework order")
    removed = {key: os.environ.pop(key, None) for key in (
        "PYTORCH_MPS_FAST_MATH", "PYTORCH_MPS_PREFER_METAL", "PYTORCH_ENABLE_MPS_FALLBACK",
        "LUISA_ENABLE_VALIDATION", "MTL_DEBUG_LAYER", "MTL_SHADER_VALIDATION", "DYLD_PRINT_LIBRARIES",
        "LUISA_TILE_BENCH_DUMP_SOURCE")}
    for key in ("TVM_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = "8"
    import numpy as np
    import torch
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no fallback")
    extra = [p for p in (tvm / "lib").iterdir() if p.is_file() and p.suffix == ".dylib"]
    hashes = artifact_hashes([*binaries.values(), system], [timing, evidence, Path(__file__), *extra])
    report["metadata"].update(
        platform=platform.platform(), torch_version=torch.__version__, torch_git_version=torch.version.git_version,
        torch_config=torch.__config__.show(), threads=8, removed_environment=removed,
        native_variants={v: {"binary": str(b), "loader": loaders[v]} for v, b in binaries.items()},
        artifacts_before=hashes, frozen_report=str(evidence),
        samples=9, sample_ms=30, warmup_ms=100, capture_sources=True, metal_device_timing=str(timing),
        frozen_plans=[dict(backend=k[0], name=k[1], **p) for k, p in plans.items()],
        timing="separate batched/single E2E, no-counter command-buffer GPU control, and instrumented diagnostics; not pure kernel time")
    save()
    failed = False
    for round_index, (backend, name), variant, ordinal in events:
        config = dict(plans[backend, name])
        case = Case(**config.pop("case"))
        run_args = argparse.Namespace(**config, native=binaries[variant], system_baseline=system,
                                      output=output, samples=9, sample_ms=30, warmup_ms=100, threads=8,
                                      timeout=300, metal_device_timing=timing, capture_sources=True)
        started = time.time()
        print(f"{args.phase} {round_index + 1}: {name} {variant}", flush=True)
        try:
            with loader_path(loaders[variant]):
                row = run_case(torch, np, run_args, case, backend, ordinal)
            print(f"  valid; GPU {row['native']['device_timing']['control']['command_buffer_throughput_us_p50']:.3f} us; "
                  f"E2E {row['native']['throughput_us_p50']:.3f} us", flush=True)
        except Exception as error:
            failed = True
            row = dict(backend=backend, name=name, case=vars(case), valid=False, error=str(error))
            print(f"  FAILED: {error}", flush=True)
        row.update(round=round_index, variant=variant, loader=loaders[variant],
                   started_at_unix=started, finished_at_unix=time.time())
        report["results"].append(row)
        save()
    after = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in hashes}
    report["metadata"].update(artifacts_after=after, artifacts_unchanged=hashes == after,
                              finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
    report["passed"] = not failed and hashes == after
    save()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
