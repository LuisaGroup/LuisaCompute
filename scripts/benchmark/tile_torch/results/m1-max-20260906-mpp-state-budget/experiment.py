#!/usr/bin/env python3
"""Build-gated state-budget experiment; old/new loaders stay independent."""
import argparse
from contextlib import contextmanager
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[1]
sys.path.insert(0, str(BENCH))
from repeat import artifact_hashes, load_plan, order_for_round
from run import Case, implementation_order, run_case

COHORTS = {
    "search": ((1024, 1024, 1024), (4096, 4096, 4096), (1025, 1025, 1024), (4096, 4096, 11008)),
    "heldout": ((257, 769, 113), (2049, 4097, 1025), (4097, 4097, 4096), (8192, 8192, 8192)),
}


@contextmanager
def loader_path(value):
    previous = os.environ.get("DYLD_LIBRARY_PATH")
    os.environ["DYLD_LIBRARY_PATH"] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("DYLD_LIBRARY_PATH", None)
        else:
            os.environ["DYLD_LIBRARY_PATH"] = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("check", "search", "heldout", "replay"), required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--plans", type=Path, nargs="+", help="Completed search/heldout experiment directories for frozen replay")
    parser.add_argument("--build", type=Path, default=Path("/tmp/luisa-tvm-mpp.VaKmzx/luisa-build"))
    parser.add_argument("--tvm-build", type=Path, default=Path("/tmp/luisa-tvm-mpp.VaKmzx/build"))
    parser.add_argument("--baseline", type=Path, default=Path("/tmp/luisa-mpp-budget-baseline.ISAho9"))
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9-]+", args.tag):
        parser.error("use a new lowercase hyphenated tag")
    output = HERE / args.tag
    output.mkdir(exist_ok=False)
    build, tvm, baseline = args.build.resolve(), args.tvm_build.resolve(), args.baseline.resolve()
    report = {"metadata": {"phase": args.phase, "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                           "builds": [], "platform": platform.platform()}, "results": []}
    def save():
        (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    def execute(command, log, loader=None, expected_failure=False):
        env = dict(os.environ)
        if loader:
            env["DYLD_LIBRARY_PATH"] = loader
        start = time.time()
        completed = subprocess.run([str(v) for v in command], env=env, capture_output=True, text=True)
        text = completed.stdout + completed.stderr
        (output / log).write_text(text)
        row = dict(command=[str(v) for v in command], loader=loader, log=log, exit_code=completed.returncode,
                   started_at_unix=start, finished_at_unix=time.time(), expected_failure=expected_failure)
        return row, text
    for tree in (tvm, build):
        row, text = execute(["cmake", "--build", tree, "--parallel", "8"], "build-" + tree.name + ".log")
        report["metadata"]["builds"].append(row)
        save()
        if row["exit_code"]:
            print(text, flush=True)
            return 1
    binaries = {"reference": baseline / "benchmark_tile_tirx", "candidate": build / "bin/benchmark_tile_tirx"}
    loaders = {"reference": f"{baseline}:{build / 'bin'}", "candidate": f"{build / 'bin'}:{tvm / 'lib'}"}
    system, timing = build / "bin/benchmark_tile_system", build / "bin/libluisa-benchmark-metal-timing.dylib"
    extra = [p for p in (tvm / "lib").iterdir() if p.is_file() and p.suffix == ".dylib"]
    extra += [build / "bin" / name for name in ("test_tile_tirx_planner", "test_tile_tirx_matrix", "test_tile_tirx_execution",
                                               "test_tile_tirx_poc", "test_tile_tirx_poc_neural", "test_tile_tirx_poc_algorithms", "test_tile_native_runtime")]
    extra += [BENCH / "run.py", BENCH / "repeat.py", HERE / "protocol.md"]
    for directory in args.plans or []:
        extra += [directory / "results.json", *directory.glob("*-reference/results.json"), *directory.glob("*-candidate/results.json")]
    hashes = artifact_hashes([*binaries.values(), system], [Path(__file__), timing, *extra])
    report["metadata"].update(artifacts_before=hashes, variants={v: dict(binary=str(b), loader=loaders[v]) for v, b in binaries.items()},
                              baseline_commit="52e17083a", user_barrier_edit_held_constant=True,
                              timing="E2E batch/single; no-counter command-buffer GPU batch/single; instrumented diagnostics are not pure kernel timestamps")
    for key in ("TVM_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = "8"
    removed = {key: os.environ.pop(key, None) for key in (
        "PYTORCH_MPS_FAST_MATH", "PYTORCH_MPS_PREFER_METAL", "PYTORCH_ENABLE_MPS_FALLBACK",
        "LUISA_ENABLE_VALIDATION", "MTL_DEBUG_LAYER", "MTL_SHADER_VALIDATION", "DYLD_PRINT_LIBRARIES", "LUISA_TILE_BENCH_DUMP_SOURCE")}
    report["metadata"]["removed_environment"] = removed
    failed = False
    if args.phase == "check":
        tests = [("candidate", "test_tile_tirx_planner", [], False),
                 ("reference", "test_tile_tirx_planner", ["tile_planner_realization_fragment_state_budget"], True),
                 ("reference", "test_tile_tirx_matrix", ["metal", "tile_matrix_mpp_output_only_fragment_budget"], True)]
        tests += [("candidate", name, [backend], False) for name in
                  ("test_tile_tirx_matrix", "test_tile_tirx_execution", "test_tile_tirx_poc", "test_tile_tirx_poc_neural", "test_tile_tirx_poc_algorithms")
                  for backend in ("metal", "cpu")]
        tests.append(("candidate", "test_tile_native_runtime", ["metal"], False))
        for variant, name, arguments, negative in tests:
            row, text = execute([build / "bin" / name, *arguments], f"{variant}-{name}-{'-'.join(arguments) or 'unit'}.log", loaders[variant], negative)
            match = re.search(r"all tests passed \((\d+) asserts in (\d+) tests\)", re.sub(r"\x1b\[[0-9;]*m", "", text))
            assertions = int(match.group(1)) if match else 0
            row.update(variant=variant, passed_assertions=assertions,
                       passed=(row["exit_code"] != 0 and "FAILED" in text) if negative else (row["exit_code"] == 0 and assertions > 0))
            report["results"].append(row)
            failed |= not row["passed"]
            save()
            print(json.dumps(row), flush=True)
    elif args.phase in COHORTS:
        report["metadata"].update(cohort=COHORTS[args.phase], blocks=[[128, 32, 4096], [128, 64, 4096], [64, 128, 4096], [64, 64, 4096]], threads=[64, 128, 256])
        for index, shape in enumerate(COHORTS[args.phase]):
            for variant in (("reference", "candidate") if index % 2 == 0 else ("candidate", "reference")):
                label = "x".join(map(str, shape)) + "-" + variant
                print(label, flush=True)
                command = [sys.executable, BENCH / "run.py", "--native", binaries[variant], "--system-baseline", system,
                           "--output", output / label, "--backends", "metal", "--operations", "gemm", "--gemm-shapes", "x".join(map(str, shape)),
                           "--execution-scope", "group", "--cooperative-matrix", "--matrix-realization", "mpp-views", "--pipeline-window", "1",
                           "--tune-gemm-blocks", "128,32,4096;128,64,4096;64,128,4096;64,64,4096", "--tune-group-threads", "64,128,256",
                           "--copy-batch", "1", "--capture-sources", "--metal-device-timing", timing,
                           "--tuning-metric", "gpu-control", "--samples", "5", "--sample-ms", "20", "--warmup-ms", "100", "--timeout", "300"]
                row, text = execute(command, label + ".log", loaders[variant])
                row.update(variant=variant, shape=shape, report=label + "/results.json", passed=row["exit_code"] == 0)
                report["results"].append(row)
                failed |= not row["passed"]
                save()
                print(text[-1800:], flush=True)
    else:
        if not args.plans:
            raise ValueError("replay requires completed --plans")
        plans = {v: {} for v in binaries}
        for directory in args.plans:
            source = json.loads((directory / "results.json").read_text())
            if not source.get("passed") or source["metadata"]["phase"] not in COHORTS:
                raise ValueError("only completed search/heldout receipts can freeze plans")
            for row in source["results"]:
                selected = load_plan(directory / row["report"], {"gemm"})
                if set(selected) & set(plans[row["variant"]]):
                    raise ValueError("duplicate frozen case")
                plans[row["variant"]].update(selected)
        keys = list(plans["reference"])
        if set(keys) != set(plans["candidate"]):
            raise ValueError("unpaired frozen plans")
        import numpy as np
        import torch
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
        report["metadata"].update(torch_version=torch.__version__, torch_git_version=torch.version.git_version, rounds=6,
                                  samples=9, sample_ms=30, warmup_ms=100,
                                  frozen_plans={v: [dict(backend=k[0], name=k[1], **p) for k, p in values.items()] for v, values in plans.items()})
        for round_index in range(6):
            for (backend, name), variant, _ in order_for_round(keys, round_index):
                config = dict(plans[variant][backend, name])
                case = Case(**config.pop("case"))
                run_args = argparse.Namespace(**config, native=binaries[variant], system_baseline=system, output=output, samples=9,
                                              sample_ms=30, warmup_ms=100, threads=8, timeout=300, metal_device_timing=timing, capture_sources=True)
                ordinal = round_index + keys.index((backend, name)) + (3 if variant == "candidate" else 0)
                print(f"round {round_index + 1}: {name} {variant}", flush=True)
                start = time.time()
                try:
                    with loader_path(loaders[variant]):
                        row = run_case(torch, np, run_args, case, backend, ordinal)
                    if tuple(row["implementation_order"]) != implementation_order(ordinal, True):
                        raise ValueError("framework order mismatch")
                except Exception as error:
                    failed = True
                    row = dict(backend=backend, name=name, case=vars(case), valid=False, error=str(error))
                row.update(round=round_index, variant=variant, loader=loaders[variant], started_at_unix=start, finished_at_unix=time.time())
                report["results"].append(row)
                save()
                print("valid" if row.get("valid") else row, flush=True)
    after = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in hashes}
    report["metadata"].update(artifacts_after=after, artifacts_unchanged=hashes == after, finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
    report["passed"] = not failed and hashes == after
    save()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
