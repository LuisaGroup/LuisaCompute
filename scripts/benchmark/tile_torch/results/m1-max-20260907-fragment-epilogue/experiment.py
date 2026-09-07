#!/usr/bin/env python3
"""Build-gated closed matrix epilogue checks and fixed-schedule replay."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[1]
sys.path.insert(0, str(BENCH))
from run import Case, MATRIX_OPERATIONS, run_case
from repeat import artifact_hashes

SHAPES = ((128, 128, 128), (127, 193, 61), (1024, 1024, 1024),
          (4096, 4096, 4096), (128, 2048, 512), (2048, 128, 512))
TESTS = [("test_tile_tirx", []), ("test_tile_tirx_planner", [])]
TESTS += [(name, [backend]) for name in ("test_tile_tirx_execution", "test_tile_tirx_matrix", "test_tile_tirx_poc",
          "test_tile_tirx_poc_neural", "test_tile_tirx_poc_algorithms", "test_tile_tirx_pipeline", "test_tile_tirx_memory",
          "test_tile_tirx_cooperative") for backend in ("metal", "cpu")]
TESTS += [("test_tile_native_runtime", ["metal"])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("check", "replay", "controls"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build", type=Path, default=Path("/tmp/luisa-tvm-mpp.VaKmzx/luisa-build"))
    parser.add_argument("--tvm-lib", type=Path, default=Path("/Users/mike/.cache/luisa-tile/tvm-fragment-build.0ToXYr/lib"))
    parser.add_argument("--baseline", type=Path, default=Path("/tmp/luisa-fragment-baseline.VtSxFD"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    build, tvm = args.build.resolve(strict=True), args.tvm_lib.resolve(strict=True)
    loader = f"{build / 'bin'}:{tvm}"
    os.environ["DYLD_LIBRARY_PATH"] = loader
    for key in ("VECLIB_MAXIMUM_THREADS", "TVM_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[key] = "8"
    removed = {key: os.environ.pop(key, None) for key in ("MTL_DEBUG_LAYER", "MTL_SHADER_VALIDATION", "LUISA_ENABLE_VALIDATION",
               "DYLD_PRINT_LIBRARIES", "LUISA_TILE_BENCH_DUMP_SOURCE", "PYTORCH_MPS_FAST_MATH", "PYTORCH_MPS_PREFER_METAL", "PYTORCH_ENABLE_MPS_FALLBACK")}
    report = {"metadata": {"phase": args.phase, "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                           "loader": loader, "removed_environment": removed}, "results": []}

    def save():
        (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def execute(command, label):
        process = subprocess.run(list(map(str, command)), capture_output=True, text=True, timeout=900)
        (args.output / (label + ".log")).write_text(process.stdout + process.stderr)
        return process.returncode

    command = ["cmake", "--build", build, "--parallel", "8"]
    code = execute(command, "build")
    report["metadata"]["build"] = dict(command=list(map(str, command)), exit_code=code)
    save()
    if code:
        return 1
    native = build / "bin/benchmark_tile_tirx"
    binaries = [native] + [build / "bin" / name for name, _ in TESTS]
    if args.phase == "controls":
        binaries.append(args.baseline / native.name)
    extra = [Path(__file__), HERE / "protocol.md", BENCH / "run.py", BENCH / "repeat.py", *tvm.glob("*.dylib")]
    before = artifact_hashes(binaries, extra)
    report["metadata"]["artifacts_before"] = before
    failed = False
    if args.phase == "check":
        for name, arguments in TESTS:
            label = name + "-" + ("-".join(arguments) or "unit")
            print(label, flush=True)
            command = [build / "bin" / name, *arguments]
            code = execute(command, label)
            log = (args.output / (label + ".log")).read_text()
            passed = re.search(r"all tests passed \((\d+) asserts in (\d+) tests\)", re.sub(r"\x1b\[[0-9;]*m", "", log))
            row = dict(command=list(map(str, command)), log=label + ".log", exit_code=code,
                       passed=code == 0 and passed is not None, assertions=int(passed[1]) if passed else 0)
            failed |= not row["passed"]
            report["results"].append(row)
            save()
    else:
        import numpy as np
        import torch
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
        report["metadata"].update(torch_version=torch.__version__, torch_commit=torch.version.git_version,
                                  shapes=SHAPES, rounds=4 if args.phase == "replay" else 1,
                                  selection="none; 64x64x4096 / 256 workers fixed for all matrix graphs")

        def measure(case, backend, variant, ordinal, round_index):
            matrix = case.operation in MATRIX_OPERATIONS
            binary = args.baseline / native.name if variant == "baseline" else native
            os.environ["DYLD_LIBRARY_PATH"] = f"{args.baseline.resolve()}:{build / 'bin'}" if variant == "baseline" else loader
            options = argparse.Namespace(native=binary.resolve(), output=args.output, samples=7 if args.phase == "replay" else 3,
                sample_ms=20 if args.phase == "replay" else 1, warmup_ms=100 if args.phase == "replay" else 1,
                threads=8, timeout=300, execution_scope="group" if matrix and backend == "metal" else "auto", pipeline_window=1,
                cooperative_matrix=matrix and backend == "metal", matrix_realization="mpp-views" if matrix and backend == "metal" else "simdgroup",
                no_vectorize=False, auto_vectorize=backend == "cpu", gemm_block=(64, 64, 4096) if backend == "metal" else (8, 8, 16),
                group_threads=256 if matrix and backend == "metal" else 0, copy_batch=1, capture_sources=True,
                fuse_matrix_epilogues=variant == "fused", metal_subgroup_reductions=backend == "metal" and case.operation in ("softmax", "sum"),
                metal_device_timing=build / "bin/libluisa-benchmark-metal-timing.dylib" if args.phase == "replay" else None)
            print(f"{round_index} {backend} {case.name} {variant}", flush=True)
            try:
                row = run_case(torch, np, options, case, backend, ordinal)
            except Exception as error:
                row = dict(valid=False, case=vars(case), backend=backend, error=str(error))
            row.update(variant=variant, round=round_index)
            report["results"].append(row)
            save()
            return row.get("valid", False)

        if args.phase == "replay":
            cases = [Case(op, *shape) for shape in SHAPES for op in MATRIX_OPERATIONS]
            for r in range(4):
                for ci, case in enumerate(cases if r % 2 == 0 else cases[::-1]):
                    variants = ("reference", "fused") if r % 2 == 0 else ("fused", "reference")
                    # Each variant sees both native/Torch orders twice; the
                    # two order factors cover all four combinations.
                    for variant in variants:
                        failed |= not measure(case, "metal", variant, r // 2, r)
        else:
            controls = [("metal", Case(op, *shape)) for op in MATRIX_OPERATIONS
                        for shape in ((128, 128, 128), (127, 193, 61))]
            controls += [(backend, Case(op, 17, 257)) for backend in ("metal", "cpu") for op in ("add", "gelu_pair", "sum", "softmax")]
            for backend, case in controls:
                for variant in ("baseline", "reference"):
                    failed |= not measure(case, backend, variant, 0, 0)
    after = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in before}
    report["metadata"].update(artifacts_after=after, artifacts_unchanged=before == after,
                              finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
    report["passed"] = not failed and before == after
    save()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
