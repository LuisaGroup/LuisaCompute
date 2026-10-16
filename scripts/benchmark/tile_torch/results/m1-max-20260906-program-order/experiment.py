#!/usr/bin/env python3
"""Build-gated correctness and fixed traversal/K diagnostic."""
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
from run import Case, run_case
from repeat import artifact_hashes

SHAPES = ((512, 512, 512), (4096, 4096, 4096), (8192, 8192, 8192),
          (4096, 4096, 11008), (2049, 4097, 1025), (257, 769, 113))
CONFIGS = tuple((k, r, c) for k in (512, 4096) for r, c in ((1, 1), (2, 4), (4, 8), (8, 16)))
TVM_LIB = Path("/private/tmp/luisa-tvm-mpp.VaKmzx/build/lib")
TESTS = [("test_tile_tirx", []), ("test_tile_tirx_planner", [])]
TESTS += [(name, [backend]) for name in ("test_tile_tirx_execution", "test_tile_tirx_matrix", "test_tile_tirx_poc",
            "test_tile_tirx_poc_neural", "test_tile_tirx_poc_algorithms", "test_tile_tirx_pipeline", "test_tile_tirx_memory",
            "test_tile_tirx_cooperative") for backend in ("metal", "cpu")]
TESTS += [("test_tile_native_runtime", ["metal"])]


def canonical_llvm_labels(code):
    # TVM uses per-process allocation addresses as TBAA labels. Preserve the
    # alias graph, identities, width/offset suffixes and every instruction.
    labels = {}
    pattern = r'(^!\d+ = !\{!")(0x[0-9a-f]+)((?:\.w\d+\.b\d+)?", !\d+, i64 0\}$)'
    def replace(match):
        identity = labels.setdefault(match[2], len(labels))
        return match[1] + f"allocation_{identity}" + match[3]
    return re.sub(pattern, replace, code, flags=re.MULTILINE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("check", "screen", "defaults", "syntax"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build", type=Path, default=Path("/tmp/luisa-tvm-mpp.VaKmzx/luisa-build"))
    parser.add_argument("--baseline", type=Path, default=Path("/tmp/luisa-program-walk-baseline.xgtt5L"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    build = args.build.resolve(strict=True)
    loader = f"{build / 'bin'}:{TVM_LIB}"
    os.environ["DYLD_LIBRARY_PATH"] = loader
    for key in ("VECLIB_MAXIMUM_THREADS", "TVM_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[key] = "8"
    removed = {key: os.environ.pop(key, None) for key in ("MTL_DEBUG_LAYER", "MTL_SHADER_VALIDATION", "LUISA_ENABLE_VALIDATION",
               "DYLD_PRINT_LIBRARIES", "LUISA_TILE_BENCH_DUMP_SOURCE", "PYTORCH_MPS_FAST_MATH", "PYTORCH_MPS_PREFER_METAL", "PYTORCH_ENABLE_MPS_FALLBACK")}
    report = {"metadata": {"started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "phase": args.phase,
                           "loader": loader, "removed_environment": removed}, "results": []}

    def save():
        (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def execute(command, label):
        process = subprocess.run(list(map(str, command)), capture_output=True, text=True)
        text = process.stdout + process.stderr
        (args.output / (label + ".log")).write_text(text)
        return process.returncode, text

    command = ["cmake", "--build", build, "--parallel", "8"]
    code, text = execute(command, "build")
    report["metadata"]["build"] = {"command": list(map(str, command)), "exit_code": code}
    save()
    if code:
        print(text, flush=True)
        return 1
    native = build / "bin/benchmark_tile_tirx"
    system = build / "bin/benchmark_tile_system"
    binaries = [native, system] + [build / "bin" / name for name, _ in TESTS]
    if args.phase == "defaults":
        binaries.append(args.baseline.resolve(strict=True) / native.name)
    before = artifact_hashes(binaries, [Path(__file__), HERE / "protocol.md", BENCH / "run.py", BENCH / "repeat.py",
                                        *TVM_LIB.glob("*.dylib")])
    report["metadata"]["artifacts_before"] = before
    failed = False
    if args.phase == "check":
        for name, arguments in TESTS:
            label = name + "-" + ("-".join(arguments) or "unit")
            command = [build / "bin" / name, *arguments]
            print(label, flush=True)
            code, text = execute(command, label)
            passed = re.search(r"all tests passed \((\d+) asserts in (\d+) tests\)", re.sub(r"\x1b\[[0-9;]*m", "", text))
            row = dict(command=list(map(str, command)), exit_code=code, log=label + ".log", passed=code == 0 and passed is not None,
                       assertions=int(passed.group(1)) if passed else 0)
            failed |= not row["passed"]
            report["results"].append(row)
            save()
            print(row, flush=True)
    elif args.phase == "syntax":
        source_root = BENCH.parents[2]
        files = ["src/tests/benchmark/benchmark_tile_tirx.cpp"]
        files += [f"src/tests/unit/tile/bridge/test_tirx_{name}.cpp" for name in ("execution", "layout", "matrix")]
        files += [f"src/tile/bridge/tirx/{name}.cpp" for name in
                  ("compiler", "cooperative", "execution", "lower", "mapping", "planner", "views")]
        commands = [(Path(f).stem, [sys.executable, source_root / "scripts/check_cpp_syntax.py",
                     "--compile-commands-dir", build, f]) for f in files]
        commands.append(("python-tests", [sys.executable, "-m", "unittest", "discover", "-s", BENCH, "-p", "test_*.py"]))
        for label, command in commands:
            print(label, flush=True)
            code, text = execute(command, label)
            report["results"].append(dict(command=list(map(str, command)), log=label + ".log", exit_code=code, passed=code == 0))
            failed |= code != 0
            save()
    elif args.phase == "defaults":
        import numpy as np
        import torch
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
        cohort = [("metal", Case("gemm", *shape)) for shape in ((512, 512, 512), (4096, 4096, 4096), (2049, 4097, 1025))]
        cohort += [(backend, Case(op, m, n, 1)) for backend in ("metal", "cpu")
                   for op, m, n in (("add", 17, 257), ("gelu_pair", 17, 257), ("softmax", 37, 1537))]
        for backend, case in cohort:
            sources = []
            for label, binary in (("baseline", args.baseline / native.name), ("current", native)):
                matrix = case.operation == "gemm"
                os.environ["DYLD_LIBRARY_PATH"] = f"{binary.parent.resolve()}:{loader}"
                options = argparse.Namespace(native=binary.resolve(), output=args.output, samples=3, sample_ms=1, warmup_ms=1,
                    threads=8, timeout=300, execution_scope="group" if matrix else "auto", pipeline_window=1,
                    cooperative_matrix=matrix, matrix_realization="mpp-views" if matrix else "simdgroup",
                    no_vectorize=False, auto_vectorize=backend == "cpu", gemm_block=(128, 64, 4096),
                    group_threads=256 if matrix else 0, copy_batch=1, capture_sources=True,
                    metal_subgroup_reductions=backend == "metal" and case.operation == "softmax")
                print(f"default {backend} {case.name} {label}", flush=True)
                try:
                    row = run_case(torch, np, options, case, backend, len(sources))
                except Exception as error:
                    row = dict(valid=False, case=vars(case), error=str(error))
                row["variant"] = label
                failed |= not row.get("valid", False)
                sources.append(row.get("native_source_sha256"))
                report["results"].append(row)
                save()
            raw_equal = len(sources) == 2 and sources[0] is not None and sources[0] == sources[1]
            equivalent = raw_equal
            if backend == "cpu" and all(sources):
                codes = [(args.output / "sources" / (sha + ".ll")).read_text() for sha in sources]
                equivalent = canonical_llvm_labels(codes[0]) == canonical_llvm_labels(codes[1])
            failed |= not equivalent
            report["results"][-1].update(default_source_unchanged=raw_equal, default_source_equivalent=equivalent)
            save()
        os.environ["DYLD_LIBRARY_PATH"] = loader
    else:
        import numpy as np
        import torch
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
        report["metadata"].update(shapes=SHAPES, configs=CONFIGS, rounds=2, samples=5, sample_ms=20, warmup_ms=100,
                                  torch_version=torch.__version__, selection="none; fixed exploratory cohort")
        for round_index in range(2):
            for shape in SHAPES if round_index == 0 else SHAPES[::-1]:
                shift = SHAPES.index(shape)
                order = CONFIGS[shift:] + CONFIGS[:shift]
                if round_index:
                    order = order[::-1]
                for ordinal, (k, r, c) in enumerate(order):
                    case = Case("gemm", *shape)
                    options = argparse.Namespace(native=native, system_baseline=system, output=args.output, samples=5, sample_ms=20,
                        warmup_ms=100, threads=8, timeout=300, execution_scope="group", pipeline_window=1, cooperative_matrix=True,
                        matrix_realization="mpp-views", no_vectorize=False, auto_vectorize=False, gemm_block=(128, 64, k),
                        group_threads=256, copy_batch=1, program_order_rows=r, program_order_columns=c, capture_sources=True,
                        metal_device_timing=build / "bin/libluisa-benchmark-metal-timing.dylib")
                    print(f"round {round_index} {shape} K={k} order={r}x{c}", flush=True)
                    try:
                        row = run_case(torch, np, options, case, "metal", ordinal + shift + round_index)
                    except Exception as error:
                        row = dict(valid=False, case=vars(case), error=str(error))
                    row.update(round=round_index, configuration=[k, r, c], order=order)
                    failed |= not row.get("valid", False)
                    report["results"].append(row)
                    save()
                    print(row.get("native", {}).get("gpu_throughput_us_p50", row.get("error", "validated")), flush=True)
    after = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in before}
    report["metadata"].update(artifacts_after=after, artifacts_unchanged=before == after,
                              finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
    report["passed"] = not failed and before == after
    save()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
