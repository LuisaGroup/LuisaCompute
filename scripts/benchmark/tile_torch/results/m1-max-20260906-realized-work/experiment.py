#!/usr/bin/env python3
"""Full-build gated correctness, model selection and frozen old/new replay."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[1]
sys.path.insert(0, str(BENCH))
from repeat import artifact_hashes, load_plan, order_for_round
from run import Case, run_case

SHAPES = ((512, 512, 512), (4096, 4096, 4096), (1025, 1025, 1024), (4096, 4096, 11008),
          (257, 769, 113), (2049, 4097, 1025), (4097, 4097, 4096), (8192, 8192, 8192))
BLOCKS = tuple((m, n, k) for m, n in ((32, 64), (64, 64), (128, 32), (128, 64), (64, 128)) for k in (128, 512, 4096))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("check", "select", "replay"), required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--build", type=Path, default=Path("/tmp/luisa-tvm-mpp.VaKmzx/luisa-build"))
    parser.add_argument("--tvm-build", type=Path, default=Path("/tmp/luisa-tvm-mpp.VaKmzx/build"))
    parser.add_argument("--baseline", type=Path, default=Path("/tmp/luisa-mpp-realized-baseline.mbTNjX"))
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9-]+", args.tag):
        parser.error("use a new lowercase hyphenated tag")
    output = HERE / args.tag
    output.mkdir(exist_ok=False)
    build, tvm, baseline = args.build.resolve(), args.tvm_build.resolve(), args.baseline.resolve()
    binaries = {"reference": baseline / "benchmark_tile_tirx", "candidate": build / "bin/benchmark_tile_tirx"}
    loaders = {"reference": f"{baseline}:{build / 'bin'}", "candidate": f"{build / 'bin'}:{tvm / 'lib'}"}
    system = build / "bin/benchmark_tile_system"
    timing = build / "bin/libluisa-benchmark-metal-timing.dylib"
    report = {"metadata": {"phase": args.phase, "started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "builds": [],
                           "baseline_commit": "71bd1d6a5", "user_barrier_edit_held_constant": True,
                           "variants": {v: dict(binary=str(b), loader=loaders[v]) for v, b in binaries.items()}}, "results": []}
    def save():
        (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    def execute(command, name, loader=None):
        env = dict(os.environ)
        if loader:
            env["DYLD_LIBRARY_PATH"] = loader
        start = time.time()
        completed = subprocess.run(list(map(str, command)), env=env, capture_output=True, text=True)
        text = completed.stdout + completed.stderr
        (output / (name + ".log")).write_text(text)
        return dict(command=list(map(str, command)), loader=loader, log=name + ".log", exit_code=completed.returncode,
                    started_at_unix=start, finished_at_unix=time.time()), text
    for tree in (tvm, build):
        row, text = execute(["cmake", "--build", tree, "--parallel", "8"], "build-" + tree.name)
        report["metadata"]["builds"].append(row)
        save()
        if row["exit_code"]:
            print(text, flush=True)
            return 1
    tests = [("test_tile_tirx_planner", [])]
    tests += [(name, [backend]) for name in ("test_tile_tirx_matrix", "test_tile_tirx_execution", "test_tile_tirx_poc",
                                            "test_tile_tirx_poc_neural", "test_tile_tirx_poc_algorithms") for backend in ("metal", "cpu")]
    tests.append(("test_tile_native_runtime", ["metal"]))
    extra = [Path(__file__), HERE / "protocol.md", BENCH / "run.py", BENCH / "repeat.py", timing]
    extra += [build / "bin" / name for name, _ in tests]
    extra += [p for p in (tvm / "lib").iterdir() if p.is_file() and p.suffix == ".dylib"]
    if args.selection:
        extra += [args.selection / "results.json", *args.selection.glob("*/results.json")]
    before = artifact_hashes([*binaries.values(), system], extra)
    report["metadata"]["artifacts_before"] = before
    for key in ("TVM_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = "8"
    report["metadata"]["removed_environment"] = {key: os.environ.pop(key, None) for key in (
        "PYTORCH_MPS_FAST_MATH", "PYTORCH_MPS_PREFER_METAL", "PYTORCH_ENABLE_MPS_FALLBACK", "LUISA_ENABLE_VALIDATION",
        "MTL_DEBUG_LAYER", "MTL_SHADER_VALIDATION", "DYLD_PRINT_LIBRARIES", "LUISA_TILE_BENCH_DUMP_SOURCE")}
    failed = False
    if args.phase == "check":
        for name, arguments in tests:
            row, text = execute([build / "bin" / name, *arguments], name + "-" + ("-".join(arguments) or "unit"), loaders["candidate"])
            match = re.search(r"all tests passed \((\d+) asserts in (\d+) tests\)", re.sub(r"\x1b\[[0-9;]*m", "", text))
            row["passed_assertions"] = int(match.group(1)) if match else 0
            row["passed"] = row["exit_code"] == 0 and row["passed_assertions"] > 0
            failed |= not row["passed"]
            report["results"].append(row)
            save()
            print(json.dumps(row), flush=True)
    elif args.phase == "select":
        report["metadata"].update(shapes=SHAPES, blocks=BLOCKS, selection_metric="model; timings are diagnostics only", coefficients_fitted=False)
        for i, shape in enumerate(SHAPES):
            for variant in (("reference", "candidate") if i % 2 == 0 else ("candidate", "reference")):
                label = "x".join(map(str, shape)) + "-" + variant
                print(label, flush=True)
                command = [sys.executable, BENCH / "run.py", "--native", binaries[variant], "--system-baseline", system,
                           "--output", output / label, "--backends", "metal", "--operations", "gemm", "--gemm-shapes", "x".join(map(str, shape)),
                           "--execution-scope", "group", "--cooperative-matrix", "--matrix-realization", "mpp-views", "--pipeline-window", "1",
                           "--tune-gemm-blocks", ";".join(",".join(map(str, b)) for b in BLOCKS), "--copy-batch", "1", "--capture-sources",
                           "--tuning-metric", "model", "--samples", "1", "--sample-ms", "1", "--warmup-ms", "1", "--timeout", "300"]
                row, text = execute(command, label, loaders[variant])
                row.update(shape=shape, variant=variant, report=label + "/results.json", passed=row["exit_code"] == 0)
                failed |= not row["passed"]
                report["results"].append(row)
                save()
                print(text[-1200:], flush=True)
    else:
        if not args.selection:
            raise ValueError("replay requires --selection")
        source = json.loads((args.selection / "results.json").read_text())
        if not source.get("passed") or source["metadata"]["phase"] != "select":
            raise ValueError("selection did not complete")
        plans = {v: {} for v in binaries}
        for row in source["results"]:
            selected_path = args.selection / row["report"]
            loaded = load_plan(selected_path, {"gemm"})
            selected, = json.loads(selected_path.read_text())["results"]
            execution, = selected["native"]["execution_plans"]
            for plan in loaded.values():
                # Freeze the solved width, not the original automatic request.
                plan["group_threads"] = execution["threads"]
            plans[row["variant"]].update(loaded)
        keys = list(plans["reference"])
        if set(keys) != set(plans["candidate"]) or len(keys) != len(SHAPES):
            raise ValueError("incomplete or unpaired selection")
        import numpy as np
        import torch
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
        report["metadata"].update(torch_version=torch.__version__, rounds=6, samples=9, sample_ms=30, warmup_ms=100,
                                  frozen_plans={v: [dict(backend=k[0], name=k[1], **p) for k, p in values.items()] for v, values in plans.items()})
        for r in range(6):
            for (backend, name), variant, _ in order_for_round(keys, r):
                config = dict(plans[variant][backend, name])
                case = Case(**config.pop("case"))
                options = argparse.Namespace(**config, native=binaries[variant], system_baseline=system, output=output,
                                             samples=9, sample_ms=30, warmup_ms=100, threads=8, timeout=300,
                                             metal_device_timing=timing, capture_sources=True)
                ordinal = r + keys.index((backend, name)) + (3 if variant == "candidate" else 0)
                print(f"round {r + 1}: {name} {variant}", flush=True)
                previous = os.environ.get("DYLD_LIBRARY_PATH")
                start = time.time()
                try:
                    os.environ["DYLD_LIBRARY_PATH"] = loaders[variant]
                    row = run_case(torch, np, options, case, backend, ordinal)
                except Exception as error:
                    failed = True
                    row = dict(case=vars(case), backend=backend, name=name, valid=False, error=str(error))
                finally:
                    if previous is None:
                        os.environ.pop("DYLD_LIBRARY_PATH", None)
                    else:
                        os.environ["DYLD_LIBRARY_PATH"] = previous
                row.update(round=r, variant=variant, loader=loaders[variant], started_at_unix=start, finished_at_unix=time.time())
                failed |= not row.get("valid", False)
                report["results"].append(row)
                save()
                print("valid" if row.get("valid") else row, flush=True)
    after = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in before}
    report["metadata"].update(artifacts_after=after, artifacts_unchanged=before == after, finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
    report["passed"] = not failed and before == after
    save()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
