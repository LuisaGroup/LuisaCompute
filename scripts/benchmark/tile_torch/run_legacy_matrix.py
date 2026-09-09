#!/usr/bin/env python3
"""Screen old/current Tile across shapes and Runtime lowering paths.

Sequential, alternating paired visits; complete outputs/guards are checked by
benchmark_tile_migrated. Failures and unsupported routes remain in the matrix.
This is a diagnostic comparison, not a Torch/MPS acceptance cohort. No tuning
or winner selection takes place here. CPU time includes Runtime dispatch.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time


def sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def cases():
    result = []

    def add(operation, shapes):
        for shape in shapes:
            result.append(dict(id=operation + "-" + "x".join(map(str, shape)), operation=operation, shape=list(shape)))

    for op in ["copy", "add", "saxpy", "clamp", "exp"]:
        add(op, [(17, 65, 1), (8192, 512, 1)])
    for op in ["rmsnorm", "sum", "max", "min", "abssum", "absmax"]:
        add(op, [(17, 65, 1), (512, 4096, 1)])
    for op in ["cumsum", "cummax"]:
        add(op, [(8, 32, 1), (128, 1024, 1)])
    add("transpose", [(16, 32, 1), (64, 64, 1), (72, 72, 1)])
    add("gemm", [(32, 32, 32), (127, 193, 61), (512, 512, 512), (1024, 2048, 256), (4096, 4096, 4096)])
    add("gemm_fp16", [(32, 32, 32), (127, 193, 61), (512, 512, 512), (1024, 1024, 1024)])
    return result


def routes(case):
    result = ["legacy-metal", "metal", "legacy-simd", "simd"]
    if case["operation"].startswith("gemm") or case["id"] in ["copy-17x65x1", "sum-17x65x1", "rmsnorm-17x65x1"]:
        result += ["metal-native", "metal-tirx-mpp"]
    if case["operation"] == "gemm":
        result += ["metal-direct", "metal-native-direct", "metal-tirx-mpp-direct"]
    return result


def run_command(argv, log, env, timeout):
    record = dict(argv=list(map(str, argv)), started_unix=time.time(), load_average=os.getloadavg())
    with log.open("x") as file:
        process = subprocess.Popen(record["argv"], env=env, stdout=file, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            record["exit_code"] = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            record.update(exit_code=None, timed_out=True)
    record.update(finished_unix=time.time(), log=log.name, log_sha256=sha(log))
    if record.get("exit_code") != 0:
        record["failure_tail"] = log.read_text(errors="replace")[-4000:]
    return record


def summarize(report):
    result = {}
    for visit in report["visits"]:
        key = visit["case"] + "/" + visit["route"]
        group = result.setdefault(key, dict(case=visit["case"], route=visit["route"], attempts=0, passed=0, failed=0,
                                             not_attempted=0, gpu_us=[], batched_e2e_us=[], single_e2e_us=[], compile_ms=[]))
        if "not_attempted" in visit:
            group["not_attempted"] += 1
            continue
        group["attempts"] += 1
        if visit.get("exit_code") != 0:
            group["failed"] += 1
            continue
        group["passed"] += 1
        data = visit["measurement"]
        group["batched_e2e_us"].append(statistics.median(data["throughput_us"]))
        group["single_e2e_us"].append(statistics.median(data["latency_us"]))
        group["compile_ms"].append(data["compile_ms"])
        if "device_timing" in data:
            gpu = data["device_timing"]
            group["gpu_us"].append(statistics.median(s["compute_ns"] / 1000 / gpu["repetitions"] for s in gpu["throughput"]))
    return list(result.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--exporter", type=Path, required=True, help="emit_legacy_tile_sized")
    parser.add_argument("--source-receipt", type=Path, required=True, help="Exact current source overlay/build receipt")
    parser.add_argument("--legacy-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory only")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--sample-ms", type=int, default=10)
    parser.add_argument("--warmup-ms", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--case-id", action="append", help="Explicit subset, recorded before measurement")
    parser.add_argument("--route", action="append", help="Explicit subset, recorded before measurement")
    args = parser.parse_args()
    if not (1 <= args.rounds <= 10 and 1 <= args.samples <= 101 and 1 <= args.sample_ms <= 10000 and
            1 <= args.warmup_ms <= 60000 and 1 <= args.timeout <= 600):
        parser.error("invalid sampling limits")
    binary, exporter = args.binary.resolve(), args.exporter.resolve()
    suite = cases()
    if args.case_id:
        if set(args.case_id) - {case["id"] for case in suite}:
            parser.error("unknown case id")
        suite = [case for case in suite if case["id"] in args.case_id]
    available_routes = {route for case in suite for route in routes(case)}
    if args.route and set(args.route) - available_routes:
        parser.error("unknown or inapplicable route")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LUISA_", "DYLD_", "TVM_"))}
    env["TVM_NUM_THREADS"] = "8"
    artifacts = [binary, exporter, Path(__file__).resolve(), args.source_receipt.resolve(), args.legacy_provenance.resolve()]
    artifacts += sorted(p for p in binary.parent.iterdir() if p.suffix in [".dylib", ".so"])
    report = dict(status="diagnostic_only_not_goal_acceptance", cases=suite, options=vars(args).copy(),
                  source_receipt=json.loads(args.source_receipt.read_text()), legacy_provenance=json.loads(args.legacy_provenance.read_text()),
                  binaries_and_receipts={str(p): sha(p) for p in artifacts}, planned=[], exports=[], visits=[],
                  cpu_metric="synchronized batched/single host wall including Runtime dispatch; NOT pure native entry",
                  gpu_metric="compute-encoder GPU interval sum per dispatch, separately instrumented; controls retained",
                  modern_geometry="pointwise/transpose 16x16; rows=4; scan chunks=32; Metal pipelined GEMM=16x16x32, SIMD GEMM=2x2x4; separate full-K direct FP32 capture=32x32",
                  legacy_geometry="unchanged original per-op block/threads/pipeline; no autotuning")
    for key, value in list(report["options"].items()):
        if isinstance(value, Path):
            report["options"][key] = str(value.resolve())
    for case in suite:
        for route in routes(case):
            if not args.route or route in args.route:
                report["planned"].append(dict(case=case["id"], route=route))

    def save():
        report["summary"] = summarize(report)
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    # Complete capture/export before timing; do not overlap compiler processes
    # with another measured process. Each benchmark's own JIT precedes timing.
    for case in suite:
        prefix = output / (case["id"] + "-legacy")
        record = run_command([exporter, case["operation"], prefix, *case["shape"]], prefix.with_suffix(".export.log"), env, args.timeout)
        record["case"] = case["id"]
        if record.get("exit_code") == 0:
            record["export"] = json.loads((output / record["log"]).read_text().strip().splitlines()[-1])
            record["ast"] = prefix.name + ".ast.json"
            record["ast_sha256"] = sha(output / record["ast"])
        report["exports"].append(record)
        save()
        print("export", case["id"], record.get("exit_code"), flush=True)
    failed = set()
    for case, exported in zip(suite, report["exports"]):
        cohort = [item["route"] for item in report["planned"] if item["case"] == case["id"]]
        for round_id in range(args.rounds):
            order = cohort if round_id % 2 == 0 else list(reversed(cohort))
            for route in order:
                record = dict(case=case["id"], route=route, round=round_id)
                report["visits"].append(record)
                if (case["id"], route) in failed:
                    record["not_attempted"] = "first visit failed; no timing replacement or winner substitution"
                    save()
                    continue
                if route.startswith("legacy-") and exported.get("exit_code") != 0:
                    record["not_attempted"] = "legacy capture/export failed"
                    save()
                    continue
                backend_route = route.removeprefix("legacy-")
                direct = backend_route.endswith("-direct")
                if direct:
                    backend_route = backend_route.removesuffix("-direct")
                block = [32, 32, 32] if direct else [2, 2, 4] if backend_route == "simd" and case["operation"].startswith("gemm") else [16, 16, 32]
                prefix = output / f"{case['id']}-{route}-r{round_id}"
                command = [binary, backend_route, "gemm_direct" if direct else case["operation"], *case["shape"], *block,
                           args.samples, args.sample_ms, args.warmup_ms, prefix]
                if route.startswith("legacy-"):
                    command += [output / exported["ast"], *exported["export"]["dispatch"]]
                run_env = dict(env)
                if backend_route.startswith("metal"):
                    run_env["LUISA_TILE_BENCH_METAL_TIMING"] = str(binary.parent / "libluisa-benchmark-metal-timing.dylib")
                record.update(run_command(command, Path(str(prefix) + ".log"), run_env, args.timeout))
                if record.get("exit_code") == 0:
                    record["measurement"] = json.loads((output / record["log"]).read_text().strip().splitlines()[-1])
                    data = record["measurement"]
                    if data["dimensions"] != case["shape"] or data["correctness"]["checks"] != 2:
                        raise RuntimeError("benchmark receipt does not match the requested complete-check case")
                else:
                    failed.add((case["id"], route))
                record["artifacts"] = {p.name: sha(p) for p in sorted(output.glob(prefix.name + ".*"))}
                save()
                print(case["id"], route, round_id, record.get("exit_code"), flush=True)
    report["artifacts_unchanged"] = all(sha(Path(p)) == digest for p, digest in report["binaries_and_receipts"].items())
    report["finished_unix"] = time.time()
    save()
    if not report["artifacts_unchanged"]:
        raise RuntimeError("source receipt or binaries changed during the experiment")


if __name__ == "__main__":
    main()
