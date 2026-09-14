#!/usr/bin/env python3
"""Interleaved frozen-binary comparison; CPU Runtime wall time, not a device counter."""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from compare_llm import check_metadata, parse_case, reference, shapes_for, validate_output


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifacts(directory):
    return {str(p): digest(p) for p in sorted(directory.iterdir())
            if p.is_file() and (p.suffix in {".so", ".dylib"} or p.name == "benchmark_tile_xir")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensors", type=Path, required=True)
    parser.add_argument("--cases", type=parse_case, nargs="+", required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    if min(args.rounds, args.samples, args.timeout) < 1:
        parser.error("rounds, samples and timeout must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    args.tensors.mkdir(parents=True, exist_ok=False)
    binaries = {name: getattr(args, name).resolve() for name in ("baseline", "candidate")}
    before = {name: artifacts(path) for name, path in binaries.items()}
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LUISA_SIMD_", "LUISA_TILE_", "DYLD_"))}
    env.update(LUISA_SIMD_WORKER_COUNT="8", LUISA_SIMD_WARP_WIDTH="8")
    report = dict(metadata=dict(platform=platform.platform(), date=dt.datetime.now(dt.timezone.utc).isoformat(),
                               rounds=args.rounds, samples=args.samples, timeout_seconds=args.timeout,
                               sample_ms=30, warmup_ms=100, packet_width=8, workers=8,
                               timing="synchronized_host_wall", kernel_only=False,
                               cases=args.cases, artifacts_sha256=before), results=[], summaries=[])
    record = args.output / "report.json"

    def save():
        record.write_text(json.dumps(report, indent=2) + "\n")

    failed = False
    for case_id, (op, dims) in enumerate(args.cases):
        input_hashes = None
        pairs = []
        disabled = set()
        for round_id in range(args.rounds):
            order = ("baseline", "candidate") if round_id % 2 == 0 else ("candidate", "baseline")
            pair = {}
            for name in order:
                tag = f"{case_id:02}-{op}-{'x'.join(map(str, dims))}-{round_id}-{name}"
                row = dict(operation=op, dimensions=dims, round=round_id, variant=name, order=order, valid=False)
                report["results"].append(row)
                if name in disabled:
                    row["error"] = "not retried after failed visit in this cohort"
                    save()
                    continue
                output = args.tensors / (tag + ".f32")
                command = [str(binaries[name] / "benchmark_tile_xir"), "llm", op, ",".join(map(str, dims)),
                           "1", "1", str(args.samples), "30", "100", str(output)]
                row["command"] = command
                start = time.monotonic()
                try:
                    with (args.output / (tag + ".json")).open("w") as stdout, (args.output / (tag + ".log")).open("w") as stderr:
                        completed = subprocess.run(command, env=env, stdout=stdout, stderr=stderr, timeout=args.timeout, check=False)
                    row["returncode"] = completed.returncode
                    if completed.returncode:
                        raise ValueError(f"benchmark exit {completed.returncode}")
                    measurement = json.loads((args.output / (tag + ".json")).read_text())
                    check_metadata(measurement, "cpu", op, dims, (1, 1), args.samples)
                    if "W8," not in measurement["realization"] or "8 CPU workers;" not in measurement["realization"]:
                        raise ValueError("wrong SIMD execution controls")
                    inputs, shape = shapes_for(op, dims)
                    paths = [Path(str(output) + f".input{i}.f32") for i in range(3)]
                    hashes = [digest(p) for p in paths]
                    if input_hashes is not None and input_hashes != hashes:
                        raise ValueError("input bits changed across variants/rounds")
                    input_hashes = hashes
                    arrays = [np.fromfile(p, dtype=np.float32).reshape(s) for p, s in zip(paths, inputs)]
                    actual = np.fromfile(output, dtype=np.float32).reshape(shape)
                    oracle = validate_output(actual, reference(op, dims, arrays))
                    for key in ("throughput_us", "latency_us"):
                        samples = measurement[key]
                        if len(samples) != args.samples or not all(math.isfinite(v) and v > 0 for v in samples):
                            raise ValueError("invalid timing samples")
                    row.update(valid=True, measurement=measurement, oracle=oracle,
                               input_sha256=hashes, output_sha256=digest(output))
                    pair[name] = statistics.median(measurement["throughput_us"])
                    print(tag, "PASS", round(pair[name], 3), "us", flush=True)
                except (ValueError, OSError, subprocess.TimeoutExpired) as error:
                    row["error"] = str(error)
                    disabled.add(name)
                    failed = True
                    print(tag, "FAILED", str(error), flush=True)
                finally:
                    row["process_and_validation_wall_seconds"] = time.monotonic() - start
                    save()
            if len(pair) == 2:
                pairs.append(pair["baseline"] / pair["candidate"])
        report["summaries"].append(dict(operation=op, dimensions=dims, complete_pairs=len(pairs),
                                        baseline_over_candidate_throughput_ratios=pairs,
                                        median_ratio=statistics.median(pairs) if pairs else None))
        save()
    after = {name: artifacts(path) for name, path in binaries.items()}
    report["metadata"]["artifacts_unchanged"] = before == after
    if before != after:
        failed = True
        report["metadata"]["artifacts_after_sha256"] = after
    save()
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
