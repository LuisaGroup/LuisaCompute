#!/usr/bin/env python3
"""Predeclared standalone grid permutation screen, no adaptive selection."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from compare_mpp import oracle, validate_metadata, validate_output

BUILD = Path("/tmp/luisa-tvm-mpp.VaKmzx/luisa-build/bin")
CONFIG = (32, 32, 1, 1, 0, 1, 4, 4)
SHAPES = [(1024, 1024, 1537), (4096, 4096, 4096), (8192, 8192, 8192),
          (4096, 4096, 11008), (2049, 4097, 1025)]
PATHS = ("legacy", "linear", "stripe4", "stripe8", "mps")
WALKS = {"legacy": 0, "linear": 1, "stripe4": 4, "stripe8": 8}
RECTANGLES = {"rectangle2x8": (2, 8), "rectangle4x16": (4, 16), "rectangle8x32": (8, 32)}


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def measure(np, shape, path, reference, config=CONFIG, samples=5, sample_ms=20, warmup_ms=100):
    row = dict(shape=list(shape), path=path, config=list(config) if path != "mps" else None,
               valid=False)
    try:
        with tempfile.TemporaryDirectory(prefix="mpp-walk-") as temporary:
            output = Path(temporary) / "output.f32"
            source = Path(temporary) / "kernel.metal"
            binary = BUILD / ("benchmark_tile_system" if path == "mps" else "benchmark_tile_mpp")
            command = [str(binary), "metal" if path == "mps" else "fp32", *map(str, shape),
                       str(samples), str(sample_ms), str(warmup_ms), str(output)]
            if path != "mps":
                tm, tn, sg, coop, static, inline, group, cm = config
                command += list(map(str, (tm, tn, sg, coop, 0, static, inline, group, cm)))
                walk_rows, walk_columns = RECTANGLES[path] if path in RECTANGLES else (WALKS[path], 1)
                if walk_rows:
                    command.append(str(walk_rows))
                    if walk_columns != 1:
                        command.append(str(walk_columns))
            environment = os.environ.copy()
            environment.pop("LUISA_TILE_BENCH_METAL_TIMING", None)
            environment.pop("DYLD_INSERT_LIBRARIES", None)
            environment["LUISA_TILE_BENCH_DUMP_SOURCE"] = str(source)
            row["command"] = command
            process = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=300)
            row.update(returncode=process.returncode, stderr=process.stderr)
            if process.returncode:
                raise RuntimeError(f"benchmark exit {process.returncode}: {process.stderr}")
            result = json.loads(process.stdout)
            row["measurement"] = result
            validate_metadata(result, shape, config if path != "mps" else None, samples,
                              walk=(walk_rows, walk_columns) if path != "mps" else (0, 1))
            if path != "mps":
                if result.get("walk_rows") != walk_rows or result.get("walk_columns", 1) != walk_columns:
                    raise ValueError("physical program walk differs from request")
                sha = digest(source)
                destination = ROOT / "sources" / (sha + ".metal")
                destination.parent.mkdir(exist_ok=True)
                if not destination.exists():
                    destination.write_bytes(source.read_bytes())
                row["source_sha256"] = sha
            actual = np.fromfile(output, dtype="<f4").reshape(shape[:2])
            row["correctness"] = validate_output(np, actual, reference)
            row["medians"] = {key: statistics.median(result[key]) for key in (
                "throughput_us", "latency_us", "gpu_throughput_us", "gpu_latency_us")}
            row["valid"] = True
    except Exception as error:
        row["error"] = str(error)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--correctness-name", default="correctness.json")
    parser.add_argument("--rectangles", action="store_true")
    args = parser.parse_args()
    destination = ROOT / (args.correctness_name if args.correctness else
                          "results-rectangles.json" if args.rectangles else "results.json")
    paths = ("linear", *RECTANGLES, "mps") if args.rectangles else PATHS
    if destination.exists():
        raise FileExistsError(destination)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[variable] = "8"
    import numpy as np
    binaries = [BUILD / name for name in ("benchmark_tile_mpp", "benchmark_tile_system")]
    evidence = dict(timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
                    platform=platform.platform(), numpy=np.__version__,
                    artifacts_before={str(path): digest(path) for path in binaries}, rows=[])
    if args.correctness:
        for shape in ((1, 1, 1), (129, 65, 7), (513, 97, 61), (1025, 129, 33)):
            reference = oracle(np, shape)
            for config in (CONFIG, (32, 32, 1, 1, 0, 1, 4, 1), (64, 64, 4, 1, 0, 1, 4, 1)):
                for path in paths[:-1]:
                    row = measure(np, shape, path, reference, config, 1, 1, 1)
                    evidence["rows"].append(row)
                    print("correctness", shape, config, path, row["valid"], flush=True)
    else:
        references = {shape: oracle(np, shape) for shape in SHAPES}
        for round_index in range(2):
            for shape in SHAPES if round_index == 0 else reversed(SHAPES):
                offset = SHAPES.index(shape) % len(paths)
                order = paths[offset:] + paths[:offset]
                if round_index:
                    order = order[::-1]
                for path in order:
                    row = measure(np, shape, path, references[shape])
                    row.update(round=round_index, order=list(order))
                    evidence["rows"].append(row)
                    print(round_index, shape, path, row.get("medians", row.get("error")), flush=True)
                    destination.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    evidence["artifacts_after"] = {str(path): digest(path) for path in binaries}
    evidence["artifacts_unchanged"] = evidence["artifacts_before"] == evidence["artifacts_after"]
    destination.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    return 0 if evidence["artifacts_unchanged"] and all(row["valid"] for row in evidence["rows"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
