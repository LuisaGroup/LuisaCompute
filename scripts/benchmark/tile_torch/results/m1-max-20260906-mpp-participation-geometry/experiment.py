#!/usr/bin/env python3
"""Matched-geometry MPP participation diagnostic; no schedule selection."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[1]
sys.path.insert(0, str(BENCH))
from compare_mpp import measure, oracle

SHAPES = ((512, 512, 512), (4096, 4096, 4096), (8192, 8192, 8192),
          (256, 11008, 4096), (2049, 4097, 1025))
CONFIGS = {
    "independent_128x64": (32, 32, 1, 1, 0, 1, 8, 4),
    "collective_128x64": (128, 64, 8, 1, 0, 1, 8, 1),
    "independent_128x32": (32, 32, 1, 1, 0, 1, 4, 4),
    "collective_128x32": (128, 32, 4, 1, 0, 1, 4, 1),
    "independent_64x64": (32, 32, 1, 1, 0, 1, 4, 2),
    "collective_64x64": (64, 64, 4, 1, 0, 1, 4, 1),
    "mps": None,
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build", type=Path, default=Path("/tmp/luisa-tvm-mpp.VaKmzx/luisa-build"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    args.build = args.build.resolve(strict=True)
    args.mpp = args.build / "bin/benchmark_tile_mpp"
    args.mps = args.build / "bin/benchmark_tile_system"
    args.samples, args.sample_ms, args.warmup_ms, args.timeout = 5, 20, 100, 300
    report = {"metadata": {"started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                           "shapes": SHAPES, "configs": CONFIGS, "rounds": 2,
                           "samples": 5, "sample_ms": 20, "warmup_ms": 100,
                           "selection": "none; exploratory diagnostic only"}, "results": []}

    def save():
        (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    command = ["cmake", "--build", str(args.build), "--parallel", "8"]
    build = subprocess.run(command, capture_output=True, text=True)
    (args.output / "build.log").write_text(build.stdout + build.stderr)
    report["metadata"]["build"] = {"command": command, "exit_code": build.returncode}
    save()
    if build.returncode:
        return 1
    artifacts = (args.mpp, args.mps, Path(__file__), HERE / "protocol.md", BENCH / "compare_mpp.py", BENCH / "run.py")
    before = {str(p): digest(p) for p in artifacts}
    report["metadata"]["artifacts_before"] = before
    report["metadata"]["removed_environment"] = {key: os.environ.pop(key, None) for key in (
        "MTL_DEBUG_LAYER", "MTL_SHADER_VALIDATION", "LUISA_ENABLE_VALIDATION", "DYLD_PRINT_LIBRARIES",
        "LUISA_TILE_BENCH_DUMP_SOURCE")}
    os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
    import numpy as np
    references = {s: oracle(np, s) for s in SHAPES}
    failed = False
    labels = list(CONFIGS)
    for round_index in range(2):
        for shape in SHAPES if round_index == 0 else SHAPES[::-1]:
            shift = SHAPES.index(shape)
            order = labels[shift:] + labels[:shift]
            if round_index:
                order.reverse()
            for label in order:
                name = f"r{round_index}-{'x'.join(map(str, shape))}-{label}"
                source = args.output / (name + ".metal")
                os.environ["LUISA_TILE_BENCH_DUMP_SOURCE"] = str(source)
                print(name, flush=True)
                row = measure(args, np, shape, CONFIGS[label], references[shape])
                row.update(round=round_index, variant=label, order=order)
                if label != "mps" and row["valid"]:
                    row["source"] = source.name
                    row["source_sha256"] = digest(source)
                failed |= not row["valid"]
                report["results"].append(row)
                save()
                print(row["measurement"]["gpu_throughput_us_p50"] if row["valid"] else row["error"], flush=True)
    after = {str(p): digest(p) for p in artifacts}
    report["metadata"].update(artifacts_after=after, artifacts_unchanged=before == after,
                              finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
    report["passed"] = not failed and before == after
    save()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
