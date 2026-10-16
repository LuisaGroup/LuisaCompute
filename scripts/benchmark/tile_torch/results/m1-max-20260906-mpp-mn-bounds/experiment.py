#!/usr/bin/env python3
"""ABBA comparison of frozen pre-M/N and extended TIRx MPP compilers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="shuffle-replay", choices=("shuffle-replay", "cooperative-replay", "replay"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    harness = root.parents[1] / "run.py"
    build = Path("/tmp/luisa-tvm-mpp.VaKmzx/luisa-build")
    binaries = build / "bin"
    old = Path("/tmp/luisa-mpp-mn-baseline.gFce8i")
    current = Path("/tmp/luisa-tvm-mpp.VaKmzx/build/lib")
    output = root / args.tag
    output.mkdir(exist_ok=False)
    artifacts = [binaries / name for name in ("benchmark_tile_tirx", "benchmark_tile_system",
                 "libluisa-tile.dylib", "libluisa-tile-bridge-tirx.dylib",
                 "libluisa-benchmark-metal-timing.dylib")]
    artifacts += [old / name for name in ("benchmark_tile_tirx", "libluisa-tile.dylib",
                  "libluisa-tile-bridge-tirx.dylib")]
    for directory in (old, current):
        artifacts += [directory / name for name in ("libtvm_compiler.dylib", "libtvm_runtime.dylib",
                      "libtvm_runtime_metal.dylib", "libtvm_runtime_extra.dylib", "libtvm_ffi.dylib")]
    artifacts = sorted(set(artifacts) | set(binaries.glob("*.dylib")))

    def hashes():
        return {str(path): hashlib.file_digest(path.open("rb"), "sha256").hexdigest() for path in artifacts}

    build_command = ["cmake", "--build", str(build), "--parallel", "8"]
    built = subprocess.run(build_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (output / "benchmark-build.log").write_text(built.stdout)
    if built.returncode:
        print(built.stdout)
        return built.returncode
    evidence = {"build_command": build_command, "build_exit": built.returncode,
                "before": hashes(), "runs": []}
    shapes = ["129x257x61", "1025x1025x1024", "2049x4097x1025", "4097x4097x4096", "1024x1024x1024"]
    blocks = ["128,32,16", "128,32,1024", "128,32,4096"]
    for version, reverse in (("old", False), ("new", False), ("new", True), ("old", True)):
        name = version + ("-reverse" if reverse else "-forward")
        binary = (old if version == "old" else binaries) / "benchmark_tile_tirx"
        libraries = old if version == "old" else current
        loader_path = str(libraries) + os.pathsep + str(binaries)
        environment = dict(os.environ, DYLD_LIBRARY_PATH=loader_path)
        command = [sys.executable, str(harness), "--native", str(binary),
                   "--system-baseline", str(binaries / "benchmark_tile_system"),
                   "--output", str(output / name), "--backends", "metal", "--operations", "gemm",
                   "--execution-scope", "group", "--cooperative-matrix", "--matrix-realization", "mpp-views",
                   "--pipeline-window", "1", "--group-threads", "128", "--copy-batch", "1",
                   "--tune-gemm-blocks", ";".join(reversed(blocks) if reverse else blocks),
                   "--gemm-shapes", ",".join(reversed(shapes) if reverse else shapes),
                   "--max-tuning-candidates", "3", "--tuning-metric", "gpu-control",
                   "--samples", "5", "--sample-ms", "20", "--warmup-ms", "100",
                   "--threads", "8", "--timeout", "300", "--capture-sources",
                   "--metal-device-timing", str(binaries / "libluisa-benchmark-metal-timing.dylib")]
        print(name, flush=True)
        process = subprocess.run(command, env=environment, check=False)
        evidence["runs"].append({"name": name, "command": command,
                                 "loader_environment": {"DYLD_LIBRARY_PATH": loader_path},
                                 "returncode": process.returncode})
        (output / "execution.json").write_text(json.dumps(evidence, indent=2) + "\n")
        if process.returncode:
            break
    evidence["after"] = hashes()
    evidence["artifacts_unchanged"] = evidence["before"] == evidence["after"]
    (output / "execution.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return int(not evidence["artifacts_unchanged"] or len(evidence["runs"]) != 4 or
               any(run["returncode"] for run in evidence["runs"]))


if __name__ == "__main__":
    sys.exit(main())
