#!/usr/bin/env python3
"""Run the predeclared two-order diagnostic with prebuilt artifacts only."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    root = Path(__file__).resolve().parent
    harness = root.parents[1] / "run.py"
    build = Path("/tmp/luisa-tvm-mpp.VaKmzx")
    binaries = build / "luisa-build/bin"
    libraries = build / "build/lib"
    artifacts = [libraries / name for name in (
        "libtvm_compiler.dylib", "libtvm_runtime.dylib",
        "libtvm_runtime_metal.dylib", "libtvm_runtime_extra.dylib",
        "libtvm_ffi.dylib")]

    def hashes() -> dict[str, str]:
        return {str(path): hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
                for path in artifacts}

    evidence = {"compiler_artifacts_before": hashes(), "runs": []}
    environment = os.environ.copy()
    environment["DYLD_LIBRARY_PATH"] = str(libraries)
    shapes = ["1024x1024x1537", "4096x4096x4096", "4096x4096x11008"]
    blocks = ["128,32,128", "128,32,512", "128,32,1024", "128,32,4096"]
    for reverse in (False, True):
        directory = root / ("reverse" if reverse else "forward")
        command = [sys.executable, str(harness), "--native", str(binaries / "benchmark_tile_tirx"),
                   "--system-baseline", str(binaries / "benchmark_tile_system"),
                   "--output", str(directory), "--backends", "metal", "--operations", "gemm",
                   "--execution-scope", "group", "--cooperative-matrix", "--matrix-realization", "mpp-views",
                   "--pipeline-window", "1", "--group-threads", "128", "--copy-batch", "1",
                   "--tune-gemm-blocks", ";".join(reversed(blocks) if reverse else blocks),
                   "--gemm-shapes", ",".join(reversed(shapes) if reverse else shapes),
                   "--max-tuning-candidates", "4", "--tuning-metric", "gpu-control",
                   "--samples", "5", "--sample-ms", "20", "--warmup-ms", "100",
                   "--threads", "8", "--timeout", "300", "--capture-sources",
                   "--metal-device-timing", str(binaries / "libluisa-benchmark-metal-timing.dylib")]
        code = subprocess.run(command, env=environment, check=False).returncode
        evidence["runs"].append({"command": command, "returncode": code})
        (root / "execution.json").write_text(json.dumps(evidence, indent=2) + "\n")
    evidence["compiler_artifacts_after"] = hashes()
    evidence["compiler_artifacts_unchanged"] = (
        evidence["compiler_artifacts_before"] == evidence["compiler_artifacts_after"])
    (root / "execution.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return 0 if evidence["compiler_artifacts_unchanged"] and all(
        run["returncode"] == 0 for run in evidence["runs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
