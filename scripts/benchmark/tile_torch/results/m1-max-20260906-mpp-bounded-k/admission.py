"""Record fixed-request admission with the frozen pre-extension binaries.

This is not a speed comparison. A successful request would require separate
complete-output validation before its timing could be used.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--library-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    libraries = args.library_dir.resolve(strict=True)
    artifacts = {binary}
    for directory in (binary.parent, libraries):
        artifacts.update(p.resolve() for p in directory.iterdir()
                         if p.is_file() and p.suffix in (".dylib", ".so"))
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    before = {str(p): digest(p) for p in sorted(artifacts)}
    environment = os.environ.copy()
    environment["DYLD_LIBRARY_PATH"] = str(libraries)
    for key in ("LUISA_TILE_BENCH_METAL_TIMING", "LUISA_TILE_BENCH_DUMP_SOURCE",
                "LUISA_ENABLE_VALIDATION", "MTL_DEBUG_LAYER", "MTL_SHADER_VALIDATION"):
        environment.pop(key, None)
    report = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "purpose": "admission only; no validated baseline execution or speed ratio",
        "loader_environment": {"DYLD_LIBRARY_PATH": str(libraries)},
        "artifacts_sha256": before,
        "results": [],
    }
    with tempfile.TemporaryDirectory(prefix="tile-bounded-k-admission-") as folder:
        for shape in ((128, 128, 61), (1024, 1024, 1537), (4096, 4096, 11008)):
            command = [str(binary), "metal", "gemm", *map(str, shape),
                       "128", "32", "1024", "1", "1", "1",
                       str(Path(folder) / "output.f32"), "group", "1",
                       "mpp-views", "vectorize", "128", "1"]
            process = subprocess.run(command, env=environment, text=True,
                                     capture_output=True, timeout=300)
            report["results"].append({
                "shape": shape, "command": command, "returncode": process.returncode,
                "admitted": process.returncode == 0, "output_validated": False,
                "stdout": process.stdout, "stderr": process.stderr,
            })
            print(shape, "admitted (not validated)" if process.returncode == 0 else "rejected", flush=True)
    report["artifacts_unchanged"] = all(p.is_file() and digest(p) == before[str(p)] for p in artifacts)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return int(not report["artifacts_unchanged"] or any(r["admitted"] for r in report["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
