#!/usr/bin/env python3
"""Full-build gate and archived correctness receipts for the fusion change."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--build", type=Path, default=Path("/tmp/luisa-tvm-mpp.VaKmzx/luisa-build"))
    parser.add_argument("--libraries", type=Path, default=Path("/tmp/luisa-tvm-mpp.VaKmzx/build/lib"))
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9-]+", args.tag):
        parser.error("use a new lowercase hyphenated tag")
    output = Path(__file__).resolve().parent / args.tag
    output.mkdir(exist_ok=False)
    build, libraries = args.build.resolve(), args.libraries.resolve()
    command = ["cmake", "--build", str(build), "--parallel", "8"]
    built = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (output / "build.log").write_text(built.stdout)
    report = {"build_command": command, "build_exit": built.returncode, "tests": [],
              "measurement_scope": "correctness only; no timing comparison"}
    receipt = output / "receipt.json"
    receipt.write_text(json.dumps(report, indent=2) + "\n")
    if built.returncode:
        print(built.stdout)
        return 1
    names = ["test_tile_tirx_execution", "test_tile_tirx_matrix", "test_tile_tirx_poc",
             "test_tile_tirx_poc_neural", "test_tile_tirx_poc_algorithms"]
    artifacts = [build / "bin" / name for name in names]
    artifacts += [build / "bin" / name for name in ("libluisa-tile.dylib", "libluisa-tile-bridge-tirx.dylib")]
    artifacts += [libraries / name for name in ("libtvm_compiler.dylib", "libtvm_runtime.dylib",
                  "libtvm_runtime_metal.dylib", "libtvm_runtime_extra.dylib", "libtvm_ffi.dylib")]
    before = {str(path): digest(path) for path in artifacts}
    report["artifacts_before"] = before
    env = dict(os.environ, DYLD_LIBRARY_PATH=str(libraries))
    for name in names:
        for backend in ("metal", "cpu"):
            command = [str(build / "bin" / name), backend]
            start = time.time()
            try:
                tested = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, timeout=600)
                clean = re.sub(r"\x1b\[[0-9;]*m", "", tested.stdout)
                code = tested.returncode
            except subprocess.TimeoutExpired as error:
                clean = (error.stdout or b"")
                clean = clean.decode(errors="replace") if isinstance(clean, bytes) else clean
                clean += "\nTIMEOUT after 600 seconds\n"
                code = 124
            (output / f"{name}-{backend}.log").write_text(clean)
            match = re.search(r"all tests passed \((\d+) asserts in (\d+) tests\)", clean)
            assertions = int(match.group(1)) if match else 0
            result = dict(command=command, exit_code=code, started_at_unix=start,
                          finished_at_unix=time.time(), passed_assertions=assertions,
                          passed_tests=int(match.group(2)) if match else 0,
                          passed=code == 0 and assertions > 0)
            report["tests"].append(result)
            receipt.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(result), flush=True)
    report["artifacts_after"] = {str(path): digest(path) for path in artifacts}
    report["artifacts_unchanged"] = before == report["artifacts_after"]
    report["passed"] = report["artifacts_unchanged"] and all(row["passed"] for row in report["tests"])
    receipt.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
