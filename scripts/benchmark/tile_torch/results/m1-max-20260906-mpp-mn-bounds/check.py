#!/usr/bin/env python3
"""Archive a fully built native Metal matrix test, without performance claims."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--filter", default="tile_matrix_mpp_bounded_mn_views")
    parser.add_argument("--all", action="store_true", help="run the complete matrix suite")
    parser.add_argument("--backend", choices=("metal", "cpu"), default="metal")
    parser.add_argument("--build", type=Path, default=Path("/tmp/luisa-tvm-mpp.VaKmzx/luisa-build"))
    parser.add_argument("--libraries", type=Path, default=Path("/tmp/luisa-tvm-mpp.VaKmzx/build/lib"))
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9-]+", args.tag):
        parser.error("tag must be a new lowercase hyphenated name")
    root = Path(__file__).resolve().parent
    output = root / args.tag
    output.mkdir(exist_ok=False)
    build = args.build.resolve()
    libraries = args.libraries.resolve()
    full_build = ["cmake", "--build", str(build), "--parallel", "8"]
    command = [str(build / "bin/test_tile_tirx_matrix"), args.backend]
    if not args.all:
        command.append(args.filter)
    built = subprocess.run(full_build, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (output / "build.log").write_text(built.stdout)
    if built.returncode:
        print(built.stdout)
        return built.returncode
    artifacts = [build / "bin/test_tile_tirx_matrix", build / "bin/libluisa-tile.dylib",
                 build / "bin/libluisa-tile-bridge-tirx.dylib"]
    artifacts += [libraries / name for name in ("libtvm_compiler.dylib", "libtvm_runtime.dylib",
                  "libtvm_runtime_metal.dylib", "libtvm_runtime_extra.dylib", "libtvm_ffi.dylib")]
    before = {str(path): digest(path) for path in artifacts}
    env = dict(os.environ, DYLD_LIBRARY_PATH=str(libraries))
    start = time.time()
    tested = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
    clean = re.sub(r"\x1b\[[0-9;]*m", "", tested.stdout)
    (output / "test.log").write_text(clean)
    after = {str(path): digest(path) for path in artifacts}
    assertions = re.search(r"all tests passed \((\d+) asserts in (\d+) tests\)", clean)
    assertion_count = int(assertions.group(1)) if assertions else 0
    receipt = {"build_command": full_build, "build_exit": built.returncode,
               "test_command": command, "test_exit": tested.returncode,
               "started_at_unix": start, "finished_at_unix": time.time(),
               "before": before, "after": after, "artifacts_unchanged": before == after,
               "passed_assertions": assertion_count,
               "passed": tested.returncode == 0 and before == after and assertion_count > 0,
               "measurement_scope": "correctness test only; no performance measurement"}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"output": str(output), "test_exit": tested.returncode,
                      "passed": receipt["passed"], "passed_assertions": assertion_count,
                      "artifacts_unchanged": before == after}))
    print("\n".join(clean.splitlines()[-8:]))
    return tested.returncode or (0 if receipt["passed"] else 1)


if __name__ == "__main__":
    sys.exit(main())
