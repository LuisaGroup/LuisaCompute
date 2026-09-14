#!/usr/bin/env python3
"""Archive full-build gates and nonzero native correctness receipts."""
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
    parser.add_argument("--tvm-build", type=Path, default=Path("/tmp/luisa-tvm-mpp.VaKmzx/build"))
    parser.add_argument("--libraries", type=Path)
    parser.add_argument("--tests", nargs="+", default=["test_tile_tirx_matrix"])
    parser.add_argument("--backends", nargs="+", default=["metal", "cpu"])
    parser.add_argument("--filter", help="Exact native test name (the bundled matcher does not implement '*')")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9-]+", args.tag):
        parser.error("use a new lowercase hyphenated tag")
    if args.filter and any(character in args.filter for character in "*?\\"):
        parser.error("use an exact native test name or omit --filter")
    output = Path(__file__).resolve().parent / args.tag
    output.mkdir(exist_ok=False)
    build, tvm = args.build.resolve(), args.tvm_build.resolve()
    libraries = args.libraries.resolve() if args.libraries else tvm / "lib"
    report = {"builds": [], "tests": [], "scope": "correctness only"}
    receipt = output / "receipt.json"
    for tree in (tvm, build):
        command = ["cmake", "--build", str(tree), "--parallel", "8"]
        built = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log = f"build-{tree.name}.log"
        (output / log).write_text(built.stdout)
        report["builds"].append(dict(command=command, exit_code=built.returncode, log=log))
        receipt.write_text(json.dumps(report, indent=2) + "\n")
        if built.returncode:
            print(built.stdout, flush=True)
            return 1
    paths = [build / "bin" / name for name in args.tests]
    paths += [build / "bin" / name for name in ("libluisa-tile.dylib", "libluisa-tile-bridge-tirx.dylib")]
    paths += [libraries / name for name in ("libtvm_compiler.dylib", "libtvm_runtime.dylib", "libtvm_runtime_extra.dylib", "libtvm_runtime_metal.dylib", "libtvm_ffi.dylib")]
    report["artifacts_before"] = {str(path): digest(path) for path in paths}
    env = dict(os.environ, DYLD_LIBRARY_PATH=f"{build / 'bin'}:{libraries}")
    report["loader"] = env["DYLD_LIBRARY_PATH"]
    for name in args.tests:
        for backend in args.backends:
            command = [str(build / "bin" / name), backend]
            if args.filter:
                command.append(args.filter)
            start = time.time()
            try:
                tested = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
                clean = re.sub(r"\x1b\[[0-9;]*m", "", tested.stdout)
                code = tested.returncode
            except subprocess.TimeoutExpired as error:
                clean = error.stdout or b""
                clean = clean.decode(errors="replace") if isinstance(clean, bytes) else clean
                clean += "\nTIMEOUT after 600 seconds\n"
                code = 124
            log = f"{name}-{backend}.log"
            (output / log).write_text(clean)
            match = re.search(r"all tests passed \((\d+) asserts in (\d+) tests\)", clean)
            assertions = int(match.group(1)) if match else 0
            row = dict(command=command, exit_code=code, started_at_unix=start, finished_at_unix=time.time(), log=log,
                       passed_assertions=assertions, passed_tests=int(match.group(2)) if match else 0,
                       passed=code == 0 and assertions > 0)
            report["tests"].append(row)
            receipt.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    report["artifacts_after"] = {str(path): digest(path) for path in paths}
    report["artifacts_unchanged"] = report["artifacts_before"] == report["artifacts_after"]
    report["passed"] = report["artifacts_unchanged"] and all(row["passed"] for row in report["tests"])
    receipt.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
