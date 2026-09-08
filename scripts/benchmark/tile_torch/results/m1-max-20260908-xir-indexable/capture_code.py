#!/usr/bin/env python3
"""Capture one verified native object separately from uninstrumented timings."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from compare_llm import check_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LUISA_SIMD_", "LUISA_TILE_", "DYLD_"))}
    env.update(LUISA_SIMD_WORKER_COUNT="8", LUISA_SIMD_WARP_WIDTH="8",
               LUISA_SIMD_DUMP_ASSEMBLY_DIR=str(args.output / "object"),
               LUISA_TILE_BENCH_DUMP_SOURCE=str(args.output / "kernel.ll"),
               DYLD_PRINT_LIBRARIES="1")
    command = [str(args.binary.resolve()), "llm", "rmsnorm", "64,256", "1", "1", "2", "5", "10", str(args.output / "output.f32")]
    with (args.output / "measurement.json").open("w") as stdout, (args.output / "capture.log").open("w") as stderr:
        subprocess.run(command, env=env, stdout=stdout, stderr=stderr, timeout=60, check=True)
    check_metadata(json.loads((args.output / "measurement.json").read_text()), "cpu", "rmsnorm", (64, 256), (1, 1), 2)
    objects = list((args.output / "object").glob("*.o"))
    if len(objects) != 1:
        raise ValueError("expected one actual ORC object")
    with (args.output / "object.asm").open("w") as stdout:
        subprocess.run(["/opt/homebrew/opt/llvm/bin/llvm-objdump", "--disassemble", "--demangle", str(objects[0])], stdout=stdout, check=True)
    print("PASS: complete output checked; code inspection only, no performance ranking")


if __name__ == "__main__":
    main()
