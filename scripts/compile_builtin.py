#!/usr/bin/env python3
"""Compile builtin kernels to .dxil (DX) or .spv (VK) via lc_compile_builtin.

Builds the lc_compile_builtin target in release mode, then runs it with the
specified backend to produce AOT-compiled shader bytecode.

Usage:
    python scripts/compile_builtin.py dx output.dxil
    python scripts/compile_builtin.py vk output.spv
    python scripts/compile_builtin.py dx output.dxil --no-build
    python scripts/compile_builtin.py vk output.spv --name my_kernel
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
TARGET = "lc_compile_builtin"


def run(cmd, **kwargs):
    """Run a command, printing it first."""
    print(f"$ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, **kwargs)


def build_release():
    """Configure release mode and build lc_compile_builtin."""
    result = run(
        ["xmake", "f", "-p", "windows", "-a", "x64", "-m", "release", "-c", "-y"],
        cwd=ROOT_DIR,
    )
    if result.returncode != 0:
        print("Error: xmake config (release) failed", file=sys.stderr)
        sys.exit(1)

    result = run(["xmake", "build", TARGET], cwd=ROOT_DIR)
    if result.returncode != 0:
        print(f"Error: xmake build {TARGET} failed", file=sys.stderr)
        sys.exit(1)


def compile_builtin(backend: str, dest: str, kernel_name: str | None = None):
    """Run lc_compile_builtin to compile a kernel to .dxil or .spv."""
    cmd = ["xmake", "run", TARGET, backend, dest]
    if kernel_name:
        cmd.append(kernel_name)
    result = run(cmd, cwd=ROOT_DIR)
    if result.returncode != 0:
        print(f"Error: lc_compile_builtin failed (backend={backend})", file=sys.stderr)
        sys.exit(1)

    dest_path = Path(dest)
    if not dest_path.is_absolute():
        dest_path = ROOT_DIR / dest_path
    if dest_path.exists():
        print(f"OK: {dest_path} ({dest_path.stat().st_size} bytes)")
    else:
        print(f"Error: output file not found: {dest_path}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Compile builtin kernels to .dxil or .spv via lc_compile_builtin."
    )
    parser.add_argument(
        "backend",
        choices=["dx", "vk"],
        help="Backend: 'dx' for DXIL, 'vk' for SPIR-V",
    )
    parser.add_argument(
        "destination",
        help="Output file path (e.g. output.dxil or output.spv)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional shader name (default: test_builtin)",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Skip the build step (assume target is already built in release)",
    )
    args = parser.parse_args()

    if not args.no_build:
        build_release()

    compile_builtin(args.backend, args.destination, args.name)


if __name__ == "__main__":
    main()
