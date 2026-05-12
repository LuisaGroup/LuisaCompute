#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
import subprocess
import platform
from pathlib import Path

ROOT_DIR = Path('.')
BIN_DEBUG = ROOT_DIR / "bin" / "debug"

CACHE_PATH = BIN_DEBUG / ".cache"
HLSL_PATH = BIN_DEBUG / "hlsl_output.hlsl"
_DEFAULT_spv_path = BIN_DEBUG / "shader.spv"
DXC_PATH = BIN_DEBUG / "dxc.exe"

def delete_if_exists(path: Path):
    if path.exists():
        path.unlink(missing_ok=True)
        print(f"Deleted: {path}")

def main():
    parser = argparse.ArgumentParser(description="Generate HLSL and compile to SPIR-V.")
    parser.add_argument(
        "--spv-path",
        type=Path,
        default=_DEFAULT_spv_path,
        help=f"Output SPIR-V path (default: {_DEFAULT_spv_path})",
    )
    args = parser.parse_args()

    spv_path = args.spv_path
    spv_path = BIN_DEBUG / spv_path
    # 1. Set env var
    env = os.environ.copy()
    env["LUISA_DUMP_SOURCE"] = "1"

    # 2. Delete existing outputs
    shutil.rmtree(CACHE_PATH, ignore_errors=True)
    delete_if_exists(HLSL_PATH)
    delete_if_exists(spv_path)

    # 3. Run xmake
    print("Running: xmake run test_dsl vk")
    result = subprocess.run(["xmake", "run", "test_dsl", "vk"], cwd=ROOT_DIR, env=env)
    if result.returncode != 0:
        print(f"Error: xmake run failed with code {result.returncode}", file=sys.stderr)
        sys.exit(1)

    # 4. Check HLSL exists
    if not HLSL_PATH.exists():
        print(f"Error: {HLSL_PATH} does not exist after running test_dsl", file=sys.stderr)
        sys.exit(1)
    print(f"Found: {HLSL_PATH}")

    # 5. Compile HLSL to SPIR-V
    cmd = [
        str(DXC_PATH),
        "-spirv",
        "-T", "cs_6_5",
        "-E", "main",
        str(HLSL_PATH),
        "-enable-16bit-types",
        "-HV", "2021",
        "-Fc", str(spv_path),
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT_DIR)
    if result.returncode != 0:
        print(f"Error: dxc failed with code {result.returncode}", file=sys.stderr)
        sys.exit(1)

    # 6. Check SPIR-V exists
    if not spv_path.exists():
        print(f"Error: {spv_path} does not exist after dxc compilation", file=sys.stderr)
        sys.exit(1)
    print(f"Found: {spv_path}")

    # 7. Open in text editor
    print("Opening shader.spv in text editor...")
    if platform.system() == "Windows":
        os.startfile(str(spv_path))
    else:
        editor = os.environ.get("EDITOR", "xdg-open")
        subprocess.Popen([editor, str(spv_path)])

    print("Done.")

if __name__ == "__main__":
    main()
