"""
Agent script: configure and build LuisaCompute with CMake on Windows.
Mirrors the CI workflow in .github/workflows/build-cmake.yml.

Usage:
    python scripts/agent_windows_cmake.py              # configure + build + verify Release
    python scripts/agent_windows_cmake.py --config     # configure only
    python scripts/agent_windows_cmake.py --build      # build only (requires prior --config)
    python scripts/agent_windows_cmake.py --clean      # clean build cache before configure
    python scripts/agent_windows_cmake.py --verify     # verify key targets exist
    python scripts/agent_windows_cmake.py --type Debug # build Debug
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT / "build"

# Default CMake flags matching CI workflow
CMAKE_FLAGS = [
    "-D", "LUISA_COMPUTE_ENABLE_RUST=OFF",
    "-D", "LUISA_COMPUTE_ENABLE_REMOTE=OFF",
    "-D", "LUISA_COMPUTE_ENABLE_CPU=OFF",
]

# Key output files to verify after build
VERIFY_FILES = [
    BUILD_DIR / "lib" / "SPIRV-Tools-opt.lib",
    BUILD_DIR / "lib" / "SPIRV-Tools.lib",
    BUILD_DIR / "bin" / "luisa-ast.dll",
    BUILD_DIR / "bin" / "luisa-core.dll",
]


def run(cmd: list[str], cwd=None, timeout: int = 300, env=None,
        check: bool = True) -> subprocess.CompletedProcess:
    """Run a command, streaming output. Exits on failure if check=True."""
    print(f"[RUN] {' '.join(str(c) for c in cmd)}")
    proc = subprocess.run(cmd, cwd=cwd or ROOT, timeout=timeout,
                          env=env or os.environ, capture_output=False)
    if check and proc.returncode != 0:
        sys.exit(proc.returncode)
    return proc


def run_capture(cmd: list[str], cwd=None, timeout: int = 30) -> str:
    """Run a command, capture stdout, raise on failure."""
    proc = subprocess.run(cmd, cwd=cwd or ROOT, timeout=timeout,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        sys.exit(proc.returncode)
    return proc.stdout.strip()


def find_program(name: str) -> str | None:
    """Find a program in PATH, .deps, or common locations."""
    # 1. PATH
    path = shutil.which(name)
    if path:
        print(f"[FOUND] {name}: {path}")
        return path

    # 2. Bootstrap .deps directory
    deps = ROOT / ".deps"
    if deps.is_dir():
        candidate = deps / f"{name}.exe" if sys.platform == "win32" else deps / name
        if candidate.is_file():
            print(f"[FOUND] {name}: {candidate}")
            return str(candidate)

    # 3. Python scripts dir (for ninja installed via pip)
    if name == "ninja":
        pip_ninja = Path(sys.executable).parent / "ninja.exe"
        if pip_ninja.is_file():
            print(f"[FOUND] {name}: {pip_ninja}")
            return str(pip_ninja)

    print(f"[MISSING] {name}")
    return None


def prepare_msvc_environment() -> dict:
    """Detect and activate MSVC environment using vswhere, like bootstrap.py."""

    def find_msvc(pattern: str) -> list[str]:
        vswhere = find_program("vswhere.exe")
        if not vswhere:
            deps_vswhere = ROOT / ".deps" / "vswhere.exe"
            if deps_vswhere.is_file():
                vswhere = str(deps_vswhere)
            else:
                return []

        result = run_capture([
            vswhere, "-format", "json", "-utf8", "-nologo", "-sort",
            "-products", "*", "-find", pattern, "-latest",
        ])
        data = json.loads(result)
        return [x.replace("\\", "/") for x in data]

    vcvars = find_msvc("**/Auxiliary/Build/vcvars64.bat")
    if not vcvars:
        print("[WARN] Could not find vcvars64.bat. Proceeding without MSVC environment.")
        return os.environ.copy()

    vcvars_bat = vcvars[0]
    print(f"[MSVC] Using: {vcvars_bat}")

    dump_cmd = (
        f'"{vcvars_bat}" && python -c '
        f'"import os, json; print(json.dumps(dict(os.environ)))"'
    )
    result = subprocess.run(
        dump_cmd, shell=True, capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        print(f"[WARN] vcvars failed: {result.stderr}")
        return os.environ.copy()

    env_vars = json.loads(result.stdout.strip())
    env = os.environ.copy()
    env.update(env_vars)
    print("[MSVC] Environment prepared.")
    return env


def configure(env: dict, build_type: str = "Release"):
    """Run CMake configure."""
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    cmake = find_program("cmake")
    if not cmake:
        sys.exit("ERROR: cmake not found. Install it or run from a VS Developer Command Prompt.")

    ninja = find_program("ninja")

    cmd = [
        cmake, "-S", str(ROOT), "-B", str(BUILD_DIR), "-G", "Ninja",
        f"-DCMAKE_BUILD_TYPE={build_type}",
    ]
    if ninja:
        cmd += [f"-DCMAKE_MAKE_PROGRAM={ninja}"]
    cmd += CMAKE_FLAGS

    run(cmd, env=env)


def build(env: dict, jobs: int = None):
    """Run CMake build."""
    if jobs is None:
        jobs = os.cpu_count() or 8

    cmake = find_program("cmake")
    if not cmake:
        sys.exit("ERROR: cmake not found.")

    run([cmake, "--build", str(BUILD_DIR), "-j", str(jobs)], env=env)


def clean():
    """Remove CMake cache to force re-configure."""
    cache = BUILD_DIR / "CMakeCache.txt"
    if cache.is_file():
        print(f"[CLEAN] Removing {cache}")
        cache.unlink()
    cmake_files = BUILD_DIR / "CMakeFiles"
    if cmake_files.is_dir():
        print(f"[CLEAN] Removing {cmake_files}")
        shutil.rmtree(cmake_files, ignore_errors=True)


def verify(env: dict):
    """Verify key build output files exist."""
    print("\n[VERIFY] Checking key build outputs...")
    all_good = True
    for f in VERIFY_FILES:
        if f.is_file():
            size_kb = f.stat().st_size // 1024
            print(f"  [OK] {f.relative_to(ROOT)}  ({size_kb} KB)")
        else:
            print(f"  [MISSING] {f.relative_to(ROOT)}")
            all_good = False

    if all_good:
        print("[VERIFY] All key outputs present.\n")
    else:
        print("[VERIFY] Some outputs missing! Build may have failed.\n")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Configure and build LuisaCompute with CMake (Windows)")
    parser.add_argument("--config", action="store_true", help="Run CMake configure")
    parser.add_argument("--build", action="store_true", help="Run CMake build")
    parser.add_argument("--clean", action="store_true", help="Clean build cache before configure")
    parser.add_argument("--verify", action="store_true", help="Verify key build output files")
    parser.add_argument("--type", default="Release", choices=["Release", "Debug"],
                        help="Build type (default: Release)")
    parser.add_argument("-j", type=int, default=None,
                        help="Parallel jobs (default: cpu_count)")
    args = parser.parse_args()

    # Default: config + build + verify if no flags given
    if not any([args.config, args.build, args.clean, args.verify]):
        args.config = True
        args.build = True
        args.verify = True

    env = prepare_msvc_environment()

    if args.clean:
        clean()

    if args.config:
        configure(env, args.type)

    if args.build:
        build(env, args.j)

    if args.verify:
        verify(env)

    print("[DONE] All steps completed successfully.")


if __name__ == "__main__":
    main()
