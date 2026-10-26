"""Assemble a runnable Python project tree from LuisaCompute sources + built artifacts.

Links `src/py/luisa` (the package) and `src/tests/python` (the test suite) into a single
self-contained tree (default: `bin/py`) by copying:

  * the `luisa` package (all *.py / *.pyi)            -> <out>/luisa/**
  * compiled extension + runtime DLLs + SDK DLLs       -> <out>/luisa/dylibs/**
  * test + helper .py files                            -> <out>/*.py

Pure stdlib. Idempotent (recreates the output tree each run). Never deletes anything
outside the output directory.

Usage:
    python scripts/assemble_py_project.py [--mode debug|release|releasedbg]
                                          [--out DIR] [--src-root DIR] [--no-tests]
"""

from __future__ import annotations

import argparse
import fnmatch
import shutil
import sys
from pathlib import Path

# luisa-*.dll pattern to copy, minus this stale leftover backend (not built in this config).
LOADED_DLL_EXCLUDE = {"luisa-backend-fallback.dll"}

# SDK dlls needed by the dx backend (DXC / DirectStorage).
SDK_DLLS = ["dxcompiler.dll", "dxil.dll", "dstorage.dll", "dstoragecore.dll"]

PACKAGE_GLOBS = ("*.py", "*.pyi")
TEST_GLOBS = ("*.py",)


def _is_pycache(path: Path) -> bool:
    return "__pycache__" in path.parts or path.suffix == ".pyc"


def wipe_tree(target: Path) -> None:
    """Remove the output tree if it exists (targeted dir only)."""
    if target.exists():
        shutil.rmtree(target)


def _iter_matching(root: Path, globs: tuple[str, ...]):
    for entry in sorted(root.iterdir()):
        if _is_pycache(entry):
            continue
        if entry.is_file() and any(fnmatch.fnmatch(entry.name, g) for g in globs):
            yield entry


def copy_package(src_pkg: Path, dst_pkg: Path, manifest: list[str]) -> int:
    """Copy package *.py / *.pyi files. Returns count."""
    n = 0
    for f in _iter_matching(src_pkg, PACKAGE_GLOBS):
        dst = dst_pkg / f.name
        shutil.copy2(f, dst)
        manifest.append(f"package   {f.name}")
        n += 1
    return n


def copy_tests(src_tests: Path, dst_root: Path, manifest: list[str]) -> int:
    """Copy test + helper *.py files to the project root. Returns count."""
    n = 0
    for f in _iter_matching(src_tests, TEST_GLOBS):
        dst = dst_root / f.name
        shutil.copy2(f, dst)
        manifest.append(f"test      {f.name}")
        n += 1
    return n




# Data-file assets referenced by the python tests. ``from_src_tests_dir`` are
# paths relative to <src_root>/src/tests; ``from_src_tests_python_dir`` relative
# to <src_root>/src/tests/python. Copied to the output root so tests that load
# external data (e.g. test-texture.py -> Image2D.from_ldr_image("logo.png")) run
# without a missing-file crash. Extend this list when a new test gains an asset.
ASSET_FILES = {
    "from_src_tests_dir": ["logo.png"],
    "from_src_tests_python_dir": [],
}


def copy_assets(src_tests_parent: Path, src_tests: Path, dst_root: Path,
                manifest: list[str]) -> int:
    """Copy the declared test data-asset files to the project root."""
    n = 0
    for rel in ASSET_FILES["from_src_tests_dir"]:
        src = src_tests_parent / rel
        if src.is_file():
            shutil.copy2(src, dst_root / Path(rel).name)
            manifest.append(f"asset     {Path(rel).name}")
            n += 1
    for rel in ASSET_FILES["from_src_tests_python_dir"]:
        src = src_tests / rel
        if src.is_file():
            shutil.copy2(src, dst_root / Path(rel).name)
            manifest.append(f"asset     {Path(rel).name}")
            n += 1
    return n


def copy_artifacts(bin_dir: Path, dst_dylibs: Path, manifest: list[str]) -> int:
    """Copy lcapi.pyd, luisa_nvrtc.exe, and luisa-*.dll (excl. fallback). Returns count."""
    n = 0
    required = ["lcapi.pyd", "luisa_nvrtc.exe"]
    for name in required:
        src = bin_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Required artifact missing: {src}")
        shutil.copy2(src, dst_dylibs / name)
        manifest.append(f"artifact  {name}")
        n += 1
    for src in sorted(bin_dir.glob("luisa-*.dll")):
        if src.name in LOADED_DLL_EXCLUDE:
            continue
        shutil.copy2(src, dst_dylibs / src.name)
        manifest.append(f"dll       {src.name}")
        n += 1
    return n


def newest_dx_sdk(sdks_root: Path) -> Path:
    """Pick the newest dx_sdk_* directory (lexicographic timestamp suffix wins)."""
    candidates = [
        d
        for d in sdks_root.glob("dx_sdk_*")
        if d.is_dir() and not d.name.startswith("linux_")
    ]
    if not candidates:
        raise FileNotFoundError(f"No dx_sdk_* directory under {sdks_root}")
    return max(candidates, key=lambda d: d.name)


def copy_sdk_dlls(sdk_dir: Path, dst_dylibs: Path, manifest: list[str]) -> int:
    n = 0
    for name in SDK_DLLS:
        src = sdk_dir / name
        if not src.exists():
            raise FileNotFoundError(f"SDK dll missing: {src}")
        shutil.copy2(src, dst_dylibs / name)
        manifest.append(f"sdk       {name}")
        n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assemble a runnable Python project tree from LuisaCompute sources + built artifacts.",
    )
    parser.add_argument(
        "--mode",
        choices=["debug", "release", "releasedbg"],
        default="debug",
        help="Build configuration directory under bin/ to draw artifacts from.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output tree root (default: <src-root>/bin/py).",
    )
    parser.add_argument(
        "--src-root",
        default=None,
        help="Project root (default: parent of this script).",
    )
    parser.add_argument(
        "--no-tests",
        action="store_true",
        help="Skip copying the test/helper .py files.",
    )
    args = parser.parse_args(argv)

    src_root = Path(args.src_root).resolve() if args.src_root else Path(__file__).resolve().parent.parent
    out = Path(args.out).resolve() if args.out else (src_root / "bin" / "py")

    src_pkg = src_root / "src" / "py" / "luisa"
    src_tests = src_root / "src" / "tests" / "python"
    bin_dir = src_root / "bin" / args.mode
    sdks_root = src_root / "SDKs"

    for label, p in (("package source", src_pkg), ("build artifacts", bin_dir), ("SDKs", sdks_root)):
        if not p.is_dir():
            print(f"ERROR: {label} not found: {p}", file=sys.stderr)
            return 2

    dst_pkg = out / "luisa"
    dst_dylibs = dst_pkg / "dylibs"

    # Recreate the output tree (idempotent). Only ever touch `out`.
    wipe_tree(out)
    dst_dylibs.mkdir(parents=True)

    manifest: list[str] = []
    total = 0
    total += copy_package(src_pkg, dst_pkg, manifest)
    if not args.no_tests:
        total += copy_tests(src_tests, out, manifest)
        total += copy_assets(src_tests.parent, src_tests, out, manifest)
    total += copy_artifacts(bin_dir, dst_dylibs, manifest)
    sdk_dir = newest_dx_sdk(sdks_root)
    total += copy_sdk_dlls(sdk_dir, dst_dylibs, manifest)

    print(f"Assembled Python project tree at: {out}")
    print(f"  artifacts from : {bin_dir}")
    print(f"  dx SDK from    : {sdk_dir}")
    print(f"  {total} files copied")
    for line in manifest:
        print(f"    {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
