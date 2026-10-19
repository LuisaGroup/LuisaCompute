#!/usr/bin/env python3
"""Build DirectXShaderCompiler, pack the LuisaCompute DX/VK SDKs, update the
cmake/xmake SDK configs, and publish the packages to GitHub.

Usage:
    python scripts/pack_dxsdk.py <dxc_source_dir> [--date YYYYMMDD]
                                 [--skip-build] [--skip-upload]

Steps:
    1. Build DXC in Release mode:
         - Windows: dxcompiler.dll + dxil.dll  (via build.py)
         - Linux:   libdxcompiler.so          (via build.py --wsl, from Windows)
    2. Stage packages under SDKs/:
         - dx_sdk_<date>/        (dxcompiler.dll, dxil.dll, plus dstorage.dll
                                  and dstoragecore.dll copied from the previous
                                  SDK package)
         - linux_dxc_<date>_x86_64/ (libdxcompiler.so)
    3. Compress with 7-Zip (maximum compression) into:
         - SDKs/dx_sdk_<date>.zip
         - SDKs/linux_dxc_<date>_x86_64.zip
    4. Update scripts/sdks.cmake (URL + SHA256 of LUISA_COMPUTE_DX_SDK) and
       scripts/find_sdk.lua (dx_sdk / vk_sdk zip names).
    5. Upload the zips to https://github.com/LuisaGroup/SDKs/releases/tag/sdk
       with `gh`.
"""

import argparse
import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SDKS_DIR = REPO_ROOT / "SDKs"
GITHUB_REPO = "LuisaGroup/SDKs"
RELEASE_TAG = "sdk"

# Files that are not produced by the DXC build; copied from the previous SDK.
CARRIED_OVER_DLLS = ["dstorage.dll", "dstoragecore.dll"]


def log(msg: str) -> None:
    print(f"[pack_dxsdk] {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"[pack_dxsdk][ERROR] {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def run(cmd, cwd=None, check=True) -> subprocess.CompletedProcess:
    log(f"run: {' '.join(str(c) for c in cmd)}" + (f" (cwd={cwd})" if cwd else ""))
    return subprocess.run(cmd, cwd=cwd, check=check)


def find_7z() -> Path:
    exe = shutil.which("7z") or shutil.which("7z.exe")
    if exe:
        return Path(exe)
    for candidate in (r"C:\Program Files\7-Zip\7z.exe",
                      r"C:\Program Files (x86)\7-Zip\7z.exe"):
        p = Path(candidate)
        if p.exists():
            return p
    fail("7-Zip (7z.exe) not found on PATH or in 'C:\\Program Files\\7-Zip'.")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 1. Build
# ---------------------------------------------------------------------------

def build_windows(dxc_root: Path, jobs: int) -> None:
    log("Building DXC for Windows (Release) ...")
    run([sys.executable, "build.py",
         "--build-type", "Release",
         "--build-dir", "build/Release",
         "--jobs", str(jobs)], cwd=dxc_root)
    for name in ("dxcompiler.dll", "dxil.dll"):
        if not (dxc_root / "build/Release/bin" / name).exists():
            fail(f"Windows build finished but {name} is missing.")


LINUX_BUILD_DIR = "build-wsl"
LINUX_SO_REL = Path(LINUX_BUILD_DIR) / "lib" / "libdxcompiler.so"


def build_linux(dxc_root: Path, jobs: int) -> None:
    # Pass an explicit --build-dir so the artifact location is deterministic:
    # build.py's own WSL default is also "build-wsl".
    if platform.system() == "Windows":
        log("Building DXC for Linux via WSL (Release) ...")
        cmd = [sys.executable, "build.py", "--wsl",
               "--build-type", "Release",
               "--build-dir", LINUX_BUILD_DIR,
               "--jobs", str(jobs)]
        distro = os.environ.get("WSL_DISTRO")
        if distro:
            cmd += ["--wsl-distro", distro]
        run(cmd, cwd=dxc_root)
    else:
        log("Building DXC for Linux (Release) ...")
        run([sys.executable, "build.py",
             "--build-type", "Release",
             "--build-dir", LINUX_BUILD_DIR,
             "--jobs", str(jobs)], cwd=dxc_root)
    so = dxc_root / LINUX_SO_REL
    if not so.exists():
        fail(f"Linux build finished but {so} is missing.")


# ---------------------------------------------------------------------------
# 2. Stage + 3. Zip
# ---------------------------------------------------------------------------

def previous_sdk_dir(new_dir: Path) -> Path | None:
    """Latest existing dx_sdk_* staging dir, excluding the one being created."""
    candidates = sorted(p for p in SDKS_DIR.glob("dx_sdk_2*")
                        if p.is_dir() and p != new_dir)
    return candidates[-1] if candidates else None


def copy_carried_over_dlls(staging: Path) -> None:
    prev = previous_sdk_dir(staging)
    copied = []
    if prev is not None:
        for name in CARRIED_OVER_DLLS:
            src = prev / name
            if src.exists():
                shutil.copy2(src, staging / name)
                copied.append(name)
    missing = [n for n in CARRIED_OVER_DLLS if n not in copied]
    if missing:
        fail(f"Could not carry over {missing} from previous SDK "
             f"(checked {prev}). Build them once or place them manually.")


def make_zip(staging: Path, zip_path: Path, seven_zip: Path) -> None:
    files = sorted(p.name for p in staging.iterdir() if p.is_file())
    if not files:
        fail(f"Staging dir {staging} is empty, nothing to zip.")
    # Files are added from inside the staging dir so they land at the zip
    # root (matching the layout of the existing SDK zips).
    run([str(seven_zip), "a", "-tzip", "-mx=9", "-mfb=258", "-mpass=15",
         str(zip_path), *files], cwd=staging)
    if not zip_path.exists():
        fail(f"7z reported success but {zip_path} does not exist.")
    log(f"Created {zip_path.name} ({zip_path.stat().st_size} bytes, "
        f"sha256={sha256_of(zip_path)})")


# ---------------------------------------------------------------------------
# 4. Update configs
# ---------------------------------------------------------------------------

def update_sdks_cmake(zip_name: str, sha256: str) -> None:
    sdks_file = SCRIPT_DIR / "sdks.cmake"
    if not sdks_file.exists():
        fail(f"{sdks_file} not found.")
    content = sdks_file.read_text(encoding="utf-8")
    url = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}/{zip_name}"
    new_content, count = re.subn(
        r'(set\s*\(\s*LUISA_COMPUTE_DX_SDK\s+")[^"]+("\s+")[^"]+("\s*\))',
        rf"\g<1>{url}\g<2>{sha256}\g<3>",
        content, count=1)
    if count == 0:
        fail("Could not find LUISA_COMPUTE_DX_SDK in scripts/sdks.cmake.")
    sdks_file.write_text(new_content, encoding="utf-8")
    log(f"Updated scripts/sdks.cmake: {zip_name} {sha256}")


def update_find_sdk_lua(dx_zip: str, linux_zip: str) -> None:
    lua_file = SCRIPT_DIR / "find_sdk.lua"
    if not lua_file.exists():
        fail(f"{lua_file} not found.")
    content = lua_file.read_text(encoding="utf-8")
    new_content, n1 = re.subn(
        r"(dx_sdk\s*=\s*\{\s*\n\s*name\s*=\s*')[^']+(')",
        rf"\g<1>{dx_zip}\g<2>", content, count=1)
    new_content, n2 = re.subn(
        r"(vk_sdk\s*=\s*\{\s*\n\s*name\s*=\s*')[^']+(')",
        rf"\g<1>{linux_zip}\g<2>", new_content, count=1)
    if n1 == 0 or n2 == 0:
        fail("Could not find dx_sdk/vk_sdk entries in scripts/find_sdk.lua.")
    lua_file.write_text(new_content, encoding="utf-8")
    log(f"Updated scripts/find_sdk.lua: dx_sdk={dx_zip}, vk_sdk={linux_zip}")


# ---------------------------------------------------------------------------
# 5. Publish
# ---------------------------------------------------------------------------

def publish(zips: list[Path]) -> None:
    if not shutil.which("gh"):
        fail("'gh' (GitHub CLI) not found on PATH.")
    view = subprocess.run(["gh", "release", "view", RELEASE_TAG,
                           "--repo", GITHUB_REPO], capture_output=True)
    if view.returncode != 0:
        log(f"Release '{RELEASE_TAG}' does not exist yet, creating it.")
        run(["gh", "release", "create", RELEASE_TAG,
             "--repo", GITHUB_REPO,
             "--title", RELEASE_TAG,
             "--notes", "LuisaCompute SDK packages (DXC/dxil/dstorage)."])
    # --clobber makes same-day re-runs (same date-stamped asset names)
    # replace the previously uploaded packages instead of failing.
    run(["gh", "release", "upload", RELEASE_TAG,
         *[str(z) for z in zips], "--repo", GITHUB_REPO, "--clobber"])
    log(f"Uploaded {len(zips)} package(s) to "
        f"https://github.com/{GITHUB_REPO}/releases/tag/{RELEASE_TAG}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build DXC, pack LuisaCompute SDKs, update configs, publish.")
    parser.add_argument("dxc_root", type=Path,
                        help="Absolute path to the DirectXShaderCompiler source tree.")
    parser.add_argument("--date", default=None, metavar="YYYYMMDD",
                        help="Package date stamp (default: today).")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2),
                        help="Parallel build jobs (default: cpu count - 2).")
    parser.add_argument("--skip-build", action="store_true",
                        help="Reuse existing build outputs under the DXC build dirs.")
    parser.add_argument("--skip-upload", action="store_true",
                        help="Do everything except the gh release upload.")
    args = parser.parse_args()

    dxc_root: Path = args.dxc_root.resolve()
    if not dxc_root.is_dir():
        fail(f"DXC source directory not found: {dxc_root}")
    if not (dxc_root / "build.py").exists():
        fail(f"{dxc_root} does not look like the DXC source tree (no build.py).")

    date = args.date or datetime.now().strftime("%Y%m%d")
    if not re.fullmatch(r"\d{8}", date):
        fail(f"--date must be YYYYMMDD, got '{date}'.")

    is_windows = platform.system() == "Windows"
    dx_name = f"dx_sdk_{date}"
    linux_name = f"linux_dxc_{date}_x86_64"
    dx_staging = SDKS_DIR / dx_name
    linux_staging = SDKS_DIR / linux_name
    dx_zip = SDKS_DIR / f"{dx_name}.zip"
    linux_zip = SDKS_DIR / f"{linux_name}.zip"

    SDKS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Build
    if not args.skip_build:
        if is_windows:
            build_windows(dxc_root, args.jobs)
        build_linux(dxc_root, args.jobs)

    # 2. Stage
    if is_windows:
        dx_bin = dxc_root / "build/Release/bin"
        dx_staging.mkdir(parents=True, exist_ok=True)
        for name in ("dxcompiler.dll", "dxil.dll"):
            src = dx_bin / name
            if not src.exists():
                fail(f"{src} not found (build DXC first or drop --skip-build).")
            shutil.copy2(src, dx_staging / name)
        copy_carried_over_dlls(dx_staging)

    so = dxc_root / LINUX_SO_REL
    if not so.exists():
        fail(f"{so} not found (build DXC first or drop --skip-build).")
    linux_staging.mkdir(parents=True, exist_ok=True)
    shutil.copy2(so, linux_staging / "libdxcompiler.so")

    # 3. Zip (maximum compression)
    seven_zip = find_7z()
    zips = []
    if is_windows:
        make_zip(dx_staging, dx_zip, seven_zip)
        zips.append(dx_zip)
    make_zip(linux_staging, linux_zip, seven_zip)
    zips.append(linux_zip)

    # 4. Update configs
    if is_windows:
        update_sdks_cmake(dx_zip.name, sha256_of(dx_zip))
    update_find_sdk_lua(dx_zip.name, linux_zip.name)

    # 5. Publish
    if not args.skip_upload:
        publish(zips)
    else:
        log("--skip-upload given; packages left in SDKs/.")

    log("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
