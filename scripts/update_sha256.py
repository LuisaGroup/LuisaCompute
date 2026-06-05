#!/usr/bin/env python3
"""Update DX SDK SHA256 and zip name across all relevant files.

Usage:
    python scripts/update_sha256.py [zip_name]

If zip_name is given, it must match a file under SDKs/ (e.g. dx_sdk_20260511.zip).
If omitted, the zip name is parsed from LUISA_COMPUTE_DX_SDK_DOWNLOAD_URL in
src/backends/dx/CMakeLists.txt (existing behaviour).

Files updated:
    1. src/backends/dx/CMakeLists.txt          — URL + SHA256
    2. scripts/download_sdks.cmake             — URL + SHA256
    3. scripts/find_sdk.lua                    — dx_sdk name
"""

import hashlib
import re
import sys
from pathlib import Path


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    # ── 1. Determine zip name ──────────────────────────────────────────
    if len(sys.argv) > 1:
        zip_name = sys.argv[1]
        if not zip_name.endswith(".zip"):
            print(f"[ERROR] zip_name must end with .zip, got: '{zip_name}'")
            return 1
        zip_path = repo_root / "SDKs" / zip_name
        if not zip_path.exists():
            print(f"[ERROR] Local SDK file not found: {zip_path}")
            return 1
        print(f"[INFO] Using zip name from command line: {zip_name}")
    else:
        # Fall back to parsing from CMakeLists.txt
        cmake_file = repo_root / "src" / "backends" / "dx" / "CMakeLists.txt"
        if not cmake_file.exists():
            print(f"[ERROR] CMake file not found: {cmake_file}")
            return 1
        content = cmake_file.read_text(encoding="utf-8")
        url_match = re.search(
            r'set\s*\(\s*LUISA_COMPUTE_DX_SDK_DOWNLOAD_URL\s+"([^"]+)"\s*\)',
            content,
        )
        if not url_match:
            print("[ERROR] Could not parse LUISA_COMPUTE_DX_SDK_DOWNLOAD_URL from CMakeLists.txt.")
            return 1
        url = url_match.group(1)
        zip_name = url.split("/")[-1]
        if not zip_name.endswith(".zip"):
            print(f"[ERROR] Extracted filename does not look like a zip: '{zip_name}'")
            return 1
        print(f"[INFO] Detected SDK zip from URL: {zip_name}")
        zip_path = repo_root / "SDKs" / zip_name

    if not zip_path.exists():
        print(f"[ERROR] Local SDK file not found: {zip_path}")
        return 1

    sha256 = compute_sha256(zip_path)
    print(f"[INFO] SHA256 of {zip_path.relative_to(repo_root)}: {sha256}")

    # ── 2. Update src/backends/dx/CMakeLists.txt ───────────────────────
    cmake_file = repo_root / "src" / "backends" / "dx" / "CMakeLists.txt"
    if cmake_file.exists():
        content = cmake_file.read_text(encoding="utf-8")
        new_url = f"https://github.com/LuisaGroup/SDKs/releases/download/sdk/{zip_name}"

        # Update URL
        new_content, url_count = re.subn(
            r'(set\s*\(\s*LUISA_COMPUTE_DX_SDK_DOWNLOAD_URL\s+")[^"]+("\s*\))',
            rf'\g<1>{new_url}\g<2>',
            content,
            count=1,
        )
        if url_count == 0:
            print("[ERROR] Could not find LUISA_COMPUTE_DX_SDK_DOWNLOAD_URL in CMakeLists.txt.")
            return 1

        # Update SHA256
        new_content, sha_count = re.subn(
            r'(set\s*\(\s*LUISA_COMPUTE_DX_SDK_SHA256\s+")([^"]*)("\s*\))',
            lambda m: f'{m.group(1)}{sha256}{m.group(3)}',
            new_content,
            count=1,
        )
        if sha_count == 0:
            print("[ERROR] Could not find LUISA_COMPUTE_DX_SDK_SHA256 in CMakeLists.txt.")
            return 1

        if new_content != content:
            cmake_file.write_text(new_content, encoding="utf-8")
            print(f"[INFO] Updated CMakeLists.txt: URL -> {new_url}, SHA256 -> {sha256}")
        else:
            print("[INFO] CMakeLists.txt already up-to-date.")
    else:
        print(f"[WARN] CMake file not found, skipping: {cmake_file}")

    # ── 3. Update scripts/download_sdks.cmake ──────────────────────────
    dl_file = repo_root / "scripts" / "download_sdks.cmake"
    if dl_file.exists():
        content = dl_file.read_text(encoding="utf-8")
        new_url = f"https://github.com/LuisaGroup/SDKs/releases/download/sdk/{zip_name}"

        # Replace the URL inside the dx block
        new_content = re.sub(
            r'("https://github\.com/LuisaGroup/SDKs/releases/download/sdk/dx_sdk_[^"]+\.zip")',
            f'"{new_url}"',
            content,
            count=1,
        )

        # Replace the SHA256 inside the dx block
        new_content, sha_count = re.subn(
            r'(download_sdk\(\$\{sdk\}\s*\n\s*"[^"]*"\s*\n\s*)"[0-9a-f]{64}"',
            lambda m: f'{m.group(1)}"{sha256}"',
            new_content,
            count=1,
        )

        if sha_count == 0:
            print("[WARN] Could not find SHA256 in download_sdks.cmake.")

        if new_content != content:
            dl_file.write_text(new_content, encoding="utf-8")
            print(f"[INFO] Updated download_sdks.cmake: URL -> {new_url}")
        else:
            print("[INFO] download_sdks.cmake already up-to-date.")
    else:
        print(f"[WARN] download_sdks.cmake not found, skipping: {dl_file}")

    # ── 4. Update scripts/find_sdk.lua ─────────────────────────────────
    lua_file = repo_root / "scripts" / "find_sdk.lua"
    if lua_file.exists():
        content = lua_file.read_text(encoding="utf-8")
        new_content, lua_count = re.subn(
            r"(dx_sdk\s*=\s*\{\s*\n\s*name\s*=\s*')dx_sdk_[^']+\.zip(')",
            rf"\g<1>{zip_name}\g<2>",
            content,
            count=1,
        )

        if lua_count == 0:
            print("[WARN] Could not find dx_sdk name in find_sdk.lua.")

        if new_content != content:
            lua_file.write_text(new_content, encoding="utf-8")
            print(f"[INFO] Updated find_sdk.lua: dx_sdk name -> '{zip_name}'")
        else:
            print("[INFO] find_sdk.lua already up-to-date.")
    else:
        print(f"[WARN] find_sdk.lua not found, skipping: {lua_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
