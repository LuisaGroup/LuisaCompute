#!/usr/bin/env python3
"""Update LUISA_COMPUTE_DX_SDK_SHA256 in src/backends/dx/CMakeLists.txt from local SDKs/<zip>."""

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
    cmake_file = repo_root / "src" / "backends" / "dx" / "CMakeLists.txt"

    if not cmake_file.exists():
        print(f"[ERROR] CMake file not found: {cmake_file}")
        return 1

    content = cmake_file.read_text(encoding="utf-8")

    # 1. Parse LUISA_COMPUTE_DX_SDK_DOWNLOAD_URL
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

    # 2. Compute SHA256 of local SDKs/<zip_name>
    zip_path = repo_root / "SDKs" / zip_name
    if not zip_path.exists():
        print(f"[ERROR] Local SDK file not found: {zip_path}")
        return 1

    sha256 = compute_sha256(zip_path)
    print(f"[INFO] SHA256 of {zip_path.relative_to(repo_root)}: {sha256}")

    # 3. Rewrite LUISA_COMPUTE_DX_SDK_SHA256 default value
    new_content, count = re.subn(
        r'(set\s*\(\s*LUISA_COMPUTE_DX_SDK_SHA256\s+")([^"]*)("\s*\))',
        lambda m: f'{m.group(1)}{sha256}{m.group(3)}',
        content,
        count=1,
    )

    if count == 0:
        print("[ERROR] Could not find LUISA_COMPUTE_DX_SDK_SHA256 in CMakeLists.txt.")
        return 1

    if new_content == content:
        print("[INFO] SHA256 value is already up-to-date. No changes written.")
        return 0

    cmake_file.write_text(new_content, encoding="utf-8")
    print(f"[INFO] Updated LUISA_COMPUTE_DX_SDK_SHA256 in {cmake_file.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
