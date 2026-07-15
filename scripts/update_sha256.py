#!/usr/bin/env python3
"""Update DX SDK SHA256 and zip name across all relevant files.

Usage:
    python scripts/update_sha256.py [zip_name]

If zip_name is given, it must match a file under SDKs/ (e.g. dx_sdk_20260511.zip).
If omitted, the zip name is parsed from LUISA_COMPUTE_DX_SDK in
scripts/sdks.cmake (existing behaviour).

Files updated:
    1. scripts/sdks.cmake                      — URL + SHA256
    2. scripts/find_sdk.lua                    — dx_sdk name
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
        # Fall back to parsing from scripts/sdks.cmake
        sdks_file = repo_root / "scripts" / "sdks.cmake"
        if not sdks_file.exists():
            print(f"[ERROR] SDK metadata file not found: {sdks_file}")
            return 1
        content = sdks_file.read_text(encoding="utf-8")
        url_match = re.search(
            r'set\s*\(\s*LUISA_COMPUTE_DX_SDK\s+"([^"]+)"\s+"[^"]+"\s*\)',
            content,
        )
        if not url_match:
            print("[ERROR] Could not parse LUISA_COMPUTE_DX_SDK from scripts/sdks.cmake.")
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

    # ── 2. Update scripts/sdks.cmake ───────────────────────────────────
    sdks_file = repo_root / "scripts" / "sdks.cmake"
    if sdks_file.exists():
        content = sdks_file.read_text(encoding="utf-8")
        new_url = f"https://github.com/LuisaGroup/SDKs/releases/download/sdk/{zip_name}"

        new_content, count = re.subn(
            r'(set\s*\(\s*LUISA_COMPUTE_DX_SDK\s+")[^"]+("\s+")[^"]+("\s*\))',
            rf'\g<1>{new_url}\g<2>{sha256}\g<3>',
            content,
            count=1,
        )
        if count == 0:
            print("[ERROR] Could not find LUISA_COMPUTE_DX_SDK in scripts/sdks.cmake.")
            return 1

        if new_content != content:
            sdks_file.write_text(new_content, encoding="utf-8")
            print(f"[INFO] Updated scripts/sdks.cmake: URL -> {new_url}, SHA256 -> {sha256}")
        else:
            print("[INFO] scripts/sdks.cmake already up-to-date.")
    else:
        print(f"[WARN] SDK metadata file not found, skipping: {sdks_file}")

    # ── 3. Update scripts/find_sdk.lua ─────────────────────────────────
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
