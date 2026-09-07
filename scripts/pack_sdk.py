#!/usr/bin/env python3
"""Pack an SDK directory into a .zip archive.

The archive contains only the bare files placed directly in the source
directory (no subdirectory, no recursion). 7-Zip is preferred when available
for ultra compression; otherwise Python's builtin zipfile module is used.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def find_7zip() -> str | None:
    """Locate a 7-Zip executable on the current platform.

    Searches PATH first, then falls back to common installation directories
    on Windows.
    """
    # Names used on different platforms / distributions.
    names = ["7z.exe", "7za.exe", "7zz.exe", "7zzs.exe", "7z", "7za", "7zz", "7zzs"]
    for name in names:
        path = shutil.which(name)
        if path:
            return path

    # Common Windows installation paths when 7-Zip is not on PATH.
    if sys.platform == "win32":
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        candidates = [
            os.path.join(program_files, r"7-Zip\7z.exe"),
            os.path.join(program_files_x86, r"7-Zip\7z.exe"),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate

    return None


def collect_files(source_dir: Path, pattern: str) -> list[Path]:
    """Return the list of top-level files to pack, sorted for stability."""
    if pattern == "*":
        files = [p for p in source_dir.iterdir() if p.is_file()]
    else:
        files = sorted(source_dir.glob(pattern))
        files = [p for p in files if p.is_file()]
    return sorted(files)


def pack_with_7zip(seven_zip: str, zip_path: Path, source_dir: Path, files: list[Path]) -> None:
    """Create the zip archive using 7-Zip with ultra LZMA compression."""
    # Run from inside the source directory so 7-Zip stores bare file names.
    cmd = [
        seven_zip,
        "a",                 # add files to archive
        "-tzip",             # zip format
        "-mx=9",             # ultra compression level
        "-m0=LZMA",          # LZMA for best compression ratio
        str(zip_path),
    ] + [f.name for f in files]

    subprocess.run(cmd, cwd=source_dir, check=True)


def pack_with_python(zip_path: Path, source_dir: Path, files: list[Path]) -> None:
    """Create the zip archive using Python's builtin zipfile module.

    LZMA is preferred for the best compression ratio; if the lzma module is
    unavailable, DEFLATED level 9 is used as a safe fallback.
    """
    try:
        compression = zipfile.ZIP_LZMA
        with zipfile.ZipFile(zip_path, "w", compression=compression) as zf:
            for src in files:
                zf.write(src, arcname=src.name)
    except (RuntimeError, NotImplementedError):
        compression = zipfile.ZIP_DEFLATED
        with zipfile.ZipFile(
            zip_path,
            "w",
            compression=compression,
            compresslevel=9,
        ) as zf:
            for src in files:
                zf.write(src, arcname=src.name)


def resolve_source_dir(value: str) -> Path:
    """Resolve the user-supplied SDK directory path.

    If a relative path is given and it does not already exist, it is looked up
    under the ``SDKs/`` directory located next to this script.
    """
    path = Path(value)
    if not path.is_absolute():
        # First try the path as-is; if that does not exist, look under SDKs/.
        if not path.exists():
            script_dir = Path(__file__).resolve().parent
            sdk_dir = script_dir.parent / "SDKs" / path
            if sdk_dir.exists():
                return sdk_dir.resolve()
    return path.resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack an SDK directory into a .zip archive with bare files (no subdirectory).",
    )
    parser.add_argument(
        "source",
        help="SDK directory to pack. May be a name under SDKs/, a relative path, or an absolute path.",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="Output zip file path. Defaults to <SOURCE>.zip",
    )
    parser.add_argument(
        "--use-7z",
        choices=["auto", "yes", "no"],
        default="auto",
        help="Control 7-Zip usage: auto (default), yes, or no.",
    )
    parser.add_argument(
        "--include",
        default="*",
        help="Glob pattern for files to include (default: '*', all top-level files).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    source_dir = resolve_source_dir(args.source)
    if not source_dir.exists():
        print(f"[Error] SDK directory not found: {args.source} (resolved: {source_dir})", file=sys.stderr, flush=True)
        return 1
    if not source_dir.is_dir():
        print(f"[Error] Path is not a directory: {source_dir}", file=sys.stderr, flush=True)
        return 1

    output_zip = Path(args.output) if args.output else source_dir.with_suffix(".zip")
    output_zip = output_zip.resolve()

    if output_zip.exists() and not args.overwrite:
        print(
            f"[Error] Output file already exists: {output_zip}\n"
            "        Use --overwrite to replace it.",
            file=sys.stderr,
            flush=True,
        )
        return 1

    # Ensure the output directory exists.
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    files = collect_files(source_dir, args.include)
    if not files:
        print(f"[Warning] No top-level files matched '{args.include}' in {source_dir}; creating empty zip.", flush=True)

    # Determine whether to use 7-Zip.
    seven_zip: str | None = None
    if args.use_7z in ("auto", "yes"):
        seven_zip = find_7zip()
        if args.use_7z == "yes" and seven_zip is None:
            print("[Error] --use-7z=yes requested but 7-Zip was not found.", file=sys.stderr, flush=True)
            return 1

    # Remove any pre-existing output so we do not append to an old archive.
    if output_zip.exists():
        print(f"[Info] Removing existing archive: {output_zip}", flush=True)
        output_zip.unlink()

    print(f"[Info] Packing SDK: {source_dir}", flush=True)
    print(f"[Info] Output:        {output_zip}", flush=True)
    print(f"[Info] Files to pack: {len(files)}", flush=True)

    try:
        if seven_zip:
            print(f"[Info] Using 7-Zip:   {seven_zip}", flush=True)
            pack_with_7zip(seven_zip, output_zip, source_dir, files)
        else:
            print("[Info] 7-Zip not found; using Python builtin zipfile.", flush=True)
            pack_with_python(output_zip, source_dir, files)
    except subprocess.CalledProcessError as exc:
        print(f"[Warning] 7-Zip failed ({exc}), falling back to Python builtin zipfile.", flush=True)
        if output_zip.exists():
            output_zip.unlink()
        pack_with_python(output_zip, source_dir, files)
    except (OSError, PermissionError) as exc:
        print(f"[Error] Failed to create archive: {exc}", file=sys.stderr, flush=True)
        return 1

    # Print a small summary.
    try:
        size = output_zip.stat().st_size
        print(f"[Success] Created {output_zip} ({size:,} bytes)", flush=True)
    except OSError:
        print(f"[Success] Created {output_zip}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
