#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check C++ syntax for all files in compile_commands.json in parallel.
"""

import argparse
import orjson
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_JOBS = min(8, os.cpu_count() or 1)


def load_compile_command_files(compile_commands_path: str) -> list[str]:
    database = Path(compile_commands_path).resolve()
    with open(database, "r", encoding="utf-8") as f:
        data = orjson.loads(f.read())
    files = []
    seen = set()
    cpp_exts = {".cpp", ".cc", ".cxx", ".c++", ".h", ".hpp", ".hh", ".hxx", ".h++"}
    for entry in data:
        file_path = entry.get("file")
        if not isinstance(file_path, str) or not file_path:
            continue
        source = Path(file_path)
        if not source.is_absolute():
            directory = entry.get("directory")
            base = (
                Path(directory)
                if isinstance(directory, str) and directory
                else database.parent
            )
            if not base.is_absolute():
                base = database.parent / base
            source = base / source
        source = source.resolve()
        if source.suffix.lower() not in cpp_exts or source in seen:
            continue
        seen.add(source)
        files.append(str(source))
    return files


def check_file(
    file_path: str,
    script_path: str,
    project_root: str | None,
    clangd_path: str | None,
    compile_commands_path: str | None,
    diagnostic_timeout: float = 30.0,
    clang_tidy: bool = False,
) -> tuple[str, int, str, str]:
    cmd = [sys.executable, str(script_path), file_path]
    if project_root is not None:
        cmd += ["--project-root", project_root]
    if clangd_path is not None:
        cmd += ["--clangd", clangd_path]
    if compile_commands_path is not None:
        cmd += ["--compile-commands-dir", compile_commands_path]
    cmd += ["--diagnostic-timeout", str(diagnostic_timeout)]
    if clang_tidy:
        cmd.append("--clang-tidy")

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return file_path, result.returncode, result.stdout, result.stderr


def main():
    parser = argparse.ArgumentParser(
        description="Check C++ syntax for all files in compile_commands.json",
    )
    parser.add_argument(
        "--compile-commands",
        default=".vscode/compile_commands.json",
        help="Path to compile_commands.json (default: .vscode/compile_commands.json)",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Project root directory",
    )
    parser.add_argument(
        "--clangd",
        default=None,
        help="Path to clangd executable",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=DEFAULT_JOBS,
        help=f"Maximum parallel jobs (default: {DEFAULT_JOBS})",
    )
    parser.add_argument(
        "--diagnostic-timeout",
        type=float,
        default=30.0,
        help="Per-file clangd timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--clang-tidy",
        action="store_true",
        help="Include clang-tidy diagnostics (disabled by default)",
    )

    args = parser.parse_args()

    compile_commands_path = Path(args.compile_commands)
    if not compile_commands_path.exists():
        print(f"Error: compile_commands.json not found: {compile_commands_path}", file=sys.stderr)
        sys.exit(1)

    script_path = Path(__file__).parent / "check_cpp_syntax.py"
    if not script_path.exists():
        print(f"Error: check_cpp_syntax.py not found: {script_path}", file=sys.stderr)
        sys.exit(1)

    files = load_compile_command_files(str(compile_commands_path))
    if not files:
        print("No files found in compile_commands.json", file=sys.stderr)
        sys.exit(1)

    total = len(files)
    max_workers = max(1, args.jobs)
    print(f"Checking {total} files with up to {max_workers} parallel workers...")

    errors = 0
    failures = 0
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                check_file,
                f,
                script_path,
                args.project_root,
                args.clangd,
                str(compile_commands_path.resolve()),
                args.diagnostic_timeout,
                args.clang_tidy,
            ): f
            for f in files
        }
        for future in as_completed(futures):
            file_path, returncode, stdout, stderr = future.result()
            completed += 1

            if returncode != 0:
                if stdout or stderr:
                    print(f"[{completed}/{total}] {file_path} (exit={returncode})")
                    if stdout:
                        print(stdout, end="")
                    if stderr:
                        print(stderr, end="")
                if returncode == 1:
                    errors += 1
                else:
                    failures += 1
            # else:
            #     if stdout:
            #         print(f"[{completed}/{total}] {file_path}")
            #         print(stdout, end="")

    print("=" * 60)
    print(f"Finished: {total} files checked, {errors} with errors, {failures} failures")

    if errors > 0 or failures > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
