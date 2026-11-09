#!/usr/bin/env python3
"""Opt in to repository hooks without replacing existing hooks."""
import pathlib
import subprocess
import sys

root = pathlib.Path(subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], text=True).strip())
current = subprocess.run(["git", "config", "--get", "core.hooksPath"],
                         capture_output=True, text=True)
if current.returncode not in (0, 1):
    sys.exit(current.stderr)
if current.stdout.strip() not in ("", ".githooks"):
    sys.exit("Existing core.hooksPath found; integrate .githooks manually rather than overwrite it.")
hooks_dir = pathlib.Path(subprocess.check_output(
    ["git", "rev-parse", "--git-path", "hooks"], text=True).strip())
if not hooks_dir.is_absolute():
    hooks_dir = root / hooks_dir
if not current.stdout.strip() and any(
        p.is_file() and not p.name.endswith(".sample") for p in hooks_dir.glob("*")):
    sys.exit("Existing Git hooks found; integrate .githooks manually rather than disable them.")
for name in ("pre-commit", "pre-merge-commit", "pre-push"):
    path = root / ".githooks" / name
    path.chmod(path.stat().st_mode | 0o111)
subprocess.run(["git", "config", "--local", "core.hooksPath", ".githooks"], check=True)
print("Enabled llvm-downgrade guard hooks for this clone.")
