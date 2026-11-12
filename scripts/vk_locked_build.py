#!/usr/bin/env python3
"""Serialize xmake invocations across concurrent agents.

xmake keeps a per-project build lock, so two concurrent `xmake build` runs in
the same tree fail or interleave unpredictably. Every agent/build step that
touches this repository's xmake tree must go through this wrapper, which takes
an exclusive lock file first and then runs the requested command.

Usage:
    python scripts/vk_locked_build.py build lc-backend-vk
    python scripts/vk_locked_build.py run test_vk_ext_layouts
    python scripts/vk_locked_build.py --timeout 3600 build test_vk_optional_extensions

The lock file lives under .kimix_cache/ and is never part of the build tree.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCK_PATH = REPO_ROOT / ".kimix_cache" / "vk_build.lock"
DEFAULT_TIMEOUT = 3600.0


def _try_lock(handle) -> bool:
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock(handle) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


def main(argv: list[str]) -> int:
    args = list(argv[1:])
    timeout = DEFAULT_TIMEOUT
    if args and args[0] == "--timeout":
        timeout = float(args[1])
        args = args[2:]
    if not args:
        print(__doc__)
        return 2
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    handle = open(LOCK_PATH, "a+b")
    deadline = time.time() + timeout
    waited = False
    while not _try_lock(handle):
        if time.time() > deadline:
            print("vk_locked_build: timed out waiting for the xmake lock", file=sys.stderr)
            return 3
        if not waited:
            print("vk_locked_build: waiting for another xmake build ...", flush=True)
            waited = True
        time.sleep(1.0)
    try:
        cmd = ["xmake", *args]
        print("vk_locked_build: " + " ".join(cmd), flush=True)
        return subprocess.call(cmd, cwd=str(REPO_ROOT))
    finally:
        _unlock(handle)
        handle.close()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
