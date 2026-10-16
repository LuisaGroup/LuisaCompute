#!/usr/bin/env python3
"""Clone or update every submodule (recursively) to the latest commit of its branch.

Each submodule is handled manually with plain git commands (no `git submodule
update`), so we always end up on the tip of the tracked branch instead of a
detached recorded SHA:

  * missing / empty directory  -> ``git clone``
  * existing repo              -> ``git fetch`` + checkout branch + ``reset --hard origin/<branch>``

Branch resolution order:
  1. ``branch =`` entry in the containing ``.gitmodules``
  2. the submodule's current local branch (if not detached)
  3. the remote's default HEAD branch

Nested submodules (a ``.gitmodules`` inside a submodule) are processed the same
way, and independent submodules are updated in parallel threads.
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Terminal colors (disabled when not a TTY or when NO_COLOR is set)
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def red(s: str) -> str:
    return _c("31", s)


def green(s: str) -> str:
    return _c("32", s)


def yellow(s: str) -> str:
    return _c("33", s)


def blue(s: str) -> str:
    return _c("34", s)


def cyan(s: str) -> str:
    return _c("36", s)


_print_lock = threading.Lock()


def log_lines(lines: list[str]) -> None:
    """Print a whole submodule's buffered log atomically (no interleaving)."""
    with _print_lock:
        print("\n".join(lines), flush=True)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Submodule:
    name: str
    path: str  # relative to the containing repo
    url: str
    branch: str | None  # from .gitmodules, may be None
    update: str | None = None  # 'none' means skip


@dataclass
class Result:
    name: str
    ok: bool
    head: str = ""
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# git helpers
# ---------------------------------------------------------------------------


class GitError(Exception):
    pass


def git(args: list[str], cwd: Path, log: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a git command, append it (and failures) to the submodule's log."""
    log.append(f"  {yellow('$ git ' + ' '.join(args))}")
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        if check:
            raise GitError(f"git {' '.join(args)} failed: {stderr}")
        log.append(f"  {red('[fail]')} {stderr}")
    return proc


def is_git_repo(path: Path) -> bool:
    """True only if `path` itself is the top level of a git working tree.

    Note: `rev-parse --git-dir` alone is not enough — it searches parent
    directories, so an empty (uninitialized) submodule dir inside a repo
    would be misdetected as a repo.
    """
    if not path.exists():
        return False
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=str(path), capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return False
    try:
        return Path(proc.stdout.strip()).resolve() == path.resolve()
    except OSError:
        return False


def force_rmtree(path: Path) -> None:
    """rmtree that copes with read-only files (Windows git object dirs)."""

    def _onerror(func, target, _exc):
        try:
            os.chmod(target, stat.S_IWRITE)
            func(target)
        except OSError:
            raise GitError(f"could not remove {target}")

    shutil.rmtree(path, onerror=_onerror)


def parse_gitmodules(gitmodules: Path) -> list[Submodule]:
    """Parse a .gitmodules file via `git config --get-regexp`.

    Using git itself (instead of configparser) handles value escaping
    correctly, e.g. Windows paths stored as `C:\\\\...` in the file.
    """
    proc = subprocess.run(
        ["git", "config", "--file", str(gitmodules), "--get-regexp",
         r"^submodule\..*\."],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    # returncode 1 just means "no matches"
    if proc.returncode not in (0, 1):
        raise GitError(f"failed to parse {gitmodules}: {proc.stderr.strip()}")

    entries: dict[str, dict[str, str]] = {}
    for line in proc.stdout.splitlines():
        key, _, value = line.partition(" ")
        parts = key.split(".")
        if len(parts) < 3:
            continue
        name, field_ = parts[1], parts[-1]
        entries.setdefault(name, {})[field_] = value

    result: list[Submodule] = []
    for name, fields in entries.items():
        path, url = fields.get("path"), fields.get("url")
        if not path or not url:
            continue
        result.append(Submodule(name=name, path=path.replace("\\", "/"), url=url,
                                branch=fields.get("branch"),
                                update=fields.get("update")))
    return result


def resolve_branch(sm: Submodule, sm_path: Path, log: list[str]) -> str:
    """Pick the branch to track: .gitmodules > current local branch > remote HEAD."""
    if sm.branch:
        return sm.branch

    cur = git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=sm_path, log=log).stdout.strip()
    if cur and cur != "HEAD":
        return cur

    # Detached with no configured branch: ask the remote for its default.
    try:
        head = git(["symbolic-ref", "--short", "refs/remotes/origin/HEAD"],
                   cwd=sm_path, log=log, check=False)
        if head.returncode == 0 and head.stdout.strip():
            return head.stdout.strip().removeprefix("origin/")
    except GitError:
        pass
    git(["remote", "set-head", "origin", "--auto"], cwd=sm_path, log=log, check=False)
    head = git(["symbolic-ref", "--short", "refs/remotes/origin/HEAD"],
               cwd=sm_path, log=log, check=False)
    if head.returncode == 0 and head.stdout.strip():
        return head.stdout.strip().removeprefix("origin/")
    raise GitError("detached HEAD and could not determine remote default branch")


def head_commit(sm_path: Path, log: list[str]) -> str:
    return git(["log", "--oneline", "-1"], cwd=sm_path, log=log).stdout.strip()


# ---------------------------------------------------------------------------
# Per-submodule processing
# ---------------------------------------------------------------------------


def update_submodule(sm: Submodule, repo_root: Path, max_workers: int,
                     depth: int, max_depth: int) -> Result:
    """Clone-or-update one submodule, then recurse into its own submodules."""
    indent = "  " * depth
    log: list[str] = [f"{cyan(f'{indent}[{sm.name}]')} path={sm.path} url={sm.url}"
                      + (f" branch={sm.branch}" if sm.branch else "")]
    sm_path = repo_root / sm.path

    if sm.update == "none":
        log.append(f"{indent}  {yellow('[skip]')} update = none")
        log_lines(log)
        return Result(sm.name, True, "(skipped)")

    try:
        if not is_git_repo(sm_path):
            # --- Clone -----------------------------------------------------
            if sm_path.exists():
                if (sm_path / ".git").exists():
                    # Left-over partial clone (objects but no HEAD/config): re-clone.
                    log.append(f"{indent}  {yellow('[clean]')} removing broken .git in {sm.path}")
                    force_rmtree(sm_path)
                elif any(sm_path.iterdir()):
                    raise GitError(f"{sm_path} exists and is not empty but is not a git "
                                   "repo; remove it manually")
            log.append(f"{indent}  {blue('[clone]')}")
            clone_cmd = ["clone"]
            if sm.branch:
                clone_cmd += ["--branch", sm.branch]
            git(clone_cmd + [sm.url, str(sm_path)], cwd=repo_root, log=log)
            # Fresh clone: make sure URL matches .gitmodules (it does), branch checked out.
        else:
            # --- Existing repo: fetch + move to branch tip -----------------
            # Point origin at the .gitmodules URL (create it if missing).
            cur_url = git(["remote", "get-url", "origin"], cwd=sm_path, log=log,
                          check=False).stdout.strip()
            if cur_url and cur_url != sm.url:
                log.append(f"{indent}  {yellow('[url]')} {cur_url} -> {sm.url}")
                git(["remote", "set-url", "origin", sm.url], cwd=sm_path, log=log)
            elif not cur_url:
                log.append(f"{indent}  {yellow('[url]')} adding missing origin -> {sm.url}")
                git(["remote", "add", "origin", sm.url], cwd=sm_path, log=log)

            branch = resolve_branch(sm, sm_path, log)
            log.append(f"{indent}  {blue('[fetch]')} origin/{branch}")
            git(["fetch", "origin", branch, "--prune"], cwd=sm_path, log=log)

            # Checkout the branch, creating/resetting a local tracking branch if needed.
            cur = git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=sm_path,
                      log=log).stdout.strip()
            if cur != branch:
                git(["checkout", "-B", branch, f"origin/{branch}"], cwd=sm_path, log=log)

            # Hard-sync to the remote tip (discards any local drift).
            git(["reset", "--hard", f"origin/{branch}"], cwd=sm_path, log=log)

        head = head_commit(sm_path, log)
        log.append(f"{indent}  {green('[ ok ]')} {head}")

        # --- Recurse into nested submodules --------------------------------
        nested_gitmodules = sm_path / ".gitmodules"
        if nested_gitmodules.exists() and depth < max_depth:
            nested = parse_gitmodules(nested_gitmodules)
            # Only recurse into submodules that are actually initialized/present,
            # plus clone the missing ones — same policy as the top level.
            if nested:
                log.append(f"{indent}  {cyan(f'[{len(nested)} nested submodule(s)]')}")
                log_lines(log)
                nested_results = update_all(nested, sm_path, max_workers,
                                            depth + 1, max_depth)
                failures = [r for r in nested_results if not r.ok]
                if failures:
                    return Result(sm.name, False, head,
                                  errors=[e for r in failures for e in r.errors])
                return Result(sm.name, True, head)
        elif nested_gitmodules.exists():
            log.append(f"{indent}  {yellow(f'[warn] max depth {max_depth} reached, not recursing')}")

        log_lines(log)
        return Result(sm.name, True, head)

    except (GitError, OSError) as e:
        log.append(f"{indent}  {red('[error]')} {e}")
        log_lines(log)
        return Result(sm.name, False, errors=[f"{sm.name}: {e}"])


def update_all(submodules: list[Submodule], repo_root: Path, max_workers: int,
               depth: int = 0, max_depth: int = 5) -> list[Result]:
    """Update a list of submodules in parallel; returns per-submodule results."""
    if not submodules:
        return []
    with ThreadPoolExecutor(max_workers=max_workers,
                            thread_name_prefix="sm-update") as pool:
        futures = [
            pool.submit(update_submodule, sm, repo_root, max_workers, depth, max_depth)
            for sm in submodules
        ]
        return [f.result() for f in futures]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-j", "--jobs", type=int, default=8,
                    help="parallel workers (default: 8)")
    ap.add_argument("--max-depth", type=int, default=5,
                    help="max submodule recursion depth (default: 5)")
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent,
                    help="repository root (default: parent of scripts/)")
    args = ap.parse_args()

    repo_root: Path = args.repo
    gitmodules = repo_root / ".gitmodules"
    if not gitmodules.exists():
        print(red(f"[ERROR] .gitmodules not found at {gitmodules}"))
        return 1

    try:
        submodules = parse_gitmodules(gitmodules)
    except GitError as e:
        print(red(f"[ERROR] {e}"))
        return 1

    if not submodules:
        print("[INFO] No submodules found in .gitmodules.")
        return 0

    print(f"[INFO] Updating {len(submodules)} submodule(s) recursively "
          f"with {args.jobs} parallel worker(s).\n")

    results = update_all(submodules, repo_root, args.jobs, 0, args.max_depth)

    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    print()
    print(green(f"[INFO] {len(ok)}/{len(results)} submodule(s) updated:"))
    for r in ok:
        print(f"  {green('OK  ')} {r.name}: {r.head}")
    if failed:
        print(red(f"[ERROR] {len(failed)} submodule(s) failed:"))
        for r in failed:
            for e in r.errors:
                print(f"  {red('FAIL')} {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
