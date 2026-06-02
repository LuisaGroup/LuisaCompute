#!/usr/bin/env python3
"""Clone or pull all submodules in .gitmodules to the latest commit on their current branch."""

import configparser
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command, print it, and return the result."""
    print(f"  \033[33m$ {' '.join(cmd)}\033[0m")
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=check)


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    gitmodules = repo_root / ".gitmodules"

    if not gitmodules.exists():
        print(f"[ERROR] .gitmodules not found at {gitmodules}")
        return 1

    # Parse .gitmodules (standard git-config INI format)
    parser = configparser.ConfigParser()
    parser.read(str(gitmodules))

    submodules: list[dict] = []
    for section in parser.sections():
        if section.startswith('submodule '):
            name = section.split('"')[1] if '"' in section else section[len("submodule "):]
            submodules.append({
                "name": name,
                "path": parser.get(section, "path"),
                "url": parser.get(section, "url"),
                "branch": parser.get(section, "branch", fallback=None),
            })

    if not submodules:
        print("[INFO] No submodules found in .gitmodules.")
        return 0

    print(f"[INFO] Found {len(submodules)} submodule(s).\n")

    for i, sm in enumerate(submodules, 1):
        sm_path = repo_root / sm["path"]
        print(f"\033[36m[{i}/{len(submodules)}] {sm['name']}\033[0m")
        print(f"  Path: {sm['path']}")
        print(f"  URL : {sm['url']}")

        if sm["branch"]:
            print(f"  Branch (from .gitmodules): {sm['branch']}")

        if not sm_path.exists():
            # --- Clone ---
            print(f"  \033[34m[ACTION] Cloning...\033[0m")
            clone_cmd = ["git", "clone"]
            if sm["branch"]:
                clone_cmd += ["--branch", sm["branch"]]
            clone_cmd += [sm["url"], str(sm_path)]
            result = run(clone_cmd, cwd=repo_root, check=False)
            if result.returncode != 0:
                print(f"  \033[31m[ERROR] Clone failed:\033[0m\n{result.stderr.strip()}")
                return 1
            print(f"  \033[32m[ OK ] Cloned.\033[0m")
        else:
            # --- Pull ---
            # Determine current branch
            branch_result = run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=sm_path, check=False,
            )
            current_branch = branch_result.stdout.strip()
            print(f"  Current branch: {current_branch}")

            if current_branch == "HEAD":
                # Detached HEAD — use .gitmodules branch or remote default
                target_branch = sm.get("branch")
                if target_branch:
                    print(f"  \033[33m[WARN] Detached HEAD; using .gitmodules branch: {target_branch}\033[0m")
                    # Checkout the branch
                    checkout_result = run(
                        ["git", "checkout", target_branch],
                        cwd=sm_path, check=False,
                    )
                    if checkout_result.returncode != 0:
                        # Try fetching and then checkout
                        run(["git", "fetch", "origin"], cwd=sm_path)
                        run(["git", "checkout", target_branch], cwd=sm_path)
                    current_branch = target_branch
                else:
                    # Determine remote default branch
                    print(f"  \033[33m[WARN] Detached HEAD, no branch in .gitmodules.\033[0m")
                    remote_show = run(
                        ["git", "remote", "show", "origin"],
                        cwd=sm_path, check=False,
                    )
                    for line in remote_show.stdout.splitlines():
                        line = line.strip()
                        if line.startswith("HEAD branch:"):
                            default_branch = line.split(":", 1)[1].strip()
                            print(f"  Remote HEAD branch: {default_branch}")
                            run(["git", "checkout", default_branch], cwd=sm_path)
                            current_branch = default_branch
                            break

            # Fetch and pull
            print(f"  \033[34m[ACTION] Fetching + pulling '{current_branch}'...\033[0m")
            fetch_result = run(["git", "fetch", "origin", current_branch], cwd=sm_path, check=False)
            if fetch_result.returncode != 0:
                print(f"  \033[31m[ERROR] Fetch failed:\033[0m\n{fetch_result.stderr.strip()}")
                continue

            pull_result = run(
                ["git", "pull", "--rebase", "origin", current_branch],
                cwd=sm_path, check=False,
            )
            if pull_result.returncode != 0:
                # Fallback: reset --hard (e.g. unrelated histories after detached HEAD)
                print(f"  \033[33m[WARN] Pull --rebase failed; trying reset --hard...\033[0m")
                reset_result = run(
                    ["git", "reset", "--hard", f"origin/{current_branch}"],
                    cwd=sm_path, check=False,
                )
                if reset_result.returncode != 0:
                    print(f"  \033[31m[ERROR] Reset also failed:\033[0m\n{reset_result.stderr.strip()}")
                    continue

            # Show new HEAD
            log_result = run(
                ["git", "log", "--oneline", "-1"],
                cwd=sm_path,
            )
            print(f"  \033[32m[ OK ] Now at: {log_result.stdout.strip()}\033[0m")

        print()

    print("\033[32m[INFO] All submodules processed.\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
