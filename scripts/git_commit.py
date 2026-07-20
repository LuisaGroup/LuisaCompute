"""Show changed files and diffs for a given commit hash."""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def _runcmd(args: list[str]) -> str:
    """Run a git command and return stripped stdout. Exit on failure."""
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        print(f"git error: {stderr}", file=sys.stderr)
        sys.exit(1)
    return (result.stdout or "").strip()


def _print_header(title: str) -> None:
    width = 60
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _is_root_commit(commit: str) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", f"{commit}^"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    return result.returncode != 0


def _categorize_files(diff_stat: str) -> tuple[list[str], list[str], list[str]]:
    """Parse --name-status output into (created, deleted, updated) lists."""
    created: list[str] = []
    deleted: list[str] = []
    updated: list[str] = []

    for line in diff_stat.splitlines():
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status = parts[0]
        filepath = parts[-1]  # handles renames: R100\told\tnew
        if status.startswith("A"):
            created.append(filepath)
        elif status.startswith("D"):
            deleted.append(filepath)
        else:
            updated.append(filepath)

    return created, deleted, updated


def process_commit(commit: str) -> None:
    """Show the file changes and diffs for a single commit."""
    # Verify commit exists
    verify = subprocess.run(
        ["git", "cat-file", "-t", commit],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(PROJECT_ROOT),
    )
    if verify.returncode != 0 or (verify.stdout or "").strip() != "commit":
        print(f"Invalid commit: {commit}", file=sys.stderr)
        sys.exit(1)

    root = _is_root_commit(commit)

    # Get file change statuses
    if root:
        diff_stat = _runcmd(
            ["git", "diff-tree", "--no-commit-id", "--name-status", "-r", commit],
        )
    else:
        diff_stat = _runcmd(
            ["git", "diff-tree", "--no-commit-id", "--name-status", "-r", f"{commit}^", commit],
        )

    created, deleted, updated = _categorize_files(diff_stat)

    # Print created files
    if created:
        _print_header("CREATED FILES")
        for f in created:
            print(f"  + {f}")

    # Print deleted files
    if deleted:
        _print_header("DELETED FILES")
        for f in deleted:
            print(f"  - {f}")

    # Print updated files with diffs
    if updated:
        _print_header("UPDATED FILES")
        for f in updated:
            print(f"\n  >> {f}")
            print("  " + "-" * 50)
            if root:
                diff = _runcmd(
                    ["git", "show", "--format=", "-p", commit, "--", f],
                )
            else:
                diff = _runcmd(
                    ["git", "diff", f"{commit}^", commit, "--", f],
                )
            for diff_line in diff.splitlines():
                print(f"  {diff_line}")
            print("  " + "-" * 50)

    if not created and not deleted and not updated:
        print("No changes found in this commit.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show changed files and diffs for a given commit hash."
    )
    parser.add_argument("commit", help="Commit hash to inspect")
    args = parser.parse_args()

    process_commit(args.commit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
