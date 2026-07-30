"""Show uncommitted and committed changes for the given files via git diff."""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def _runcmd(args: list[str], label: str = "") -> str:
    """Run a command and return stripped stdout (or stderr on failure)."""
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0 and result.stderr:
        return result.stderr.strip()
    return result.stdout.strip()


def _print_header(title: str) -> None:
    width = 60
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def process_file(file_path: str) -> None:
    """Show diff information for a single file."""
    fpath = Path(file_path)
    rel = str(fpath)

    # 1. File does not exist on disk
    if not fpath.exists():
        _print_header(f"DELETED: {rel}")
        # Try to show the last committed version
        tracked = _runcmd(["git", "ls-files", "--", rel])
        if tracked:
            # It was tracked; show what was removed from HEAD
            diff_out = _runcmd(["git", "diff", "HEAD", "--", rel])
            if diff_out:
                print(diff_out)
            else:
                print(f"(file tracked but no diff to HEAD — may be staged deletion)")
        else:
            print(f"(file never tracked by git)")
        return

    # 2. Is this file tracked by git?
    tracked = _runcmd(["git", "ls-files", "--", rel])
    if not tracked:
        _print_header(f"NEW (untracked): {rel}")
        try:
            content = fpath.read_text(encoding="utf-8")
            print(content)
        except Exception:
            print(f"(binary or unreadable — {fpath.stat().st_size} bytes)")
        return

    # 3. Tracked file — show uncommitted diff (working-tree vs index, index vs HEAD)
    _print_header(f"UNCOMMITTED: {rel}")
    uncommitted = _runcmd(["git", "diff", "--", rel])
    staged = _runcmd(["git", "diff", "--cached", "--", rel])
    if not uncommitted and not staged:
        print("(no uncommitted changes)")
    else:
        if staged:
            _print_header("  staged changes")
            print(staged)
        if uncommitted:
            _print_header("  working-tree changes")
            print(uncommitted)

    # 4. Show last committed diff for this file
    _print_header(f"LAST COMMIT: {rel}")
    committed = _runcmd(["git", "diff", "HEAD~", "HEAD", "--", rel])
    if committed:
        print(committed)
    else:
        print("(no diff in last commit for this file)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show git-diff information for given files (uncommitted / committed / new / deleted)."
    )
    parser.add_argument(
        "files", nargs="+", help="File paths to inspect"
    )
    args = parser.parse_args()

    for f in args.files:
        process_file(f)

    return 0


if __name__ == "__main__":
    sys.exit(main())
