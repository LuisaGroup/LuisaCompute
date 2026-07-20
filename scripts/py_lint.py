"""Python syntax check CLI. Run py_compile on a target file and optionally execute it if checks pass."""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

logger = logging.getLogger(__name__)


def _resolve_target(path_str: str) -> Path:
    input_path = Path(path_str)
    if input_path.is_absolute():
        try:
            relative_path = input_path.relative_to(PROJECT_ROOT)
        except ValueError:
            logger.error(
                "Absolute path %s is not within the project directory %s",
                input_path,
                PROJECT_ROOT,
            )
            sys.exit(1)
    else:
        relative_path = input_path
    return PROJECT_ROOT / relative_path


def _run_py_compile(target_file: Path) -> tuple[int, str, str]:
    """Run python -m py_compile on the target file and return (returncode, stdout, stderr)."""
    cmd = [sys.executable, "-m", "py_compile", str(target_file)]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    return result.returncode, result.stdout, result.stderr


def _run_python(target_file: Path) -> tuple[int, str, str]:
    """Execute the target Python file and return (returncode, stdout, stderr)."""
    cmd = [sys.executable, str(target_file)]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    return result.returncode, result.stdout, result.stderr


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Python syntax check on a target file. Use --exec to also execute it if checks pass."
    )
    parser.add_argument("target_file", help="Python file to check")
    parser.add_argument(
        "--exec", "-e",
        action="store_true",
        help="Execute the file after successful syntax check",
    )
    args = parser.parse_args()

    target_file = _resolve_target(args.target_file)

    returncode, stdout, stderr = _run_py_compile(target_file)
    if stdout:
        print(stdout)
    if stderr:
        print(stderr)

    if returncode != 0:
        logger.error("Syntax check failed; aborting execution.")
        return returncode

    print(f"[py_lint] Syntax OK: {target_file}")

    if args.exec:
        print(f"[py_lint] Executing {target_file} ...")
        exec_rc, exec_out, exec_err = _run_python(target_file)
        if exec_out:
            print(exec_out)
        if exec_err:
            print(exec_err, file=sys.stderr)
        return exec_rc

    return returncode


if __name__ == '__main__':
    sys.exit(main())
