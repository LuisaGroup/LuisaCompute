import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUPPORTED_BACKENDS = ("cuda", "dx", "fallback", "hip", "metal", "vk")


def run_cmd(cmd: list[str], cwd: Path = PROJECT_ROOT) -> int:
    print(f">>> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def get_backends(defaults: list[str] | None = None) -> list[str]:
    """Parse the requested backend matrix, failing on unsupported names."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "backends",
        metavar="backend",
        nargs="*",
        choices=SUPPORTED_BACKENDS,
        help="backend(s) to test (default: script-specific matrix)",
    )
    args = parser.parse_args()
    if args.backends:
        return args.backends
    return list(defaults) if defaults is not None else []


def format_backends(backends: list[str]) -> str:
    return ", ".join(backends) if backends else "none"


def get_targets(cwd: Path = PROJECT_ROOT) -> list[str]:
    result = subprocess.run(
        ["xmake", "show", "--list=targets"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("ERROR: failed to list targets")
        return []
    return result.stdout.split()
