import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def run_cmd(cmd: list[str], cwd: Path = PROJECT_ROOT) -> int:
    print(f">>> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def get_backends(defaults: list[str] | None = None) -> list[str]:
    """Parse backend arguments from trivial positional text and return the list of backends to test."""
    import argparse
    parser = argparse.ArgumentParser()
    args, remaining = parser.parse_known_args()
    valid = {"dx", "vk", "cuda", "metal"}
    backends = [a for a in remaining if a in valid]
    if backends:
        return backends
    return defaults if defaults is not None else []


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
