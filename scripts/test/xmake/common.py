import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def run_cmd(cmd: list[str], cwd: Path = PROJECT_ROOT) -> int:
    print(f">>> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


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
