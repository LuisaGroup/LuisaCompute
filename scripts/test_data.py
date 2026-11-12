#!/usr/bin/env python3
"""Manage LuisaCompute test and example data through GitHub Releases.

Data directories are directories named ``data`` below ``src/tests`` and
``examples``. They are stored together as repository-relative paths in one ZIP
asset on the LuisaGroup/SDKs ``Data`` release.
"""

import argparse
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile


REPO_ROOT = Path(__file__).resolve().parent.parent
SEARCH_ROOTS = (Path("src/tests"), Path("examples"))
GITHUB_REPO = "LuisaGroup/SDKs"
RELEASE_TAG = "Data"
ARCHIVE_NAME = "luisa-compute-test-data.zip"


class TestDataError(RuntimeError):
    """A test-data operation cannot be completed."""


def find_data_dirs(repo_root: Path = REPO_ROOT) -> list[Path]:
    """Return repository-relative data directories in deterministic order."""
    found: list[Path] = []
    for search_root in SEARCH_ROOTS:
        absolute_root = repo_root / search_root
        if not absolute_root.is_dir():
            continue
        for current, dirnames, _ in os.walk(absolute_root, followlinks=False):
            remaining = []
            for dirname in sorted(dirnames):
                directory = Path(current) / dirname
                if dirname == "data":
                    if directory.is_symlink():
                        raise TestDataError(f"Data directory must not be a symlink: {directory}")
                    found.append(directory.relative_to(repo_root))
                else:
                    remaining.append(dirname)
            # A data directory is archived as one unit; do not report nested
            # directories named data as separate top-level data sets.
            dirnames[:] = remaining
    return sorted(found, key=lambda path: path.as_posix())


def create_archive(
    data_dirs: list[Path],
    archive_path: Path,
    repo_root: Path = REPO_ROOT,
) -> None:
    """Compress data directories while preserving repository-relative paths."""
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for data_dir in data_dirs:
            absolute_dir = repo_root / data_dir
            if not absolute_dir.is_dir():
                raise TestDataError(f"Data directory does not exist: {absolute_dir}")
            for current, dirnames, filenames in os.walk(absolute_dir):
                dirnames.sort()
                filenames.sort()
                current_path = Path(current)
                relative_dir = current_path.relative_to(repo_root)
                archive.writestr(f"{relative_dir.as_posix()}/", "")
                for filename in filenames:
                    source = current_path / filename
                    if source.is_symlink():
                        raise TestDataError(f"Data files must not be symlinks: {source}")
                    archive.write(source, source.relative_to(repo_root).as_posix())


def _member_path(info: zipfile.ZipInfo, repo_root: Path) -> tuple[Path, Path]:
    """Validate an archive member and return its relative and absolute paths."""
    name = info.filename
    raw_parts = name.split("/")
    if (
        not name
        or "\\" in name
        or name.startswith("/")
        or any(part in {"", ".", ".."} for part in raw_parts[:-1])
        or raw_parts[-1] in {".", ".."}
    ):
        raise TestDataError(f"Unsafe path in data archive: {name!r}")

    relative = PurePosixPath(name.rstrip("/"))
    parts = relative.parts
    in_test_data = len(parts) >= 3 and parts[:2] == ("src", "tests")
    in_example_data = len(parts) >= 2 and parts[0] == "examples"
    if (
        not parts
        or "data" not in parts
        or not (in_test_data or in_example_data)
        or stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK
    ):
        raise TestDataError(f"Disallowed path or file type in data archive: {name!r}")

    relative_path = Path(*parts)
    absolute_path = repo_root / relative_path
    try:
        absolute_path.resolve().relative_to(repo_root.resolve())
    except ValueError as error:
        raise TestDataError(f"Data archive path escapes the repository: {name!r}") from error
    return relative_path, absolute_path


def extract_archive(archive_path: Path, repo_root: Path = REPO_ROOT) -> list[Path]:
    """Safely extract an archive and return the restored data directories."""
    restored: set[Path] = set()
    with zipfile.ZipFile(archive_path, "r") as archive:
        members = [
            (info, *_member_path(info, repo_root)) for info in archive.infolist()
        ]
        for info, relative, destination in members:
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info, "r") as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
            parts = relative.parts
            data_index = parts.index("data")
            restored.add(Path(*parts[: data_index + 1]))
    return sorted(restored, key=lambda path: path.as_posix())


def run_gh(arguments: list[str]) -> None:
    """Run GitHub CLI from the repository root."""
    if shutil.which("gh") is None:
        raise TestDataError("GitHub CLI ('gh') was not found on PATH.")
    command = ["gh", *arguments, "--repo", GITHUB_REPO]
    print(f"+ {' '.join(command)}", flush=True)
    try:
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    except subprocess.CalledProcessError as error:
        raise TestDataError(
            f"GitHub CLI failed with exit code {error.returncode}. "
            "Check 'gh auth status' and your release permissions."
        ) from error


def download_data() -> None:
    """Download and extract the release archive."""
    with tempfile.TemporaryDirectory(prefix="luisa-test-data-") as temp_dir:
        archive_path = Path(temp_dir) / ARCHIVE_NAME
        run_gh(
            [
                "release",
                "download",
                RELEASE_TAG,
                "--pattern",
                ARCHIVE_NAME,
                "--dir",
                temp_dir,
                "--clobber",
            ]
        )
        if not archive_path.is_file():
            raise TestDataError(f"Release asset was not downloaded: {ARCHIVE_NAME}")
        restored = extract_archive(archive_path)
    print("Restored data directories:")
    for directory in restored:
        print(directory.as_posix())


def upload_data(data_dirs: list[Path]) -> None:
    """Compress all local data directories and replace the release asset."""
    if not data_dirs:
        raise TestDataError("No test or example data directories were found.")
    with tempfile.TemporaryDirectory(prefix="luisa-test-data-") as temp_dir:
        archive_path = Path(temp_dir) / ARCHIVE_NAME
        create_archive(data_dirs, archive_path)
        print(
            f"Created {ARCHIVE_NAME} from {len(data_dirs)} data director"
            f"{'y' if len(data_dirs) == 1 else 'ies'} "
            f"({archive_path.stat().st_size} bytes)."
        )
        run_gh(
            [
                "release",
                "upload",
                RELEASE_TAG,
                str(archive_path),
                "--clobber",
            ]
        )
    print(
        f"Uploaded {ARCHIVE_NAME} to "
        f"https://github.com/{GITHUB_REPO}/releases/tag/{RELEASE_TAG}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list local test and example data folders; do not access the network.",
    )
    action.add_argument(
        "--download",
        action="store_true",
        help=f"Download and extract {ARCHIVE_NAME} from the {RELEASE_TAG} release.",
    )
    action.add_argument(
        "--upload",
        action="store_true",
        help=f"Compress all local data folders and replace {ARCHIVE_NAME} on the {RELEASE_TAG} release.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        data_dirs = find_data_dirs()
        if args.dry_run:
            for directory in data_dirs:
                print(directory.as_posix())
        elif args.download:
            download_data()
        else:
            upload_data(data_dirs)
    except (TestDataError, OSError, zipfile.BadZipFile) as error:
        print(f"test_data.py: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
