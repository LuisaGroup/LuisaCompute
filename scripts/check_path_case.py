#!/usr/bin/env python3
"""Check literal submodule directory references in CMake and XMake scripts.

Read canonical spellings from .gitmodules so this also works on a checkout
without initialized submodules, including on case-insensitive filesystems.
"""

import configparser
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
# Keep strings intact while masking line comments, preserving diagnostic lines.
COMMENTS = re.compile(r'"(?:\\[\s\S]|[^"\\])*"|\'(?:\\[\s\S]|[^\'\\])*\''
                      r'|\#[^\n]*|--[^\n]*')


def violations(source: str, build_file: str, submodules: list[str]):
    source = COMMENTS.sub(
        lambda m: " " * len(m.group()) if m.group().startswith(("#", "--"))
        else m.group(), source)
    file = PurePosixPath(build_file)
    directory = file.parent
    # Included CMake helpers can operate on another directory's sources (or
    # match paths with REGEX); only entry points have a known relative base.
    local_paths = file.name in {"CMakeLists.txt", "xmake.lua"}
    patterns = {}
    for module in map(PurePosixPath, submodules):
        # Repository paths and ../ext/... references used by both build systems.
        for path in (module, *module.parents):
            if len(path.parts) < 3:
                continue
            spelling = str(path)
            patterns[(r"(?<![\w.-])", spelling)] = r"(?![\w.-])"
            if path.parts[0] == "src":
                patterns[(r"(?<![\w.-])", str(PurePosixPath(*path.parts[1:])))] = r"(?![\w.-])"
        # Local references such as xxHash/xxhash.h in src/ext/CMakeLists.txt.
        if local_paths and directory in module.parents:
            relative = module.relative_to(directory)
            for path in (relative, *relative.parents):
                if str(path) != ".":
                    patterns[(r"(?<![\w./}{>-])", str(path))] = r"(?=/)"

    found = set()
    matched_ranges = []
    for (prefix, spelling), suffix in sorted(patterns.items(), key=lambda p: -len(p[0][1])):
        pattern = prefix + re.escape(spelling) + suffix
        for match in re.finditer(pattern, source, re.IGNORECASE):
            if any(start <= match.start() and match.end() <= end
                   for start, end in matched_ranges):
                continue
            matched_ranges.append(match.span())
            if match.group() != spelling:
                found.add((source.count("\n", 0, match.start()) + 1,
                           match.group(), spelling))
    # Bare local directory arguments must not be confused with target names.
    for module in map(PurePosixPath, submodules):
        if not local_paths or directory not in module.parents:
            continue
        spelling = str(module.relative_to(directory))
        prefix = (r'(?:\b(?:add_subdirectory|includes)\s*\(\s*["\']?'
                  r'|\$\{CMAKE_CURRENT_(?:SOURCE|LIST)_DIR\}/)')
        pattern = prefix + "(" + re.escape(spelling) + r")(?=[/\s\)\"'>]|$)"
        for match in re.finditer(pattern, source, re.IGNORECASE):
            if match.group(1) != spelling:
                found.add((source.count("\n", 0, match.start(1)) + 1,
                           match.group(1), spelling))
    yield from sorted(found)


def main() -> int:
    config = configparser.ConfigParser(interpolation=None)
    config.read(ROOT / ".gitmodules", encoding="utf-8")
    submodules = [config[section]["path"] for section in config.sections()]
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    checked = failures = 0
    for path in paths:
        file = ROOT / path
        if not file.is_file() or not (file.name == "CMakeLists.txt"
                                      or file.suffix in {".cmake", ".lua"}):
            continue
        checked += 1
        for line, actual, expected in violations(file.read_text(encoding="utf-8"), path, submodules):
            print(f"{path}:{line}: directory {actual!r} must be spelled {expected!r}",
                  file=sys.stderr)
            failures += 1
    print(f"Checked {checked} build scripts for submodule directory case.")
    return int(failures != 0)


if __name__ == "__main__":
    sys.exit(main())
