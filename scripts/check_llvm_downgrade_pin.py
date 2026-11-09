#!/usr/bin/env python3
"""Check the staged tree (default) or a commit, without checking out submodules."""
import argparse
import re
import subprocess
import sys

SUBMODULE = "src/ext/llvm-downgrade"
CMAKE = "src/ext/CMakeLists.txt"


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def check(ref=None):
    if ref is None:
        cmake = git("show", ":" + CMAKE)
        entries = git("ls-files", "--stage", "--", SUBMODULE).splitlines()
        if len(entries) != 1:
            raise ValueError("missing or conflicted llvm-downgrade gitlink in index")
        metadata, path = entries[0].split("\t")
        mode, actual, stage = metadata.split()
        if stage != "0":
            raise ValueError("unresolved llvm-downgrade merge conflict")
    else:
        # Resolve first: no arbitrary revision expressions reach path lookups.
        ref = git("rev-parse", "--verify", ref + "^{commit}")
        cmake = git("show", ref + ":" + CMAKE)
        entry = git("ls-tree", ref, "--", SUBMODULE)
        metadata, path = entry.split("\t")
        mode, kind, actual = metadata.split()
        if kind != "commit":
            raise ValueError("llvm-downgrade is not a submodule")
    if mode != "160000" or path != SUBMODULE:
        raise ValueError("llvm-downgrade must be a gitlink")
    # Ignore full-line CMake comments; require one explicit pin.
    cmake = "\n".join(line for line in cmake.splitlines()
                      if not line.lstrip().startswith("#"))
    pins = re.findall(
        r'set\s*\(\s*LUISA_LLVM_DOWNGRADE_EXPECTED_COMMIT\s+"([0-9a-f]{40})"\s*\)',
        cmake)
    if len(pins) != 1:
        raise ValueError("expected exactly one explicit CMake llvm-downgrade pin")
    if actual != pins[0]:
        raise ValueError(
            f"llvm-downgrade gitlink {actual} does not match CMake pin {pins[0]}.\n"
            "Restore the validated submodule revision and stage it. Intentional upgrades "
            "must update the CMake pin, source fingerprint and AIR overlay together, "
            "then pass Metal4/iOS tests. Do not bypass this check to update submodules.")
    print(f"llvm-downgrade pin OK: {actual}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", help="commit to check instead of staged index")
    args = parser.parse_args()
    try:
        check(args.ref)
    except (ValueError, subprocess.CalledProcessError) as exc:
        print(f"llvm-downgrade pin check failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
