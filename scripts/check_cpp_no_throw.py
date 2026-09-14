#!/usr/bin/env python3
"""Check tracked project C++ sources for exception-raising syntax and helpers."""

from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx",
                   ".inl", ".inc", ".mm", ".cu", ".cuh"}
# Git submodules are not recursively listed by git ls-files. These are the
# additional vendored sources; do not exempt all src/ext (it also has our glue).
VENDORED_DIRECTORIES = ("src/ext/BTree/", "src/ext/half/", "src/ext/liblmdb/",
                        "src/ext/stb/", "src/ext/volk/")
VENDORED_FILES = {
    "examples/extension/clangcxx_compiler/simplecpp.cpp",
    "examples/extension/clangcxx_compiler/simplecpp.h",
    "include/luisa/core/stl/functional_impl.h",
    "include/luisa/core/stl/pdqsort.h",
    "include/luisa/core/stl/unordered_dense.h",
    "src/backends/common/hlsl/WinAdapter.h",
    "src/backends/dx/DXApi/DirectMLX.h",
    "src/backends/dx/DXApi/d3dx12.h",
    "src/backends/vk/vk_mem_alloc.h",
    "src/tests/common/tinyexr.h",
    "src/tests/common/tiny_obj_loader.h",
    "src/tests/ut/ut.hpp",
}
FORBIDDEN = {"throw", "rethrow_exception", "throw_with_nested", "__cxa_throw",
             "_CxxThrowException", "throws"}
# Consume comments, quoted strings, raw strings, and numeric literals before
# identifiers. Numeric literals matter because C++ permits 1'000 separators.
TOKENS = re.compile(
    r'//[^\n]*|/\*[\s\S]*?\*/'
    r'|(?:u8|u|U|L)?R"(?P<delimiter>[^ ()\\\t\r\n]{0,16})\([\s\S]*?\)(?P=delimiter)"'
    r'|(?:u8|u|U|L)?"(?:\\[\s\S]|[^"\\])*"'
    r"|\b[0-9][\w'.]*"
    r"|(?:u8|u|U|L)?'(?:\\[\s\S]|[^'\\\r\n])*'"
    r'|[A-Za-z_][A-Za-z_0-9]*'
)


def is_third_party(path: str, source: str) -> bool:
    if path in VENDORED_FILES or path.startswith(VENDORED_DIRECTORIES):
        return True
    # Captured TorchInductor output is benchmark evidence, not Luisa code.
    # Keep our benchmark drivers under the same results directory in the scan.
    return (path.startswith("scripts/benchmark/tile_torch/results/")
            and '#include <torch/csrc/inductor/cpp_prefix.h>' in source)


def violations(source: str):
    # C/C++ line splicing precedes tokenization, including in identifiers and
    # comments. Retain an offset map so diagnostics use original line numbers.
    chars, offsets = [], []
    for match in re.finditer(r'\\\r?\n|[\s\S]', source):
        if match.group().startswith("\\") and match.group().endswith("\n"):
            continue
        chars.append(match.group())
        offsets.append(match.start())
    for token in TOKENS.finditer("".join(chars)):
        if token.group() in FORBIDDEN:
            yield source.count("\n", 0, offsets[token.start()]) + 1, token.group()


def main() -> int:
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    checked = excluded = failures = 0
    for path in paths:
        file = ROOT / path
        suffix = file.suffix
        if suffix == ".in":
            suffix = file.with_suffix("").suffix
        if suffix not in SOURCE_SUFFIXES or not file.is_file():
            continue
        source = file.read_text(encoding="utf-8", errors="replace")
        if is_third_party(path, source):
            excluded += 1
            continue
        checked += 1
        for line, token in violations(source):
            print(f"{path}:{line}: project C++ must not use {token!r}", file=sys.stderr)
            failures += 1
    print(f"Checked {checked} project C/C++ files; excluded {excluded} third-party files.")
    return int(failures != 0)


if __name__ == "__main__":
    sys.exit(main())
