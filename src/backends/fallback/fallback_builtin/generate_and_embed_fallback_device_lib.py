#!/usr/bin/env python3

"""Build the fallback device API bitcode with the selected LLVM toolchain.

The fallback backend parses this module with LLVM's C++ API. LLVM bitcode is
not an ABI-stable interchange format across downstream LLVM distributions, so
the producer must be the clang/llvm-as pair belonging to the same LLVM package
that is linked into the backend. CMake resolves those tools and passes their
absolute paths here.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import tempfile


_WRAPPER_PREFIX = "luisa_fallback_wrapper_"
_IMPLEMENTATION_PREFIX = "luisa_fallback_"


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clang", required=True, type=pathlib.Path)
    parser.add_argument("--llvm-as", required=True, type=pathlib.Path)
    parser.add_argument("--source", required=True, type=pathlib.Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--deployment-target", default="")
    parser.add_argument("--output-ll", required=True, type=pathlib.Path)
    parser.add_argument("--output-bc", required=True, type=pathlib.Path)
    parser.add_argument("--output-symbol-map", required=True, type=pathlib.Path)
    return parser.parse_args()


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _normalize_ir(raw_ir: str) -> tuple[str, list[str]]:
    # llvm.used only keeps the source-level wrappers alive until this point;
    # every wrapper is linked by its canonical luisa.* name after generation.
    lines = [
        line
        for line in raw_ir.splitlines()
        if not line.strip().startswith("@llvm.used")
        and not line.strip().startswith("@llvm.compiler.used")
        and not line.strip().startswith("; ModuleID")
        and not line.strip().startswith("source_filename")
    ]
    ir = "\n".join(lines) + "\n"

    wrapper_pattern = re.compile(
        rf"@{re.escape(_WRAPPER_PREFIX)}([A-Za-z0-9_]+)(?=\()"
    )
    wrapper_names = list(dict.fromkeys(wrapper_pattern.findall(ir)))
    if not wrapper_names:
        raise RuntimeError("no fallback device API wrappers found in generated IR")

    ir = wrapper_pattern.sub(
        lambda match: "@luisa." + match.group(1).replace("_", "."), ir
    )
    # Hidden definitions cannot be internalized by the linker. The wrappers
    # are implementation details and are reached through generated luisa.*
    # calls only, so private linkage is the exact intended visibility.
    ir = re.sub(r"\bdefine(?:\s+dso_local)?\s+hidden\b", "define private", ir)

    mapped_symbols: list[str] = []
    for wrapper_name in wrapper_names:
        implementation_pattern = re.compile(
            rf"@{re.escape(_IMPLEMENTATION_PREFIX + wrapper_name)}(?=\()"
        )
        dotted_name = wrapper_name.replace("_", ".")
        ir, replacement_count = implementation_pattern.subn(
            f"@luisa.{dotted_name}.impl", ir
        )
        if replacement_count:
            mapped_symbols.append(wrapper_name)

    return ir, mapped_symbols


def _write_if_different(path: pathlib.Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == content:
        return
    path.write_bytes(content)


def _generate_symbol_map(wrapper_names: list[str]) -> bytes:
    lines = []
    for wrapper_name in wrapper_names:
        dotted_name = wrapper_name.replace("_", ".")
        lines.append(
            f'map_symbol("luisa.{dotted_name}.impl", '
            f'&api::luisa_fallback_{wrapper_name});'
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def main() -> None:
    args = _parse_arguments()
    for tool in (args.clang, args.llvm_as):
        if not tool.is_file():
            raise FileNotFoundError(f"LLVM tool does not exist: {tool}")
    if not args.source.is_file():
        raise FileNotFoundError(f"wrapper source does not exist: {args.source}")

    args.output_ll.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="fallback-device-api-", dir=args.output_ll.parent
    ) as temporary_directory:
        raw_ll = pathlib.Path(temporary_directory) / "raw.ll"
        command = [
                str(args.clang),
                "-emit-llvm",
                "-std=c++20",
                "-ffast-math",
                "-O3",
                "-S",
                str(args.source),
                "-o",
                str(raw_ll),
                "-fomit-frame-pointer",
                "-fno-stack-protector",
                "-fno-rtti",
                "-fno-exceptions",
                f"--target={args.target}",
                "-nostdinc",
                "-nostdlib",
                "-nostdinc++",
                "-nostdlib++",
            ]
        if args.deployment_target:
            command.append(f"-mmacosx-version-min={args.deployment_target}")
        _run(command)
        normalized_ir, mapped_symbols = _normalize_ir(
            raw_ll.read_text(encoding="utf-8")
        )
        _write_if_different(args.output_ll, normalized_ir.encode("utf-8"))
        _write_if_different(
            args.output_symbol_map, _generate_symbol_map(mapped_symbols)
        )
        # No textual-IR fallback is permitted here. If llvm-as rejects clang's
        # output, the producer package is internally inconsistent and the
        # build must fail before an incompatible payload reaches the runtime.
        _run(
            [
                str(args.llvm_as),
                str(args.output_ll),
                "-o",
                str(args.output_bc),
            ]
        )


if __name__ == "__main__":
    main()
