#!/usr/bin/env python3
"""Compile dumped HLSL (LUISA_DUMP_SOURCE=1) to DXIL with dxc.

The DX/VK backends dump HLSL sources when LUISA_DUMP_SOURCE=1:
  - DX: bin/debug/hlsl_output_<name>.hlsl (one shader per file)
  - VK: bin/debug/hlsl_output.hlsl (shaders appended, each re-emitting the
        immutable header, so the file is split on the header delimiter)

Each part is compiled with the same flags the DX backend passes to dxc
(see src/backends/common/hlsl/shader_compiler.cpp) except -spirv, so the
output is DXIL instead of SPIR-V.

Usage:
    python scripts/compile_dxil.py
    python scripts/compile_dxil.py bin/debug/hlsl_output.hlsl
    python scripts/compile_dxil.py --shader-model 6_6
    python scripts/compile_dxil.py --glob "bin/debug/hlsl_output_*.dxil.hlsl"
    python scripts/compile_dxil.py in.hlsl -o out.dxil --dxc D:/DirectXShaderCompiler/build/bin/dxc.exe
"""
import argparse
import glob as globmod
import os
import shutil
import subprocess
import sys

DEFAULT_DXC_CANDIDATES = (
    r'D:\DirectXShaderCompiler\build\bin\dxc.exe',  # local DirectXShaderCompiler build
    'dxc',                                        # global 'dxc' on PATH
    'dxc.exe',                                    # ./dxc.exe
    os.path.join('bin', 'debug', 'dxc.exe'),      # bin/debug/dxc.exe
)

# Start of the immutable HLSL header emitted by the codegen
# (src/backends/common/hlsl/builtin/hlsl_header.bytes).
HEADER_DELIMITER = '#define _INF_f (1.#INF)'


def find_dxc(explicit: str | None = None) -> str | None:
    """Locate the dxc executable."""
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    for cand in DEFAULT_DXC_CANDIDATES:
        # shutil.which handles both bare names (PATH lookup) and paths
        resolved = shutil.which(cand)
        if resolved and os.path.isfile(resolved):
            return resolved
    return None


def split_parts(content: str) -> list[str]:
    """Split a concatenated HLSL dump into self-contained parts.

    Each part gets the header delimiter re-prepended so it compiles on its own.
    """
    parts = []
    for part in content.split(HEADER_DELIMITER):
        part = part.strip()
        if not part:
            continue
        parts.append(HEADER_DELIMITER + '\n' + part + '\n')
    return parts


def compile_dxil(dxc_path: str, hlsl_file: str, dxil_file: str,
                 shader_model: str, entry: str, extra_args: list[str]) -> bool:
    cmd = [
        dxc_path,
        '-enable-16bit-types',
        '-all_resources_bound',
        '-Zpr',
        '-Gfa',
        '-HV', '2021',
        '-T', f'cs_{shader_model}',
        '-E', entry,
        '-O3',
        '-no-warnings',
        *extra_args,
        hlsl_file,
        '-Fo', dxil_file,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and os.path.isfile(dxil_file):
        print(f'Success: {dxil_file} ({os.path.getsize(dxil_file)} bytes)')
        return True
    print(f'Error compiling {hlsl_file}:')
    if result.stderr:
        print(result.stderr)
    if result.stdout:
        print(result.stdout)
    return False


def main():
    parser = argparse.ArgumentParser(
        description='Compile dumped HLSL to DXIL via dxc.')
    parser.add_argument('source', nargs='?', default=None,
                        help='HLSL file or output dir to compile '
                             '(default: bin/debug)')
    parser.add_argument('-o', '--output', default=None,
                        help='Destination .dxil path (single-file mode only)')
    parser.add_argument('--glob', default=None,
                        help='Glob for HLSL files to compile '
                             '(default: <source or bin/debug>/hlsl_output*.hlsl)')
    parser.add_argument('--shader-model', default='6_5',
                        help='dxc compute shader model suffix (default: 6_5; '
                             'use 6_6 for warp/async-copy kernels, 6_9 for tensor ops)')
    parser.add_argument('-E', '--entry', default='main',
                        help='Entry function name (default: main)')
    parser.add_argument('--dxc', default=None,
                        help='Path to dxc.exe (default: auto-search)')
    parser.add_argument('--split', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Split concatenated dumps on the HLSL header '
                             'delimiter and compile each part (default: on)')
    parser.add_argument('--dxc-arg', dest='dxc_args', action='append', default=[],
                        metavar='ARG',
                        help='Extra argument forwarded to dxc (repeatable). '
                             'Arguments after a literal "--" are also forwarded.')
    args, unknown = parser.parse_known_args()
    # Everything after an explicit "--" separator is forwarded to dxc.
    if unknown:
        if '--' in unknown:
            unknown = unknown[unknown.index('--') + 1:]
        args.dxc_args = list(args.dxc_args) + list(unknown)

    dxc_path = find_dxc(args.dxc)
    if dxc_path is None:
        print('Error: dxc not found. Searched: ' +
              ', '.join(DEFAULT_DXC_CANDIDATES) +
              '\nPass --dxc <path> to specify it explicitly.')
        sys.exit(1)
    print(f'Using dxc: {dxc_path}')

    extra_args = list(args.dxc_args)

    # Resolve the list of HLSL files to process.
    if args.source and os.path.isfile(args.source) and not args.glob:
        hlsl_files = [args.source]
    else:
        base = args.source if args.source else os.path.join('bin', 'debug')
        if os.path.isdir(base):
            pattern = args.glob or os.path.join(base, 'hlsl_output*.hlsl')
        else:
            pattern = args.glob or base
        hlsl_files = sorted(globmod.glob(pattern))
        if not hlsl_files:
            print(f'Error: no HLSL files matching {pattern}')
            print('Tip: run a test with LUISA_DUMP_SOURCE=1 to dump HLSL, e.g.')
            print('     xmake run test_dsl dx')
            sys.exit(1)

    compiled = 0
    attempted = 0
    for hlsl_file in hlsl_files:
        with open(hlsl_file, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        if args.output and len(hlsl_files) == 1:
            # Single explicit destination
            ok = compile_dxil(dxc_path, hlsl_file, args.output,
                              args.shader_model, args.entry, extra_args)
            attempted += 1
            compiled += int(ok)
            continue

        parts = split_parts(content) if args.split else [content]
        if len(parts) == 0:
            print(f'Skip (empty): {hlsl_file}')
            continue

        out_dir = os.path.dirname(hlsl_file) or '.'
        stem = os.path.basename(hlsl_file)
        # Strip the extension and any dialect tag: dumps are named
        # hlsl_output_<name>[_<dialect>].hlsl (e.g. "...dxil.hlsl", "...spv.hlsl").
        for ext in ('.hlsl', '.dxil', '.spv'):
            if stem.endswith(ext):
                stem = stem[:-len(ext)]

        for i, part in enumerate(parts):
            multi = len(parts) > 1
            suffix = f'_{i}' if multi else ''
            # A single part that matches the original file is compiled from it
            # directly (no redundant temp .hlsl).
            if multi or part.strip() != content.strip():
                part_file = os.path.join(out_dir, f'{stem}_part{suffix}.hlsl')
                with open(part_file, 'w', encoding='utf-8') as f:
                    f.write(part)
            else:
                part_file = hlsl_file
            dxil_file = os.path.join(out_dir, f'{stem}{suffix}.dxil')
            attempted += 1
            if compile_dxil(dxc_path, part_file, dxil_file,
                            args.shader_model, args.entry, extra_args):
                compiled += 1

    print(f'\nTotal compiled: {compiled}/{attempted}')
    if compiled == 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
