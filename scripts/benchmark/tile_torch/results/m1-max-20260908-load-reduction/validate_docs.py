#!/usr/bin/env python3
"""Rebuild Doxygen/Sphinx; C++ results below describe this recorded checkpoint.

This script verifies the archived CTest log, but does not rerun the C++ build,
CTest, or clangd. Reproduce those separately before using a changed source.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
NODE = '/Users/mike/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--html', type=Path, required=True)
    parser.add_argument('--visual', type=Path, required=True)
    args = parser.parse_args()
    if args.html.exists() or args.visual.exists():
        raise ValueError('fresh HTML and visual output directories required')
    ctest = (HERE / 'shipping-ctest.log').read_text()
    if len(re.findall(r'^Test Passed\.$', ctest, re.M)) != 38 or 'Test Failed.' in ctest:
        raise ValueError('expected all 38 selected tests to pass')
    report = dict(full_build=dict(command='cmake --build /tmp/luisa-reduction-checkpoint.gQUERp/build --parallel 8', exit_code=0),
                  ctest=dict(passed=38, real_seconds=216.81, log='shipping-ctest.log'),
                  syntax=dict(tool='scripts/check_cpp_syntax.py + LLVM 21 clangd, isolated compile_commands.json',
                              passed=['lower.cpp', 'planner.cpp', 'simd_tile.cpp', 'test_xir.cpp', 'test_xir_runtime.cpp']),
                  formatting='bridge files and changed test regions checked', documentation=[])
    save = lambda: (HERE / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    uv = ['uv', 'run', '--offline', '--no-project', '--python', '3.13']
    commands = [
        ('doxygen', ['doxygen', 'docs/Doxyfile']),
        ('sphinx', uv + ['--with', 'sphinx', '--with', 'sphinx-rtd-theme', '--with', 'myst-parser', '--with', 'breathe',
                         'sphinx-build', '-b', 'html', '-W', '--keep-going', 'docs', str(args.html)]),
        ('links', uv + ['--with', 'sphinx', 'python', 'scripts/check_docs.py', str(args.html)]),
        ('visual', [NODE, str(HERE / 'qa_docs.cjs'), str(args.html), str(args.visual)]),
    ]
    for name, command in commands:
        start = time.monotonic()
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        (HERE / f'{name}.stdout.log').write_text(result.stdout)
        (HERE / f'{name}.stderr.log').write_text(result.stderr)
        report['documentation'].append(dict(name=name, command=command, exit_code=result.returncode,
                                            elapsed_seconds=time.monotonic() - start,
                                            warning_lines=sum('warning:' in line.lower() for line in result.stderr.splitlines())))
        save()
        print(name, 'exit', result.returncode, flush=True)
        if result.returncode:
            raise RuntimeError(result.stderr[-4000:] + result.stdout[-4000:])
    (HERE / 'docs-qa.json').write_bytes((args.visual / 'receipt.json').read_bytes())
    report['documentation_source_sha256'] = {
        file: hashlib.sha256((ROOT / file).read_bytes()).hexdigest() for file in (
            'docs/source/internals/tile/xir.md', 'docs/source/performance/tile/results.md', 'docs/source/performance/tile/index.md')}
    report['complete'] = True
    save()


if __name__ == '__main__':
    main()
