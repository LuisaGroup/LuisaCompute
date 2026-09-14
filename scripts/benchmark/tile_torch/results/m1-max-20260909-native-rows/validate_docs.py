#!/usr/bin/env python3
"""Validate benchmark helpers and the existing Sphinx report surface."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
NODE = '/Users/mike/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--html', type=Path, required=True)
    parser.add_argument('--visual', type=Path, required=True)
    parser.add_argument('--label', default='validation')
    args = parser.parse_args()
    if not args.label.replace('-', '').isalnum():
        raise ValueError('invalid receipt label')
    receipt = HERE / (args.label + '.json')
    if args.html.exists() or args.visual.exists() or receipt.exists():
        raise ValueError('fresh output/receipt paths required')
    audit = json.loads((HERE / 'audit.json').read_text())
    assert audit['status'] == 'passed' and audit['native_timed_visits'] == 432
    report = dict(assessment='Share with caveats: fixed opt-in native entries; 12 winning and 12 losing cases; no new compiler optimization.',
                  audience='technical', surface='existing repository Sphinx docs, as requested',
                  structure='current results; exact metric and ABI; numerical contracts; generated-code observations; generic candidate priorities; evidence and limits',
                  visual_choice='Exact comparison tables: six operator ranges in Sphinx, all 24 shapes/paired ranges in the linked table; no misleading cross-size absolute-time chart.',
                  caveats=['fixed mappings, not calibrated planner choices', 'native callbacks/traversal/internal allocations remain timed',
                           'different variance/reduction/math implementations', 'finite deterministic inputs, not a complete numerical stress suite',
                           'six orders are descriptive, not confidence intervals', 'no new Metal/MPS/GEMM/attention or E2E measurements'],
                  compiler_tests='No compiler source changes. Full build before capture recorded in build-capture-prepare-logs.json.gz. Predecessor CTest is not relabeled as a new run.',
                  data_validation='audit.json: 48 capture and 72 unique native outputs re-read; 432 visits recalculated; nine invalid evidence mutations rejected.',
                  incomplete_handoff_blockers=[], commands=[])
    logs = {}
    uv = ['uv', 'run', '--offline', '--no-project', '--python', '3.13']
    commands = [
        ('python-tests', uv + ['--with', 'numpy', 'python', '-m', 'unittest', 'discover', '-s', 'scripts/benchmark/tile_torch', '-p', 'test_*.py', '-v']),
        ('cpp-syntax', ['clang++', '-std=c++20', '-Wall', '-Wextra', '-Werror', '-fsyntax-only', '-Isrc', 'scripts/benchmark/tile_torch/native_rows_replay.cpp']),
        ('cpp-format', ['clang-format', '--dry-run', '--Werror', 'scripts/benchmark/tile_torch/native_rows_replay.cpp']),
        ('doxygen', ['doxygen', 'docs/Doxyfile']),
        ('sphinx', uv + ['--with', 'sphinx', '--with', 'sphinx-rtd-theme', '--with', 'myst-parser', '--with', 'breathe',
                         'sphinx-build', '-b', 'html', '-W', '--keep-going', 'docs', str(args.html)]),
        ('links', uv + ['--with', 'sphinx', 'python', 'scripts/check_docs.py', str(args.html)]),
        ('visual', [NODE, str(HERE / 'qa_docs.cjs'), str(args.html), str(args.visual)]),
    ]
    for name, command in commands:
        start = time.monotonic()
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        logs[name] = dict(stdout=result.stdout, stderr=result.stderr)
        (HERE / (args.label + '-logs.json.gz')).write_bytes(gzip.compress((json.dumps(logs, indent=2) + '\n').encode(), mtime=0))
        report['commands'].append(dict(name=name, command=command, exit_code=result.returncode,
                                       elapsed_seconds=time.monotonic() - start,
                                       warning_lines=sum('warning:' in line.lower() for line in result.stderr.splitlines())))
        receipt.write_text(json.dumps(report, indent=2) + '\n')
        print(name, 'exit', result.returncode, flush=True)
        if result.returncode:
            raise RuntimeError(result.stderr[-4000:] + result.stdout[-4000:])
    (HERE / (args.label + '-visual.json')).write_bytes((args.visual / 'receipt.json').read_bytes())
    paths = ['docs/source/performance/tile/results.md', 'docs/source/performance/tile/index.md',
             'scripts/benchmark/tile_torch/README.md', 'scripts/benchmark/tile_torch/native_rows.py',
             'scripts/benchmark/tile_torch/native_rows_replay.cpp', 'scripts/benchmark/tile_torch/test_native_rows.py',
             'scripts/benchmark/tile_torch/audit_native_rows.py']
    report['source_sha256'] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
    report['complete'] = True
    receipt.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
