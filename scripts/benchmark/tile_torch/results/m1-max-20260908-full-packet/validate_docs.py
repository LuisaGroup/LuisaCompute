#!/usr/bin/env python3
"""Validate the existing Sphinx surface after the source/evidence audit."""
import argparse
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
    args = parser.parse_args()
    if args.html.exists() or args.visual.exists():
        raise ValueError('fresh output directories required')
    audit = json.loads((HERE / 'audit.json').read_text())
    assert audit['passed'] and audit['full_outputs_rechecked'] == 256
    report = dict(ctest='final-ctest.log', build='final-build.log',
                  source_audit='provenance.json', evidence_audit='audit.json',
                  audience='technical', surface='existing repository Sphinx docs, as requested by user',
                  structure='technical summary; definitions; factorial native results; E2E limits; model implications; follow-ups',
                  visuals='exact cell-lookup tables for multiple fixed factors, plus native text diagrams; no cross-size absolute-time chart',
                  documentation=[])
    save = lambda: (HERE / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
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
                                            warning_lines=sum('warning:' in s.lower() for s in result.stderr.splitlines())))
        save()
        print(name, 'exit', result.returncode, flush=True)
        if result.returncode:
            raise RuntimeError(result.stderr[-4000:] + result.stdout[-4000:])
    (HERE / 'docs-qa.json').write_bytes((args.visual / 'receipt.json').read_bytes())
    report['source_sha256'] = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in (
        'docs/source/internals/tile/xir.md', 'docs/source/performance/tile/results.md', 'docs/source/performance/tile/index.md')}
    report['complete'] = True
    save()


if __name__ == '__main__':
    main()
