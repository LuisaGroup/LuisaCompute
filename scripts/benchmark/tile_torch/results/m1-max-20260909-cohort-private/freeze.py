#!/usr/bin/env python3
"""Freeze the tested overlay before timing; preserve the isolated-base chain."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
FILES = (
    'src/backends/simd/schedule/xir_to_schedule.h',
    'src/backends/simd/schedule/xir_to_schedule.cpp',
    'src/backends/simd/schedule/schedule_ir.h',
    'src/backends/simd/schedule/schedule_ir.cpp',
    'src/backends/simd/llvm/llvm_schedule_emitter_memory.cpp',
    'src/backends/simd/simd_compiler.cpp',
    'src/tests/unit/simd/test_xir_to_schedule.cpp',
    'src/tests/unit/simd/test_schedule_ir.cpp',
    'src/tests/unit/simd/test_llvm_schedule_codegen.cpp',
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--isolated', type=Path, required=True)
    args = parser.parse_args()
    if (HERE / 'provenance.json').exists():
        raise ValueError('immutable source freeze; use a new evidence directory')
    test_log = args.raw / 'ctest-all.log'
    test_text = test_log.read_text()
    test_results = dict(re.findall(r'^\s*\d+/\d+ Test\s+#\d+: (\S+)\s+\.{2,}\s*(.+)$', test_text, re.M))
    assert len(test_results) == 209, 'complete broad CTest results required'
    relevant = {n: s for n, s in test_results.items() if n.startswith(('test_simd_', 'simd_test_', 'test_tile_xir'))}
    assert len(relevant) == 64 and all(s.startswith('Passed') for s in relevant.values())
    failed = {n: s for n, s in test_results.items() if not s.startswith('Passed')}
    assert set(failed) == {'metal_example_coro_external_stage_debug', 'metal_example_coro_external_stage_neural_sdf',
                           'tutorial_01_mandelbrot', 'tutorial_02_path_tracing'}
    recheck = args.raw / 'ctest-metal-recheck.log'
    assert '100% tests passed out of 2' in recheck.read_text()
    predecessor = json.loads((HERE.parent / 'm1-max-20260908-ragged-cfg/provenance.json').read_text())
    inherited = predecessor['inherited_source_sha256'] | predecessor['tested_source_sha256']
    inherited = {f: h for f, h in inherited.items() if f not in FILES}
    for file, sha in inherited.items():
        assert digest(ROOT / file) == sha == digest(args.isolated / 'source' / file), file
    snapshots = {}
    for file in (*FILES, 'src/tests/common/tile_llm_benchmark.h'):
        assert (ROOT / file).read_bytes() == (args.isolated / 'source' / file).read_bytes(), file
        snapshots[file] = digest(ROOT / file)
    binary = args.isolated / 'build/bin/benchmark_tile_xir'
    closure = {str(p.resolve()): digest(p) for p in [binary, *sorted(p for p in binary.parent.glob('libluisa-*') if p.is_file())]}
    patch = subprocess.check_output(['git', 'diff', '--', *FILES], cwd=ROOT)
    (HERE / 'source-overlay.patch.gz').write_bytes(gzip.compress(patch, mtime=0))
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode='w') as tar:
        for file in sorted(snapshots):
            data = (ROOT / file).read_bytes()
            info = tarfile.TarInfo(file)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    (HERE / 'measured-sources.tar.gz').write_bytes(gzip.compress(archive.getvalue(), mtime=0))
    report = dict(parent_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  frozen_unix=time.time(), raw=str(args.raw.resolve()), isolated=str(args.isolated.resolve()),
                  isolated_baseline_context='../m1-max-20260908-ragged-cfg/provenance.json',
                  tested_source_sha256=snapshots, inherited_source_sha256=inherited,
                  overlay_matches_isolated=True, binary_closure_sha256=closure,
                  cmake_cache_sha256=digest(args.isolated / 'build/CMakeCache.txt'),
                  runner_sha256=digest(HERE / 'measure.py'),
                  compiler_options='cohort private access opt-in; predicated effects and full-packet on; reduction fusion off; no fast math',
                  default='cohort private access disabled; same-block consuming epoch required',
                  baseline_definition='same frozen binary and fixed execution mapping, experimental feature disabled',
                  tests={'ctest': '205/209 initial broad run; 64/64 SIMD/Tile XIR checks passed',
                         'initial_failures': failed, 'ctest_log_sha256': digest(test_log),
                         'metal_recheck': '2/2 timeout cases pass unchanged', 'metal_recheck_log_sha256': digest(recheck),
                         'remaining_failures': 'two tutorials require fallback backend, absent from this isolated build',
                         'codegen': 'including equal-index private loops, divergent exits, cross-block and varying-start rejection, empty/non-prefix masks'},
                  source_caveat='isolated source archive with recorded predecessor overlays, not an asserted clean checkout of parent commit')
    (HERE / 'provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS: frozen current/inherited source identities and binary closure')


if __name__ == '__main__':
    main()
