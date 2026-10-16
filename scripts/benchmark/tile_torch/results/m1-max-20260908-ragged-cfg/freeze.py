#!/usr/bin/env python3
"""Freeze the tested overlay before timing; preserve the isolated-base chain."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
FILES = (
    'src/backends/simd/llvm/llvm_schedule_emitter.h',
    'src/backends/simd/llvm/llvm_schedule_emitter_predication.cpp',
    'src/backends/simd/llvm/llvm_schedule_emitter_direct.cpp',
    'src/backends/simd/llvm/llvm_schedule_emitter_control.cpp',
    'src/backends/simd/llvm/llvm_schedule_emitter_terminators.cpp',
    'src/backends/simd/schedule/xir_to_schedule.h',
    'src/backends/simd/schedule/xir_to_schedule.cpp',
    'src/backends/simd/simd_compiler.cpp',
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
    predecessor = json.loads((HERE.parent / 'm1-max-20260908-task-grain/provenance.json').read_text())
    full_packet = json.loads((HERE.parent / 'm1-max-20260908-full-packet/provenance.json').read_text())
    inherited = full_packet['tested_source_sha256'] | predecessor['final_source_sha256']
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
                  isolated_baseline_context='../m1-max-20260908-task-grain/provenance.json',
                  tested_source_sha256=snapshots, inherited_source_sha256=inherited,
                  overlay_matches_isolated=True, binary_closure_sha256=closure,
                  cmake_cache_sha256=digest(args.isolated / 'build/CMakeCache.txt'),
                  runner_sha256=digest(HERE / 'measure.py'),
                  compiler_options='experimental predicated effects opt-in; full-packet on; reduction fusion off; no fast math',
                  default='predicated memory effects disabled; existing cohort-uniform header facts also admitted by direct CFG',
                  baseline_definition='same frozen binary and fixed execution mapping, experimental feature disabled',
                  tests={'ctest': '66/66 passed', 'codegen': 'including private triangles, empty/non-prefix masks, varying-loop fallback'},
                  source_caveat='isolated source archive with recorded predecessor overlays, not an asserted clean checkout of parent commit')
    (HERE / 'provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS: frozen current/inherited source identities and binary closure')


if __name__ == '__main__':
    main()
