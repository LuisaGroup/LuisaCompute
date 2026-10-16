#!/usr/bin/env python3
"""Archive actual generated entries and bounded measurement evidence."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
FILES = [
    'include/luisa/tile/bridge/xir/lower.h', 'include/luisa/tile/bridge/xir/planner.h',
    'src/tile/bridge/xir/lower.cpp', 'src/tile/bridge/xir/planner.cpp', 'src/tile/bridge/xir/representation.h',
    'src/backends/simd/runtime/simd_tile.cpp', 'src/backends/simd/runtime/simd_shader.cpp',
    'src/backends/simd/simd_compiler.cpp', 'src/backends/simd/simd_compiler.h',
    'src/backends/simd/llvm/llvm_schedule_codegen.h', 'src/backends/simd/llvm/llvm_schedule_codegen.cpp',
    'src/backends/simd/llvm/llvm_schedule_emitter.h', 'src/backends/simd/llvm/llvm_schedule_emitter.cpp',
    'src/backends/simd/llvm/llvm_schedule_emitter_control.cpp', 'src/backends/simd/llvm/llvm_schedule_emitter_memory.cpp',
    'src/tests/unit/tile/bridge/test_xir.cpp', 'src/tests/unit/tile/bridge/test_xir_runtime.cpp',
    'src/tests/unit/tile/bridge/test_xir_llm.cpp', 'src/tests/unit/simd/test_llvm_schedule_codegen.cpp',
    'src/tests/benchmark/benchmark_tile_xir.cpp',
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--isolated', type=Path, required=True)
    args = parser.parse_args()
    source = args.isolated / 'source'
    hashes = {}
    for path in FILES:
        data = (ROOT / path).read_bytes()
        if data != (source / path).read_bytes():
            raise ValueError('untested source: ' + path)
        hashes[path] = digest(data)
    patch = subprocess.check_output(['git', 'diff', '3b96c263d', '--', *FILES], cwd=ROOT)
    save(HERE / 'source-overlay.patch.gz', gzip.compress(patch, mtime=0))
    provenance = dict(base_source='f5daf25e6', helper_checkpoint='db52dbc59', overlay_base='3b96c263d',
                      previous_checkpoint='3df09347e', tested_source_sha256=hashes, overlay_matches_isolated=True,
                      build_cache_sha256=digest((args.isolated / 'build/CMakeCache.txt').read_bytes()))
    save(HERE / 'provenance.json', (json.dumps(provenance, indent=2) + '\n').encode())
    for name in ('factorial', 'capture-final', 'native-final64', 'native-final1024'):
        directory = args.run / name
        save(HERE / name / 'results.json', (directory / 'results.json').read_bytes())
    # Keep all failures and stdout, but avoid duplicating hundreds of MB of
    # deterministic tensors. Fingerprint every raw input/output. Numerical
    # checks ran over complete arrays during measurement, not this archive.
    raw = {}
    for path in sorted((args.run / 'factorial').rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(args.run / 'factorial')
        if path.name.endswith('.f32'):
            raw[str(relative)] = dict(bytes=path.stat().st_size, sha256=digest(path.read_bytes()))
        elif path.name.endswith('.log'):
            save(HERE / 'factorial' / relative, path.read_bytes())
    save(HERE / 'raw-tensor-fingerprints.json', (json.dumps(raw, indent=2) + '\n').encode())
    for case in sorted((args.run / 'capture-final').iterdir()):
        if not case.is_dir():
            continue
        target = HERE / 'capture-final' / case.name
        for filename in ('measurement.json', 'kernel.ll', 'stderr.log', 'stdout.log'):
            data = (case / filename).read_bytes()
            save(target / (filename + '.gz' if filename.endswith('.ll') else filename), gzip.compress(data, mtime=0) if filename.endswith('.ll') else data)
        for suffix in ('o', 's'):
            paths = list((case / 'object').glob('*.' + suffix))
            if len(paths) != 1:
                raise ValueError('expected one actual ORC object/assembly')
            save(target / ('kernel.' + suffix + '.gz'), gzip.compress(paths[0].read_bytes(), mtime=0))
    for name in ('native-final64', 'native-final1024'):
        directory = args.run / name
        report = json.loads((directory / 'results.json').read_text())
        source = Path(report['artifacts']['inductor']['source'])
        save(HERE / name / 'inductor.cpp', source.read_bytes())
        save(HERE / name / 'inductor.so.gz', gzip.compress(source.with_suffix('.so').read_bytes(), mtime=0))
        for variant in ('baseline', 'candidate'):
            save(HERE / name / (variant + '.dylib.gz'), gzip.compress((directory / (variant + '.dylib')).read_bytes(), mtime=0))
        save(HERE / name / 'replay.dylib.gz', gzip.compress((directory / 'replay.dylib').read_bytes(), mtime=0))
    print('Archived', len(hashes), 'tested source files and all final visits')


if __name__ == '__main__':
    main()
