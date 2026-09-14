#!/usr/bin/env python3
"""Archive actual objects, complete visit records, and tested source identities."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PREVIOUS = HERE.parent / 'm1-max-20260908-xir-packet-local'
NATIVE = ('native-64-l1', 'native-64-l8', 'native-1024-l1', 'native-1024-l8')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, data, compress=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(data, mtime=0) if compress else data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--isolated', type=Path, required=True)
    args = parser.parse_args()
    previous = json.loads((PREVIOUS / 'provenance.json').read_text())
    files = sorted(previous['tested_source_sha256'])
    hashes = {}
    for file in files:
        data = (ROOT / file).read_bytes()
        if data != (args.isolated / 'source' / file).read_bytes():
            raise ValueError('source differs from tested overlay: ' + file)
        hashes[file] = digest(data)
    patch = subprocess.check_output(['git', 'diff', '30cadb4a4', '--', *files], cwd=ROOT)
    save(HERE / 'source-overlay.patch.gz', patch, True)
    provenance = dict(previous_checkpoint='30cadb4a4', isolated_base=previous,
                      tested_source_sha256=hashes, overlay_matches_isolated=True,
                      build_cache_sha256=digest((args.isolated / 'build/CMakeCache.txt').read_bytes()),
                      native_helper_source=str(PREVIOUS / 'replay_native.cpp'),
                      native_helper_sha256=digest((PREVIOUS / 'replay_native.cpp').read_bytes()),
                      initial_candidate='rejected masked-vector address reconstruction; actual LLVM/object retained, no C++ source snapshot')
    save(HERE / 'provenance.json', (json.dumps(provenance, indent=2) + '\n').encode())
    fingerprints = {}
    for name in ('capture', 'capture-final', 'matrix'):
        directory = args.run / name
        for path in sorted(directory.rglob('*')):
            if not path.is_file():
                continue
            relative = path.relative_to(directory)
            if path.name.endswith('.f32'):
                fingerprints[f'{name}/{relative}'] = dict(bytes=path.stat().st_size, sha256=digest(path.read_bytes()))
            elif path.suffix in ('.json', '.log'):
                save(HERE / name / relative, path.read_bytes())
            elif path.name == 'kernel.ll':
                save(HERE / name / relative.with_suffix('.ll.gz'), path.read_bytes(), True)
            elif path.suffix in ('.o', '.s') and path.parent.name == 'object':
                save(HERE / name / relative.parent.parent / f'kernel{path.suffix}.gz', path.read_bytes(), True)
    for name in NATIVE:
        directory = args.run / name
        report = json.loads((directory / 'results.json').read_text())
        save(HERE / name / 'results.json', (directory / 'results.json').read_bytes())
        source = Path(report['artifacts']['inductor']['source'])
        save(HERE / name / 'inductor.cpp', source.read_bytes())
        save(HERE / name / 'inductor.so.gz', source.with_suffix('.so').read_bytes(), True)
        assembly = subprocess.check_output(['/opt/homebrew/opt/llvm@21/bin/llvm-objdump', '--disassemble', '--no-show-raw-insn', str(source.with_suffix('.so'))])
        save(HERE / name / 'inductor.asm.gz', assembly, True)
        for variant in ('baseline', 'candidate', 'replay'):
            save(HERE / name / f'{variant}.dylib.gz', (directory / f'{variant}.dylib').read_bytes(), True)
    save(HERE / 'raw-tensor-fingerprints.json', (json.dumps(fingerprints, indent=2) + '\n').encode())
    save(HERE / 'ctest.log', (args.isolated / 'build/Testing/Temporary/LastTest.log').read_bytes())
    print('Archived', len(hashes), 'source identities and', len(fingerprints), 'tensor fingerprints')


if __name__ == '__main__':
    main()
