#!/usr/bin/env python3
"""Freeze all visits, actual emitted code, baseline identity and source provenance."""
import argparse
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PREVIOUS = HERE.parent / 'm1-max-20260908-load-reduction'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == '.gz' else data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--isolated', type=Path, required=True)
    args = parser.parse_args()
    if (HERE / 'provenance.json').exists():
        raise ValueError('immutable checkpoint already archived')
    files = json.loads((PREVIOUS / 'shipping.json').read_text())['tested_source_sha256']
    hashes = {}
    for file in files:
        data = (ROOT / file).read_bytes()
        if data != (args.isolated / 'source' / file).read_bytes():
            raise ValueError('main/isolated source mismatch: ' + file)
        hashes[file] = digest(data)
    patch = subprocess.check_output(['git', 'diff', '3a012b547', '--', *files], cwd=ROOT)
    save(HERE / 'source-overlay.patch.gz', patch)
    provenance = dict(parent_commit='3a012b547', tested_source_sha256=hashes,
                      isolated_context='../m1-max-20260908-load-reduction/provenance.json',
                      overlay_matches_isolated=True, raw=str(args.raw),
                      cmake_cache_sha256=digest((args.isolated / 'build/CMakeCache.txt').read_bytes()),
                      defaults=dict(full_packet_specialization=False, load_reduction_fusion=False),
                      caveat='4096-instruction clone budget is a construction cap, not measured profitability')
    for file in ('ctest.log', 'final-ctest.log', 'final-build.log', 'final-syntax-test.log', *[p.name for p in args.raw.glob('syntax-*.log')]):
        save(HERE / file, (args.raw / file).read_bytes())
    save(HERE / 'ctest-details.log', (args.isolated / 'build/Testing/Temporary/LastTest.log').read_bytes())
    for name in ('measure.py', 'run_native.py', 'replay_native.py'):
        provenance[name + '_sha256'] = digest((HERE / name).read_bytes())
    replay = HERE.parent / 'm1-max-20260908-xir-packet-local'
    provenance['replay_source_sha256'] = {n: digest((replay / n).read_bytes()) for n in ('replay_native.py', 'replay_native.cpp')}
    fingerprints, logs, assembly, identities = {}, {}, [], []
    for section in ('capture', 'matrix'):
        report = json.loads((args.raw / section / 'results.json').read_text())
        save(HERE / (section + '.json.gz'), json.dumps(report, indent=2).encode() + b'\n')
        for directory in sorted((args.raw / section).iterdir()):
            if not directory.is_dir():
                continue
            label = f'{section}/{directory.name}'
            logs[label] = {n: (directory / n).read_text() for n in ('stdout.log', 'stderr.log')}
            for path in directory.glob('*.f32'):
                fingerprints[f'{label}/{path.name}'] = dict(bytes=path.stat().st_size, sha256=digest(path.read_bytes()))
            if section != 'capture':
                continue
            destination = HERE / section / directory.name
            save(destination / 'measurement.json', (directory / 'measurement.json').read_bytes())
            save(destination / 'kernel.ll.gz', (directory / 'kernel.ll').read_bytes())
            objects = list((directory / 'object').glob('*.o'))
            if len(objects) != 1:
                raise ValueError('expected one actual ORC object: ' + label)
            obj = objects[0]
            save(destination / 'kernel.o.gz', obj.read_bytes())
            disassembly = subprocess.check_output(['/opt/homebrew/opt/llvm@21/bin/llvm-objdump', '--macho', '--disassemble', '--no-show-raw-insn', str(obj)], text=True)
            save(destination / 'native.asm.gz', disassembly.encode())
            instructions = [line.split(':', 1)[1].strip().split()[0] for line in disassembly.splitlines() if re.match(r'\s*[0-9a-f]+:\s+\S', line)]
            assembly.append(dict(case=directory.name, symbols=re.findall(r'^(_llm_rows[^\n]*):$', disassembly, re.M),
                                 object_sha256=digest(obj.read_bytes()), instructions=len(instructions),
                                 conditional_branches=sum(i in ('tbz', 'tbnz', 'cbz', 'cbnz') or i.startswith('b.') for i in instructions)))
            if directory.name.endswith('-p0'):
                old_name = re.sub(r'-f([01])-p0$', r'-v\1', directory.name)
                old_dir = PREVIOUS / 'capture-final' / old_name
                same_object = obj.read_bytes() == gzip.decompress((old_dir / 'kernel.o.gz').read_bytes())
                same_llvm = (directory / 'kernel.ll').read_bytes() == gzip.decompress((old_dir / 'kernel.ll.gz').read_bytes())
                identities.append(dict(case=directory.name, previous_case=old_name, object_byte_identical=same_object, llvm_byte_identical=same_llvm))
    for rows, local, fusion in itertools.product((64, 1024), (1, 8), (0, 1)):
        name = f'native-{rows}-l{local}-f{fusion}'
        directory = args.raw / f'native-libc-{rows}-l{local}-f{fusion}'
        report = json.loads((directory / 'results.json').read_text())
        save(HERE / name / 'results.json', (directory / 'results.json').read_bytes())
        source = Path(report['artifacts']['inductor']['source'])
        save(HERE / name / 'inductor.cpp', source.read_bytes())
        save(HERE / name / 'inductor.so.gz', source.with_suffix('.so').read_bytes())
        for variant in ('baseline', 'candidate', 'replay'):
            save(HERE / name / (variant + '.dylib.gz'), (directory / (variant + '.dylib')).read_bytes())
    save(HERE / 'raw-tensor-fingerprints.json.gz', json.dumps(fingerprints, indent=2).encode())
    save(HERE / 'runtime-logs.json.gz', json.dumps(logs, indent=2).encode())
    save(HERE / 'assembly-summary.json', json.dumps(dict(metric='static_instruction_sites', scope='complete native text section',
         dynamic_profile=False, caveat='Static counts do not measure executed branches, stalls or spilling.', records=assembly), indent=2).encode())
    provenance['default_code_identity'] = identities
    preflight = []
    for directory in sorted(args.raw.glob('native-*')):
        if directory.name.startswith('native-libc-'):
            continue
        record = dict(directory=directory.name, included_in_performance_summary=False)
        if (directory / 'results.json').exists():
            data = (directory / 'results.json').read_bytes()
            save(HERE / 'preflight' / directory.name / 'results.json', data)
            record['completed_visits'] = len(json.loads(data)['results'])
        else:
            record.update(completed_visits=0, stopped='old replay rejects compiler-emitted _memcpy before timing')
        preflight.append(record)
    save(HERE / 'preflight.json', json.dumps(dict(records=preflight,
         reason='Original no-import replay aborted on the first local p1 object. All eight cells were rerun with the exact memcpy allowlist, not selected by timings.'), indent=2).encode())
    save(HERE / 'provenance.json', json.dumps(provenance, indent=2).encode() + b'\n')
    print('Archived', len(hashes), 'source hashes,', len(logs), 'Runtime visits,', len(fingerprints), 'tensor digests')


if __name__ == '__main__':
    main()
