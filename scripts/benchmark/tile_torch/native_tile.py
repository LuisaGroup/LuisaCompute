#!/usr/bin/env python3
"""Prepare frozen migrated FP32 Tile ORC objects and replay native entries.

Preparation links actual captured objects, never recompiles their LLVM. Replay
uses one common C++ timer and an ABBA sequence per cycle. No Runtime, Python,
JIT or caller allocation executes inside that timer. Inputs and complete FP64
oracles come from benchmark_tile_migrated, not operator-name reconstruction.
"""
from __future__ import annotations

import argparse
import array
import ctypes as c
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import shutil
import statistics
import struct
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LLVM = Path('/opt/homebrew/opt/llvm@21/bin')
ALLOWED_IMPORTS = {'_memcpy', '_memset', '_bzero', '___chkstk_darwin'}
HELPER_SOURCE = HERE / 'native_tile_replay.cpp'
ABI_HEADER = ROOT / 'src/backends/simd/llvm/llvm_schedule_codegen.h'


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def command(argv, directory, stem):
    result = subprocess.run(list(map(str, argv)), capture_output=True, text=True, timeout=300)
    (directory / (stem + '.stdout')).write_text(result.stdout)
    (directory / (stem + '.stderr')).write_text(result.stderr)
    save(directory / (stem + '.command.json'), dict(argv=list(map(str, argv)), returncode=result.returncode))
    if result.returncode:
        raise RuntimeError(f'{stem}: exit {result.returncode}: {result.stderr[-2000:]}')
    return result.stdout


def numbers(path, code):
    result = array.array(code)
    result.frombytes(Path(path).read_bytes())
    return result


def check_output(actual, expected, atol, rtol):
    if len(actual) != len(expected) or not expected:
        raise ValueError('output/oracle extent mismatch')
    maximum = 0.0
    for index, (value, reference) in enumerate(zip(actual, expected)):
        error = abs(value - reference)
        if not math.isfinite(value) or not math.isfinite(reference) or error > atol + rtol * abs(reference):
            raise ValueError(f'full oracle mismatch at element {index}: {value} vs {reference}')
        maximum = max(maximum, error)
    return dict(elements=len(expected), max_abs_error=maximum, atol=atol, rtol=rtol)


def prepare(args):
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    report = dict(status='preparing', format='native-tile-entry-v1', name=args.name, started_unix=time.time())
    save(directory / 'prepared.json', report)
    try:
        if platform.system() != 'Darwin' or platform.machine() != 'arm64':
            raise ValueError('this inspected object-link path currently requires Darwin arm64')
        rows = [json.loads(line) for line in args.log.read_text().splitlines() if line.startswith('{')]
        if len(rows) != 1:
            raise ValueError('expected exactly one benchmark JSON record')
        metadata = rows[0]
        if (metadata.get('implementation') != 'tile_xir_simd' or metadata.get('precision') != 'fp32' or
                metadata.get('source_kind') != 'tile_lowering_source' or metadata.get('fast_math') is not False or
                metadata.get('correctness', {}).get('checks') != 2):
            raise ValueError('expected fully checked, strict FP32 migrated XIR capture')
        dispatch = metadata['dispatch']
        if len(dispatch) != 3 or any(type(x) is not int or not 0 < x <= 0xffffffff for x in dispatch):
            raise ValueError('invalid actual dispatch metadata')
        realization = metadata['realization']
        match = re.search(r'\bW(\d+), (\d+) workers/block', realization)
        workspace = re.search(r'\bprivate_workspace_bytes=(\d+)', realization)
        if not match or not workspace:
            raise ValueError('missing actual packet/block/workspace metadata')
        width, block = map(int, match.groups())
        if width not in (2, 4, 8, 16, 32, 64) or block == 0 or block % width:
            raise ValueError('unsupported packet-batch launch')
        prefix = args.prefix.resolve()
        source_path = Path(str(prefix) + '.source.txt')
        source = source_path.read_text()
        symbols = re.findall(r'^define dso_local void @([-a-zA-Z$._0-9]+\.packet_batch(?:\.blocks)?)\(', source, re.M)
        blocks = [s for s in symbols if s.endswith('.blocks')]
        symbols = blocks if blocks else symbols
        if len(symbols) != 1 or re.search(r'cooperative[._]|@llvm\.coro\.', source):
            raise ValueError('expected one non-cooperative packet-batch entry')
        symbol, abi = symbols[0], 0 if blocks else 2
        objects = list(args.objects.resolve().glob('*.o'))
        if len(objects) != 1:
            raise ValueError('object directory must contain exactly one captured ORC object')
        original = [args.log.resolve(), source_path, objects[0], HELPER_SOURCE, ABI_HEADER, Path(__file__).resolve()]
        inputs = [Path(str(prefix) + '.input0.f32')]
        second = Path(str(prefix) + '.input1.f32')
        if second.exists():
            inputs.append(second)
        oracle = Path(str(prefix) + '.expected.f64')
        output = Path(str(prefix) + '.output.f32')
        original += [*inputs, oracle, output]
        original_hashes = {str(p): sha(p) for p in original}
        for path in inputs:
            if not path.stat().st_size or path.stat().st_size % 4:
                raise ValueError('invalid FP32 input extent')
        actual, expected = numbers(output, 'f'), numbers(oracle, 'd')
        check = metadata['correctness']
        validation = check_output(actual, expected, check['atol'], check['rtol'])
        if validation['elements'] != check['elements_per_check']:
            raise ValueError('capture correctness extent differs from oracle')
        for index, path in enumerate(inputs):
            shutil.copy2(path, directory / f'input{index}.f32')
        for source_file, name in ((oracle, 'expected.f64'), (output, 'captured.f32'), (source_path, 'kernel.ll'),
                                  (objects[0], 'kernel.o'), (args.log, 'capture.log'), (HELPER_SOURCE, 'native_tile_replay.cpp'),
                                  (Path(__file__), 'native_tile.py')):
            shutil.copy2(source_file, directory / name)
        header = directory / 'backends/simd/llvm/llvm_schedule_codegen.h'
        header.parent.mkdir(parents=True)
        shutil.copy2(ABI_HEADER, header)
        imports = command([LLVM / 'llvm-nm', '--undefined-only', '--just-symbol-name', directory / 'kernel.o'], directory, 'imports').split()
        if set(imports) - ALLOWED_IMPORTS:
            raise ValueError('uninspected ORC imports: ' + str(imports))
        exported = command([LLVM / 'llvm-nm', '--defined-only', '--extern-only', '--just-symbol-name', directory / 'kernel.o'], directory, 'exports').split()
        if '_' + symbol not in exported:
            raise ValueError('LLVM entry is not exported by the actual captured object')
        command(['clang++', '--version'], directory, 'compiler')
        command(['clang++', '-dynamiclib', directory / 'kernel.o', '-o', directory / 'kernel.dylib'], directory, 'link')
        command(['clang++', '-std=c++20', '-O3', '-dynamiclib', '-I' + str(directory), directory / 'native_tile_replay.cpp',
                 '-o', directory / 'replay.dylib'], directory, 'helper')
        if original_hashes != {str(p): sha(p) for p in original}:
            raise ValueError('capture or helper source changed during preparation')
        report.update(status='prepared', finished_unix=time.time(), metadata=metadata, symbol=symbol, abi=abi,
                      packet_width=width, block=[block, 1, 1], dispatch=dispatch, workspace_bytes=int(workspace[1]),
                      input_files=[f'input{i}.f32' for i in range(len(inputs))], output_elements=len(expected),
                      atol=check['atol'], rtol=check['rtol'], capture_correctness=validation,
                      original_sha256=original_hashes, imports=imports,
                      files={str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob('*'))
                             if p.is_file() and p.name != 'prepared.json'})
        save(directory / 'prepared.json', report)
        print(directory / 'prepared.json', flush=True)
    except Exception as error:
        report.update(status='error', error=str(error), finished_unix=time.time())
        save(directory / 'prepared.json', report)
        raise


def verify(manifest):
    path = Path(manifest)
    value = json.loads(path.read_text())
    if value.get('status') != 'prepared' or value.get('format') != 'native-tile-entry-v1':
        raise ValueError('incomplete/unsupported preparation')
    for relative, digest in value['files'].items():
        file = path.parent / relative
        if Path(relative).is_absolute() or '..' in Path(relative).parts or file.is_symlink() or sha(file) != digest:
            raise ValueError('prepared artifact changed: ' + relative)
    return value


def replay(args):
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    manifests = [path.resolve() for path in args.prepared]
    report = dict(status='running', started_unix=time.time(), runner_sha256=sha(__file__),
                  prepared_sha256={str(path): sha(path) for path in manifests}, platform=platform.platform(),
                  metric='validation_only' if args.validate_only else 'single_thread_native_entry_host_wall_us',
                  boundary='Common C++ timer; native entries, launch resets, block traversal and compiler-emitted libc/allocations included. Runtime/Python/JIT/caller allocation/validation excluded.',
                  cpu_threads=1, order_policy='ABBA per cycle; two matched pairs per cycle', options=vars(args).copy(), visits=[])
    report['options'] = {key: str(value) if isinstance(value, Path) else [str(p) for p in value] if key == 'prepared' else value
                         for key, value in report['options'].items()}
    save(directory / 'results.json', report)
    try:
        if len(manifests) != 2 or len(set(manifests)) != 2:
            raise ValueError('provide two distinct prepared entries for one case')
        entries = [verify(path) for path in manifests]
        if entries[0]['name'] == entries[1]['name']:
            raise ValueError('variant names must differ')
        for key in ('operation', 'dimensions', 'precision', 'fast_math', 'source_schedule', 'block', 'input_distribution'):
            if entries[0]['metadata'][key] != entries[1]['metadata'][key]:
                raise ValueError('source case differs: ' + key)
        for key in ('input_files', 'output_elements', 'atol', 'rtol'):
            if entries[0][key] != entries[1][key]:
                raise ValueError('validation contract differs: ' + key)
        for relative in [*entries[0]['input_files'], 'expected.f64', 'native_tile_replay.cpp', 'backends/simd/llvm/llvm_schedule_codegen.h']:
            if entries[0]['files'][relative] != entries[1]['files'][relative]:
                raise ValueError('input/oracle/helper ABI differs: ' + relative)
        p, u32, size, d = c.c_void_p, c.c_uint32, c.c_size_t, c.c_double
        libraries = [c.CDLL(str(path.parent / 'kernel.dylib')) for path in manifests]
        helper_library = c.CDLL(str(manifests[0].parent / 'replay.dylib'))
        helper = helper_library.replay_native_tile
        helper.argtypes = [p, u32, u32, c.POINTER(p), c.POINTER(size), c.POINTER(u32), c.POINTER(c.POINTER(d)),
                          c.POINTER(u32), c.POINTER(u32), u32, size, u32, u32, u32, d, d, c.POINTER(d), c.POINTER(c.c_uint64), c.POINTER(d)]
        helper.restype = c.c_int
        expected_bytes = (manifests[0].parent / 'expected.f64').read_bytes()
        expected = (d * entries[0]['output_elements']).from_buffer_copy(expected_bytes)
        inputs = [(manifests[0].parent / name).read_bytes() for name in entries[0]['input_files']]
        nan_output = struct.pack('f', float('nan')) * len(expected)
        owners = [c.create_string_buffer(raw, len(raw)) for raw in [*inputs, nan_output]]
        count = len(owners)
        pointers = (p * count)(*(c.addressof(owner) for owner in owners))
        sizes = (size * count)(*(c.sizeof(owner) for owner in owners))
        writable = (u32 * count)(*([0] * (count - 1) + [1]))
        oracles = (c.POINTER(d) * count)(*([None] * (count - 1) + [expected]))
        output_hashes = {}
        sample_count = 0 if args.validate_only else args.samples
        for cycle in range(args.cycles):
            for position, index in enumerate((0, 1, 1, 0)):
                entry = entries[index]
                c.memmove(owners[-1], nan_output, len(nan_output))
                row = dict(cycle=cycle, position=position, variant=entry['name'], valid=False)
                report['visits'].append(row)
                samples, repetitions, error = (d * sample_count)(), c.c_uint64(), d()
                result = helper(c.cast(getattr(libraries[index], entry['symbol']), p), entry['abi'], count,
                                pointers, sizes, writable, oracles, (u32 * 3)(*entry['dispatch']), (u32 * 3)(*entry['block']),
                                entry['packet_width'], entry['workspace_bytes'], sample_count,
                                0 if args.validate_only else args.warmup_ms, 0 if args.validate_only else args.target_ms,
                                entry['atol'], entry['rtol'], samples, c.byref(repetitions), c.byref(error))
                output = bytes(owners[-1])
                filename = f'visit-{cycle}-{position}.f32'
                (directory / filename).write_bytes(output)
                row.update(returncode=result, output=filename, output_sha256=sha(directory / filename))
                save(directory / 'results.json', report)
                if result or (sample_count and (repetitions.value == 0 or any(not math.isfinite(v) or v <= 0 for v in samples))):
                    raise ValueError(f'native replay failed: code={result}, variant={entry["name"]}')
                actual = array.array('f', output)
                validation = check_output(actual, expected, entry['atol'], entry['rtol'])
                if any(bytes(owner) != raw for owner, raw in zip(owners[:-1], inputs)):
                    raise ValueError('caller input changed')
                previous = output_hashes.setdefault(entry['name'], row['output_sha256'])
                if previous != row['output_sha256']:
                    raise ValueError('fixed entry output changed between visits')
                row.update(valid=True, samples_us=list(samples), repetitions=repetitions.value,
                           median_us=statistics.median(samples) if sample_count else None,
                           correctness=validation, helper_max_abs_error=error.value, all_guards_passed=True, inputs_unchanged=True)
                save(directory / 'results.json', report)
                print(entry['name'], cycle, position, row['median_us'], flush=True)
        for path in manifests:
            verify(path)
        if report['prepared_sha256'] != {str(path): sha(path) for path in manifests} or report['runner_sha256'] != sha(__file__):
            raise ValueError('manifest or runner changed during replay')
        if sample_count:
            report['summary_us'] = {entry['name']: statistics.median(v['median_us'] for v in report['visits'] if v['variant'] == entry['name']) for entry in entries}
            ratios = []
            for cycle in range(args.cycles):
                visit = report['visits'][cycle * 4:cycle * 4 + 4]
                ratios.extend((visit[1]['median_us'] / visit[0]['median_us'], visit[2]['median_us'] / visit[3]['median_us']))
            report['candidate_over_baseline'] = dict(baseline=entries[0]['name'], candidate=entries[1]['name'], pairs=ratios,
                                                    median=statistics.median(ratios), minimum=min(ratios), maximum=max(ratios))
        report.update(status='passed', finished_unix=time.time(), artifacts_unchanged=True)
        save(directory / 'results.json', report)
    except Exception as error:
        report.update(status='error', error=str(error), finished_unix=time.time())
        save(directory / 'results.json', report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest='command', required=True)
    prep = subcommands.add_parser('prepare')
    for name in ('prefix', 'log', 'objects', 'output'):
        prep.add_argument('--' + name, type=Path, required=True)
    prep.add_argument('--name', required=True)
    run = subcommands.add_parser('replay')
    run.add_argument('--prepared', type=Path, action='append', required=True)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--cycles', type=int, default=2)
    run.add_argument('--samples', type=int, default=7)
    run.add_argument('--warmup-ms', type=int, default=100)
    run.add_argument('--target-ms', type=int, default=30)
    run.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.command == 'replay' and (not 1 <= args.cycles <= 20 or not 1 <= args.samples <= 100 or
                                   not 0 <= args.warmup_ms <= 10000 or not 1 <= args.target_ms <= 10000):
        parser.error('timing parameters out of bounds')
    (prepare if args.command == 'prepare' else replay)(args)


if __name__ == '__main__':
    main()
