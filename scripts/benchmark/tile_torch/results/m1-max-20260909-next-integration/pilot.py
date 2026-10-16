"""Fixed-local native fusion probe; retained controls are never relabelled."""
import argparse
import ctypes
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import sys
import time

import numpy as np

REPO = Path('/Users/mike/CLionProjects/luisa')
RAW = Path(__file__).resolve().parent
ISOLATED = Path('/tmp/luisa-reduction-checkpoint.gQUERp')
PREVIOUS = Path('/tmp/luisa-native-rows.LuX3yX/prepared-v2/manifest.json')
sys.path.insert(0, str(REPO / 'scripts/benchmark/tile_torch'))
import native_rows as nr


def source_hashes():
    baseline = json.loads((RAW / 'baseline.json').read_text())
    expected = baseline['source_sha256'] | json.loads((RAW / 'current-source.json').read_text())
    for name, sha in expected.items():
        if nr.digest(REPO / name) != sha or nr.digest(ISOLATED / 'source' / name) != sha:
            raise ValueError('source changed: ' + name)
    return expected


def capture(directory, selected):
    directory.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, directory / 'pilot.py')
    nr.snapshot_sources(directory)
    previous = json.loads(PREVIOUS.read_text())
    source = source_hashes()
    binary = ISOLATED / 'build/bin/benchmark_tile_xir'
    closure = {str(p): nr.digest(p) for p in [binary, *sorted(binary.parent.glob('libluisa-*'))] if p.is_file()}
    report = dict(started_unix=time.time(), previous=str(PREVIOUS), previous_sha256=nr.digest(PREVIOUS),
                  source_sha256=source, binary_closure_sha256=closure,
                  pilot_sha256=nr.digest(__file__), native_rows_sha256=nr.digest(nr.__file__),
                  build_log=str(RAW / 'build-v3.log'), build_log_sha256=nr.digest(RAW / 'build-v3.log'),
                  source_caveat='Isolated archive with recorded predecessor overlays; not a clean checkout.',
                  cpu_threads=1, packet_width=8, block_size=32, local_lanes=8, cases=[],
                  helper=previous['helper'], helper_sha256=previous['helper_sha256'],
                  torch_version=previous['torch_version'], torch_git_version=previous['torch_git_version'])
    if (nr.digest(previous['helper']) != previous['helper_sha256'] or
            nr.digest(REPO / 'src/backends/simd/llvm/llvm_schedule_codegen.h') != previous['abi_header_sha256'] or
            nr.digest(nr.HERE / 'native_rows_replay.cpp') != previous['helper_source_sha256']):
        raise ValueError('native helper ABI/source mismatch')
    nr.save(directory / 'manifest.json', report)
    for case in previous['cases']:
        op, dims = case['operation'], case['dimensions']
        if selected and nr.case_name(op, dims) not in selected:
            continue
        target = directory / nr.case_name(op, dims)
        target.mkdir()
        arrays = nr.load_inputs(Path(case['inputs']), case['input_shapes'])
        if [nr.array_digest(a) for a in arrays] != case['input_sha256']:
            raise ValueError('previous input changed')
        entries = {}
        for variant in ('off', 'on'):
            folder = target / variant
            folder.mkdir()
            env = {k: v for k, v in os.environ.items() if not k.startswith(('LUISA_SIMD_', 'LUISA_TILE_BENCH_', 'DYLD_'))}
            env.update(LUISA_SIMD_WARP_WIDTH='8', LUISA_SIMD_WORKER_COUNT='1',
                       LUISA_TILE_BENCH_XIR_LOCAL_LANES='8', LUISA_TILE_BENCH_XIR_BLOCK_SIZE='32',
                       LUISA_TILE_BENCH_XIR_BLOCKS_PER_TASK='0', LUISA_SIMD_DISABLE_LOAD_REDUCTION_FUSION='1',
                       LUISA_SIMD_ENABLE_FULL_PACKET_SPECIALIZATION='1', LUISA_SIMD_ENABLE_PREDICATED_MEMORY_EFFECTS='1',
                       LUISA_SIMD_ENABLE_COHORT_PRIVATE_ACCESS='1', LUISA_TILE_BENCH_DUMP_SOURCE=str(folder / 'kernel.ll'),
                       LUISA_SIMD_DUMP_ASSEMBLY_DIR=str(folder / 'object'))
            env['LUISA_SIMD_' + ('ENABLE' if variant == 'on' else 'DISABLE') + '_POINTWISE_FUSION'] = '1'
            text = nr.command([binary, 'llm', op, ','.join(map(str, dims)), '1', '1', '3', '1', '1', folder / 'output.f32'],
                              folder, 'capture', env=env, timeout=300)
            measurement = json.loads(text)
            nr.check_metadata(measurement, 'cpu', op, dims, (1, 1), 3)
            nr.save(folder / 'measurement.json', measurement)
            inputs = nr.load_inputs(folder, case['input_shapes'])
            if [nr.array_digest(a) for a in inputs] != case['input_sha256']:
                raise ValueError('capture input changed')
            check = nr.validate_output(np.fromfile(folder / 'output.f32', np.float32).reshape(case['output_shape']),
                                       nr.reference(op, dims, arrays))
            llvm = (folder / 'kernel.ll').read_text()
            symbol, abi = 'llm_rows.packet_batch.blocks', 0
            if f'define dso_local void @{symbol}(' not in llvm:
                symbol, abi = 'llm_rows.packet_batch', 2
            if f'define dso_local void @{symbol}(' not in llvm:
                raise ValueError('unknown native entry')
            realization = measurement['realization']
            if 'local_lanes=8;' not in realization or 'W8, 32 workers/block' not in realization:
                raise ValueError('mapping changed')
            if f'pointwise_fusion={str(variant == "on").lower()};' not in realization:
                raise ValueError('feature not acknowledged')
            objects = list((folder / 'object').glob('*.o'))
            if len(objects) != 1:
                raise ValueError('expected actual ORC object')
            imports = nr.command([nr.LLVM_BIN / 'llvm-nm', '--undefined-only', '--just-symbol-name', objects[0]], folder, 'imports').split()
            if set(imports) - {'_memcpy', '_memset', '_bzero', '___chkstk_darwin'}:
                raise ValueError('uninspected imports: ' + str(imports))
            library = folder / 'entry.dylib'
            nr.command([nr.LLVM_BIN / 'clang++', '-dynamiclib', objects[0], '-o', library], folder, 'link', timeout=60)
            nr.command([nr.LLVM_BIN / 'llvm-objdump', '--disassemble', '--no-show-raw-insn', objects[0]], folder, 'assembly')
            workspace = re.search(r'private_workspace_bytes=(\d+)', realization)
            entries[variant] = dict(library=str(library), library_sha256=nr.digest(library), symbol=symbol, abi=abi,
                                    block_size=32, local_lanes=8, workspace_bytes=int(workspace[1]) if workspace else 0,
                                    capture=str(folder), realization=realization, correctness=check,
                                    environment={k: v for k, v in env.items() if k.startswith('LUISA_')},
                                    object_sha256=nr.digest(objects[0]), llvm_sha256=nr.digest(folder / 'kernel.ll'),
                                    system_imports=imports)
        entries['inductor'] = case['entries']['inductor']
        if nr.digest(entries['inductor']['library']) != entries['inductor']['library_sha256']:
            raise ValueError('Inductor object changed')
        previous_entry = case['entries']['local']
        row = dict(case, entries=entries,
                   off_matches_previous_llvm=entries['off']['llvm_sha256'] == previous_entry['llvm_sha256'],
                   off_matches_previous_object=entries['off']['object_sha256'] == previous_entry['object_sha256'])
        report['cases'].append(row)
        nr.save(directory / 'manifest.json', report)
        print(target.name, 'captured', 'previous object identical:', row['off_matches_previous_object'], flush=True)
    report.update(finished_unix=time.time(), source_unchanged=source_hashes() == source,
                  closure_unchanged=all(nr.digest(p) == sha for p, sha in closure.items()))
    nr.save(directory / 'manifest.json', report)
    if not report['closure_unchanged'] or not report['cases']:
        raise ValueError('incomplete capture')


def replay(prepared, directory, verification_only=False):
    manifest = json.loads((prepared / 'manifest.json').read_text())
    if not manifest.get('source_unchanged') or not manifest.get('closure_unchanged'):
        raise ValueError('incomplete capture')
    if nr.digest(manifest['helper']) != manifest['helper_sha256']:
        raise ValueError('helper changed')
    directory.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, directory / 'pilot.py')
    nr.snapshot_sources(directory)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.__version__ != manifest['torch_version'] or torch.version.git_version != manifest['torch_git_version']:
        raise ValueError('Torch build changed')
    library = ctypes.CDLL(manifest['helper'])
    helper = library.replay_native_rows
    ptr, u32, size = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_size_t
    helper.argtypes = [ptr, u32, u32, u32, ctypes.POINTER(ptr), ctypes.POINTER(size), u32, u32, u32,
                       u32, u32, size, u32, u32, u32, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint64)]
    helper.restype = ctypes.c_int
    report = dict(started_unix=time.time(), manifest=str(prepared / 'manifest.json'), manifest_sha256=nr.digest(prepared / 'manifest.json'),
                  pilot_sha256=nr.digest(__file__), native_rows_sha256=nr.digest(nr.__file__),
                  metric='single_thread_native_entry_host_wall_us', cpu_threads=1, samples=7, warmup_ms=100, target_ms=30,
                  boundary='Common C++ timer: actual entries, block traversal/reset and native internal allocation/libc included; Runtime/Python/JIT/caller allocations excluded.',
                  cases=[])
    if verification_only:
        report.update(metric='native_entry_correctness_smoke', samples=1, warmup_ms=0, target_ms=1,
                      timing_not_comparative=True, timing_exclusion_reason='Concurrent unrelated builds and high system load; do not infer performance from this smoke.')
    nr.save(directory / 'results.json', report)
    for case in manifest['cases']:
        op, dims = case['operation'], case['dimensions']
        target = directory / nr.case_name(op, dims)
        target.mkdir()
        arrays = nr.load_inputs(Path(case['inputs']), case['input_shapes'])
        if [nr.array_digest(a) for a in arrays] != case['input_sha256']:
            raise ValueError('input changed')
        expected = nr.reference(op, dims, arrays)
        inputs = {f'input{i}': nr.Guarded(a.size, a) for i, a in enumerate(arrays)}
        variants, libraries = {}, []
        for name, entry in case['entries'].items():
            if nr.digest(entry['library']) != entry['library_sha256']:
                raise ValueError('entry changed')
            native = ctypes.CDLL(entry['library'])
            libraries.append(native)
            address = ctypes.cast(getattr(native, entry['symbol']), ptr)
            owners = dict(inputs)
            if entry['abi'] == 1:
                plan = entry['plan']
                owners.update({key: nr.Guarded(value['elements']) for key, value in plan['allocations'].items()})
                arguments = [nr.view_array(v, owners) for v in plan['arguments']]
                output = nr.view_array(plan['output'], owners)
                mask, block, local, workspace = plan['const_mask'], 32, 1, 0
            else:
                owners['output'] = nr.Guarded(math.prod(case['output_shape']))
                output = owners['output'].data.reshape(case['output_shape'])
                arguments = [owners[f'input{i}'].data for i in range(3)] + [output]
                mask, block, local, workspace = 7, entry['block_size'], entry['local_lanes'], entry['workspace_bytes']
            pointers = (ptr * len(arguments))(*(a.ctypes.data for a in arguments))
            sizes = (size * len(arguments))(*(a.nbytes for a in arguments))
            variants[name] = address, entry['abi'], mask, block, local, workspace, pointers, sizes, owners, output
        row = dict(operation=op, dimensions=dims, results=[], output_sha256={})
        report['cases'].append(row)
        orders = [tuple(variants)] if verification_only else itertools.permutations(variants)
        for round_id, order in enumerate(orders):
            for name in order:
                address, abi, mask, block, local, workspace, pointers, sizes, owners, output = variants[name]
                for key, owner in owners.items():
                    if not key.startswith('input'):
                        owner.data.fill(np.nan)
                samples, repetitions = np.empty(report['samples'], np.float64), ctypes.c_uint64()
                code = helper(address, abi, len(pointers), mask, pointers, sizes, *dims, 8, block, local, workspace,
                              samples.size, report['warmup_ms'], report['target_ms'],
                              samples.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.byref(repetitions))
                if code or not np.isfinite(samples).all() or np.any(samples <= 0):
                    raise ValueError('native helper failed: ' + str(code))
                check = nr.validate_output(output, expected)
                for owner in owners.values():
                    owner.check()
                if [nr.array_digest(inputs[f'input{i}'].data) for i in range(3)] != case['input_sha256']:
                    raise ValueError('native input mutation')
                digest = nr.array_digest(output)
                if name not in row['output_sha256']:
                    output.tofile(target / (name + '.f32'))
                    row['output_sha256'][name] = digest
                if digest != row['output_sha256'][name]:
                    raise ValueError('fixed entry output changed')
                visit = dict(round=round_id, order=order, variant=name, median_us=float(statistics.median(samples)),
                             samples_us=samples.tolist(), repetitions=repetitions.value, correctness=check,
                             output_sha256=digest, inputs_unchanged=True, guard_elements=128 * len(owners), workspace_guards_passed=True)
                row['results'].append(visit)
                nr.save(directory / 'results.json', report)
                print(target.name, round_id, name, f'{visit["median_us"]:.6f}', 'us', flush=True)
        row['off_on_bitwise_equal'] = row['output_sha256']['off'] == row['output_sha256']['on']
        if verification_only:
            nr.save(directory / 'results.json', report)
            continue
        row['paired_ratios'] = {}
        for a, b in (('on', 'off'), ('on', 'inductor'), ('off', 'inductor')):
            values = []
            for r in range(6):
                pair = {v['variant']: v['median_us'] for v in row['results'] if v['round'] == r}
                values.append(pair[a] / pair[b])
            row['paired_ratios'][a + '/' + b] = dict(median=statistics.median(values), minimum=min(values), maximum=max(values),
                                                    wins=int(sum(v < 1 for v in values)), rounds=values)
        nr.save(directory / 'results.json', report)
    report.update(finished_unix=time.time(), runner_unchanged=nr.digest(__file__) == report['pilot_sha256'] and nr.digest(nr.__file__) == report['native_rows_sha256'])
    nr.save(directory / 'results.json', report)
    if not report['runner_unchanged']:
        raise ValueError('runner changed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('capture', 'replay', 'verify'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--case', action='append', default=[])
    args = parser.parse_args()
    if args.action == 'capture':
        capture(args.output.resolve(), args.case)
    else:
        replay(args.prepared.resolve(), args.output.resolve(), verification_only=args.action == 'verify')
