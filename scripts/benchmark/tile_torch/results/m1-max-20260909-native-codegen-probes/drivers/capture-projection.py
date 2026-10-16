"""Capture actual ORC entries with late integer lane projection off/on."""
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path('/Users/mike/CLionProjects/luisa')
PREVIOUS = Path('/tmp/luisa-native-rows.LuX3yX/prepared-v2/manifest.json')
sys.path.insert(0, str(REPO / 'scripts/benchmark/tile_torch'))
import native_rows as nr


def sources():
    manifest = json.loads((HERE / 'projection-source-snapshot.json').read_text())
    for name, expected in manifest['source_sha256'].items():
        if nr.digest(HERE / 'source' / name) != expected:
            raise ValueError('isolated source changed: ' + name)
    return manifest['source_sha256']


directory = HERE / 'projection-capture'
directory.mkdir(exist_ok=False)
assert json.loads((HERE / 'projection-build.json').read_text())['passed']
previous = json.loads(PREVIOUS.read_text())
source = sources()
binary = HERE / 'build/bin/benchmark_tile_xir'
closure = {str(p): nr.digest(p) for p in [binary, *sorted(binary.parent.glob('libluisa-*'))] if p.is_file()}
report = dict(started_unix=time.time(), previous=str(PREVIOUS), previous_sha256=nr.digest(PREVIOUS),
              source_sha256=source, binary_closure_sha256=closure, cpu_threads=1, packet_width=8,
              block_size=32, local_lanes=8, cases=[], helper=previous['helper'], helper_sha256=previous['helper_sha256'],
              torch_version=previous['torch_version'], torch_git_version=previous['torch_git_version'],
              capture_driver_sha256=nr.digest(__file__), native_rows_sha256=nr.digest(nr.__file__),
              source_contract='2cfc80493 pinned archives plus separately recorded diagnostic capability/projection overlays; phase A source and binaries relocated by stage-a-frozen.json.',
              timing_not_comparative=True, feature='integer_lane_projection', full_build_sha256=nr.digest(HERE / 'projection-build.json'))
if (nr.digest(previous['helper']) != previous['helper_sha256'] or
        nr.digest(HERE / 'source/src/backends/simd/llvm/llvm_schedule_codegen.h') != previous['abi_header_sha256'] or
        nr.digest(nr.HERE / 'native_rows_replay.cpp') != previous['helper_source_sha256']):
    raise ValueError('native entry ABI/helper mismatch')
nr.save(directory / 'manifest.json', report)
for case in previous['cases']:
    op, dims = case['operation'], case['dimensions']
    target = directory / nr.case_name(op, dims)
    target.mkdir()
    arrays = nr.load_inputs(Path(case['inputs']), case['input_shapes'])
    if [nr.array_digest(a) for a in arrays] != case['input_sha256']:
        raise ValueError('frozen input changed')
    entries = {}
    for variant in ('off', 'on'):
        folder = target / variant
        folder.mkdir()
        env = {k: v for k, v in os.environ.items() if not k.startswith(('LUISA_SIMD_', 'LUISA_TILE_BENCH_', 'DYLD_'))}
        env.update(LUISA_SIMD_WARP_WIDTH='8', LUISA_SIMD_WORKER_COUNT='1',
                   LUISA_TILE_BENCH_XIR_LOCAL_LANES='8', LUISA_TILE_BENCH_XIR_BLOCK_SIZE='32',
                   LUISA_TILE_BENCH_XIR_BLOCKS_PER_TASK='0', LUISA_SIMD_DISABLE_LOAD_REDUCTION_FUSION='1',
                   LUISA_SIMD_ENABLE_POINTWISE_FUSION='1', LUISA_SIMD_DISABLE_EXPRESSION_REDUCTION_FUSION='1',
                   LUISA_SIMD_DISABLE_LINEAR_1D_BLOCK_COALESCING='1', LUISA_SIMD_ENABLE_FULL_PACKET_SPECIALIZATION='1',
                   LUISA_SIMD_ENABLE_PREDICATED_MEMORY_EFFECTS='1', LUISA_SIMD_ENABLE_COHORT_PRIVATE_ACCESS='1',
                   LUISA_TILE_BENCH_DUMP_SOURCE=str(folder / 'kernel.ll'), LUISA_SIMD_DUMP_ASSEMBLY_DIR=str(folder / 'object'))
        env['LUISA_SIMD_EXPERIMENT_INLINE_PACKETS'] = '0'
        env['LUISA_SIMD_EXPERIMENT_PROJECT_INTEGER_LANES'] = '1' if variant == 'on' else '0'
        text = nr.command([binary, 'llm', op, ','.join(map(str, dims)), '1', '1', '3', '1', '1', folder / 'output.f32'],
                          folder, 'capture', env=env, timeout=300)
        measurement = json.loads(text)
        nr.check_metadata(measurement, 'cpu', op, dims, (1, 1), 3)
        nr.save(folder / 'measurement.json', measurement)
        if [nr.array_digest(a) for a in nr.load_inputs(folder, case['input_shapes'])] != case['input_sha256']:
            raise ValueError('capture input changed')
        correctness = nr.validate_output(np.fromfile(folder / 'output.f32', np.float32).reshape(case['output_shape']), nr.reference(op, dims, arrays))
        llvm = (folder / 'kernel.ll').read_text()
        symbol, abi = 'llm_rows.packet_batch.blocks', 0
        if f'define dso_local void @{symbol}(' not in llvm:
            symbol, abi = 'llm_rows.packet_batch', 2
        if f'define dso_local void @{symbol}(' not in llvm:
            raise ValueError('unrecognized native entry')
        realization = measurement['realization']
        if 'local_lanes=8;' not in realization or 'W8, 32 workers/block' not in realization:
            raise ValueError('mapping changed')
        if 'pointwise_fusion=true;' not in realization or 'expression_reduction_fusion=false;' not in realization:
            raise ValueError('fixed fusion policy changed')
        if 'packet.batch.full.loop:' in llvm:
            raise ValueError('packet loop policy changed')
        objects = list((folder / 'object').glob('*.o'))
        if len(objects) != 1:
            raise ValueError('expected one actual ORC object')
        imports = nr.command([nr.LLVM_BIN / 'llvm-nm', '--undefined-only', '--just-symbol-name', objects[0]], folder, 'imports').split()
        if set(imports) - {'_memcpy', '_memset', '_bzero', '___chkstk_darwin'}:
            raise ValueError('uninspected native imports: ' + str(imports))
        library = folder / 'entry.dylib'
        nr.command([nr.LLVM_BIN / 'clang++', '-dynamiclib', objects[0], '-o', library], folder, 'link', timeout=60)
        nr.command([nr.LLVM_BIN / 'llvm-objdump', '--disassemble', '--no-show-raw-insn', objects[0]], folder, 'assembly')
        workspace = re.search(r'private_workspace_bytes=(\d+)', realization)
        entries[variant] = dict(library=str(library), library_sha256=nr.digest(library), symbol=symbol, abi=abi,
                                block_size=32, local_lanes=8, workspace_bytes=int(workspace[1]) if workspace else 0,
                                capture=str(folder), realization=realization, correctness=correctness,
                                environment={k: v for k, v in env.items() if k.startswith('LUISA_')},
                                object_sha256=nr.digest(objects[0]), llvm_sha256=nr.digest(folder / 'kernel.ll'), system_imports=imports)
    entries['inductor'] = case['entries']['inductor']
    if nr.digest(entries['inductor']['library']) != entries['inductor']['library_sha256']:
        raise ValueError('frozen Inductor library changed')
    report['cases'].append(dict(case, entries=entries))
    nr.save(directory / 'manifest.json', report)
    print(target.name, 'captured and checked', flush=True)
report.update(finished_unix=time.time(), source_unchanged=sources() == source,
              closure_unchanged=all(nr.digest(p) == sha for p, sha in closure.items()))
nr.save(directory / 'manifest.json', report)
assert report['source_unchanged'] and report['closure_unchanged'] and len(report['cases']) == 24
