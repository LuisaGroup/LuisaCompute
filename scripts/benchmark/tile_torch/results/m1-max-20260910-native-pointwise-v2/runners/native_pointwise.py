#!/usr/bin/env python3
"""Fixed-mapping pointwise fusion capture and actual native-entry A/B replay.

Capture never builds the project: complete the configured full build first and
provide its log. That log is an external gate receipt, not a proof inferred from
its contents. Source, binary closure, baseline inputs, ABI/helper and actual
Inductor source/library identities are checked and frozen separately.

Only pointwise fusion changes between off/on. Static snapshot totals include
the retained alias fallback; inspect the disjoint hot path, not alloca counts
alone. Replay compares fixed candidates, not automatic planner performance;
background load is recorded, never assumed absent.
"""
from __future__ import annotations

import argparse
import ctypes
import itertools
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np

import native_rows as nr

HERE = Path(__file__).resolve().parent
ABI_HEADER = Path('src/backends/simd/llvm/llvm_schedule_codegen.h')
VARIANTS = ('off', 'on', 'inductor')
ORDERS = tuple(itertools.permutations(VARIANTS))
ALLOWED_IMPORTS = {'_memcpy', '_memset', '_bzero', '___chkstk_darwin'}
SOURCE_SUFFIXES = {'.c', '.cc', '.cpp', '.cxx', '.h', '.hh', '.hpp', '.hxx', '.m', '.mm',
                   '.inc', '.inl', '.ipp', '.tpp', '.cu', '.cuh', '.metal', '.hlsl', '.glsl',
                   '.cmake', '.in', '.py'}
SOURCE_COVERAGE = 'Selected build-related sources/headers/configuration under include, src, .cmake and cmake (including vendored headers), plus top-level CMake inputs. Excludes docs/results/payloads; not a full generated/external dependency scan.'
FIXED_ENV = dict(
    LUISA_SIMD_WARP_WIDTH='8', LUISA_SIMD_WORKER_COUNT='1',
    LUISA_TILE_BENCH_XIR_BACKEND='simd', LUISA_TILE_BENCH_XIR_LOCAL_LANES='8',
    LUISA_TILE_BENCH_XIR_BLOCK_SIZE='32', LUISA_TILE_BENCH_XIR_BLOCKS_PER_TASK='0',
    LUISA_TILE_BENCH_XIR_SEARCH_TASK_GRAIN='0',
    LUISA_SIMD_DISABLE_LOAD_REDUCTION_FUSION='1',
    LUISA_SIMD_DISABLE_EXPRESSION_REDUCTION_FUSION='1', LUISA_SIMD_DISABLE_MAP_FUSION='1',
    LUISA_SIMD_DISABLE_LINEAR_1D_BLOCK_COALESCING='1',
    LUISA_SIMD_ENABLE_FULL_PACKET_SPECIALIZATION='1',
    LUISA_SIMD_ENABLE_PREDICATED_MEMORY_EFFECTS='1', LUISA_SIMD_ENABLE_COHORT_PRIVATE_ACCESS='1',
    OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def host_observation():
    # Deliberately exclude complete process command lines/arguments. This is
    # context for interpreting a comparison, not proof of hardware exclusivity.
    process = subprocess.run(['ps', '-axo', 'pid=,pcpu=,comm='], text=True, capture_output=True, timeout=10)
    return dict(unix=time.time(), loadavg=list(os.getloadavg()),
                processes=dict(fields=['pid', 'pcpu', 'comm'], returncode=process.returncode,
                               stdout=process.stdout, stderr=process.stderr))


def hashes(paths):
    return {str(p): nr.digest(p) for p in sorted(set(map(Path, paths)))}


def source_hashes(root):
    # Isolated archives have no Git index. Hash actual compiler inputs, also
    # vendored headers, without folding unrelated docs/benchmark payloads into
    # the compiler freeze. The precise coverage is recorded in the manifest.
    paths = [p for p in root.iterdir() if p.is_file() and (p.suffix == '.cmake' or p.name == 'CMakeLists.txt')]
    for name in ('include', 'src', '.cmake', 'cmake'):
        directory = root / name
        if directory.is_dir():
            paths.extend(p for p in directory.rglob('*') if p.is_file() and
                         not {'.git', 'results', '__pycache__'}.intersection(p.relative_to(directory).parts) and
                         (p.suffix in SOURCE_SUFFIXES or p.name == 'CMakeLists.txt'))
    require(paths, 'empty source inventory')
    return hashes(paths)


def binary_hashes(binary):
    return hashes([binary, *[p for p in binary.parent.glob('libluisa-*') if p.is_file()]])


def unchanged(expected):
    require(hashes(expected) == expected, 'frozen file identity changed')


def runners(directory):
    nr.snapshot_sources(directory)
    shutil.copy2(__file__, directory / 'runner-sources' / 'native_pointwise.py')
    return hashes([Path(__file__).resolve(), Path(nr.__file__).resolve(),
                   HERE / 'native_rows_replay.cpp', HERE / 'compare_llm.py'])


def cmake_configuration(binary, source_root):
    cache = binary.parent.parent / 'CMakeCache.txt'
    values = {}
    for line in cache.read_text().splitlines():
        if match := re.match(r'([^#/:][^:]*):[^=]+=(.*)$', line):
            values[match[1]] = match[2]
    require(Path(values['CMAKE_HOME_DIRECTORY']).resolve() == source_root, 'binary build belongs to another source root')
    llvm = Path(values['LLVM_DIR']).resolve().parents[2] / 'bin'
    for name in ('clang++', 'llvm-nm', 'llvm-objdump'):
        require((llvm / name).is_file(), 'missing LLVM tool: ' + str(llvm / name))
    return cache, llvm


def select_cases(baseline, names):
    by_name = {}
    for case in baseline['cases']:
        op, dims = case['operation'], case['dimensions']
        require(op in nr.OPS and len(dims) == 2 and all(type(n) is int and n > 0 for n in dims), 'unsupported baseline case')
        require(math.prod(dims) <= nr.MAX_ELEMENTS and (op != 'rope' or dims[1] % 2 == 0), 'invalid case extent')
        key = nr.case_name(op, dims)
        require(key not in by_name, 'duplicate baseline case')
        shapes, output = nr.shapes_for(op, dims)
        require(json.loads(json.dumps(shapes)) == case['input_shapes'] and list(output) == case['output_shape'], 'baseline shape contract changed')
        by_name[key] = case
    selected = list(by_name) if names == ['all'] else names
    require(selected and len(set(selected)) == len(selected), 'empty or repeated requested cases')
    require(all(name in by_name for name in selected), 'case is absent from the frozen baseline')
    return [by_name[name] for name in selected]


def baseline_identity(path, baseline, source_root, cases):
    require(baseline.get('runner_unchanged') is True and 'finished_unix' in baseline, 'unfinished baseline preparation')
    require(baseline['cpu_threads'] == 1, 'baseline is not single-thread native entry')
    files = [path, Path(baseline['helper']), source_root / ABI_HEADER]
    require(nr.digest(files[1]) == baseline['helper_sha256'], 'baseline helper binary changed')
    require(nr.digest(files[2]) == baseline['abi_header_sha256'], 'current Tile launch ABI differs from baseline helper')
    for name, key in (('native_rows_replay.cpp', 'helper_source_sha256'), ('native_rows.py', 'source_sha256')):
        current, archived = HERE / name, path.parent / 'runner-sources' / name
        require(nr.digest(current) == nr.digest(archived) == baseline[key], 'baseline/helper source identity differs: ' + name)
        files.extend((current, archived))
    for case in cases:
        entry = case['entries']['inductor']
        base = Path(entry['library']).parent
        require(entry['abi'] == 1 and entry['symbol'] == 'kernel', 'unsupported Inductor entry ABI')
        for file, key in ((Path(entry['library']), 'library_sha256'), (base / 'inductor.cpp', 'source_sha256'),
                          (base / 'inductor-wrapper.py', 'wrapper_sha256')):
            require(nr.digest(file) == entry[key], 'frozen Inductor identity changed: ' + str(file))
            files.append(file)
        plan = nr.parse_inductor_wrapper((base / 'inductor-wrapper.py').read_text(), entry['graph']['inputs'], case['input_shapes'])
        cpp = plan.pop('cpp')
        require(cpp.strip() in (base / 'inductor.cpp').read_text(), 'wrapper C++ is not the frozen generated source')
        require(json.loads(json.dumps(plan)) == entry['plan'], 'Inductor ABI/alias plan differs from frozen wrapper')
        inputs = nr.load_inputs(Path(case['inputs']), case['input_shapes'])
        require([nr.array_digest(a) for a in inputs] == case['input_sha256'], 'frozen input bits changed')
        files.extend(Path(case['inputs']) / f'output.f32.input{i}.f32' for i in range(3))
    return hashes(files)


def check_realization(measurement, enabled):
    realization = measurement['realization']
    require('W8, 32 workers/block, 1 CPU workers;' in realization, 'packet/block/worker mapping changed')
    fields = dict(re.findall(r'\b([a-z_]+)=([^;]+)', realization))
    for name, expected in dict(local_lanes='8', blocks_per_task='0', max_unrolled_tile_elements='64',
                               unordered_reduction_partitions='4', load_reduction_fusion='false',
                               expression_reduction_fusion='false', map_fusion='false', fast_math='false',
                               pointwise_fusion=str(enabled).lower(), custom_cost_policy='false').items():
        require(fields.get(name) == expected, 'fixed realization policy changed: ' + name)
    require(int(fields.get('full_packet_specializations', '0')) > 0, 'full-packet specialization was not realized')
    for name in ('private_workspace_bytes', 'fused_pointwise_regions', 'fused_pointwise_loads', 'fused_pointwise_stores'):
        require(name in fields and fields[name].isdigit(), 'missing realization counter: ' + name)
    if not enabled:
        require(int(fields['fused_pointwise_regions']) == 0, 'off control still has pointwise fusion')
    if enabled and measurement['operation'] == 'rope':
        for name, count in dict(fused_pointwise_regions=1, fused_pointwise_loads=4,
                                fused_pointwise_stores=2, pointwise_alias_checks=3).items():
            require(fields.get(name) == str(count), 'RoPE did not realize the expected pointwise DAG: ' + name)
    return fields


def capture(args):
    require(platform.system() == 'Darwin' and platform.machine() == 'arm64', 'inspected native object path requires Darwin arm64')
    binary, source_root = args.binary.resolve(), args.source_root.resolve()
    require(binary.is_file() and source_root.is_dir(), 'missing binary/source root')
    directory = args.output.resolve()
    require(not directory.is_relative_to(source_root), 'capture output must not change the frozen source tree')
    directory.mkdir(parents=True, exist_ok=False)
    report = dict(status='capturing', format='native-pointwise-v1', started_unix=time.time(), argv=sys.argv,
                  performance_qualified=False, timing_not_comparative=True,
                  qualification='Diagnostic-only: no automatic performance acceptance, policy promotion, or cost calibration.',
                  cpu_threads=1, packet_width=8, block_size=32, local_lanes=8,
                  host_before=host_observation(), cases=[])
    manifest = directory / 'manifest.json'
    save(manifest, report)
    try:
        runner = runners(directory)
        cache, llvm = cmake_configuration(binary, source_root)
        baseline_path = args.baseline_manifest.resolve()
        baseline = json.loads(baseline_path.read_text())
        selected = select_cases(baseline, args.cases)
        inherited = baseline_identity(baseline_path, baseline, source_root, selected)
        source, closure = source_hashes(source_root), binary_hashes(binary)
        gate = args.full_build_log.resolve()
        require(gate.is_file() and gate.stat().st_size > 0, 'missing configured full-build log')
        tools = hashes([llvm / name for name in ('clang++', 'llvm-nm', 'llvm-objdump')])
        report.update(binary=str(binary), source_root=str(source_root), source_sha256=source, source_coverage=SOURCE_COVERAGE,
                      binary_closure_sha256=closure,
                      runner_sha256=runner, inherited_sha256=inherited, configuration_sha256=hashes([cache, gate]),
                      full_build_gate=dict(log=str(gate), external_attestation=True,
                                           limitation='Caller completed the configured full build before capture; log hash alone does not prove binary/source correspondence.'),
                      baseline_manifest=str(baseline_path), baseline_manifest_sha256=nr.digest(baseline_path),
                      selected_cases=[nr.case_name(c['operation'], c['dimensions']) for c in selected],
                      llvm_tools_sha256=tools, helper=str(directory / 'native_rows_replay.dylib'),
                      helper_sha256=baseline['helper_sha256'], helper_source_sha256=baseline['helper_source_sha256'],
                      abi_header_sha256=baseline['abi_header_sha256'], torch_version=baseline['torch_version'],
                      torch_git_version=baseline['torch_git_version'], fixed_environment=FIXED_ENV)
        shutil.copy2(baseline['helper'], report['helper'])
        shutil.copy2(source_root / ABI_HEADER, directory / 'llvm_schedule_codegen.h')
        nr.command([llvm / 'clang++', '--version'], directory, 'llvm-version')
        save(manifest, report)
        for case in selected:
            op, dims = case['operation'], case['dimensions']
            target = directory / nr.case_name(op, dims)
            target.mkdir()
            inputs = nr.load_inputs(Path(case['inputs']), case['input_shapes'])
            expected = nr.reference(op, dims, inputs)
            input_dir = target / 'inputs'
            input_dir.mkdir()
            for i, array in enumerate(inputs):
                array.tofile(input_dir / f'output.f32.input{i}.f32')
            expected.tofile(target / 'expected.f64')
            record = dict(case, inputs=str(input_dir), expected_f64=str(target / 'expected.f64'),
                          expected_sha256=nr.digest(target / 'expected.f64'), entries={})
            report['cases'].append(record)
            save(manifest, report)
            for variant in ('off', 'on'):
                folder = target / variant
                folder.mkdir()
                env = {k: v for k, v in os.environ.items() if not k.startswith(('LUISA_', 'DYLD_'))}
                env.update(FIXED_ENV)
                env['LUISA_SIMD_' + ('ENABLE' if variant == 'on' else 'DISABLE') + '_POINTWISE_FUSION'] = '1'
                env.update(LUISA_TILE_BENCH_DUMP_SOURCE=str(folder / 'kernel.ll'), LUISA_SIMD_DUMP_ASSEMBLY_DIR=str(folder / 'object'))
                output = folder / 'output.f32'
                text = nr.command([binary, 'llm', op, ','.join(map(str, dims)), '1', '1', '3', '1', '1', output],
                                  folder, 'capture', env=env, timeout=300)
                measurement = json.loads(text)
                nr.check_metadata(measurement, 'cpu', op, dims, (1, 1), 3)
                fields = check_realization(measurement, variant == 'on')
                save(folder / 'measurement.json', measurement)
                require([nr.array_digest(a) for a in nr.load_inputs(folder, case['input_shapes'])] == case['input_sha256'], 'capture input differs from baseline')
                validation = nr.validate_output(np.fromfile(output, np.float32).reshape(case['output_shape']), expected)
                code = (folder / 'kernel.ll').read_text()
                symbol, abi = 'llm_rows.packet_batch.blocks', 0
                if f'define dso_local void @{symbol}(' not in code:
                    symbol, abi = 'llm_rows.packet_batch', 2
                require(f'define dso_local void @{symbol}(' in code, 'unknown emitted native entry')
                objects = list((folder / 'object').glob('*.o'))
                require(len(objects) == 1, 'expected exactly one actual ORC object')
                imports = nr.command([llvm / 'llvm-nm', '--undefined-only', '--just-symbol-name', objects[0]], folder, 'imports').split()
                require(not set(imports) - ALLOWED_IMPORTS, 'uninspected ORC imports: ' + str(imports))
                exports = nr.command([llvm / 'llvm-nm', '--defined-only', '--extern-only', '--just-symbol-name', objects[0]], folder, 'exports').split()
                require('_' + symbol in exports, 'LLVM entry is not exported by the captured object')
                library = folder / 'entry.dylib'
                nr.command([llvm / 'clang++', '-dynamiclib', objects[0], '-o', library], folder, 'link', timeout=60)
                nr.command([llvm / 'llvm-objdump', '--disassemble', '--no-show-raw-insn', objects[0]], folder, 'assembly')
                record['entries'][variant] = dict(library=str(library), library_sha256=nr.digest(library), symbol=symbol, abi=abi,
                    block_size=32, local_lanes=8, workspace_bytes=int(fields['private_workspace_bytes']),
                    capture=str(folder), realization=measurement['realization'], realization_fields=fields, correctness=validation,
                    environment={k: v for k, v in env.items() if k.startswith('LUISA_') or k in FIXED_ENV},
                    object=str(objects[0]), object_sha256=nr.digest(objects[0]), llvm_sha256=nr.digest(folder / 'kernel.ll'),
                    assembly_sha256=nr.digest(folder / 'assembly.stdout.log'), system_imports=imports,
                    capture_output_sha256=nr.digest(output), capture_timing_not_comparative=True)
                save(manifest, report)
            entry = dict(case['entries']['inductor'])
            base = Path(entry['library']).parent
            destination = target / 'inductor'
            destination.mkdir()
            for name in ('inductor.so', 'inductor.cpp', 'inductor-wrapper.py'):
                shutil.copy2(base / name, destination / name)
            entry.update(library=str(destination / 'inductor.so'), source=str(destination / 'inductor.cpp'))
            record['entries']['inductor'] = entry
            off, on = (record['entries'][name] for name in ('off', 'on'))
            previous = case['entries']['local']
            record.update(off_on_capture_bitwise_equal=off['capture_output_sha256'] == on['capture_output_sha256'],
                          off_matches_baseline_llvm=off['llvm_sha256'] == previous['llvm_sha256'],
                          off_matches_baseline_object=off['object_sha256'] == previous['object_sha256'],
                          off_on_llvm_identical=off['llvm_sha256'] == on['llvm_sha256'],
                          off_on_object_identical=off['object_sha256'] == on['object_sha256'])
            save(manifest, report)
            print(target.name, 'captured and FP64-checked', flush=True)
        require(source_hashes(source_root) == source and binary_hashes(binary) == closure, 'source or binary closure changed during capture')
        for identity in (runner, inherited, report['configuration_sha256'], tools):
            unchanged(identity)
        report.update(status='captured', source_unchanged=True, closure_unchanged=True, runner_unchanged=True,
                      artifact_sha256=hashes(p for p in directory.rglob('*') if p.is_file() and p != manifest),
                      host_after=host_observation(), finished_unix=time.time())
        save(manifest, report)
    except BaseException as error:
        report.update(status='error', error=f'{type(error).__name__}: {error}', finished_unix=time.time())
        save(manifest, report)
        raise


def verify_manifest(path):
    manifest = json.loads(path.read_text())
    require(manifest.get('format') == 'native-pointwise-v1' and manifest.get('status') == 'captured', 'incomplete or unknown capture')
    require(manifest['source_unchanged'] is True and manifest['closure_unchanged'] is True and manifest['runner_unchanged'] is True, 'capture identities changed')
    for key in ('artifact_sha256', 'runner_sha256', 'inherited_sha256', 'configuration_sha256', 'llvm_tools_sha256'):
        unchanged(manifest[key])
    require(source_hashes(Path(manifest['source_root'])) == manifest['source_sha256'], 'captured source tree changed')
    require(binary_hashes(Path(manifest['binary'])) == manifest['binary_closure_sha256'], 'captured binary closure changed')
    require(manifest['fixed_environment'] == FIXED_ENV, 'fixed option contract changed')
    require([nr.case_name(c['operation'], c['dimensions']) for c in manifest['cases']] == manifest['selected_cases'], 'capture cohort changed')
    for case in manifest['cases']:
        require(set(case['entries']) == set(VARIANTS), 'missing capture variant')
    return manifest


def invoke_variant(helper, data, dims, samples_count, warmup_ms, target_ms, expected, input_sha256):
    address, abi, mask, block, local, workspace, pointers, sizes, owners, output = data
    for key, owner in owners.items():
        if not key.startswith('input'):
            owner.data.fill(np.nan)
    samples, repetitions = np.empty(samples_count, np.float64), ctypes.c_uint64()
    code = helper(address, abi, len(pointers), mask, pointers, sizes, *dims, 8, block, local, workspace,
                  samples_count, warmup_ms, target_ms,
                  samples.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.byref(repetitions))
    require(code == 0 and np.isfinite(samples).all() and np.all(samples > 0), 'native helper/timing/workspace guard failed: ' + str(code))
    require(1 <= repetitions.value <= 1048576, 'invalid native repeat count')
    validation = nr.validate_output(output, expected)
    for owner in owners.values():
        owner.check()
    require([nr.array_digest(owners[f'input{i}'].data) for i in range(3)] == input_sha256, 'native entry mutated an input')
    return dict(samples_us=samples.tolist(), median_us=float(statistics.median(samples)), repetitions=repetitions.value,
                correctness=validation, output_sha256=nr.array_digest(output), inputs_unchanged=True,
                guard_elements=128 * len(owners), workspace_guards_passed=True)


def replay(args):
    path = args.prepared.resolve() / 'manifest.json'
    manifest = verify_manifest(path)
    manifest_sha256 = nr.digest(path)
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    only_verify = args.mode == 'verify'
    report = dict(status='running', started_unix=time.time(), argv=sys.argv, manifest=str(path), manifest_sha256=manifest_sha256,
                  runner_sha256=runners(directory), timing_not_comparative=only_verify,
                  fixed_candidate_comparison=not only_verify, default_planner_performance=False,
                  metric='native_entry_correctness_smoke' if only_verify else 'single_thread_native_entry_host_wall_us',
                  cpu_threads=1, samples=1 if only_verify else args.samples, warmup_ms=0 if only_verify else args.warmup_ms,
                  target_ms=1 if only_verify else args.target_ms, aligned_payload_bytes=64, per_allocation_guard_elements=128,
                  boundary='Common C++ timer: actual native entries, block traversal/reset, compiler-emitted libc/internal allocations included; Runtime/Python/JIT/caller allocations excluded.',
                  qualification='Fixed-candidate native-entry comparison with observed background load, not a quiet-machine claim or default-planner result. Guards are checked during each call; their released storage is not retained.',
                  host_before=host_observation(), cases=[])
    result_path = directory / 'results.json'
    save(result_path, report)
    try:
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            os.environ[name] = '1'
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        require(torch.__version__ == manifest['torch_version'] and torch.version.git_version == manifest['torch_git_version'], 'Torch build differs from frozen Inductor entry')
        library = ctypes.CDLL(manifest['helper'])
        helper = library.replay_native_rows
        ptr, u32, size = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_size_t
        helper.argtypes = [ptr, u32, u32, u32, ctypes.POINTER(ptr), ctypes.POINTER(size), u32, u32, u32,
                           u32, u32, size, u32, u32, u32, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint64)]
        helper.restype = ctypes.c_int
        for case in manifest['cases']:
            op, dims = case['operation'], case['dimensions']
            target = directory / nr.case_name(op, dims)
            target.mkdir()
            arrays = nr.load_inputs(Path(case['inputs']), case['input_shapes'])
            require([nr.array_digest(a) for a in arrays] == case['input_sha256'], 'replay input identity changed')
            expected = nr.reference(op, dims, arrays)
            require(nr.array_digest(expected) == case['expected_sha256'], 'FP64 oracle identity changed')
            inputs = {f'input{i}': nr.Guarded(a.size, a) for i, a in enumerate(arrays)}
            variants, libraries = {}, []
            for name in VARIANTS:
                entry = case['entries'][name]
                require(nr.digest(entry['library']) == entry['library_sha256'], 'native library changed')
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
            row = dict(operation=op, dimensions=dims, prechecks={}, results=[], output_sha256={})
            report['cases'].append(row)
            # Native prechecks are separate from the balanced timing cohort.
            # A previously validated Runtime wrapper is not enough to validate
            # the replay ABI, aliased output views or private workspace setup.
            for name, data in variants.items():
                row['prechecks'][name] = invoke_variant(helper, data, dims, 1, 0, 1, expected, case['input_sha256'])
            save(result_path, report)
            for round_id, order in enumerate((VARIANTS,) if only_verify else ORDERS):
                for name in order:
                    visit = invoke_variant(helper, variants[name], dims, report['samples'], report['warmup_ms'], report['target_ms'], expected, case['input_sha256'])
                    output_hash = visit['output_sha256']
                    require(output_hash == row['prechecks'][name]['output_sha256'], 'native output changed after precheck')
                    if name not in row['output_sha256']:
                        variants[name][-1].tofile(target / (name + '.f32'))
                        row['output_sha256'][name] = output_hash
                    require(row['output_sha256'][name] == output_hash, 'fixed entry output changed between rounds')
                    visit.update(round=round_id, order=order, variant=name)
                    row['results'].append(visit)
                    save(result_path, report)
                    print(target.name, round_id, name, f'{visit["median_us"]:.6f}', 'us (diagnostic)', flush=True)
            row['off_on_bitwise_equal'] = row['output_sha256']['off'] == row['output_sha256']['on']
            if not only_verify:
                row['summary_us'] = {name: statistics.median(v['median_us'] for v in row['results'] if v['variant'] == name) for name in VARIANTS}
                row['paired_ratios'] = {}
                for a, b in (('on', 'off'), ('on', 'inductor'), ('off', 'inductor')):
                    values = []
                    for round_id in range(len(ORDERS)):
                        pair = {v['variant']: v['median_us'] for v in row['results'] if v['round'] == round_id}
                        values.append(pair[a] / pair[b])
                    row['paired_ratios'][a + '/' + b] = dict(median=statistics.median(values), minimum=min(values), maximum=max(values), wins=sum(v < 1 for v in values), rounds=values)
            save(result_path, report)
        require(nr.digest(path) == manifest_sha256, 'manifest changed during replay')
        verify_manifest(path)
        unchanged(report['runner_sha256'])
        report.update(status='passed', finished_unix=time.time(), runner_unchanged=True,
                      artifact_sha256=hashes(p for p in directory.rglob('*') if p.is_file() and p != result_path),
                      host_after=host_observation())
        save(result_path, report)
    except BaseException as error:
        report.update(status='error', error=f'{type(error).__name__}: {error}', finished_unix=time.time())
        save(result_path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='mode', required=True)
    cap = commands.add_parser('capture')
    for name in ('binary', 'source-root', 'baseline-manifest', 'full-build-log', 'output'):
        cap.add_argument('--' + name, type=Path, required=True)
    cap.add_argument('--cases', nargs='+', required=True, help='Canonical names, e.g. rope-17x66 rope-1024x4098; or all frozen baseline cases')
    for name in ('verify', 'replay'):
        run = commands.add_parser(name)
        run.add_argument('--prepared', type=Path, required=True)
        run.add_argument('--output', type=Path, required=True)
        run.add_argument('--samples', type=int, default=7)
        run.add_argument('--warmup-ms', type=int, default=100)
        run.add_argument('--target-ms', type=int, default=30)
    args = parser.parse_args()
    if args.mode == 'capture':
        capture(args)
    else:
        require(1 <= args.samples <= 100 and 0 <= args.warmup_ms <= 10000 and 1 <= args.target_ms <= 10000, 'invalid native timing parameters')
        replay(args)


if __name__ == '__main__':
    main()
