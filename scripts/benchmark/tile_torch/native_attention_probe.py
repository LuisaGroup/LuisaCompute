#!/usr/bin/env python3
"""Frozen Tile native-entry versus handwritten FP32 CPU attention probes.

This is a benchmark, not a Tile lowering. online_neon explicitly changes the QK
reduction tree; dense_accelerate changes both the algorithm and storage, and may
use BLAS-internal FMA/packing/allocation. Neither reduces the FP32 input precision.
The common C++ timer excludes Python, JIT, caller allocations and validation, but
includes the entire native entry and any library-internal work. No Torch arm is
implemented: a future ATen adapter must declare its different allocation boundary.

Run each command in a fresh externally supervised subprocess; the experiment
driver must retain its complete stdout/stderr and impose a process-group timeout.
The replay itself invokes native code in-process and cannot recover from a native
crash or hang. selftest only invokes the helper's validation-only protocol.
"""
from __future__ import annotations

import argparse
import array
import ctypes as c
import json
import math
from pathlib import Path
import platform
import re
import shutil
import statistics
import struct
import subprocess
import sys
import time

import native_tile as tile

HERE = Path(__file__).resolve().parent
LLVM = Path('/opt/homebrew/opt/llvm@22/bin')
SOURCE = HERE / 'native_attention_probe.cpp'
HELPER = HERE / 'native_tile_replay.cpp'
ABI_RELATIVE = 'backends/simd/llvm/llvm_schedule_codegen.h'
ABI = HERE.parents[2] / 'src' / ABI_RELATIVE
FORMAT = 'native-attention-probe-v1'
MAX_ELEMENTS = 1 << 26
MAX_WORKSPACE = 16 * 1024 * 1024
ATOL = RTOL = 5e-5
VARIANTS = ('online_neon', 'dense_accelerate')
MACROS = ('B', 'H', 'KH', 'Q', 'K', 'D', 'DV')
SELFTEST_SHAPES = ((2, 6, 2, 3, 17, 7, 9), (1, 4, 1, 1, 1, 1, 1),
                   (1, 2, 2, 5, 5, 16, 12), (1, 1, 1, 2, 9, 5, 17))
PATTERNS = ('random', 'zero', 'alternating_large')
BOUNDARY = ('Common unchanged C++ timer; entire native entry, launch resets/block traversal, '
            'and BLAS-internal packing/allocations included. Python, Runtime, JIT, caller '
            'allocations/copies and validation excluded. Single requested calling thread, '
            'not CPU cycles, GPU time or multithread Runtime throughput.')
MATH = dict(input_precision='fp32', output_precision='fp32', reduced_precision=False,
            fast_math=False, compiler_fp_contract='off', finite_inputs_only=True,
            probe_reassociation='explicit four-lane QK tree for online_neon; BLAS reduction/FMA for dense_accelerate',
            baseline_math_unchanged=True, causal_alignment='bottom_right', dropout=0,
            cross_implementation_bitwise_required=False,
            numerical_scope='full oracle checks on retained bounded inputs, not universal equivalence for all finite FP32 values; extreme overflow and Tile -1e30 versus probe -infinity masking are not generalized',
            interpretation='handwritten diagnostic; not proof of a legal strict-MMA compiler rewrite')


def save(path, value):
    tile.save(path, value)


def sha(path):
    return tile.sha(path)


def integer(value, name, lower=1, upper=MAX_ELEMENTS):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f'{name}: expected integer in [{lower}, {upper}]')
    return value


def dimensions(value):
    if not isinstance(value, (list, tuple)) or len(value) != 7:
        raise ValueError('expected seven dimensions B,H,KH,Q,K,D,DV')
    b, h, kh, q, k, d, dv = (integer(x, name) for name, x in zip(MACROS, value))
    if q > k or h % kh:
        raise ValueError('attention requires Q <= K and H % KH == 0')
    inputs = [[b, h, q, d], [b, kh, k, d], [b, kh, k, dv]]
    output = [b, h, q, dv]
    if any(math.prod(shape) > MAX_ELEMENTS for shape in [*inputs, output, [q, k]]):
        raise ValueError('tensor extent exceeds bounded FP32 contract')
    workspace = max(q * k, 16) * 4
    if workspace > MAX_WORKSPACE:
        raise ValueError('probe workspace exceeds common helper capacity')
    return dict(dimensions=list(value), input_shapes=inputs, output_shape=output,
                output_elements=math.prod(output), workspace_bytes=workspace)


def launch(entry):
    if type(entry.get('abi')) is not int or entry['abi'] not in (0, 2):
        raise ValueError('unsupported Tile native ABI')
    for name in ('dispatch', 'block'):
        values = entry.get(name)
        if not isinstance(values, list) or len(values) != 3:
            raise ValueError('invalid Tile ' + name)
        for value in values:
            integer(value, name, upper=0xffffffff)
    width = integer(entry.get('packet_width'), 'packet_width', upper=64)
    if width & (width - 1) or math.prod(entry['block']) % width:
        raise ValueError('invalid packet width/block relation')
    if math.prod(entry['block']) > 0xffffffff:
        raise ValueError('Tile block overflow')
    grid = [(x - 1) // y + 1 for x, y in zip(entry['dispatch'], entry['block'])]
    if math.prod(grid) > 0xffffffff:
        raise ValueError('Tile grid overflow')
    integer(entry.get('workspace_bytes'), 'workspace_bytes', 0, MAX_WORKSPACE)
    if not isinstance(entry.get('symbol'), str) or not re.fullmatch(r'[-a-zA-Z$._0-9]+', entry['symbol']):
        raise ValueError('invalid Tile symbol')


def baseline_contract(entry):
    metadata = entry.get('metadata', {})
    required = dict(implementation='tile_xir_simd', operation='attention', precision='fp32',
                    source_kind='tile_lowering_source', fast_math=False, relaxed_precision=False,
                    attention_qk='mma', attention_pv='mma', source_reduction_policy='unordered_tree')
    if entry.get('capture_kind') != 'llm':
        raise ValueError('expected frozen LLM attention capture')
    for key, expected in required.items():
        if type(metadata.get(key)) is not type(expected) or metadata[key] != expected:
            raise ValueError('baseline math/metadata mismatch: ' + key)
    shape = dimensions(metadata.get('dimensions'))
    for name in ('input_shapes', 'output_shape'):
        if metadata.get(name) != shape[name]:
            raise ValueError('baseline tensor shape mismatch: ' + name)
    for extent in [*metadata['input_shapes'], metadata['output_shape']]:
        if not isinstance(extent, list) or any(type(x) is not int for x in extent):
            raise ValueError('baseline tensor dimensions must be strict integers')
    if entry.get('input_files') != ['input0.f32', 'input1.f32', 'input2.f32']:
        raise ValueError('expected exactly three ordered Q/K/V buffers')
    if type(entry.get('output_elements')) is not int or entry['output_elements'] != shape['output_elements']:
        raise ValueError('baseline output extent mismatch')
    check = metadata.get('correctness', {})
    if (type(check.get('checks')) is not int or check['checks'] != 2 or
            type(check.get('elements_per_check')) is not int or check['elements_per_check'] != shape['output_elements']):
        raise ValueError('expected complete capture correctness checks')
    for record in (entry, check):
        for key, value in (('atol', ATOL), ('rtol', RTOL)):
            if type(record.get(key)) not in (int, float) or record[key] != value:
                raise ValueError('baseline tolerance mismatch: ' + key)
    launch(entry)
    realization = metadata.get('realization', '')
    match = re.search(r'\bW(\d+), (\d+) workers/block', realization)
    workspace = re.search(r'\bprivate_workspace_bytes=(\d+)', realization)
    if (not match or not workspace or [int(match[1]), int(match[2])] != [entry['packet_width'], entry['block'][0]] or
            entry['block'][1:] != [1, 1] or int(workspace[1]) != entry['workspace_bytes'] or
            metadata.get('dispatch') != entry['dispatch']):
        raise ValueError('prepared launch differs from actual capture metadata')
    return shape


def safe_file(directory, relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError('unsafe artifact path')
    name = Path(relative)
    if name.is_absolute() or '..' in name.parts:
        raise ValueError('unsafe artifact path')
    file = directory / name
    if any(part.is_symlink() for part in [file, *file.parents] if part != directory.parent):
        # resolve the bundle directory once at entry; no symlink inside it is admitted.
        raise ValueError('symlinked artifact path: ' + relative)
    if not file.is_file():
        raise ValueError('missing artifact: ' + relative)
    return file


def verify_files(directory, files):
    if not isinstance(files, dict) or not files:
        raise ValueError('missing artifact inventory')
    for relative, digest in files.items():
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('invalid artifact digest')
        if sha(safe_file(directory, relative)) != digest:
            raise ValueError('artifact changed: ' + relative)


def inventory(directory, exclude=()):
    return {str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob('*'))
            if p.is_file() and str(p.relative_to(directory)) not in exclude}


def payloads(directory, shape):
    result = []
    for i, extent in enumerate(shape['input_shapes']):
        raw = (directory / f'input{i}.f32').read_bytes()
        if len(raw) != 4 * math.prod(extent):
            raise ValueError('FP32 input byte extent mismatch')
        values = array.array('f', raw)
        if any(not math.isfinite(value) for value in values):
            raise ValueError('finite FP32 inputs required')
        result.append(raw)
    return result


def oracle(inputs, shape):
    """Independent dense NumPy FP64 reference; no reuse of probe recurrence."""
    import numpy as np
    b, h, kh, q, k, d, dv = shape['dimensions']
    arrays = []
    for raw, extent in zip(inputs, shape['input_shapes']):
        if len(raw) != 4 * math.prod(extent):
            raise ValueError('oracle input byte extent mismatch')
        values = np.frombuffer(raw, dtype='<f4').reshape(extent).astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError('nonfinite oracle input')
        arrays.append(values)
    if len(arrays) != 3:
        raise ValueError('oracle requires three inputs')
    query, key, value = arrays
    result = np.empty(shape['output_shape'], dtype=np.float64)
    scale = float(np.float32(1) / np.sqrt(np.float32(d)))
    valid = np.arange(k)[None, :] <= np.arange(q)[:, None] + k - q
    for batch in range(b):
        for head in range(h):
            group = head // (h // kh)
            score = query[batch, head] @ key[batch, group].T * scale
            score[~valid] = -np.inf
            probability = np.exp(score - score.max(axis=1, keepdims=True))
            probability /= probability.sum(axis=1, keepdims=True)
            result[batch, head] = probability @ value[batch, group]
    if not np.isfinite(result).all():
        raise ValueError('nonfinite independent FP64 oracle')
    return result.astype('<f8').tobytes(), dict(implementation='independent_numpy_dense_fp64',
                                             numpy_version=np.__version__, scale=scale,
                                             causal_alignment='bottom_right', elements=result.size)


def check_bytes(actual, expected):
    if len(actual) % 4 or len(expected) % 8:
        raise ValueError('invalid output/oracle byte extent')
    return tile.check_output(array.array('f', actual), array.array('d', expected), ATOL, RTOL)


def command(argv, directory, name):
    argv = list(map(str, argv))
    record = dict(argv=argv, started_unix=time.time(), cwd=str(Path.cwd()))
    save(directory / f'{name}.command.json', record)
    try:
        completed = subprocess.run(argv, capture_output=True, timeout=300)
        stdout, stderr = completed.stdout, completed.stderr
        record['returncode'] = completed.returncode
    except (subprocess.TimeoutExpired, OSError) as error:
        stdout = getattr(error, 'stdout', None) or b''
        stderr = getattr(error, 'stderr', None) or b''
        record.update(error=str(error), timeout=isinstance(error, subprocess.TimeoutExpired))
        raise
    finally:
        (directory / f'{name}.stdout').write_bytes(locals().get('stdout', b''))
        (directory / f'{name}.stderr').write_bytes(locals().get('stderr', b''))
        record['finished_unix'] = time.time()
        save(directory / f'{name}.command.json', record)
    if completed.returncode:
        raise RuntimeError(f'{name}: exit {completed.returncode}: {stderr.decode(errors="replace")[-2000:]}')
    return stdout.decode(errors='replace')


def require_host():
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise ValueError('native probes require Darwin arm64')
    version = platform.mac_ver()[0].split('.')
    if not version or not version[0].isdigit() or int(version[0]) < 15:
        raise ValueError('BLAS threading contract requires macOS >= 15')
    if sys.byteorder != 'little':
        raise ValueError('little-endian payload ABI required')


def compile_flags(directory):
    return ['-std=c++20', '-O3', '-fno-fast-math', '-ffp-contract=off',
            '-dynamiclib', '-mmacosx-version-min=15.0', '-I' + str(directory)]


def compile_bundle(directory, shape, llvm):
    source_hashes = {name: sha(directory / name) for name in
                     ('native_attention_probe.cpp', 'native_tile_replay.cpp', ABI_RELATIVE)}
    # Preserve argv[0] == clang++: resolving its final symlink to clang would
    # silently change the driver mode and omit automatic C++ runtime linkage.
    compiler = llvm.resolve(strict=True) / 'clang++'
    if not compiler.is_file():
        raise ValueError('missing LLVM C++ driver')
    version = command([compiler, '--version'], directory, 'compiler')
    if not re.search(r'clang version 22\.', version):
        raise ValueError('this experiment requires LLVM 22 clang++')
    before = sha(compiler)
    command([compiler, *compile_flags(directory), directory / 'native_tile_replay.cpp',
             '-o', directory / 'replay.dylib'], directory, 'compile-helper')
    macros = ['-DATTN_' + name + '=' + str(value) for name, value in zip(MACROS, shape['dimensions'])]
    command([compiler, *compile_flags(directory), *macros, directory / 'native_attention_probe.cpp',
             '-framework', 'Accelerate', '-o', directory / 'probe.dylib'], directory, 'compile-probe')
    command(['/usr/bin/otool', '-L', directory / 'probe.dylib'], directory, 'probe-linked-libraries')
    command(['/usr/bin/otool', '-L', directory / 'replay.dylib'], directory, 'helper-linked-libraries')
    if sha(compiler) != before:
        raise ValueError('compiler changed during preparation')
    if source_hashes != {name: sha(directory / name) for name in source_hashes}:
        raise ValueError('frozen source changed during compilation')
    return dict(path=str(compiler), sha256=before, version=version,
                flags=compile_flags(directory), macros=macros,
                loader_attestation='source/built-dylib hashes and otool dependencies; not full OS shared-cache/library closure')


def copy_sources(directory, baseline=None):
    files = [(SOURCE, 'native_attention_probe.cpp'), (Path(__file__), 'native_attention_probe.py'),
             (Path(tile.__file__), 'native_tile.py'), (HERE / 'test_native_attention_probe.py', 'test_native_attention_probe.py')]
    for live, relative in ((HELPER, 'native_tile_replay.cpp'), (ABI, ABI_RELATIVE)):
        source = live if baseline is None else baseline / relative
        if baseline is not None and sha(live) != sha(source):
            raise ValueError('live helper/ABI differs from frozen baseline: ' + relative)
        files.append((source, relative))
    before = {str(source): sha(source) for source, _ in files}
    for source, relative in files:
        destination = directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    if before != {str(source): sha(source) for source, _ in files}:
        raise ValueError('probe/helper sources changed while copying')
    return before


def prepare(args):
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    report = dict(format=FORMAT, status='preparing', started_unix=time.time(), math=MATH, boundary=BOUNDARY)
    save(directory / 'probe.json', report)
    try:
        require_host()
        original = args.tile.resolve(strict=True)
        original_digest = sha(original)
        entry = tile.verify(original)
        verify_files(original.parent, entry['files'])
        shape = baseline_contract(entry)
        target = directory / 'tile'
        target.mkdir()
        for relative in [*entry['files'], 'prepared.json']:
            source = safe_file(original.parent, relative)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        if sha(target / 'prepared.json') != original_digest:
            raise ValueError('baseline manifest changed during copy')
        tile.verify(target / 'prepared.json')
        sources = copy_sources(directory, target)
        inputs = payloads(target, shape)
        expected, reference = oracle(inputs, shape)
        original_expected = (target / 'expected.f64').read_bytes()
        tile.check_output(array.array('d', original_expected), array.array('d', expected), ATOL, RTOL)
        capture = check_bytes((target / 'captured.f32').read_bytes(), expected)
        (directory / 'expected.f64').write_bytes(expected)
        compiler = compile_bundle(directory, shape, args.llvm.resolve(strict=True))
        tile.verify(original)
        if sha(original) != original_digest:
            raise ValueError('baseline manifest changed during preparation')
        report.update(status='prepared', finished_unix=time.time(), shape=shape,
                      baseline='tile/prepared.json', baseline_sha256=original_digest,
                      original_baseline=str(original), sources_sha256=sources, compiler=compiler,
                      oracle=reference, captured_vs_independent_oracle=capture,
                      variants={name: probe_entry(name, shape) for name in VARIANTS},
                      files=inventory(directory, ('probe.json',)))
        save(directory / 'probe.json', report)
        print(directory / 'probe.json', flush=True)
    except Exception as error:
        report.update(status='error', error=str(error), finished_unix=time.time())
        save(directory / 'probe.json', report)
        raise


def probe_entry(variant, shape):
    if variant not in VARIANTS:
        raise ValueError('unsupported probe variant')
    return dict(name=variant, symbol='attention_' + variant, abi=0, dispatch=[1, 1, 1], block=[1, 1, 1],
                packet_width=1, workspace_bytes=shape['workspace_bytes'],
                source_kind='handwritten_probe_not_tile_lowering',
                algorithm='kv16_online_rows' if variant == 'online_neon' else 'dense_qk_softmax_pv_blas',
                useful_score_scratch_bytes=64 if variant == 'online_neon' else 4 * shape['dimensions'][3] * shape['dimensions'][4],
                full_declared_scratch_allocated=True)


def verify(bundle):
    path = Path(bundle).resolve(strict=True)
    report = json.loads(path.read_text())
    if report.get('format') != FORMAT or report.get('status') != 'prepared':
        raise ValueError('incomplete or unsupported probe bundle')
    verify_files(path.parent, report.get('files'))
    for source, relative in ((Path(__file__), 'native_attention_probe.py'), (Path(tile.__file__), 'native_tile.py')):
        if sha(source) != report['files'].get(relative):
            raise ValueError('running verifier/runner differs from frozen bundle')
    if report.get('baseline') != 'tile/prepared.json':
        raise ValueError('invalid baseline manifest path')
    baseline_path = path.parent / report['baseline']
    if sha(baseline_path) != report.get('baseline_sha256'):
        raise ValueError('baseline manifest identity mismatch')
    baseline = tile.verify(baseline_path)
    shape = baseline_contract(baseline)
    if report.get('shape') != shape or report.get('math') != MATH:
        raise ValueError('probe shape/math contract changed')
    if report.get('variants') != {name: probe_entry(name, shape) for name in VARIANTS}:
        raise ValueError('probe entry contract changed')
    for relative in ('native_tile_replay.cpp', ABI_RELATIVE):
        if report['files'].get(relative) != baseline['files'].get(relative):
            raise ValueError('common helper/ABI differs from frozen baseline')
    needed = ['probe.dylib', 'replay.dylib', 'expected.f64', 'native_attention_probe.cpp',
              'tile/prepared.json', 'tile/kernel.o', 'tile/kernel.dylib', 'tile/expected.f64',
              *['tile/' + name for name in baseline['input_files']]]
    if any(name not in report['files'] for name in needed):
        raise ValueError('required frozen artifact missing from inventory')
    return report, baseline


def configure_helper(library):
    p, u32, size, d = c.c_void_p, c.c_uint32, c.c_size_t, c.c_double
    helper = library.replay_native_tile
    helper.argtypes = [p, u32, u32, c.POINTER(p), c.POINTER(size), c.POINTER(u32), c.POINTER(c.POINTER(d)),
                      c.POINTER(u32), c.POINTER(u32), u32, size, u32, u32, u32, d, d,
                      c.POINTER(d), c.POINTER(c.c_uint64), c.POINTER(d)]
    helper.restype = c.c_int
    return helper


def configure_probe(library, workspace):
    for name, result in (('attention_probe_contract_version', c.c_uint32),
                         ('attention_probe_workspace_bytes', c.c_uint64),
                         ('attention_probe_set_single_threaded', c.c_int32),
                         ('attention_probe_threading', c.c_uint32)):
        function = getattr(library, name)
        function.argtypes, function.restype = [], result
    version = library.attention_probe_contract_version()
    actual_workspace = library.attention_probe_workspace_bytes()
    if version != 1 or actual_workspace != workspace:
        raise ValueError('probe binary version/workspace contract mismatch')
    result = library.attention_probe_set_single_threaded()
    mode = library.attention_probe_threading()
    if result != 0 or mode != 1:
        raise ValueError(f'BLAS single-thread admission failed: set={result}, get={mode}')
    return dict(api='BLASSetThreading/BLASGetThreading', set_status=result, observed_enum=mode,
                expected_enum=1, scope='calling-thread TLS; set/query outside common timer',
                limitation='API acknowledgement, not independent profiling of every internal library worker',
                contract_version=version, workspace_bytes=actual_workspace)


def native_visit(helper, library, entry, inputs, expected_bytes, directory, filename, *,
                 samples, warmup_ms, target_ms):
    """One common-helper visit; all caller allocation occurs before its timer."""
    p, u32, size, d = c.c_void_p, c.c_uint32, c.c_size_t, c.c_double
    expected = (d * (len(expected_bytes) // 8)).from_buffer_copy(expected_bytes)
    nan_output = struct.pack('<f', float('nan')) * len(expected)
    owners = [c.create_string_buffer(raw, len(raw)) for raw in [*inputs, nan_output]]
    pointers = (p * 4)(*(c.addressof(owner) for owner in owners))
    sizes = (size * 4)(*(c.sizeof(owner) for owner in owners))
    writable = (u32 * 4)(0, 0, 0, 1)
    oracles = (c.POINTER(d) * 4)(None, None, None, expected)
    values, repetitions, error = (d * samples)(), c.c_uint64(), d()
    result = helper(c.cast(getattr(library, entry['symbol']), p), entry['abi'], 4,
                    pointers, sizes, writable, oracles, (u32 * 3)(*entry['dispatch']), (u32 * 3)(*entry['block']),
                    entry['packet_width'], entry['workspace_bytes'], samples, warmup_ms, target_ms,
                    ATOL, RTOL, values, c.byref(repetitions), c.byref(error))
    output = bytes(owners[-1])
    (directory / filename).write_bytes(output)
    row = dict(valid=False, returncode=result, output=filename, output_sha256=sha(directory / filename),
               samples_us=list(values), repetitions=repetitions.value, helper_max_abs_error=error.value)
    # Save the helper's raw result even when it rejects a guard or numerical error.
    save(directory / (filename + '.json'), dict(row, samples_us=[x if math.isfinite(x) else None for x in values],
                                               helper_max_abs_error=error.value if math.isfinite(error.value) else None))
    if result != 0:
        raise ValueError(f'native helper rejected {entry["symbol"]}: code={result}, evidence={filename}.json')
    if samples and (repetitions.value == 0 or any(not math.isfinite(v) or v <= 0 for v in values)):
        raise ValueError('native helper returned invalid timing samples')
    if not samples and repetitions.value != 0:
        raise ValueError('validation-only call unexpectedly timed repetitions')
    if not math.isfinite(error.value):
        raise ValueError('native helper returned nonfinite error')
    validation = check_bytes(output, expected_bytes)
    if any(bytes(owner) != raw for owner, raw in zip(owners[:-1], inputs)):
        raise ValueError('caller input changed')
    row.update(valid=True, correctness=validation, all_guards_passed=True, inputs_unchanged=True,
               median_us=statistics.median(values) if samples else None)
    save(directory / (filename + '.json'), row)
    return row


def summarize(visits, cycles, candidate):
    if len(visits) != 4 * cycles:
        raise ValueError('incomplete ABBA cohort')
    ratios = []
    for cycle in range(cycles):
        group = visits[4 * cycle:4 * cycle + 4]
        for position, (row, variant) in enumerate(zip(group, ('tile', candidate, candidate, 'tile'))):
            if (row.get('valid') is not True or row.get('variant') != variant or
                    row.get('cycle') != cycle or row.get('position') != position or
                    type(row.get('median_us')) not in (int, float) or
                    not math.isfinite(row['median_us']) or row['median_us'] <= 0):
                raise ValueError('invalid or reordered ABBA visit')
        ratios.extend((group[1]['median_us'] / group[0]['median_us'], group[2]['median_us'] / group[3]['median_us']))
    return dict(summary_us={name: statistics.median(v['median_us'] for v in visits if v['variant'] == name)
                            for name in ('tile', candidate)},
                candidate_over_baseline=dict(baseline='tile', candidate=candidate, pairs=ratios,
                                             median=statistics.median(ratios), minimum=min(ratios), maximum=max(ratios)))


def replay(args):
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    bundle = args.bundle.resolve(strict=True)
    report = dict(status='running', started_unix=time.time(), metric='single_thread_native_entry_host_wall_us',
                  boundary=BOUNDARY, math=MATH, bundle=str(bundle), bundle_sha256=sha(bundle),
                  runner_sha256=sha(__file__), platform=platform.platform(),
                  options={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  order_policy='ABBA per cycle; all predeclared visits retained, no selective retry',
                  stderr_capture='external supervising process must retain stdout/stderr, including native fd diagnostics',
                  visits=[dict(cycle=cycle, position=position, variant=name, valid=False, status='NotRun')
                          for cycle in range(args.cycles) for position, name in enumerate(('tile', args.variant, args.variant, 'tile'))])
    save(directory / 'results.json', report)
    try:
        require_host()
        prepared, baseline = verify(bundle)
        base = bundle.parent
        inputs = payloads(base / 'tile', prepared['shape'])
        expected, reference = oracle(inputs, prepared['shape'])
        if expected != (base / 'expected.f64').read_bytes():
            raise ValueError('independent FP64 oracle bytes differ from frozen preparation')
        libraries = dict(tile=c.CDLL(str(base / 'tile/kernel.dylib')),
                         probe=c.CDLL(str(base / 'probe.dylib')))
        helper_library = c.CDLL(str(base / 'replay.dylib'))
        helper = configure_helper(helper_library)
        report.update(oracle=reference, threading=configure_probe(libraries['probe'], prepared['shape']['workspace_bytes']),
                      common_helper_sha256=sha(base / 'replay.dylib'),
                      entry_contracts=dict(tile={key: baseline[key] for key in ('symbol', 'abi', 'dispatch', 'block', 'packet_width', 'workspace_bytes')},
                                           probe=prepared['variants'][args.variant]),
                      interpretation='Handwritten probes versus unchanged compiler object; algorithms, reduction order and resources differ.')
        save(directory / 'results.json', report)
        output_hashes = {}
        for row in report['visits']:
            name = row['variant']
            row['status'] = 'running'
            save(directory / 'results.json', report)
            if libraries['probe'].attention_probe_threading() != 1:
                raise ValueError('BLAS thread mode changed before visit')
            entry = baseline if name == 'tile' else prepared['variants'][name]
            observed = native_visit(helper, libraries['tile' if name == 'tile' else 'probe'], entry, inputs, expected,
                                    directory, f'visit-{row["cycle"]}-{row["position"]}.f32',
                                    samples=args.samples, warmup_ms=args.warmup_ms, target_ms=args.target_ms)
            row.update(observed, status='passed')
            row['blas_threading_after'] = libraries['probe'].attention_probe_threading()
            if row['blas_threading_after'] != 1:
                raise ValueError('BLAS thread mode changed during visit')
            previous = output_hashes.setdefault(name, row['output_sha256'])
            if previous != row['output_sha256']:
                raise ValueError('fixed variant output changed between visits')
            save(directory / 'results.json', report)
            print(name, row['cycle'], row['position'], row['median_us'], flush=True)
        verify(bundle)
        if sha(bundle) != report['bundle_sha256'] or sha(__file__) != report['runner_sha256']:
            raise ValueError('bundle manifest or runner changed during replay')
        report.update(summarize(report['visits'], args.cycles, args.variant))
        report.update(status='passed', finished_unix=time.time(), artifacts_unchanged=True,
                      deterministic_output_sha256=output_hashes, files=inventory(directory, ('results.json',)))
        save(directory / 'results.json', report)
    except Exception as error:
        for row in report['visits']:
            if row['status'] == 'running':
                row.update(status='error', valid=False, error=str(error))
        report.update(status='error', error=str(error), finished_unix=time.time())
        save(directory / 'results.json', report)
        raise


def pattern_inputs(shape, pattern):
    import numpy as np
    if pattern not in PATTERNS:
        raise ValueError('unknown selftest pattern')
    random = np.random.default_rng(1729)
    result = []
    for index, extent in enumerate(shape['input_shapes']):
        count = math.prod(extent)
        if pattern == 'zero':
            values = np.zeros(count, dtype='<f4')
        elif pattern == 'random':
            values = random.uniform(-0.5, 0.5, count).astype('<f4')
        else:
            coordinates = np.arange(count)
            # Large, finite logits stress stable softmax without FP32 overflow.
            values = (((coordinates + index) % 2 * 2 - 1) * 16 + (coordinates % 7) * 0.03125).astype('<f4')
        result.append(values.tobytes())
    return result


def selftest(args):
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', metric='validation_only_no_timer', started_unix=time.time(),
                  shapes=[list(shape) for shape in SELFTEST_SHAPES], patterns=list(PATTERNS),
                  variants=list(VARIANTS), expected_checks=24, checks=[], math=MATH)
    save(directory / 'results.json', report)
    try:
        require_host()
        for index, extent in enumerate(SELFTEST_SHAPES):
            base = directory / f'shape-{index}'
            base.mkdir()
            shape = dimensions(extent)
            sources = copy_sources(base)
            compiler = compile_bundle(base, shape, args.llvm.resolve(strict=True))
            library = c.CDLL(str(base / 'probe.dylib'))
            helper_library = c.CDLL(str(base / 'replay.dylib'))
            helper = configure_helper(helper_library)
            threading = configure_probe(library, shape['workspace_bytes'])
            frozen = inventory(base)
            save(base / 'selftest-prepared.json', dict(shape=shape, sources_sha256=sources, compiler=compiler,
                                                     threading=threading, files=frozen))
            for pattern in PATTERNS:
                destination = base / pattern
                destination.mkdir()
                inputs = pattern_inputs(shape, pattern)
                for i, raw in enumerate(inputs):
                    (destination / f'input{i}.f32').write_bytes(raw)
                expected, reference = oracle(inputs, shape)
                (destination / 'expected.f64').write_bytes(expected)
                for variant in VARIANTS:
                    row = dict(shape=index, pattern=pattern, variant=variant, status='running', valid=False)
                    report['checks'].append(row)
                    save(directory / 'results.json', report)
                    if library.attention_probe_threading() != 1:
                        raise ValueError('BLAS thread mode changed during selftest')
                    row.update(native_visit(helper, library, probe_entry(variant, shape), inputs, expected,
                                            destination, variant + '.f32', samples=0, warmup_ms=0, target_ms=0),
                               status='passed', oracle=reference)
                    if library.attention_probe_threading() != 1:
                        raise ValueError('BLAS thread mode changed during selftest call')
                    row['artifact_directory'] = str(destination.relative_to(directory))
                    save(directory / 'results.json', report)
            verify_files(base, frozen)
        if len(report['checks']) != report['expected_checks'] or not all(row['valid'] for row in report['checks']):
            raise ValueError('incomplete selftest matrix')
        report.update(status='passed', finished_unix=time.time(), files=inventory(directory, ('results.json',)))
        save(directory / 'results.json', report)
    except Exception as error:
        report.update(status='error', error=str(error), finished_unix=time.time())
        save(directory / 'results.json', report)
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--tile', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--llvm', type=Path, default=LLVM)
    run = commands.add_parser('replay')
    run.add_argument('--bundle', type=Path, required=True)
    run.add_argument('--variant', choices=VARIANTS, required=True)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--cycles', type=int, default=3)
    run.add_argument('--samples', type=int, default=7)
    run.add_argument('--warmup-ms', type=int, default=30)
    run.add_argument('--target-ms', type=int, default=15)
    test = commands.add_parser('selftest')
    test.add_argument('--output', type=Path, required=True)
    test.add_argument('--llvm', type=Path, default=LLVM)
    args = parser.parse_args(argv)
    if args.command == 'replay':
        try:
            integer(args.cycles, 'cycles', 1, 20)
            integer(args.samples, 'samples', 1, 100)
            integer(args.warmup_ms, 'warmup-ms', 0, 10000)
            integer(args.target_ms, 'target-ms', 1, 10000)
        except ValueError as error:
            parser.error(str(error))
    return args


def main():
    args = parse_args()
    {'prepare': prepare, 'replay': replay, 'selftest': selftest}[args.command](args)


if __name__ == '__main__':
    main()
