#!/usr/bin/env python3
"""Offline audit of this frozen six-case native attention diagnostic, no native execution.
Only audit.json is written. Original absolute producer/tool paths are historical
receipts, never opened; all checked evidence resolves inside this archive.
"""
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import statistics as stats
import tarfile
import sys

os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
import numpy as np

ROOT = Path(__file__).resolve().parent
VARIANTS = ['online_neon', 'dense_accelerate']
CASES = [('decode-mha-d64', [1, 8, 8, 1, 2048, 64, 64], [1, 16], 0, False),
         ('decode-gqa-d80', [1, 8, 2, 1, 2053, 80, 96], [1, 16], 8, False),
         ('decode-long-kv', [1, 16, 4, 1, 8193, 128, 128], [1, 16], 8, False),
         ('prefill-q4', [1, 4, 2, 32, 65, 32, 32], [4, 16], 0, True),
         ('prefill-q8', [1, 4, 2, 64, 129, 32, 48], [8, 16], 0, True),
         ('batch-gqa-q4', [2, 6, 2, 17, 67, 40, 48], [4, 16], 8, True)]
ABI = 'backends/simd/llvm/llvm_schedule_codegen.h'
REPORT = dict(status='running', scope='six frozen CPU cases, two handwritten probes; no Torch/GPU',
              captures=[], comparisons=[dict(case=n, variant=v, status='NotRun') for n, *_ in CASES for v in VARIANTS],
              limitations=['Handwritten reference is not Tile lowering or a proven legal strict-MMA transform.',
                           'Full finite-input checks cover the archived payloads, not all finite FP32 inputs or extreme sentinel/overflow behavior.',
                           'Common native-entry wall timer excludes Python/JIT/caller allocation but includes BLAS internals and launch traversal.',
                           'BLAS TLS acknowledgement is not independent profiling of every library thread.',
                           'Historical producer/tools are fingerprint receipts, not archived binaries, loader closure or reproducible-build proof.',
                           'Paired min/max is observed spread, not a confidence interval; background host work is not controlled.'])


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(path.read_text())


def safe(base, name):
    rel = PurePosixPath(name)
    require(name and not rel.is_absolute() and '..' not in rel.parts and '\\' not in name, 'unsafe path: ' + name)
    file = base / name
    require(file.is_file() and not any((base / Path(*rel.parts[:i])).is_symlink() for i in range(1, len(rel.parts) + 1)), 'nonregular artifact: ' + name)
    return file


def files(base, inventory):
    require(bool(inventory), 'empty inventory')
    for name, expected in inventory.items():
        require(sha(safe(base, name)) == expected, 'file hash mismatch: ' + str(base / name))


def reference(base, dims):
    b, h, kh, q, k, d, dv = dims
    require(all(type(x) is int and 0 < x <= 2**26 for x in dims) and q <= k and h % kh == 0, 'shape admission')
    shapes = [(b, h, q, d), (b, kh, k, d), (b, kh, k, dv)]
    tensors = []
    for i, shape in enumerate(shapes):
        require(math.prod(shape) <= 2**26, 'input too large')
        raw = (base / f'input{i}.f32').read_bytes()
        require(len(raw) == math.prod(shape) * 4, 'input length')
        value = np.frombuffer(raw, dtype='<f4').astype(np.float64).reshape(shape)
        require(np.isfinite(value).all(), 'nonfinite input')
        tensors.append(value)
    out = np.empty((b, h, q, dv), dtype=np.float64)
    scale = float(np.float32(1) / np.sqrt(np.float32(d)))
    for batch in range(b):
        for head in range(h):
            group = head // (h // kh)
            # Independent dense FP64 formula, not the producer's online recurrence
            # or the runner's BLAS matmul: explicit einsum sums, optimize disabled.
            score = np.einsum('qd,kd->qk', tensors[0][batch, head], tensors[1][batch, group], optimize=False) * scale
            score[np.arange(k)[None, :] > np.arange(q)[:, None] + k - q] = -np.inf
            weight = np.exp(score - score.max(axis=1, keepdims=True))
            weight /= weight.sum(axis=1, keepdims=True)
            out[batch, head] = np.einsum('qk,kd->qd', weight, tensors[2][batch, group], optimize=False)
    return out.ravel()


def numerical(path, expected, dtype='<f4'):
    raw = path.read_bytes()
    require(len(raw) == len(expected) * np.dtype(dtype).itemsize, 'output length: ' + str(path))
    actual = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), 'nonfinite output/oracle')
    tolerance = 1e-11 if dtype == '<f8' else 5e-5
    error = np.abs(actual - expected)
    require(np.all(error <= tolerance * (1 + np.abs(expected))), 'full oracle failure: ' + str(path))
    return float(error.max())


def valid_output(base, row, expected):
    require(row['valid'] is True and row['returncode'] == 0 and row['all_guards_passed'] is True and row['inputs_unchanged'] is True, 'failed native checks')
    output = safe(base, row['output'])
    require(sha(output) == row['output_sha256'], 'output identity')
    check = row['correctness']
    require(check['elements'] == len(expected) and check['atol'] == check['rtol'] == 5e-5, 'correctness extent/tolerance')
    maximum = numerical(output, expected)
    require(math.isclose(maximum, check['max_abs_error'], abs_tol=1e-11), 'reported oracle error differs')
    return maximum


def field(text, key, boolean=False):
    matches = re.findall(r'\b' + key + '=' + ('(true|false)' if boolean else r'(\d+)') + r'\b', text)
    require(len(matches) == 1, 'missing/duplicate metadata ' + key)
    return matches[0] == 'true' if boolean else int(matches[0])


def sources_and_gates(plan, provenance):
    archive = ROOT / 'sources.tar.gz'
    require(sha(archive) == provenance['source_archive_sha256'] and archive.stat().st_size == provenance['source_archive_bytes'], 'source archive identity')
    inventory = provenance['source_inventory']
    require(len(inventory) == 523 and provenance['source_sha256'] == {n: r['sha256'] for n, r in inventory.items()}, 'source inventory extent/schema')
    with tarfile.open(archive, 'r:gz') as stream:
        members = stream.getmembers()
        require(len(members) == len({m.name for m in members}) == 523, 'duplicate/missing archive members')
        require(set(m.name for m in members) == set(inventory), 'archive member set')
        for member in members:
            name = PurePosixPath(member.name)
            require(member.isfile() and not name.is_absolute() and '..' not in name.parts and '\\' not in member.name, 'unsafe tar member')
            raw = stream.extractfile(member).read()
            require(len(raw) == inventory[member.name]['bytes'] and digest(raw) == inventory[member.name]['sha256'], 'source member hash: ' + member.name)
    for original, expected in plan['identities'].items():
        if original.endswith('/benchmark_tile_xir'):
            require(expected == provenance['binary_fingerprints']['benchmark_tile_xir']['sha256'], 'producer receipt differs')
        else:
            name = '_experiment/' + Path(original).name if Path(original).name in ('run.py', 'freeze.py') else str(Path(original).relative_to(provenance['main_root']))
            require(inventory[name]['sha256'] == expected, 'plan/source identity: ' + name)
    for name in ('run.py', 'freeze.py', 'plan.json'):
        require(sha(ROOT / name) == inventory['_experiment/' + name]['sha256'], 'experiment source/plan changed')
    for member in provenance['validation_members']:
        require(sha(ROOT / 'validation' / member.removeprefix('_validation/')) == inventory[member]['sha256'], 'gate evidence changed')
    validation = ROOT / 'validation'
    require(re.search(r'Ran 29 tests.*\n\nOK\s*$', (validation / 'python-tests-final.log').read_text(), re.S), 'pure unit gate')
    ctest = (validation / 'ctest.log').read_text()
    require(re.search(r'100% tests passed(?:, 0 tests failed)? out of 4', ctest), 'CTest gate')
    for name in ('test_simd_llvm_schedule_codegen', 'test_tile_xir_target_info', 'test_tile_xir_runtime', 'test_tile_xir_llm'):
        require(re.search(re.escape(name) + r'.*Passed', ctest), 'missing CTest ' + name)
    require((validation / 'build-runner-final.log').stat().st_size > 0 and 'exit 0' in (validation / 'notes.md').read_text(), 'final selected build log/declaration')
    REPORT['limitations'].append('Build logs lack a machine-readable exit receipt; final full-build success is a frozen supervisor declaration in validation/notes.md, not independently inferred from progress text.')
    test = load(validation / 'native-selftest/results.json')
    require(test['shapes'] == [[2, 6, 2, 3, 17, 7, 9], [1, 4, 1, 1, 1, 1, 1], [1, 2, 2, 5, 5, 16, 12], [1, 1, 1, 2, 9, 5, 17]] and test['patterns'] == ['random', 'zero', 'alternating_large'] and test['variants'] == VARIANTS, 'fixed selftest matrix')
    require(test['status'] == 'passed' and test['expected_checks'] == len(test['checks']) == 24, 'selftest matrix')
    require(len({(r['shape'], r['pattern'], r['variant']) for r in test['checks']}) == 24, 'selftest duplicates')
    files(validation / 'native-selftest', test['files'])
    for row in test['checks']:
        require(row['status'] == 'passed' and row['samples_us'] == [] and row['repetitions'] == 0 and row['median_us'] is None, 'selftest must be validation-only')
        base = validation / 'native-selftest' / row['artifact_directory']
        expected = reference(base, test['shapes'][row['shape']])
        numerical(base / 'expected.f64', expected, '<f8')
        valid_output(base, row, expected)
    require(test['finished_unix'] < provenance['started_unix'] and plan['frozen_unix'] < provenance['frozen_unix'], 'gate/plan chronology')
    REPORT['gates'] = dict(pure_tests=29, native_validation_checks=24, ctests=4, source_members=523,
                           ctest_seconds=float(re.search(r'Total Test time \(real\) =\s*([\d.]+)', ctest)[1]))
    return inventory


def audit():
    plan, provenance = load(ROOT / 'plan.json'), load(ROOT / 'provenance.json')
    declared = [dict(name=n, dimensions=d, block=b, cap=c, two_dimensional=t) for n, d, b, c, t in CASES]
    require(plan['cases'] == declared and plan['variants'] == VARIANTS, 'predeclared cases/variants drift')
    require([plan[k] for k in ('cycles', 'samples', 'warmup_ms', 'target_ms')] == [3, 7, 30, 15], 'timing plan drift')
    inventory = sources_and_gates(plan, provenance)
    captures, replays = load(ROOT / 'capture-summary.json'), load(ROOT / 'replay-summary.json')
    require(captures['plan_sha256'] == replays['plan_sha256'] == sha(ROOT / 'plan.json'), 'plan receipts drift')
    REPORT['raw_capture_statuses'], REPORT['raw_replay_statuses'] = captures['cases'], replays['comparisons']
    require([r['case'] for r in captures['cases']] == [n for n, *_ in CASES], 'capture coverage')
    require([(r['case'], r['variant']) for r in replays['comparisons']] == [(n, v) for n, *_ in CASES for v in VARIANTS], 'replay coverage')
    require(all(r['status'] == 'prepared' for r in captures['cases']) and all(r['status'] == 'ok' for r in replays['comparisons']), 'Error/NotRun cohort; retained, not admitted')
    require(provenance['frozen_unix'] < captures['started'] <= captures['finished'] < replays['started'] <= replays['finished'], 'source/capture/replay chronology')
    REPORT['chronology'] = {k: v for k, v in [('plan', plan['frozen_unix']), ('source', provenance['frozen_unix']), ('capture_start', captures['started']), ('capture_finish', captures['finished']), ('replay_start', replays['started']), ('replay_finish', replays['finished'])]}
    command_count = 0
    for path in ROOT.rglob('*.command.json'):
        command = load(path)
        require(command.get('returncode') == 0 and 'error' not in command, 'command failed: ' + str(path))
        prefix = str(path).removesuffix('.command.json')
        require(Path(prefix + '.stdout').is_file() and Path(prefix + '.stderr').read_bytes() == b'', 'missing stdout/nonempty stderr: ' + str(path))
        command_count += 1
    previous_finish = replays['started']
    for ci, (name, dims, block, cap, two_d) in enumerate(CASES):
        base, bundle = ROOT / name, ROOT / name / 'probe'
        metadata, command = load(base / 'capture.stdout'), load(base / 'capture.command.json')
        require(provenance['frozen_unix'] < command['started'] <= command['finished'] < replays['started'], 'per-capture chronology')
        require(command['command'][1:-1] == ['llm', 'attention', ','.join(map(str, dims)), *map(str, block), '3', '1', '1'], 'producer CLI')
        env = command['environment']
        controls = dict(LUISA_TILE_BENCH_XIR_BACKEND='simd', LUISA_TILE_BENCH_ATTENTION_QK='mma', LUISA_TILE_BENCH_ATTENTION_PV='mma', LUISA_TILE_BENCH_XIR_BLOCK_SIZE='32', LUISA_TILE_BENCH_XIR_LOCAL_LANES='1', LUISA_TILE_BENCH_XIR_REGION_WORK='4096', LUISA_TILE_BENCH_XIR_MMA_OUTPUT_BLOCK='4', LUISA_TILE_BENCH_XIR_MMA_UNROLL_TERMS=str(cap), LUISA_TILE_BENCH_XIR_MMA_2D_BLOCKING=str(int(two_d)), LUISA_SIMD_ENABLE_FULL_PACKET_SPECIALIZATION='1')
        require(all(env[k] == v for k, v in controls.items()) and all(env[k] == '1' for k in ('OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS')), 'producer environment drift')
        require(metadata['dimensions'] == dims and metadata['attention_block'] == block and metadata['operation'] == 'attention' and metadata['precision'] == 'fp32' and metadata['fast_math'] is False and metadata['relaxed_precision'] is False and metadata['attention_qk'] == metadata['attention_pv'] == 'mma' and metadata['source_reduction_policy'] == 'unordered_tree', 'source math drift')
        text = metadata['realization']
        require(re.findall(r'\bW(\d+), (\d+) workers/block', text) == [('8', '32')], 'packet controls')
        for key, value in dict(local_lanes=1, max_unrolled_tile_elements=64, max_unrolled_region_work=4096, requested_mma_output_block=4, requested_max_unrolled_mma_terms=cap).items():
            require(field(text, key) == value, 'realization ' + key)
        require(field(text, 'requested_mma_2d_blocking', True) is two_d, '2D request')
        prepared, tile = load(bundle / 'probe.json'), load(bundle / 'tile/prepared.json')
        require(prepared['status'] == 'prepared' and prepared['format'] == 'native-attention-probe-v1' and tile['metadata'] == metadata, 'prepared metadata')
        require(command['finished'] < prepared['started_unix'] <= prepared['finished_unix'] < replays['started'], 'preparation chronology')
        files(bundle, prepared['files']); files(bundle / 'tile', tile['files']); files(base / 'tile', load(base / 'tile/prepared.json')['files'])
        require(sha(bundle / 'probe.json') == captures['cases'][ci]['probe_sha256'] and sha(bundle / 'tile/prepared.json') == prepared['baseline_sha256'] == sha(base / 'tile/prepared.json'), 'prepared identity')
        objects = list((base / 'objects').glob('*.o'))
        require(len(objects) == 1 and sha(objects[0]) == sha(bundle / 'tile/kernel.o') and sha(base / 'output.f32.source.txt') == sha(bundle / 'tile/kernel.ll'), 'actual ORC/LLVM identity')
        for relative in ('native_tile_replay.cpp', ABI):
            source = 'src/' + relative if relative == ABI else 'scripts/benchmark/tile_torch/' + relative
            require(prepared['files'][relative] == tile['files'][relative] == inventory[source]['sha256'], 'common helper source/ABI')
        for filename in ('native_attention_probe.cpp', 'native_attention_probe.py', 'native_tile.py', 'test_native_attention_probe.py'):
            require(prepared['files'][filename] == inventory['scripts/benchmark/tile_torch/' + filename]['sha256'], 'probe/runner source identity')
        require(prepared['compiler']['sha256'] == provenance['tool_fingerprints']['llvm_clangxx']['sha256'], 'compiler identity')
        argv = load(bundle / 'compile-probe.command.json')['argv']
        require(all(flag in argv for flag in ('-std=c++20', '-O3', '-fno-fast-math', '-ffp-contract=off', '-dynamiclib', '-mmacosx-version-min=15.0', '-framework', 'Accelerate')), 'compile flags')
        helper_argv = load(bundle / 'compile-helper.command.json')['argv']
        require(helper_argv[0] == argv[0] and all(flag in helper_argv for flag in prepared['compiler']['flags']) and '-framework' not in helper_argv and not any(x.startswith('-DATTN_') for x in helper_argv), 'same compiler/strict flags for common helper')
        require([x for x in argv if x.startswith('-DATTN_')] == ['-DATTN_' + k + '=' + str(v) for k, v in zip(('B', 'H', 'KH', 'Q', 'K', 'D', 'DV'), dims)], 'compile dimensions')
        expected = reference(bundle / 'tile', dims)
        for path in (bundle / 'expected.f64', bundle / 'tile/expected.f64', base / 'output.f32.expected.f64'):
            numerical(path, expected, '<f8')
        capture_error = numerical(base / 'output.f32', expected)
        require(sha(base / 'output.f32') == sha(bundle / 'tile/captured.f32'), 'captured output identity')
        for i in range(3):
            require(sha(base / f'output.f32.input{i}.f32') == sha(bundle / f'tile/input{i}.f32'), 'captured input identity')
        REPORT['captures'].append(dict(case=name, max_abs_error=capture_error, actual_object_sha256=sha(objects[0]), realization=text))
        for vi, variant in enumerate(VARIANTS):
            row = REPORT['comparisons'][ci * 2 + vi]
            out, result = base / variant, load(base / variant / 'results.json')
            execution = load(base / ('replay-' + variant + '.command.json'))
            require(execution['process_group_isolated'] is True and execution['timeout_s'] == 120 and previous_finish <= execution['started'] <= result['started_unix'] <= result['finished_unix'] <= execution['finished'], 'replay isolation/order/chronology')
            previous_finish = execution['finished']
            require(result['status'] == 'passed' and result['artifacts_unchanged'] is True and len(result['visits']) == 12, 'replay status/count')
            require(sha(out / 'results.json') == replays['comparisons'][ci * 2 + vi]['results_sha256'] and result['bundle_sha256'] == sha(bundle / 'probe.json'), 'replay manifest identity')
            require(result['runner_sha256'] == prepared['files']['native_attention_probe.py'] and result['common_helper_sha256'] == sha(bundle / 'replay.dylib'), 'running runner/common-helper identity')
            require(result['threading']['set_status'] == 0 and result['threading']['observed_enum'] == result['threading']['expected_enum'] == 1 and result['threading']['contract_version'] == 1 and result['threading']['workspace_bytes'] == prepared['shape']['workspace_bytes'], 'threading/version/scratch acknowledgement')
            require([result['options'][k] for k in ('cycles', 'samples', 'warmup_ms', 'target_ms')] == [3, 7, 30, 15], 'replay timer options')
            require(result['entry_contracts']['tile'] == {k: tile[k] for k in ('symbol', 'abi', 'dispatch', 'block', 'packet_width', 'workspace_bytes')}, 'actual Tile entry launch')
            entry = result['entry_contracts']['probe']
            require(entry == prepared['variants'][variant] and entry['abi'] == 0 and entry['packet_width'] == 1 and entry['dispatch'] == entry['block'] == [1, 1, 1] and entry['workspace_bytes'] == max(dims[3] * dims[4], 16) * 4, 'probe whole-entry launch')
            files(out, result['files'])
            hashes, errors = {}, []
            for i, visit in enumerate(result['visits']):
                arm = ('tile', variant, variant, 'tile')[i % 4]
                require((visit['cycle'], visit['position'], visit['variant']) == (i // 4, i % 4, arm) and visit['status'] == 'passed' and visit['blas_threading_after'] == 1, 'ABBA order/status/TLS')
                require(len(visit['samples_us']) == 7 and all(math.isfinite(x) and x > 0 for x in visit['samples_us']) and type(visit['repetitions']) is int and visit['repetitions'] > 0, 'timing samples')
                require(stats.median(visit['samples_us']) == visit['median_us'], 'visit median')
                receipt = load(out / (visit['output'] + '.json'))
                require(all(visit.get(k) == v for k, v in receipt.items()), 'raw per-visit receipt differs')
                errors.append(valid_output(out, visit, expected))
                require(hashes.setdefault(arm, visit['output_sha256']) == visit['output_sha256'], 'nondeterministic output within arm')
            require(hashes == result['deterministic_output_sha256'], 'determinism receipt')
            medians = {arm: stats.median(v['median_us'] for v in result['visits'] if v['variant'] == arm) for arm in ('tile', variant)}
            pairs = [result['visits'][a]['median_us'] / result['visits'][b]['median_us'] for c in range(3) for a, b in ((4*c+1, 4*c), (4*c+2, 4*c+3))]
            ratio = dict(baseline='tile', candidate=variant, pairs=pairs, median=stats.median(pairs), minimum=min(pairs), maximum=max(pairs))
            require(result['summary_us'] == medians and result['candidate_over_baseline'] == ratio, 'ABBA summary/ratio')
            row.update(status='passed', tile_us=medians['tile'], probe_us=medians[variant], ratio=ratio, visits=12, samples=84, max_abs_error=max(errors))
    REPORT.update(status='passed', commands_checked=command_count, visits=144, samples=1008, source_archive_sha256=provenance['source_archive_sha256'], plan_sha256=sha(ROOT / 'plan.json'), audit_sha256=sha(Path(__file__)))


if __name__ == '__main__':
    try:
        audit()
    except Exception as error:
        REPORT.update(status='error', error=str(error))
    (ROOT / 'audit.json').write_text(json.dumps(REPORT, indent=2, allow_nan=False) + '\n')
    print('status:', REPORT['status'])
    for row in REPORT['comparisons']:
        if row['status'] == 'passed':
            ratio = row['ratio']
            print(f'{row["case"]:18s} {row["variant"]:16s} Tile={row["tile_us"]:.9f} us probe={row["probe_us"]:.9f} us ratio={ratio["median"]:.9f} [{ratio["minimum"]:.9f}, {ratio["maximum"]:.9f}]')
    if REPORT['status'] != 'passed':
        print(REPORT['error'], file=sys.stderr)
        sys.exit(1)
