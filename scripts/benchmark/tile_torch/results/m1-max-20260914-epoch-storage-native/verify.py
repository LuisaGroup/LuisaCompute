"""Offline full11/full9 identities, 1800 samples, 360 full outputs and FP64 oracles.

Requires NumPy only. Never loads native artifacts or reads current source/raw.
Use --write once to create immutable derived summaries and top-level checksums.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import time

import numpy as np
from evidence import Evidence

HERE = Path(__file__).resolve().parent


def shapes(op, dims):
    if op == 'attention':
        b, h, kh, q, k, d, v = dims
        return [(b, h, q, d), (b, kh, k, d), (b, kh, k, v)], (b, h, q, v)
    m, n = dims
    auxiliary = (1 if op == 'rmsnorm' else m, n)
    return [(m, n), auxiliary, auxiliary], (m, n)


def reference(op, dims, arrays):
    x, u, v = [a.astype(np.float64) for a in arrays]
    if op == 'rmsnorm':
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + float(np.float32(1e-5))) * u
    if op == 'masked_softmax':
        m, n = dims
        valid = np.arange(n)[None, :] <= (np.arange(m) % n)[:, None]
        scores = np.where(valid, x, float(np.float32(-1e30)))
        exponent = np.exp(scores - scores.max(axis=-1, keepdims=True))
        return exponent / exponent.sum(axis=-1, keepdims=True)
    b, h, kh, q, k, d, dv = dims
    keys, values = np.repeat(u, h // kh, axis=1), np.repeat(v, h // kh, axis=1)
    scores = (x @ keys.swapaxes(-1, -2)) * float(np.float32(1) / np.sqrt(np.float32(d)))
    valid = np.arange(k)[None, :] <= np.arange(q)[:, None] + k - q
    scores = np.where(valid, scores, -1e30)
    exponent = np.where(valid, np.exp(scores - scores.max(axis=-1, keepdims=True)), 0)
    return (exponent / exponent.sum(axis=-1, keepdims=True)) @ values


def tensor(store, name, dtype, shape):
    data = store.read(name)
    assert len(data) == np.dtype(dtype).itemsize * math.prod(shape)
    array = np.frombuffer(data, dtype=dtype).reshape(shape)
    assert np.isfinite(array).all()
    return array


def numerical(actual, expected):
    error = np.abs(actual.astype(np.float64) - expected)
    assert np.all(error <= 5e-5 + 5e-5 * np.abs(expected))
    return float(error.max())


def receipt(store, folder, admission_sha):
    result, command = store.json(folder + '/result.json'), store.json(folder + '/command.json')
    assert result['status'] == 'passed' and result['exit_code'] == 0 and not result['timed_out']
    assert result['argv'] == command['argv'] and result['admission_sha256'] == command['admission_sha256'] == admission_sha
    assert result['driver_sha256'] == command['driver_sha256']
    assert result['started'] <= result['finished'] and result['timeout_seconds'] == 180
    for channel in ('stdout', 'stderr'):
        assert store.digest(folder + '/' + channel + '.txt') == result[channel + '_sha256']
    return result


def bundle(store, folder, historical=False):
    data = store.json(folder + '/prepared.json')
    assert data['status'] == 'prepared' and data['format'] == 'native-tile-entry-v1'
    for name, digest in data['files'].items():
        assert store.digest(folder + '/' + name) == digest
    for original, digest in data['original_sha256'].items():
        assert store.digest(store.alias(original, full9=historical)) == digest
    return data


def pair(store, folder, job, expected, torch=False):
    result = store.json(folder + '/results.json')
    assert result['status'] == 'passed' and result['artifacts_unchanged'] and result['cpu_threads'] == 1
    options = result if torch else result['options']
    assert [options[k] for k in ('cycles', 'samples', 'warmup_ms', 'target_ms')] == [3, 5, 40, 20]
    assert len(result['visits']) == 12
    baseline, candidate = ('torch-inductor' if torch else 'full9-default'), 'full11-default'
    values, grouped, output_hashes = [], {baseline: [], candidate: []}, {}
    maximum = 0.0
    if torch:
        assert result['timer'] == 'native_rows' and result['historical_times_reused'] is False
        assert len(result['preflight']) == 2 and all(r['valid'] and r['all_guards_passed'] and r['inputs_unchanged'] for r in result['preflight'])
        for original, digest in result['artifacts_sha256'].items():
            assert store.digest(store.alias(original)) == digest
    else:
        assert result['metric'] == 'single_thread_native_entry_host_wall_us'
        for original, digest in result['prepared_sha256'].items():
            assert store.digest(store.alias(original)) == digest
    for i, row in enumerate(result['visits']):
        assert row['cycle'] == i // 4 and row['position'] == i % 4
        assert row['variant'] == (baseline, candidate, candidate, baseline)[i % 4]
        assert row['valid'] and row['all_guards_passed'] and row['inputs_unchanged']
        assert row.get('returncode', 0) == 0 and row['repetitions'] > 0
        samples = row['samples_us']
        assert len(samples) == 5 and all(math.isfinite(v) and v > 0 for v in samples)
        median = st.median(samples)
        assert median == row['median_us']
        values.append(median)
        grouped[row['variant']].append(median)
        output = folder + '/' + (f'visit-{i // 4}-{i % 4}.f32' if torch else row['output'])
        digest = store.digest(output)
        assert digest == row['output_sha256'] == output_hashes.setdefault(row['variant'], digest)
        maximum = max(maximum, numerical(tensor(store, output, np.float32, expected.shape), expected))
    medians = {name: st.median(samples) for name, samples in grouped.items()}
    ratios = [values[4 * c + b] / values[4 * c + a] for c in range(3) for a, b in ((0, 1), (3, 2))]
    paired = dict(baseline=baseline, candidate=candidate, pairs=ratios, median=st.median(ratios), minimum=min(ratios), maximum=max(ratios))
    assert medians == result['summary_us'] == job['summary_us']
    assert paired == result['candidate_over_baseline'] == job['candidate_over_baseline']
    assert store.digest(folder + '/results.json') == job['result_sha256']
    return dict(case=job['case'], local_lanes=job['local_lanes'], summary_us=medians, candidate_over_baseline=paired, max_abs_error=maximum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    started = time.time()
    store = Evidence(HERE)
    for name, row in store.manifest['shards'].items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == row['sha256']
    for name in store.manifest['files']:
        store.read(name)
    admission = store.json('cohort/admission.json')
    joined = store.json('cohort/joined-admission.json')
    admission_sha, joined_sha = store.digest('cohort/admission.json'), store.digest('cohort/joined-admission.json')
    assert joined['admission_sha256'] == admission_sha
    assert joined['driver_sha256'] == store.digest('cohort/driver.py') and joined['torch_driver_sha256'] == store.digest('cohort/torch_pairs.py')
    frozen = store.json(store.alias(admission['freeze']))
    assert store.digest(store.alias(admission['freeze'])) == admission['freeze_sha256'] == '29ef908f51beadddedaac6cc85735f2c260cf86978c075bd0fb875707230099c'
    assert len(frozen['source_sha256']) == 26 and frozen['source_sha256'] == admission['source_sha256']
    for name, digest in frozen['source_sha256'].items():
        assert store.digest(store.alias(str(Path(frozen['selected_source']) / name))) == digest
    for original, digest in admission['critical_files_sha256'].items() | admission['gate_sha256'].items() | joined['full11_snapshot'].items() | joined['torch']['files_sha256'].items() | joined['historical_pins'].items():
        assert store.digest(store.alias(original)) == digest
    gates = []
    for original in admission['gate_sha256']:
        result = store.json(store.alias(original))
        command = store.json(store.alias(Path(original).parent / 'command.json'))
        assert result['status'] == 'passed' and result['exit_code'] == 0 and not result.get('timed_out', False)
        assert result['source_sha256'] == command['source_sha256'] == frozen['source_sha256']
        assert result['source_freeze_sha256'] == command['source_freeze_sha256'] == admission['freeze_sha256']
        assert result['command'] == command['command'] and result['started'] >= frozen['finished']
        for channel in ('stdout', 'stderr'):
            assert store.digest(store.alias(Path(original).parent / (channel + '.txt'))) == result[channel + '_sha256']
        gates.append(result)
    build = [r for r in gates if '--build' in r['command']]
    assert len(build) == 1 and '--target' not in build[0]['command'] and len(gates) == 4
    assert all(r['started'] >= build[0]['finished'] for r in gates if '--build' not in r['command'])
    old = store.json('full9/immutable/manifest.json')
    assert old['status'] == 'passed' and old['full9_source_files'] == 23
    for original, row in old['saved'].items():
        assert store.digest(store.alias(original, full9=True)) == row['sha256']
    declared = store.json('cohort/plan_v1.json')['cases']
    expected_keys = {(c[0], local) for c in declared for local in (1, 8)}
    phases = {mode: store.json(f'cohort/{mode}-results.json') for mode in ('capture', 'prepare', 'replay', 'torch')}
    for mode, phase in phases.items():
        expected = expected_keys if mode != 'torch' else {(c[0], local) for c in declared[:4] for local in (1, 8)}
        assert phase['status'] == 'passed' and phase['joined_admission_sha256'] == joined_sha and phase['historical_times_reused'] is False
        assert len(phase['jobs']) == len(expected) and {(r['case'], r['local_lanes']) for r in phase['jobs']} == expected
        assert all(r['status'] == 'passed' for r in phase['jobs'])
    prepared = {(r['case'], r['local_lanes']): r for r in phases['prepare']['jobs']}
    captures = {(r['case'], r['local_lanes']): r for r in phases['capture']['jobs']}
    baselines = {(r['case'], r['local_lanes']): r for r in joined['baseline']}
    assert set(baselines) == expected_keys
    references, arrays_by_case = {}, {}
    for name, op, dims, block in declared:
        inputs, out_shape = shapes(op, dims)
        for local in (1, 8):
            label = f'{name}-l{local}'
            capture = 'cohort/capture-' + label
            result = receipt(store, capture, admission_sha)
            assert result['rss_samples'] and result['peak_sampled_process_tree_rss_kib'] <= 8388608
            assert not any('DISABLE_' in arg or 'ENABLE_' in arg for arg in result['argv'])
            check = store.json(capture + '/validation.json')
            assert store.digest(capture + '/validation.json') == captures[name, local]['validation_sha256']
            for original, digest in check['files_sha256'].items():
                assert store.digest(store.alias(original)) == digest
            metadata = store.json(capture + '/stdout.txt')
            assert metadata == check['metadata'] and metadata['operation'] == op and metadata['dimensions'] == dims
            assert metadata['precision'] == 'fp32' and metadata['fast_math'] is False and metadata['relaxed_precision'] is False
            assert metadata['attention_block'] == block and f'local_lanes={local};' in metadata['realization']
            arrays = [tensor(store, capture + f'/output.f32.input{i}.f32', np.float32, shape) for i, shape in enumerate(inputs)]
            expected = reference(op, dims, arrays)
            references[name], arrays_by_case[name] = expected, arrays
            stored_oracle = tensor(store, capture + '/output.f32.expected.f64', np.float64, out_shape)
            assert np.allclose(stored_oracle, expected, atol=1e-12, rtol=1e-12)
            numerical(tensor(store, capture + '/output.f32', np.float32, out_shape), expected)
            current = 'cohort/prepared-' + label
            receipt(store, 'cohort/prepare-' + label, admission_sha)
            assert store.digest(current + '/prepared.json') == prepared[name, local]['prepared_sha256']
            new_bundle = bundle(store, current)
            old_folder = 'cohort/baseline-' + label
            old_bundle = bundle(store, old_folder, historical=True)
            baseline = baselines[name, local]
            assert store.digest(old_folder + '/prepared.json') == baseline['snapshot_sha256']
            assert store.digest(old_folder + '/historical-prepared.json') == baseline['sha256'] == old_bundle['retained_parent_sha256']
            original = store.json(old_folder + '/historical-prepared.json')
            original['name'], original['retained_parent_sha256'] = 'full9-default', baseline['sha256']
            assert original == old_bundle and new_bundle['metadata'] == metadata
            for relative in (*new_bundle['input_files'], 'expected.f64', 'native_tile_replay.cpp', 'backends/simd/llvm/llvm_schedule_codegen.h'):
                assert new_bundle['files'][relative] == old_bundle['files'][relative]
            assert [new_bundle['files'][name] for name in new_bundle['input_files']] == baseline['input_sha256']
    summaries = {}
    for mode in ('replay', 'torch'):
        rows = []
        for job in phases[mode]['jobs']:
            label = f"{job['case']}-l{job['local_lanes']}"
            receipt(store, 'cohort/' + mode + '-' + label, admission_sha)
            folder = 'cohort/' + mode + '-output-' + label
            expected = references[job['case']]
            rows.append(pair(store, folder, job, expected, torch=mode == 'torch'))
            if mode == 'torch':
                saved = tensor(store, folder + '/expected.f64', np.float64, expected.shape)
                assert np.allclose(saved, expected, atol=1e-12, rtol=1e-12)
                result = store.json(folder + '/results.json')
                x = arrays_by_case[job['case']][0].astype(np.float64)
                if result['operation'] == 'rmsnorm':
                    writes = [np.sum(x * x, axis=1, keepdims=True), expected]
                else:
                    m, n = result['dimensions']
                    scores = np.where(np.arange(n)[None, :] <= (np.arange(m) % n)[:, None], x, float(np.float32(-1e30)))
                    maximum = scores.max(axis=1, keepdims=True)
                    writes = [maximum, np.exp(scores - maximum).sum(axis=1, keepdims=True), expected]
                for i, oracle in enumerate(writes):
                    assert np.allclose(tensor(store, folder + f'/torch-write-{i}.f64', np.float64, oracle.shape), oracle, atol=1e-12, rtol=1e-12)
                for visit in result['visits']:
                    counts = [v.size for v in writes] if visit['variant'] == 'torch-inductor' else [expected.size]
                    assert [v['elements'] for v in visit['correctness']] == counts
        summaries[mode] = rows
    summary = dict(status='passed', cases=declared, full11_vs_full9=summaries['replay'], full11_vs_torch=summaries['torch'],
                   phase_wall_seconds={name: value['finished'] - value['started'] for name, value in phases.items()},
                   timer_cohorts_are_separate=True, historical_times_reused=False, metric='single_thread_native_entry_host_wall_us')
    verification = dict(status='passed', started=started, finished=time.time(), logical_files=len(store.manifest['files']), unique_blobs=len(store.manifest['objects']),
                        full11_source_files=26, full9_source_files=23, captures=22, prepares=22, native_tile_pairs=22, native_rows_pairs=8,
                        native_tile_visits=264, native_rows_visits=96, total_raw_samples=1800, complete_final_outputs_rechecked=360,
                        full_output_fp64=True, all_ratios_recomputed=True, native_execution=False, reads_current_root_selected_or_raw=False,
                        boundary='Guards/input unchanged and actual scratch validation are original native receipts; their unpersisted bytes cannot be rerun offline. All saved final outputs and scratch FP64 references reread.',
                        manifest_sha256=hashlib.sha256((HERE / 'manifest.json').read_bytes()).hexdigest())
    if args.write:
        for name, value in (('summary.json', summary), ('verification.json', verification)):
            with (HERE / name).open('x') as stream:
                json.dump(value, stream, indent=2)
                stream.write('\n')
        with (HERE / 'results.md').open('x') as stream:
            stream.write(store.read('cohort/SUMMARY.md').decode())
        with (HERE / 'SHA256SUMS').open('x') as stream:
            for path in sorted(HERE.iterdir()):
                if path.is_file() and path.name != 'SHA256SUMS':
                    stream.write(hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + path.name + '\n')
    else:
        assert summary == json.loads((HERE / 'summary.json').read_text())
        assert (HERE / 'results.md').read_bytes() == store.read('cohort/SUMMARY.md')
        for line in (HERE / 'SHA256SUMS').read_text().splitlines():
            digest, name = line.split('  ', 1)
            assert '/' not in name and hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
