"""Offline hashes, FP64 full tensors, receipts and all 44 paired summaries.

Requires NumPy, not Luisa/Torch/LLVM. Does not load native code or read current
ROOT/SELECTED. --write creates immutable verification/summary/table/SHA files.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np
from evidence import Evidence, recompute

HERE = Path(__file__).resolve().parent


def shapes(op, dims):
    if op == 'attention':
        b, h, kh, q, k, d, v = dims
        return [(b, h, q, d), (b, kh, k, d), (b, kh, k, v)], (b, h, q, v)
    m, n = dims
    auxiliary = (1 if op == 'rmsnorm' else m, n)
    return [(m, n), auxiliary, auxiliary], (m, n)


def oracle(op, dims, arrays):
    x, u, v = [value.astype(np.float64) for value in arrays]
    if op == 'rmsnorm':
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + float(np.float32(1e-5))) * u
    if op == 'masked_softmax':
        m, n = dims
        valid = np.arange(n)[None, :] <= np.arange(m)[:, None] % n
        score = np.where(valid, x, -1e30)
        exponent = np.where(valid, np.exp(score - score.max(axis=-1, keepdims=True)), 0)
        return exponent / exponent.sum(axis=-1, keepdims=True)
    b, h, kh, q, k, d, dv = dims
    keys, values = np.repeat(u, h // kh, axis=1), np.repeat(v, h // kh, axis=1)
    scale = float(np.float32(1) / np.sqrt(np.float32(d)))
    score = (x @ keys.swapaxes(-1, -2)) * scale
    valid = np.arange(k)[None, :] <= np.arange(q)[:, None] + k - q
    score = np.where(valid, score, -1e30)
    exponent = np.where(valid, np.exp(score - score.max(axis=-1, keepdims=True)), 0)
    return (exponent / exponent.sum(axis=-1, keepdims=True)) @ values


def tensor(store, path, dtype, shape):
    data = store.read(path)
    assert len(data) == np.dtype(dtype).itemsize * math.prod(shape)
    value = np.frombuffer(data, dtype=dtype).reshape(shape)
    assert np.isfinite(value).all()
    return value


def validate(value, expected):
    error = np.abs(value.astype(np.float64) - expected)
    assert np.all(error <= 5e-5 + 5e-5 * np.abs(expected))
    return float(error.max())


def run_receipt(store, folder, admission_hash):
    result = store.json(folder + '/result.json')
    command = store.json(folder + '/command.json')
    assert result['status'] == 'passed' and result['exit_code'] == 0 and not result['timed_out']
    assert result['admission_sha256'] == command['admission_sha256'] == admission_hash
    assert result['argv'] == command['argv'] and result['driver_sha256'] == command['driver_sha256'] == store.digest('cohort/driver_v3.py')
    for channel in ('stdout', 'stderr'):
        assert store.digest(folder + '/' + channel + '.txt') == result[channel + '_sha256']
    return result


def table(summary):
    rows = {(r['case'], r['local_lanes'], r['disabled_option']): r for r in summary['jobs']}
    lines = ['# Full9：同二进制、两条独立消融边', '',
             '时间单位µs；时间列为默认/关闭。倍率是相邻ABBA配对的 `关闭/默认` 中位数，大于1表示默认更快。两条边的默认时间分别测得，不混用、不相乘。', '',
             '| case | local | uniform 默认/关闭 µs | uniform 倍率 | cohort 默认/关闭 µs | cohort 倍率 |',
             '|---|---:|---:|---:|---:|---:|']
    for case in summary['cases']:
        for local in (1, 8):
            fields = []
            for arm in ('disable-uniform', 'disable-cohort'):
                row = rows[case[0], local, arm]
                fields += [f"{row['median_us']['default']:.3f}/{row['median_us'][arm]:.3f}", f"{row['disabled_over_default']['median']:.4f}"]
            lines.append(f'| {case[0]} | {local} | ' + ' | '.join(fields) + ' |')
    lines += ['', 'case的完整shape/BQ/BK见 `summary.json`；全部11个case当前数值检查通过，不证明跨epoch collective或一般模型已经完备。没有Torch/MPS对照。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    started = time.time()
    store = Evidence(HERE)
    for name, item in store.manifest['shards'].items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == item['sha256']
    for digest in store.manifest['objects']:
        store.blob(digest)
    for name in store.manifest['files']:
        store.read(name)
    admission = store.json('cohort/admission.json')
    admission_hash = store.digest('cohort/admission.json')
    frozen = store.json(store.alias(admission['freeze']))
    assert store.digest(store.alias(admission['freeze'])) == admission['freeze_sha256']
    assert len(frozen['source_sha256']) == 23 and frozen['source_sha256'] == admission['source_sha256']
    selected = frozen['selected_source']
    for name, expected in frozen['source_sha256'].items():
        assert store.digest(store.alias(str(Path(selected) / name))) == expected
    for original, expected in admission['critical_files_sha256'].items():
        assert store.digest(store.alias(original)) == expected
    for original, expected in admission['gate_sha256'].items():
        assert store.digest(store.alias(original)) == expected
        gate = store.json(store.alias(original))
        assert gate['status'] == 'passed' and gate['exit_code'] == 0 and gate['source_freeze_sha256'] == admission['freeze_sha256']
        for channel in ('stdout', 'stderr'):
            assert store.digest(store.alias(Path(original).parent / (channel + '.txt'))) == gate[channel + '_sha256']
    phases = {name: store.json(f'cohort/{name}-results.json') for name in ('capture', 'prepare', 'replay')}
    for name, count in (('capture', 66), ('prepare', 66), ('replay', 44)):
        phase = phases[name]
        assert phase['status'] == 'passed' and phase['admission_sha256'] == admission_hash and len(phase['jobs']) == count
        assert all(row['status'] == 'passed' for row in phase['jobs'])
        if name != 'capture':
            assert store.digest(store.alias(phase['prior_phase'])) == phase['prior_phase_sha256']
    captures = {(r['case'], r['local_lanes'], r['arm']): r for r in phases['capture']['jobs']}
    prepares = {(r['case'], r['local_lanes'], r['arm']): r for r in phases['prepare']['jobs']}
    declared = store.json('cohort/plan_v1.json')
    expected_keys = {(case[0], local, arm) for case in declared['cases'] for local in (1, 8) for arm in ('default', 'disable-uniform', 'disable-cohort')}
    assert set(captures) == set(prepares) == expected_keys
    references, captured_bits, max_errors = {}, {}, {}
    peak_rss = 0
    for case in declared['cases']:
        name, op, dims, block = case
        inputs, output_shape = shapes(op, dims)
        reference_inputs = None
        for local in (1, 8):
            for arm in ('default', 'disable-uniform', 'disable-cohort'):
                label = f'{name}-l{local}-{arm}'
                folder = 'cohort/capture-' + label
                receipt = run_receipt(store, folder, admission_hash)
                assert receipt['rss_samples'] and receipt['timeout_seconds'] == 180
                assert receipt['peak_sampled_process_tree_rss_kib'] <= 8388608
                peak_rss = max(peak_rss, receipt['peak_sampled_process_tree_rss_kib'])
                assert store.digest(folder + '/validation.json') == captures[name, local, arm]['validation_sha256']
                checked = store.json(folder + '/validation.json')
                for original, expected in checked['files_sha256'].items():
                    assert store.digest(store.alias(original)) == expected
                payload = store.json(folder + '/stdout.txt')
                assert payload == checked['metadata'] and payload['precision'] == 'fp32' and payload['fast_math'] is False
                assert payload['relaxed_precision'] is False and payload['dimensions'] == dims and payload['operation'] == op
                assert payload['attention_block'] == block and payload['input_shapes'] == [list(v) for v in inputs]
                assert payload['output_shape'] == list(output_shape) and payload['correctness']['checks'] == 2
                assert payload['correctness']['guard_elements_per_check'] == 34 and payload['correctness']['elements_per_check'] == math.prod(output_shape)
                assert f'local_lanes={local};' in payload['realization'] and 'W8, 32 workers/block, 1 CPU workers' in payload['realization']
                programs = dims[0] if op != 'attention' else dims[0] * dims[1] * ((dims[3] + block[0] - 1) // block[0])
                assert payload['dispatch'] == [programs * local, 1, 1]
                settings = dict(item.split('=', 1) for item in receipt['argv'][1:] if item.startswith('LUISA_'))
                assert {k: v for k, v in settings.items() if k.startswith('LUISA_SIMD_DISABLE_')} == ({'LUISA_SIMD_DISABLE_UNIFORM_READ_LANE': '1'} if arm == 'disable-uniform' else {'LUISA_SIMD_DISABLE_COHORT_PRIVATE_ACCESS': '1'} if arm == 'disable-cohort' else {})
                arrays = [tensor(store, folder + f'/output.f32.input{i}.f32', np.float32, shape) for i, shape in enumerate(inputs)]
                input_hashes = [store.digest(folder + f'/output.f32.input{i}.f32') for i in range(3)]
                assert input_hashes == checked['input_sha256']
                if reference_inputs is None:
                    reference_inputs = input_hashes
                    references[name] = oracle(op, dims, arrays)
                assert input_hashes == reference_inputs
                expected = references[name]
                stored_oracle = tensor(store, folder + '/output.f32.expected.f64', np.float64, output_shape)
                assert np.allclose(stored_oracle, expected, atol=1e-12, rtol=1e-12)
                actual = tensor(store, folder + '/output.f32', np.float32, output_shape)
                max_errors[name] = max(max_errors.get(name, 0), validate(actual, expected))
                bits = store.digest(folder + '/output.f32')
                assert captured_bits.setdefault((name, local), bits) == bits == checked['output_sha256']
                preparation = 'cohort/prepared-' + label
                run_receipt(store, 'cohort/prepare-' + label, admission_hash)
                assert store.digest(preparation + '/prepared.json') == prepares[name, local, arm]['prepared_sha256']
                bundle = store.json(preparation + '/prepared.json')
                assert bundle['status'] == 'prepared' and bundle['metadata'] == payload
                for relative, expected_hash in bundle['files'].items():
                    assert store.digest(preparation + '/' + relative) == expected_hash
                for original, expected_hash in bundle['original_sha256'].items():
                    assert store.digest(store.alias(original)) == expected_hash
    for job in phases['replay']['jobs']:
        name, local, arm = job['case'], job['local_lanes'], job['arm']
        label = f'{name}-l{local}-{arm}'
        run_receipt(store, 'cohort/replay-' + label, admission_hash)
        folder = 'cohort/replayed-' + label
        replay = store.json(folder + '/results.json')
        for original, expected_hash in replay['prepared_sha256'].items():
            assert store.digest(store.alias(original)) == expected_hash
        for row in replay['visits']:
            path = folder + '/' + row['output']
            assert store.digest(path) == row['output_sha256'] == captured_bits[name, local]
            max_errors[name] = max(max_errors[name], validate(tensor(store, path, np.float32, references[name].shape), references[name]))
    summary = recompute(store)
    text = table(summary)
    result = dict(status='passed', started=started, finished=time.time(), logical_files=len(store.manifest['files']),
                  unique_blobs=len(store.manifest['objects']), captures=66, prepares=66, paired_replays=44, visits=528, samples=2640,
                  full9_source_files=23, full_output_fp64=True, max_abs_errors=max_errors,
                  peak_capture_rss_kib=peak_rss, all_same_local_ablation_outputs_bitwise_equal=True,
                  native_execution=False, current_selected_or_root_source_reads=False,
                  guard_boundary='In-memory guard assertions retained as runner receipts; unpersisted guard bytes cannot be independently reread.',
                  semantic_boundary='All current 11 cases passed; a subsequent review identified a potential cross-epoch collective gap. This cohort is not proof of general-model completeness.',
                  manifest_sha256=hashlib.sha256((HERE / 'manifest.json').read_bytes()).hexdigest())
    if args.write:
        for name, value in (('summary.json', summary), ('verification.json', result)):
            with (HERE / name).open('x') as stream:
                json.dump(value, stream, indent=2)
                stream.write('\n')
        with (HERE / 'results.md').open('x') as stream:
            stream.write(text)
        with (HERE / 'SHA256SUMS').open('x') as stream:
            for path in sorted(HERE.iterdir()):
                if path.is_file() and path.name != 'SHA256SUMS':
                    stream.write(hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + path.name + '\n')
    else:
        assert summary == json.loads((HERE / 'summary.json').read_text())
        assert text == (HERE / 'results.md').read_text()
    print(json.dumps(result, indent=2))
    store.close()


if __name__ == '__main__':
    main()
