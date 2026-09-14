#!/usr/bin/env python3
"""Read-only audit of completed Metal4 measurements; never import the runner.

Writes only a requested /tmp output file. Recomputes from raw tick/CB records,
checks input/output receipts, and evaluates complete NumPy FP64 references.
Guard regions are not exported: their coverage remains C++ runtime evidence.
"""
import argparse
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import numpy as np


def check(value, message):
    if not value:
        raise ValueError(message)


def receipt(path):
    path = Path(path)
    if not path.is_file():
        return None
    with path.open('rb') as stream:
        return {'bytes': path.stat().st_size,
                'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}


def near(actual, expected, message):
    check(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-6), message)


def oracle(op, arrays):
    x, y, z = [a.astype(np.float64) for a in arrays]
    if op in ('rmsnorm', 'layernorm'):
        center = x - x.mean(-1, keepdims=True) if op == 'layernorm' else x
        answer = center * y / np.sqrt(np.mean(center * center, -1, keepdims=True) + float(np.float32(1e-5)))
        return answer + z if op == 'layernorm' else answer
    if op == 'swiglu':
        return x * y / (1.0 + np.exp(-x))
    if op == 'masked_softmax':
        mask = np.arange(x.shape[1])[None, :] <= np.arange(x.shape[0])[:, None] % x.shape[1]
        selected = np.where(mask, x, -1e30)
        exponent = np.where(mask, np.exp(selected - selected.max(-1, keepdims=True)), 0.0)
        return exponent / exponent.sum(-1, keepdims=True)
    if op == 'gelu_residual':
        return .5 * x * (1 + np.tanh(float(np.float32(.7978845608)) * (x + float(np.float32(.044715)) * x**3))) + y
    if op == 'rope':
        a, b = np.split(x, 2, axis=-1)
        return np.concatenate((a * y - b * z, a * z + b * y), axis=-1)
    raise ValueError('unsupported reference')


def audit_row(row, directory, protocol):
    payload = json.loads((directory / row['stdout']).read_text())
    for key, expected in {'operation': row['operation'], 'dimensions': row['dimensions'],
                          'backend': 'metal4', 'precision': 'fp32', 'fast_math': False,
                          'repetitions': protocol['host_repetitions'], 'repetition_policy': 'fixed'}.items():
        check(payload[key] == expected, f'metadata mismatch {key}')
    for path, saved in row['tensor_receipts'].items():
        check(receipt(path) == saved, f'tensor receipt mismatch {path}')
    output = Path(row['command'][-1])
    m, n = row['dimensions']
    op = row['operation']
    aux = (1 if op in ('rmsnorm', 'layernorm') else m, n // 2 if op == 'rope' else n)
    shapes = [(m, n), aux, aux]
    arrays = [np.fromfile(str(output) + f'.input{i}.f32', dtype=np.float32).reshape(shape)
              for i, shape in enumerate(shapes)]
    expected = oracle(op, arrays)
    actual = np.fromfile(output, dtype=np.float32).reshape((m, n)).astype(np.float64)
    delta = np.abs(actual - expected)
    check(np.isfinite(actual).all() and np.isfinite(expected).all(), 'nonfinite output/reference')
    check((delta <= 5e-5 + 5e-5 * np.abs(expected)).all(), 'complete FP64 reference mismatch')
    check(payload['correctness']['elements_per_check'] == m * n and
          payload['correctness']['checks'] == 2 and payload['correctness']['guard_elements_per_check'] == 34,
          'incomplete C++ correctness metadata')
    timing = payload['device_timing']
    check(timing['scope'] == 'instrumented_dispatch_intervals' and
          timing['host_samples_instrumented'] is False and timing['zero_overhead_kernel_time'] is False,
          'wrong instrumentation labels')
    frequency = timing['capabilities']['timestamp_frequency_hz']
    check(type(frequency) is int and frequency > 0, 'invalid frequency')
    metrics, identities, sample_ids = {}, set(), set()
    dispatch_count = command_buffer_count = 0
    for phase in ('throughput', 'latency'):
        count = min(protocol['host_repetitions'], 64) if phase == 'throughput' else 1
        for counters, source, label in ((True, timing, 'instrumented_dispatch_ns'),
                                        (False, timing['control'], 'feedback_only_command_buffer_ns_per_dispatch')):
            values = []
            check(len(source[phase]) == protocol['samples'], 'sample count mismatch')
            for sample in source[phase]:
                check(sample['sample_id'] not in sample_ids, 'duplicate sample ID')
                sample_ids.add(sample['sample_id'])
                check(not sample['error'] and not sample['overflow'], 'failed sample included')
                check(sample['dispatch_timestamps_enabled'] == counters and
                      sample['timestamp_frequency_hz'] == (frequency if counters else 0), 'mode mismatch')
                buffers = {b['ordinal']: b for b in sample['command_buffers']}
                check(len(buffers) == len(sample['command_buffers']), 'duplicate CB ordinal')
                counts, elapsed, feedback = collections.Counter(), [], []
                check(len(sample['dispatches']) == count, 'dispatch count mismatch')
                check(len({d['ordinal'] for d in sample['dispatches']}) == count, 'duplicate dispatch ordinal')
                for dispatch in sample['dispatches']:
                    owner = dispatch['command_buffer_ordinal']
                    check(owner in buffers, 'orphan dispatch')
                    counts[owner] += 1
                    identities.add((dispatch['shader_checksum'], tuple(dispatch['dispatch_size']), tuple(dispatch['block_size'])))
                    a, b = dispatch['begin_ticks'], dispatch['end_ticks']
                    if counters:
                        check(dispatch['valid'] and type(a) is int and type(b) is int and 0 < a < b, 'invalid ticks')
                        ns = (b - a) * 1e9 / frequency
                        near(dispatch['elapsed_ns'], ns, 'tick conversion mismatch')
                        elapsed.append(ns)
                    else:
                        check(not dispatch['valid'] and a == b == dispatch['elapsed_ns'] == 0, 'control has counters')
                for ordinal, buffer in buffers.items():
                    check(counts[ordinal] == buffer['dispatch_count'], 'CB ownership count mismatch')
                    check(not buffer['contains_non_dispatch_work'], 'non-dispatch work in benchmark')
                    host = [buffer[k] for k in ('host_commit_begin_ns', 'host_feedback_begin_ns',
                                                'host_callbacks_end_ns', 'host_completion_publish_ns')]
                    check(all(type(t) is int and t > 0 for t in host) and host == sorted(host), 'host order mismatch')
                    check(buffer['host_commit_return_ns'] >= host[0], 'commit return precedes commit')
                    if buffer['dispatch_count']:
                        a, b = buffer['gpu_begin_seconds'], buffer['gpu_end_seconds']
                        check(buffer['valid'] and 0 < a < b and math.isfinite(b), 'invalid nonempty CB span')
                        feedback.append((b - a) * 1e9)
                values.append(statistics.median(elapsed) if counters else sum(feedback) / count)
                dispatch_count += count
                command_buffer_count += len(buffers)
            metric = f'{phase}_{label}'
            saved = row['metrics'][metric]
            check(len(saved['samples']) == len(values), 'metric sample count mismatch')
            for a, b in zip(saved['samples'], values):
                near(a, b, 'saved sample metric mismatch')
            near(saved['median'], statistics.median(values), 'saved median mismatch')
            metrics[metric] = statistics.median(values)
        host_key = phase + '_host_wall_us_per_dispatch'
        near(row['metrics'][host_key]['median'], statistics.median(payload[phase + '_us']), 'host median mismatch')
        metrics[host_key] = statistics.median(payload[phase + '_us'])
    check(len(identities) == 1, 'shader/geometry changed across phases')
    return {'output_elements_checked': m * n, 'max_abs_error': float(delta.max()),
            'dispatch_records_checked': dispatch_count, 'command_buffers_checked': command_buffer_count,
            'metrics': metrics, 'guards_independently_checked': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    check(str(args.output.resolve()).startswith('/private/tmp/') or str(args.output.resolve()).startswith('/tmp/'), 'output must be under /tmp')
    result = {'audited_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'method': 'independent raw timestamp recomputation and complete NumPy FP64 oracle', 'cohorts': {}}
    for name in ('cohort', 'rows'):
        directory = args.root / name
        manifest = json.loads((directory / 'results.json').read_text())
        records, issues, input_hashes = [], [], {}
        audit = {'manifest_receipt': receipt(directory / 'results.json'),
                 'status_counts': dict(collections.Counter(row['status'] for row in manifest['results'])),
                 'finished_utc': manifest.get('finished_utc'),
                 'artifacts_after_present': 'artifacts_after' in manifest,
                 'artifacts_current_match_before': {p: receipt(p) == r for p, r in manifest['artifacts_before'].items()},
                 'records': records, 'issues': issues}
        for index, row in enumerate(manifest['results']):
            if row['status'] == 'NotRun':
                continue
            record = {k: row[k] for k in ('round', 'operation', 'dimensions', 'requested_local_lanes', 'status')}
            record['index'] = index
            if row['status'] != 'OK':
                record['error_log_first_line'] = (directory / row['stdout']).read_text().splitlines()[0]
                records.append(record)
                continue
            try:
                record.update(audit_row(row, directory, manifest['protocol']))
                identity = (row['operation'], tuple(row['dimensions']))
                hashes = tuple(receipt(row['command'][-1] + f'.input{i}.f32')['sha256'] for i in range(3))
                check(identity not in input_hashes or input_hashes[identity] == hashes, 'inputs differ across variants/rounds')
                input_hashes[identity] = hashes
                record['audit'] = 'PASS'
            except Exception as error:
                record['audit'] = 'FAIL'
                record['audit_error'] = str(error)
                issues.append({'index': index, 'error': str(error)})
            records.append(record)
        audit['audited_ok_rows'] = sum(r.get('audit') == 'PASS' for r in records)
        audit['output_elements_checked'] = sum(r.get('output_elements_checked', 0) for r in records)
        audit['dispatch_records_checked'] = sum(r.get('dispatch_records_checked', 0) for r in records)
        audit['command_buffers_checked'] = sum(r.get('command_buffers_checked', 0) for r in records)
        result['cohorts'][name] = audit
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({name: {k: v for k, v in value.items() if k not in ('records', 'artifacts_current_match_before')}
                      for name, value in result['cohorts'].items()}, indent=2))


if __name__ == '__main__':
    main()
