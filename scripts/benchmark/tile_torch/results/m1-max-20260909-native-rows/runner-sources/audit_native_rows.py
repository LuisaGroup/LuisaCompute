#!/usr/bin/env python3
"""Independently re-read captured/native outputs and recompute paired timings."""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
from pathlib import Path
import statistics

import numpy as np

from compare_llm import check_metadata, reference, shapes_for, validate_output
from native_rows import array_digest, case_name, digest, load_inputs, parse_inductor_wrapper, save

VARIANTS = ('whole', 'local', 'inductor')
ORDERS = list(itertools.permutations(VARIANTS))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def statistics_only(report, expected_cases):
    require(report['cpu_threads'] == 1 and report['samples'] == 7 and report['warmup_ms'] == 100 and report['target_ms'] == 30,
            'timing contract changed')
    require(report['metric'] == 'single_thread_native_entry_host_wall_us', 'wrong metric')
    require(report['aligned_payload_bytes'] == 64 and report['per_allocation_guard_elements'] == 128, 'guard/alignment contract changed')
    keys = [(c['operation'], tuple(c['dimensions'])) for c in report['cases']]
    require(keys == expected_cases and len(set(keys)) == len(keys), 'missing/repeated/reordered cases')
    summary = []
    for case in report['cases']:
        op, dims = case['operation'], case['dimensions']
        require(len(case['results']) == 18, 'missing native visits')
        keys = [(r['round'], r['variant']) for r in case['results']]
        require(len(set(keys)) == 18 and set(keys) == {(r, v) for r in range(6) for v in VARIANTS}, 'missing/duplicate rounds')
        require(set(case['output_sha256']) == set(VARIANTS), 'missing output snapshots')
        medians = {}
        for row in case['results']:
            require(tuple(row['order']) == ORDERS[row['round']], 'unbalanced native order')
            samples = row['samples_us']
            require(len(samples) == 7 and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in samples), 'invalid sample')
            median = statistics.median(samples)
            require(row['median_us'] == median, 'wrong median')
            require(type(row['repetitions']) is int and 1 <= row['repetitions'] <= 1048576, 'invalid repetition count')
            require(row['correctness']['elements'] == math.prod(dims) and row['correctness']['atol'] == 5e-5 and row['correctness']['rtol'] == 5e-5,
                    'incorrect full-output validation contract')
            require(math.isfinite(row['correctness']['max_abs_error']) and row['correctness']['max_abs_error'] >= 0, 'invalid oracle result')
            require(row['inputs_unchanged'] is True and row['workspace_guards_passed'] is True, 'failed input/workspace checks')
            require(row['guard_elements'] >= 512 and row['guard_elements'] % 128 == 0, 'incomplete guard checks')
            require(row['output_sha256'] == case['output_sha256'][row['variant']], 'fixed-entry output bits differ')
            medians[(row['round'], row['variant'])] = median
        summary_us = {v: statistics.median(medians[(r, v)] for r in range(6)) for v in VARIANTS}
        require(summary_us == case['summary_us'], 'incorrect case summary')
        ratios = {}
        for numerator, denominator in (('local', 'inductor'), ('whole', 'inductor'), ('local', 'whole')):
            values = [medians[(r, numerator)] / medians[(r, denominator)] for r in range(6)]
            item = dict(median=statistics.median(values), minimum=min(values), maximum=max(values), wins=sum(v < 1 for v in values), rounds=values)
            key = numerator + '/' + denominator
            require(item == case['paired_ratios'][key], 'incorrect paired ratio')
            ratios[key] = item
        summary.append(dict(operation=op, dimensions=dims, summary_us=summary_us, paired_ratios=ratios))
    return summary


def mutation_checks(report, expected):
    tests = {
        'missing_visit': lambda d: d['cases'][0]['results'].pop(),
        'duplicate_visit': lambda d: d['cases'][0]['results'].__setitem__(1, copy.deepcopy(d['cases'][0]['results'][0])),
        'wrong_median': lambda d: d['cases'][0]['results'][0].__setitem__('median_us', 0.0),
        'wrong_ratio': lambda d: d['cases'][0]['paired_ratios']['local/inductor'].__setitem__('median', 0.01),
        'changed_bits': lambda d: d['cases'][0]['results'][0].__setitem__('output_sha256', 'bad'),
        'worker_mismatch': lambda d: d.__setitem__('cpu_threads', 8),
        'failed_guard': lambda d: d['cases'][0]['results'][0].__setitem__('workspace_guards_passed', False),
        'wrong_oracle_size': lambda d: d['cases'][0]['results'][0]['correctness'].__setitem__('elements', 1),
        'unbalanced_order': lambda d: d['cases'][0]['results'][0].__setitem__('order', list(reversed(VARIANTS))),
    }
    rejected = []
    for name, mutate in tests.items():
        changed = copy.deepcopy(report)
        mutate(changed)
        try:
            statistics_only(changed, expected)
        except ValueError:
            rejected.append(name)
        else:
            raise ValueError('failed to reject evidence mutation: ' + name)
    return rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    captured = json.loads((args.capture / 'results.json').read_text())
    manifest = json.loads((args.prepared / 'manifest.json').read_text())
    report = json.loads((args.replay / 'results.json').read_text())
    require(captured['closure_unchanged'] and captured['runner_unchanged'] and manifest['runner_unchanged'] and report['runner_unchanged'], 'unfinished or changing run')
    require(manifest['capture_sha256'] == digest(args.capture / 'results.json') and report['manifest_sha256'] == digest(args.prepared / 'manifest.json'), 'mixed experiments')
    for directory, metadata in ((args.capture, captured), (args.prepared, manifest), (args.replay, report)):
        require(digest(directory / 'runner-sources/native_rows.py') == metadata['source_sha256'], 'missing frozen runner identity')
    require(digest(manifest['helper']) == manifest['helper_sha256'], 'helper changed')
    require(digest(args.prepared / 'runner-sources/native_rows_replay.cpp') == manifest['helper_source_sha256'], 'helper source changed')
    expected_cases = [(op, tuple(dims)) for op, dims in captured['cases']]
    require([(c['operation'], tuple(c['dimensions'])) for c in manifest['cases']] == expected_cases, 'prepared cohort differs')
    expected_captures = {(op, tuple(dims), local) for op, dims in expected_cases for local in (1, 8)}
    actual_captures = [(r['operation'], tuple(r['dimensions']), r['local_lanes']) for r in captured['results']]
    require(len(set(actual_captures)) == len(actual_captures) and set(actual_captures) == expected_captures, 'capture coverage differs')
    capture_checks, native_checks, environment_checks = [], [], []
    for row in captured['results']:
        require(row['valid'] is True, 'failed capture')
        source = Path(row['directory'])
        op, dims = row['operation'], row['dimensions']
        measurement = json.loads((source / 'measurement.json').read_text())
        check_metadata(measurement, 'cpu', op, dims, (1, 1), 3)
        for key, value in {'LUISA_SIMD_WORKER_COUNT': '1', 'LUISA_SIMD_WARP_WIDTH': '8',
                           'LUISA_TILE_BENCH_XIR_LOCAL_LANES': str(row['local_lanes']), 'LUISA_TILE_BENCH_XIR_BLOCK_SIZE': '32',
                           'LUISA_SIMD_DISABLE_LOAD_REDUCTION_FUSION': '1', 'LUISA_SIMD_ENABLE_FULL_PACKET_SPECIALIZATION': '1',
                           'LUISA_SIMD_ENABLE_PREDICATED_MEMORY_EFFECTS': '1', 'LUISA_SIMD_ENABLE_COHORT_PRIVATE_ACCESS': '1'}.items():
            require(row['environment'][key] == value, 'wrong compiler option')
        environment_checks.append(str(source))
        shapes, output_shape = shapes_for(op, dims)
        arrays = load_inputs(source, shapes)
        require([array_digest(a) for a in arrays] == row['input_sha256'], 'capture input changed')
        require(digest(source / 'output.f32') == row['output_sha256'], 'capture output changed')
        validation = validate_output(np.fromfile(source / 'output.f32', dtype=np.float32).reshape(output_shape), reference(op, dims, arrays))
        capture_checks.append(dict(case=case_name(op, dims), local_lanes=row['local_lanes'], validation=validation))
    summaries = statistics_only(report, expected_cases)
    for case, measured in zip(manifest['cases'], report['cases']):
        op, dims = case['operation'], case['dimensions']
        arrays = load_inputs(Path(case['inputs']), case['input_shapes'])
        require([array_digest(a) for a in arrays] == case['input_sha256'], 'replay input changed')
        expected = reference(op, dims, arrays)
        for variant in VARIANTS:
            entry = case['entries'][variant]
            require(digest(entry['library']) == entry['library_sha256'], 'native library changed')
            snapshot = args.replay / case_name(op, dims) / (variant + '.f32')
            require(digest(snapshot) == measured['output_sha256'][variant], 'native snapshot changed')
            validation = validate_output(np.fromfile(snapshot, dtype=np.float32).reshape(case['output_shape']), expected)
            if variant == 'inductor':
                base = Path(entry['library']).parent
                require(digest(base / 'inductor-wrapper.py') == entry['wrapper_sha256'] and digest(base / 'inductor.cpp') == entry['source_sha256'], 'Inductor source changed')
                plan = parse_inductor_wrapper((base / 'inductor-wrapper.py').read_text(), entry['graph']['inputs'], case['input_shapes'])
                plan.pop('cpp')
                require(json.loads(json.dumps(plan)) == entry['plan'], 'ABI plan changed')
                guards = (3 + len(plan['allocations'])) * 128
            else:
                source = Path(entry['capture'])
                require(digest(source / 'kernel.ll') == entry['llvm_sha256'], 'LLVM changed')
                objects = list((source / 'object').glob('*.o'))
                require(len(objects) == 1 and digest(objects[0]) == entry['object_sha256'], 'ORC object changed')
                guards = 512
            require(all(r['guard_elements'] == guards for r in measured['results'] if r['variant'] == variant), 'wrong native guard extent')
            native_checks.append(dict(case=case_name(op, dims), variant=variant, output_sha256=digest(snapshot), validation=validation))
    receipt = dict(status='passed', capture_outputs_re_read=len(capture_checks), native_unique_outputs_re_read=len(native_checks),
                   native_timed_visits=sum(len(c['results']) for c in report['cases']),
                   native_guard_checks='Recorded during every timed visit; guard arrays themselves are not retained.',
                   repeated_outputs='All six visits of each fixed entry report the same bits as its retained snapshot; mappings and frameworks need only match FP64 tolerance.',
                   adversarial_evidence_checks=mutation_checks(report, expected_cases),
                   capture_checks=capture_checks, native_checks=native_checks, summary=summaries,
                   source_sha256=digest(__file__), replay_sha256=digest(args.replay / 'results.json'))
    save(args.output, receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('capture_checks', 'native_checks', 'summary')}, indent=2))


if __name__ == '__main__':
    main()
