#!/usr/bin/env python3
"""Recompute all task-grain measurements, coverage, and fixed-code controls."""
import argparse
from collections import defaultdict
import copy
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from compare_llm import reference, validate_output
from measure import PILOT, VALIDATION, VARIANTS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_visits(report, suite, rounds):
    cases = PILOT if suite == 'pilot' else VALIDATION
    if suite == 'fixed-grain':
        cases = [(op, dims) for op in ('rmsnorm', 'layernorm', 'masked_softmax', 'swiglu', 'gelu_residual', 'rope')
                 for dims in ((17, 66), (129, 768), (1024, 4096))]
    expected = set()
    for op, dims in cases:
        variants = {'grain_0', 'grain_1', 'grain_3', 'grain_4294967295'} if suite == 'fixed-grain' else set(VARIANTS)
        if op == 'attention':
            variants.remove('local')
        expected.update((op, dims, r, v) for r in range(rounds) for v in variants)
    actual = set()
    inputs = {}
    for row in report['results']:
        key = row['operation'], tuple(row['dimensions'])
        visit = (*key, row['round'], row['variant'])
        assert visit not in actual and row['valid'] and row['returncode'] == 0
        assert row['measurement']['operation'] == row['operation'] and row['measurement']['dimensions'] == row['dimensions']
        assert row['environment']['LUISA_SIMD_WORKER_COUNT'] == '8'
        assert row['median_us'] == statistics.median(row['measurement']['throughput_us'])
        assert key not in inputs or inputs[key] == row['input_sha256']
        inputs[key] = row['input_sha256']
        actual.add(visit)
    assert actual == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    args = parser.parse_args()
    args.raw = args.raw.resolve()
    groups = defaultdict(dict)
    fingerprints = {}
    logs = {}
    reports = {}
    closures = []
    full_checks = 0
    fixed_code_groups = defaultdict(set)
    fixed_output_groups = defaultdict(set)
    for suite, count, rounds in [('pilot', 40, 2), ('validation', 408, 4), ('fixed-grain', 144, 2)]:
        report = json.loads((args.raw / suite / 'results.json').read_text())
        verify_visits(report, suite, rounds)
        assert report['metric'] == 'runtime_e2e_synchronized_host_wall_us'
        assert report['closure_unchanged'] and report['full_packet_specialization']
        assert not report['load_reduction_fusion'] and report['cpu_workers_requested'] == 8
        assert report['runner_sha256'] == digest(HERE / 'measure.py')
        assert len(report['results']) == count
        closures.append(report['binary_closure_sha256'])
        seen = set()
        inputs_by_case = {}
        for row in report['results']:
            key = (row['operation'], tuple(row['dimensions']))
            visit = (*key, row['round'], row['variant'])
            assert visit not in seen and row['valid'] and row['returncode'] == 0
            seen.add(visit)
            assert row['round'] in range(rounds) and row['variant'] in row['order']
            assert row['environment']['LUISA_SIMD_WORKER_COUNT'] == '8'
            assert row['environment']['LUISA_SIMD_ENABLE_FULL_PACKET_SPECIALIZATION'] == '1'
            assert row['environment']['LUISA_SIMD_DISABLE_LOAD_REDUCTION_FUSION'] == '1'
            measurement = row['measurement']
            assert not measurement['fast_math'] and not measurement['relaxed_precision']
            samples = measurement['throughput_us']
            assert len(samples) == 7 and all(np.isfinite(x) and x > 0 for x in samples)
            assert row['median_us'] == statistics.median(samples)
            assert key not in inputs_by_case or row['input_sha256'] == inputs_by_case[key]
            inputs_by_case[key] = row['input_sha256']
            output = Path(row['command'][-1])
            assert output.is_relative_to(args.raw.resolve())
            paths = [Path(str(output) + f'.input{i}.f32') for i in range(3)]
            assert [digest(p) for p in paths] == row['input_sha256'] and digest(output) == row['output_sha256']
            arrays = [np.fromfile(p, dtype=np.float32).reshape(shape) for p, shape in zip(paths, measurement['input_shapes'])]
            expected = reference(row['operation'], row['dimensions'], arrays)
            validate_output(np.fromfile(output, dtype=np.float32).reshape(expected.shape), expected)
            full_checks += 1
            for path in [*paths, output]:
                fingerprints[str(path.relative_to(args.raw))] = dict(bytes=path.stat().st_size, sha256=digest(path))
            directory = output.parent
            logs[str(directory.relative_to(args.raw))] = {stream: (directory / f'{stream}.log').read_text() for stream in ('stdout', 'stderr')}
            groups[(suite, *key)].setdefault(row['variant'], []).append(row)
            if suite == 'fixed-grain':
                assert row['selected_block'] == 32 and row['selected_local'] == 8
                objects = list((directory / 'object').glob('*.o'))
                assert len(objects) == 1
                assert digest(directory / 'kernel.ll') == row['llvm_sha256'] and digest(objects[0]) == row['object_sha256']
                fixed_code_groups[key].add((row['llvm_sha256'], row['object_sha256']))
                fixed_output_groups[key].add(row['output_sha256'])
        reports[suite] = report
    assert closures[0] == closures[1] == closures[2]
    assert len(fixed_code_groups) == 18 and all(len(v) == 1 for v in fixed_code_groups.values())
    assert all(len(v) == 1 for v in fixed_output_groups.values())
    summaries = []
    for (suite, op, dims), variants in groups.items():
        rounds = 4 if suite == 'validation' else 2
        expected_variants = {'grain_0', 'grain_1', 'grain_3', 'grain_4294967295'} if suite == 'fixed-grain' else {'whole', 'local', 'joint_legacy', 'joint_tasks'}
        if op == 'attention':
            expected_variants.remove('local')
        assert set(variants) == expected_variants
        for rows in variants.values():
            assert {r['round'] for r in rows} == set(range(rounds)) and len(rows) == rounds
        medians = {v: statistics.median(r['median_us'] for r in rows) for v, rows in variants.items()}
        baseline = 'grain_0' if suite == 'fixed-grain' else 'joint_legacy'
        candidate = 'grain_4294967295' if suite == 'fixed-grain' else 'joint_tasks'
        ratios = [next(r['median_us'] for r in variants[candidate] if r['round'] == i) /
                  next(r['median_us'] for r in variants[baseline] if r['round'] == i) for i in range(rounds)]
        plans = {v: sorted({(r['selected_local'], r['selected_block'], r['selected_grain']) for r in rows}) for v, rows in variants.items()}
        summaries.append(dict(suite=suite, operation=op, dimensions=dims, medians_us=medians, plans=plans,
                              paired_candidate_over_baseline=statistics.median(ratios), paired_range=[min(ratios), max(ratios)],
                              candidate_round_wins=sum(r < 1 for r in ratios), rounds=rounds))
    rejected = []
    for name, mutate in {
        'missing_visit': lambda r: r['results'].pop(),
        'duplicate_visit': lambda r: r['results'].append(r['results'][0]),
        'failed_oracle': lambda r: r['results'][0].update(valid=False),
        'changed_median': lambda r: r['results'][0].update(median_us=-1),
        'mixed_inputs': lambda r: r['results'][0].update(input_sha256=['changed']),
        'wrong_worker_count': lambda r: r['results'][0]['environment'].update(LUISA_SIMD_WORKER_COUNT='1'),
    }.items():
        report = copy.deepcopy(reports['pilot'])
        mutate(report)
        try:
            verify_visits(report, 'pilot', 2)
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError('mutated evidence was accepted: ' + name)
    audit = dict(passed=True, visits=full_checks, fixed_llvm_and_object_identity_groups=18, fixed_output_identity_groups=18,
                 rejected_mutated_evidence=rejected,
                 metric='runtime_e2e_synchronized_host_wall_us', native_time_claim='none; fixed-grain native code is byte-identical',
                 summaries=summaries)
    (HERE / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    for name, value in [('runtime-logs', logs), ('tensor-fingerprints', fingerprints), *reports.items()]:
        (HERE / f'{name}.json.gz').write_bytes(gzip.compress((json.dumps(value, indent=2) + '\n').encode(), mtime=0))
    captures = HERE / 'captures'
    captures.mkdir(exist_ok=True)
    for key in fixed_code_groups:
        row = groups[('fixed-grain', *key)]['grain_0'][0]
        source = Path(row['command'][-1]).parent
        output = captures / f'{key[0]}-{"x".join(map(str, key[1]))}'
        output.mkdir(exist_ok=True)
        for name, path in [('kernel.ll.gz', source / 'kernel.ll'), ('kernel.o.gz', next((source / 'object').glob('*.o')))]:
            (output / name).write_bytes(gzip.compress(path.read_bytes(), mtime=0))
        (output / 'measurement.json').write_text(json.dumps(row['measurement'], indent=2) + '\n')
    print(json.dumps({k: v for k, v in audit.items() if k != 'summaries'}, indent=2))
    for row in summaries:
        print(row['suite'], row['operation'], row['dimensions'], row['medians_us'], 'paired ratio', round(row['paired_candidate_over_baseline'], 3))
    tables = ['# Task-grain 完整结果', '', '全部为 FP32、W8、8 个请求的 CPU workers、full-packet 特化开启、fusion 关闭。时间为各轮 p50 的中位数（µs），配对比值单独复算；不把不同尺寸合并成一个平均加速比。', '']
    for suite in ('pilot', 'validation', 'fixed-grain'):
        tables += ['## ' + suite, '']
        if suite == 'fixed-grain':
            columns = ['grain_0', 'grain_1', 'grain_3', 'grain_4294967295']
            tables += ['固定 local=8、block=32；最后一列候选将整个 launch 放在 caller 执行。代码与输出逐字节相同。', '']
        else:
            columns = ['whole', 'local', 'joint_legacy', 'joint_tasks']
            tables += ['候选策略见 notes.md；attention 不支持 local 候选，用 — 明确标出，未把未实现的路径计作胜利。', '']
        tables += ['| 算子 / 尺寸 | ' + ' | '.join(columns) + ' | 配对比值 | 配对范围 |', '|---|' + '---:|' * 6]
        for row in summaries:
            if row['suite'] != suite:
                continue
            values = [f'{row["medians_us"][v]:.3f}' if v in row['medians_us'] else '—' for v in columns]
            lo, hi = row['paired_range']
            tables.append('| ' + row['operation'] + ' ' + '×'.join(map(str, row['dimensions'])) + ' | ' + ' | '.join(values) + f' | {row["paired_candidate_over_baseline"]:.3f} | {lo:.3f}–{hi:.3f} |')
        tables += ['']
    (HERE / 'tables.md').write_text('\n'.join(tables))


if __name__ == '__main__':
    main()
