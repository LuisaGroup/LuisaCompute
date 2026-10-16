#!/usr/bin/env python3
"""Audit all fixed-mapping visits; re-read Runtime outputs and archive witnesses.

Native outputs are checked in the replay before each timing row is saved.
This audit recomputes native statistics, not discarded native output arrays.
"""
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

from measure import cases, NATIVE_SHAPES

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE.parents[1]))
from compare_llm import reference, validate_output


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_gz(name, value):
    (HERE / name).write_bytes(gzip.compress((json.dumps(value, indent=2) + '\n').encode(), mtime=0))


def verify_visits(report, suite):
    expected = set()
    for op, dims in cases(suite):
        for local in ((8,) if suite == 'capture' else ((1,) if op == 'attention' else (1, 8))):
            for enabled in (False, True):
                for round_id in range(1 if suite == 'capture' else 2):
                    expected.add((op, dims, local, enabled, round_id))
    actual, inputs = set(), {}
    for row in report['results']:
        key = row['operation'], tuple(row['dimensions'])
        visit = (*key, row['local_lanes'], row['predicated_memory_effects'], row['round'])
        assert visit not in actual and row['valid'] and row['returncode'] == 0
        actual.add(visit)
        env = row['environment']
        assert env['LUISA_SIMD_WORKER_COUNT'] == ('1' if suite == 'capture' else '8')
        assert env['LUISA_SIMD_WARP_WIDTH'] == '8'
        assert env['LUISA_TILE_BENCH_XIR_BLOCK_SIZE'] == '32'
        assert env['LUISA_TILE_BENCH_XIR_BLOCKS_PER_TASK'] == '0'
        assert env['LUISA_TILE_BENCH_XIR_LOCAL_LANES'] == str(row['local_lanes'])
        assert env['LUISA_SIMD_DISABLE_LOAD_REDUCTION_FUSION'] == '1'
        assert env['LUISA_SIMD_ENABLE_FULL_PACKET_SPECIALIZATION'] == '1'
        assert env['LUISA_SIMD_ENABLE_PREDICATED_MEMORY_EFFECTS'] == '1'
        assert (env.get('LUISA_SIMD_DISABLE_PREDICATED_MEMORY_EFFECTS') == '1') != row['predicated_memory_effects']
        m = row['measurement']
        assert m['operation'] == row['operation'] and m['dimensions'] == row['dimensions']
        assert not m['fast_math'] and not m['relaxed_precision']
        samples = m['throughput_us']
        assert len(samples) == 7 and all(np.isfinite(s) and s > 0 for s in samples)
        assert row['median_us'] == statistics.median(samples)
        assert key not in inputs or inputs[key] == row['input_sha256']
        inputs[key] = row['input_sha256']
    assert actual == expected


def paired(variants, numerator, denominator, rounds):
    ratios = [next(r['median_us'] for r in variants[numerator] if r['round'] == i) /
              next(r['median_us'] for r in variants[denominator] if r['round'] == i) for i in range(rounds)]
    return dict(median=statistics.median(ratios), range=[min(ratios), max(ratios)],
                round_wins=sum(x < 1 for x in ratios), ratios=ratios)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    args = parser.parse_args()
    raw = args.raw.resolve()
    provenance = json.loads((HERE / 'provenance.json').read_text())
    assert provenance['runner_sha256'] == digest(HERE / 'measure.py')
    for file, sha in provenance['tested_source_sha256'].items():
        assert sha == digest(ROOT / file) == digest(Path(provenance['isolated']) / 'source' / file)
    for file, sha in provenance['binary_closure_sha256'].items():
        assert digest(Path(file)) == sha
    logs, fingerprints, groups, reports = {}, {}, defaultdict(dict), {}
    output_groups = defaultdict(set)
    runtime_checks = 0
    for suite, count in [('broad', 344), ('capture', 8)]:
        report = json.loads((raw / suite / 'results.json').read_text())
        verify_visits(report, suite)
        assert report['closure_unchanged'] and len(report['results']) == count
        assert report['binary_closure_sha256'] == provenance['binary_closure_sha256']
        assert report['source_runner_sha256'] == provenance['runner_sha256']
        assert report['metric'] == 'runtime_e2e_synchronized_host_wall_us'
        for row in report['results']:
            directory = Path(row['directory']).resolve()
            assert directory.is_relative_to(raw)
            output = directory / 'output.f32'
            inputs = [Path(str(output) + f'.input{i}.f32') for i in range(len(row['measurement']['input_shapes']))]
            assert [digest(p) for p in inputs] == row['input_sha256'] and digest(output) == row['output_sha256']
            arrays = [np.fromfile(p, dtype=np.float32).reshape(s) for p, s in zip(inputs, row['measurement']['input_shapes'])]
            expected = reference(row['operation'], row['dimensions'], arrays)
            validate_output(np.fromfile(output, dtype=np.float32).reshape(expected.shape), expected)
            runtime_checks += 1
            for path in [output, *inputs]:
                fingerprints[str(path.relative_to(raw))] = dict(bytes=path.stat().st_size, sha256=digest(path))
            logs[str(directory.relative_to(raw))] = {s: (directory / f'{s}.log').read_text() for s in ('stdout', 'stderr')}
            key = suite, row['operation'], tuple(row['dimensions']), row['local_lanes']
            groups[key].setdefault(str(int(row['predicated_memory_effects'])), []).append(row)
            output_groups[key].add(row['output_sha256'])
            if suite == 'capture':
                objects = list((directory / 'object').glob('*.o'))
                assert len(objects) == 1
                target = HERE / 'captures' / directory.name
                target.mkdir(parents=True, exist_ok=True)
                for name, source in [('kernel.ll.gz', directory / 'kernel.ll'), ('kernel.o.gz', objects[0])]:
                    (target / name).write_bytes(gzip.compress(source.read_bytes(), mtime=0))
                (target / 'measurement.json').write_text(json.dumps(row['measurement'], indent=2) + '\n')
        reports[suite] = report
        save_gz(suite + '.json.gz', report)
    # No math-order change in this implementation: exact output bits should
    # match at fixed mapping, not necessarily across whole/local mappings.
    assert all(len(s) == 1 for s in output_groups.values())
    summaries = []
    for (suite, op, dims, local), variants in groups.items():
        if suite == 'capture':
            continue  # capture timings are not a performance cohort
        summaries.append(dict(operation=op, dimensions=dims, local_lanes=local,
                              medians_us={v: statistics.median(r['median_us'] for r in rows) for v, rows in variants.items()},
                              candidate_over_baseline=paired(variants, '1', '0', 2),
                              direct_cfg={v: sorted({'direct CFG=true' in r['measurement']['realization'] for r in rows}) for v, rows in variants.items()}))
    native_summaries, native_checks = [], 0
    for dims in NATIVE_SHAPES:
        directory = raw / ('native-' + 'x'.join(map(str, dims)))
        report = json.loads((directory / 'results.json').read_text())
        assert report['dimensions'] == list(dims) and len(report['results']) == 18
        assert report['metric'] == 'single_thread_native_entry_host_wall_us'
        assert report['runtime_excluded'] and not report['hardware_cycles'] and report['cpu_threads'] == 1
        assert report['helper_source_sha256'] == digest(HERE / 'replay_native.cpp')
        assert report['helper_sha256'] == digest(directory / 'replay.dylib')
        variants = defaultdict(list)
        seen = set()
        for row in report['results']:
            key = row['variant'], row['round']
            assert key not in seen and key[0] in ('baseline', 'candidate', 'inductor') and key[1] in range(6)
            seen.add(key)
            assert len(row['samples_us']) == 7 and all(np.isfinite(x) and x > 0 for x in row['samples_us'])
            assert row['median_us'] == statistics.median(row['samples_us'])
            assert row['guard_elements'] == 68 and row['correctness']['elements'] == dims[0] * dims[1]
            variants[row['variant']].append(row)
            native_checks += 1
        expected_rows = {('rmsnorm', tuple(r['dimensions'])): r for r in reports['capture']['results']}
        assert report['input_sha256'] == expected_rows[('rmsnorm', dims)]['input_sha256']
        medians = {v: statistics.median(r['median_us'] for r in rows) for v, rows in variants.items()}
        assert medians == report['summary_us']
        target = HERE / directory.name
        target.mkdir(exist_ok=True)
        for variant in ('baseline', 'candidate'):
            info = report['artifacts'][variant]
            assert info['replay_abi'] in (0, 2)
            assert info['exported_symbol'] == ('llm_rows.packet_batch.blocks' if info['replay_abi'] == 0 else 'llm_rows.packet_batch')
            source = Path(info['source_capture'])
            assert source.is_relative_to(raw) and info['llvm_sha256'] == digest(source / 'kernel.ll')
            objects = list((source / 'object').glob('*.o'))
            assert len(objects) == 1 and digest(objects[0]) == info['object_sha256']
            assert digest(directory / (variant + '.dylib')) == info['library_sha256']
        info = report['artifacts']['inductor']
        cpp = Path(info['source'])
        assert cpp.is_relative_to(raw) and digest(cpp) == info['source_sha256']
        assert digest(cpp.with_suffix('.so')) == info['library_sha256']
        (target / 'inductor.cpp').write_bytes(cpp.read_bytes())
        (target / 'inductor.so.gz').write_bytes(gzip.compress(cpp.with_suffix('.so').read_bytes(), mtime=0))
        (target / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
        native_summaries.append(dict(dimensions=dims, medians_us=medians,
                                     candidate_over_baseline=paired(variants, 'candidate', 'baseline', 6),
                                     candidate_over_inductor=paired(variants, 'candidate', 'inductor', 6)))
    rejected = []
    for name, mutate in {
        'missing_visit': lambda r: r['results'].pop(),
        'duplicate_visit': lambda r: r['results'].append(r['results'][0]),
        'failed_oracle': lambda r: r['results'][0].update(valid=False),
        'wrong_median': lambda r: r['results'][0].update(median_us=-1),
        'mixed_inputs': lambda r: r['results'][0].update(input_sha256=['changed']),
        'wrong_worker_count': lambda r: r['results'][0]['environment'].update(LUISA_SIMD_WORKER_COUNT='1'),
        'wrong_toggle': lambda r: r['results'][0]['environment'].pop('LUISA_SIMD_DISABLE_PREDICATED_MEMORY_EFFECTS'),
    }.items():
        report = copy.deepcopy(reports['broad'])
        mutate(report)
        try:
            verify_visits(report, 'broad')
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError('accepted mutated evidence: ' + name)
    report = dict(passed=True, runtime_full_outputs_rechecked=runtime_checks, native_full_checks_at_measurement=native_checks,
                  native_audit_limit='output arrays are not retained; recomputed timings and validated recorded full-output/guard checks',
                  fixed_mapping_bit_identical_output_groups=len(output_groups), rejected_mutated_evidence=rejected,
                  summaries=summaries, native_summaries=native_summaries)
    (HERE / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    save_gz('runtime-logs.json.gz', logs)
    save_gz('tensor-fingerprints.json.gz', fingerprints)
    save_gz('validation-logs.json.gz', {p.name: p.read_text() for p in sorted(raw.glob('*.log'))})
    preflight = raw / 'native-17x65-preflight-helper/results.json'
    save_gz('preflight-helper.json.gz', json.loads(preflight.read_text()))
    print(json.dumps({k: v for k, v in report.items() if k not in ('summaries', 'native_summaries')}, indent=2))
    tables = ['# Ragged CFG 完整结果', '', 'FP32，W8，block=32，legacy task grain，full-packet 开、fusion 关、fast math 关。时间单位 µs。两种计时边界不能混用；配对比值是逐轮比值的中位数，不是表中中位时间相除。', '',
              '## 单线程纯 native entry：RMSNorm', '', '实际 ORC / TorchInductor 入口；6 个平衡顺序，每次 7 个样本。不含 Runtime、Python、分配；保留原生函数调用、Luisa launch record 重置及 LLVM 生成的 libc 调用。', '',
              '| 尺寸 | baseline | candidate | Inductor | candidate/baseline | candidate/Inductor | 后者逐轮范围 |', '|---|---:|---:|---:|---:|---:|---:|']
    for row in native_summaries:
        values = [row['medians_us'][v] for v in ('baseline', 'candidate', 'inductor')]
        lo, hi = row['candidate_over_inductor']['range']
        tables.append('| ' + '×'.join(map(str, row['dimensions'])) + ' | ' + ' | '.join(f'{x:.3f}' for x in values) + f" | {row['candidate_over_baseline']['median']:.3f} | {row['candidate_over_inductor']['median']:.3f} | {lo:.3f}–{hi:.3f} |")
    tables += ['', '## Runtime E2E：全部 86 个固定映射对照', '', '8 个请求的 CPU workers；2 个反向顺序，每次 7 个样本；小差异与相反方向的轮次不能视为稳定收益。local=1 为 whole-program，local=8 为 packet-local；attention 仅测 local=1。RoPE 要求偶数宽度，奇数测试点向上取偶，开关两侧相同。', '',
               '| 算子 / 尺寸 | local | off | on | on/off 配对比 | 配对范围 | direct CFG off→on |', '|---|---:|---:|---:|---:|---:|---|']
    for row in summaries:
        ratio = row['candidate_over_baseline']
        lo, hi = ratio['range']
        cfg = row['direct_cfg']
        tables.append('| ' + row['operation'] + ' ' + '×'.join(map(str, row['dimensions'])) + f" | {row['local_lanes']} | {row['medians_us']['0']:.3f} | {row['medians_us']['1']:.3f} | {ratio['median']:.3f} | {lo:.3f}–{hi:.3f} | {cfg['0']}→{cfg['1']} |")
    (HERE / 'tables.md').write_text('\n'.join(tables) + '\n')


if __name__ == '__main__':
    main()
