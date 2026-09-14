#!/usr/bin/env python3
"""Check comparisons, complete-case coverage, identities and timing boundaries."""
import copy
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def p50(samples):
    require(bool(samples) and all(math.isfinite(x) and x > 0 for x in samples), 'invalid timing samples')
    return statistics.median(samples)


def inspect(report):
    require(report['metric'] == 'runtime_e2e_synchronized_host_wall_us' and report['mode'] == 'factorial', 'wrong timing boundary')
    require(report['closure_unchanged'], 'binary closure changed')
    values, inputs = {}, {}
    expected_order = [[1, False], [1, True], [8, True], [8, False]]
    for row in report['results']:
        case = row['operation'], tuple(row['dimensions'])
        key = (*case, row['round'], row['local_lanes'], row['private_array_interleaving'])
        require(key not in values, 'duplicate visit')
        values[key] = row
        require(row['valid'] and row['returncode'] == 0, 'failed visit; do not discard it')
        require(row['round'] in (0, 1) and row['order'] == (expected_order if row['round'] == 0 else expected_order[::-1]), 'unbalanced factorial order')
        require(case not in inputs or inputs[case] == row['input_sha256'], 'different input bits')
        inputs[case] = row['input_sha256']
        m, c = row['measurement'], row['correctness']
        require(c['elements'] == math.prod(case[1]) and c['atol'] == c['rtol'] == 5e-5, 'incomplete oracle')
        require(m['timing'] == 'synchronized_host_wall' and m['fast_math'] is False and m['implementation'] == 'tile_xir_simd', 'changed math/path/boundary')
        require(m['correctness']['checks'] == 2 and m['correctness']['guard_elements_per_check'] == 34, 'missing complete guarded checks')
        require('8 CPU workers;' in m['realization'] and f"local_lanes={row['local_lanes']};" in m['realization'], 'wrong execution realization')
        for metric in ('throughput_us', 'latency_us'):
            require(len(m[metric]) == 5, 'wrong sample count')
            p50(m[metric])
        require(math.isclose(row['median_us'], p50(m['throughput_us']), rel_tol=1e-12), 'incorrect saved median')
    require(len(inputs) == 13 and len(values) == 104, 'missing or extra case visits')
    summaries = []
    for op, dimensions in inputs:
        medians, rounds = {}, {}
        for local, interleave in expected_order:
            name = f'l{local}_soa{int(interleave)}'
            rows = [values[op, dimensions, r, local, interleave] for r in (0, 1)]
            rounds[name] = [p50(row['measurement']['throughput_us']) for row in rows]
            medians[name] = p50(rounds[name])
        ratios = {name: [rounds[numerator][r] / rounds[denominator][r] for r in (0, 1)] for name, numerator, denominator in (
            ('layout_new_over_old', 'l1_soa1', 'l1_soa0'),
            ('mapping_local_over_program', 'l8_soa1', 'l1_soa1'),
            ('combined_new_over_old', 'l8_soa1', 'l1_soa0'))}
        summaries.append(dict(operation=op, dimensions=dimensions, median_us=medians,
                              paired_ratio_p50={name: p50(x) for name, x in ratios.items()},
                              paired_ratio_range={name: [min(x), max(x)] for name, x in ratios.items()}))
    return summaries


def native(name):
    report = read(HERE / name / 'results.json')
    require(report['runtime_excluded'] and not report['hardware_cycles'] and report['cpu_threads'] == 1 and
            report['metric'] == 'single_thread_native_entry_host_wall_us', 'wrong native boundary')
    require(len(report['results']) == 18, 'missing native visits')
    require(digest((HERE / 'replay_native.cpp').read_bytes()) == report['helper_source_sha256'], 'changed native helper')
    require(digest(gzip.decompress((HERE / name / 'replay.dylib.gz').read_bytes())) == report['helper_sha256'], 'changed helper binary')
    require(digest((HERE / name / 'inductor.cpp').read_bytes()) == report['artifacts']['inductor']['source_sha256'], 'changed generated Torch source')
    require(digest(gzip.decompress((HERE / name / 'inductor.so.gz').read_bytes())) == report['artifacts']['inductor']['library_sha256'], 'changed generated Torch binary')
    rows, columns = report['dimensions']
    for variant, local in (('baseline', 1), ('candidate', 8)):
        captured = HERE / 'capture-final' / f'rmsnorm-{rows}x{columns}-r0-l{local}'
        require(digest(gzip.decompress((captured / 'kernel.o.gz').read_bytes())) == report['artifacts'][variant]['object_sha256'], 'changed actual ORC object')
        require(digest(gzip.decompress((HERE / name / (variant + '.dylib.gz')).read_bytes())) == report['artifacts'][variant]['library_sha256'], 'changed linked ORC entry')
    result = {}
    orders = {tuple(order) for order in itertools.permutations(('baseline', 'candidate', 'inductor'))}
    for variant in ('baseline', 'candidate', 'inductor'):
        records = [r for r in report['results'] if r['variant'] == variant]
        require(len(records) == 6 and {tuple(r['order']) for r in records} == orders, 'unbalanced native orders')
        for r in records:
            require(r['guard_elements'] == 68 and r['correctness']['elements'] == rows * columns and len(r['samples_us']) == 7, 'incomplete native validation')
            require(math.isclose(p50(r['samples_us']), r['median_us'], rel_tol=1e-12), 'incorrect native median')
        result[variant] = p50([r['median_us'] for r in records])
        require(math.isclose(result[variant], report['summary_us'][variant], rel_tol=1e-12), 'incorrect native summary')
    return dict(dimensions=report['dimensions'], median_us=result)


def main():
    report = read(HERE / 'factorial/results.json')
    summaries = inspect(report)
    for mutate in (
        lambda r: r.update(metric='kernel_only'), lambda r: r.update(closure_unchanged=False),
        lambda r: r['results'][0]['measurement']['throughput_us'].__setitem__(0, float('nan')),
        lambda r: r['results'][0]['correctness'].update(elements=1),
        lambda r: r['results'][0].update(input_sha256=['wrong']),
        lambda r: r['results'].pop(),
    ):
        broken = copy.deepcopy(report)
        mutate(broken)
        try:
            inspect(broken)
        except (ValueError, KeyError):
            continue
        raise ValueError('corrupt evidence accepted')
    result = dict(factorial=summaries, native=[native('native-final64'), native('native-final1024')],
                  visits=104, rejected_mutations=6, confidence='share with caveats: two E2E orders, fixed W8 and FP32 M1 Max; not cross-target parity')
    result['sha256'] = {str(p.relative_to(HERE)): digest(p.read_bytes()) for p in sorted(HERE.rglob('*'))
                        if p.is_file() and p.name != 'audit.json' and '__pycache__' not in p.parts}
    (HERE / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'sha256'}, indent=2))


if __name__ == '__main__':
    main()
