#!/usr/bin/env python3
"""Independently check timing boundaries, fixed controls, coverage and artifacts."""
import copy
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'm1-max-20260908-xir-packet-local'
CASES = {
    ('rmsnorm', 64, 256), ('rmsnorm', 17, 65), ('rmsnorm', 1024, 4096),
    ('rmsnorm', 17, 16384), ('rmsnorm', 1, 4096),
    ('layernorm', 1, 4096), ('layernorm', 17, 16384), ('layernorm', 1024, 4096),
    ('masked_softmax', 64, 4096), ('swiglu', 17, 65), ('swiglu', 1024, 4096),
    ('gelu_residual', 17, 65), ('gelu_residual', 1024, 4096),
    ('rope', 64, 128), ('rope', 17, 66),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def p50(values):
    require(bool(values) and all(math.isfinite(x) and x > 0 for x in values), 'invalid samples')
    return statistics.median(values)


def inspect(report):
    require(report['metric'] == 'runtime_e2e_synchronized_host_wall_us' and not report['capture_only'], 'wrong boundary')
    require(report['closure_unchanged'] and report['private_array_interleaving'], 'changed layout or binary')
    require(report['source_runner_sha256'] == digest((HERE / 'measure.py').read_bytes()), 'changed runner')
    records, inputs = {}, {}
    order = [[1, False], [1, True], [8, True], [8, False]]
    for row in report['results']:
        case = row['operation'], *row['dimensions']
        key = (*case, row['round'], row['local_lanes'], row['contiguous_private_access'])
        require(key not in records, 'duplicate visit')
        records[key] = row
        require(row['valid'] and row['returncode'] == 0, 'failed visit; retain the failure')
        require(row['order'] == (order if row['round'] == 0 else order[::-1]), 'wrong A/B order')
        require(case not in inputs or inputs[case] == row['input_sha256'], 'input bits differ')
        inputs[case] = row['input_sha256']
        env = row['environment']
        require(env['LUISA_SIMD_WORKER_COUNT'] == '8' and env['LUISA_SIMD_WARP_WIDTH'] == '8', 'changed target configuration')
        require(env['LUISA_TILE_BENCH_XIR_LOCAL_LANES'] == str(row['local_lanes']), 'mapping changed')
        require(('LUISA_SIMD_DISABLE_CONTIGUOUS_PRIVATE_ACCESS' in env) != row['contiguous_private_access'], 'incorrect A/B flag')
        m, c = row['measurement'], row['correctness']
        require(m['timing'] == 'synchronized_host_wall' and not m['fast_math'] and m['implementation'] == 'tile_xir_simd', 'wrong implementation/math')
        require(c['elements'] == math.prod(case[1:]) and c['atol'] == c['rtol'] == 5e-5, 'incomplete oracle')
        require(m['correctness']['checks'] == 2 and m['correctness']['guard_elements_per_check'] == 34, 'missing guarded native checks')
        require('8 CPU workers;' in m['realization'] and f"local_lanes={row['local_lanes']};" in m['realization'], 'wrong realization')
        for metric in ('throughput_us', 'latency_us'):
            require(len(m[metric]) == 7, 'incorrect sample count')
            p50(m[metric])
        require(math.isclose(row['median_us'], p50(m['throughput_us']), rel_tol=1e-12), 'wrong median')
    expected = {(*case, round_id, local, flag) for case in CASES for round_id in (0, 1) for local, flag in order}
    require(set(records) == expected, 'missing or unexpected case visit')
    summaries = []
    for case in sorted(CASES):
        medians, ratios, spans = {}, {}, {}
        for local in (1, 8):
            visits = {flag: [p50(records[*case, r, local, flag]['measurement']['throughput_us']) for r in (0, 1)] for flag in (False, True)}
            for flag, values in visits.items():
                medians[f'l{local}_v{int(flag)}'] = p50(values)
            paired = [visits[True][r] / visits[False][r] for r in (0, 1)]
            ratios[f'l{local}'] = p50(paired)
            spans[f'l{local}'] = [min(paired), max(paired)]
        summaries.append(dict(operation=case[0], dimensions=case[1:], median_us=medians,
                              paired_new_over_old=ratios, paired_ratio_range=spans))
    return summaries


def native(rows, columns, local):
    name = f'native-{rows}-l{local}'
    report = read(HERE / name / 'results.json')
    require(report['metric'] == 'single_thread_native_entry_host_wall_us' and report['runtime_excluded'] and
            not report['hardware_cycles'] and report['cpu_threads'] == 1 and report['packet_width'] == 8, 'wrong native boundary')
    require(report['dimensions'] == [rows, columns] and len(report['results']) == 18, 'wrong native coverage')
    require(digest((PREVIOUS / 'replay_native.cpp').read_bytes()) == report['helper_source_sha256'], 'changed native helper')
    for file, hash_value in (
        ('replay.dylib.gz', report['helper_sha256']),
        ('inductor.so.gz', report['artifacts']['inductor']['library_sha256']),
    ):
        require(digest(gzip.decompress((HERE / name / file).read_bytes())) == hash_value, 'changed actual native binary')
    require(digest((HERE / name / 'inductor.cpp').read_bytes()) == report['artifacts']['inductor']['source_sha256'], 'changed generated source')
    for variant, flag in (('baseline', 0), ('candidate', 1)):
        capture = HERE / 'capture-final' / f'rmsnorm-{rows}x{columns}-r0-l{local}-v{flag}'
        require(digest(gzip.decompress((capture / 'kernel.o.gz').read_bytes())) == report['artifacts'][variant]['object_sha256'], 'wrong actual ORC object')
        require(digest(gzip.decompress((capture / 'kernel.ll.gz').read_bytes())) == report['artifacts'][variant]['llvm_sha256'], 'wrong captured LLVM')
        require(digest(gzip.decompress((HERE / name / f'{variant}.dylib.gz').read_bytes())) == report['artifacts'][variant]['library_sha256'], 'wrong linked entry')
    orders = set(itertools.permutations(('baseline', 'candidate', 'inductor')))
    medians, by_round = {}, {}
    for variant in ('baseline', 'candidate', 'inductor'):
        visits = [r for r in report['results'] if r['variant'] == variant]
        require(len(visits) == 6 and {tuple(r['order']) for r in visits} == orders, 'wrong native orders')
        for r in visits:
            require(r['guard_elements'] == 68 and r['correctness']['elements'] == rows * columns and len(r['samples_us']) == 7, 'missing native checks')
            require(r['correctness']['atol'] == r['correctness']['rtol'] == 5e-5, 'changed tolerance')
            require(math.isclose(p50(r['samples_us']), r['median_us'], rel_tol=1e-12), 'wrong native median')
            by_round[variant, r['round']] = r['median_us']
        medians[variant] = p50([r['median_us'] for r in visits])
        require(math.isclose(medians[variant], report['summary_us'][variant], rel_tol=1e-12), 'wrong native summary')
    ratios = {denominator: [by_round['candidate', r] / by_round[denominator, r] for r in range(6)] for denominator in ('baseline', 'inductor')}
    return dict(dimensions=[rows, columns], local_lanes=local, median_us=medians,
                paired_ratio_p50={k: p50(v) for k, v in ratios.items()},
                paired_ratio_range={k: [min(v), max(v)] for k, v in ratios.items()})


def main():
    report = read(HERE / 'matrix/results.json')
    summaries = inspect(report)
    mutations = (
        lambda r: r.update(metric='kernel_only'), lambda r: r.update(closure_unchanged=False),
        lambda r: r['results'][0]['measurement']['throughput_us'].__setitem__(0, float('nan')),
        lambda r: r['results'][0]['correctness'].update(elements=1),
        lambda r: r['results'][0].update(input_sha256=['wrong']), lambda r: r['results'].pop(),
        lambda r: r['results'][0]['environment'].update(LUISA_SIMD_WORKER_COUNT='1'),
        lambda r: r.update(private_array_interleaving=False),
    )
    for mutate in mutations:
        broken = copy.deepcopy(report)
        mutate(broken)
        try:
            inspect(broken)
        except (ValueError, KeyError):
            continue
        raise ValueError('corrupt evidence accepted')
    result = dict(matrix=summaries, native=[native(rows, columns, local) for rows, columns in ((64, 256), (1024, 4096)) for local in (1, 8)],
                  runtime_visits=120, native_visits=72, rejected_mutations=len(mutations),
                  confidence='descriptive fixed-W8/FP32 M1 Max evidence; two E2E orders; not cross-target or all-operator parity')
    result['sha256'] = {str(p.relative_to(HERE)): digest(p.read_bytes()) for p in sorted(HERE.rglob('*'))
                        if p.is_file() and p.name != 'audit.json' and '__pycache__' not in p.parts}
    (HERE / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'sha256'}, indent=2))


if __name__ == '__main__':
    main()
