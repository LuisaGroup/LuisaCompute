#!/usr/bin/env python3
"""Independently audit coverage, code identity, policy state and paired ratios."""
import argparse
import copy
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics as stats
import sys

HERE = Path(__file__).resolve().parent


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == '.gz' else data)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def ratio(values):
    return dict(median=stats.median(values), minimum=min(values), maximum=max(values),
                slower_rounds=sum(x > 1 for x in values), rounds=len(values))


def audit_runtime(report, capture):
    from measure import CASES
    cases = [('rmsnorm', 64, 256), ('rmsnorm', 1024, 4096)] if capture else CASES
    expected = {(op, r, c, l, f, p, order)
                for op, r, c in cases for l, f, p, order in itertools.product((1, 8), (False, True), (False, True), range(1 if capture else 2))}
    assert report['closure_unchanged'] and report['capture_only'] == capture
    rows, inputs = {}, {}
    for row in report['results']:
        key = (row['operation'], *row['dimensions'], row['local_lanes'], row['load_reduction_fusion'], row['full_packet_specialization'], row['round'])
        assert key not in rows and row['valid'] and row['returncode'] == 0
        rows[key] = row
        op, r, c, local, fusion, packet, _ = key
        assert row['correctness']['elements'] == r * c
        assert row['correctness']['atol'] == row['correctness']['rtol'] == 5e-5
        m = row['measurement']
        assert m['operation'] == op and m['dimensions'] == [r, c]
        assert m['implementation'] == 'tile_xir_simd' and m['fast_math'] is False
        assert m['precision'] == 'fp32' and m['relaxed_precision'] is False
        assert len(m['throughput_us']) == 7 and all(math.isfinite(x) and x > 0 for x in m['throughput_us'])
        assert row['median_us'] == stats.median(m['throughput_us'])
        text = m['realization']
        assert f'local_lanes={local};' in text and f'load_reduction_fusion={str(fusion).lower()};' in text
        applied = int(re.search(r'full_packet_specializations=(\d+)', text)[1])
        assert applied in (0, 1) and (packet or applied == 0)
        env = row['environment']
        assert env['LUISA_SIMD_WORKER_COUNT'] == ('1' if capture else '8')
        assert env['LUISA_SIMD_ENABLE_FULL_PACKET_SPECIALIZATION'] == '1'
        assert ('LUISA_SIMD_DISABLE_FULL_PACKET_SPECIALIZATION' in env) != packet
        group = (op, r, c)
        assert group not in inputs or inputs[group] == row['input_sha256']
        inputs[group] = row['input_sha256']
    assert set(rows) == expected
    return rows


def main():
    if sys.flags.optimize:
        raise RuntimeError('run this audit without Python -O; assertions are required')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, help='also recheck every complete output against its FP64 oracle')
    args = parser.parse_args()
    captured = read(HERE / 'capture.json.gz')
    matrix = read(HERE / 'matrix.json.gz')
    capture_rows, matrix_rows = audit_runtime(captured, True), audit_runtime(matrix, False)
    rejected = []
    mutations = {
        'missing_visit': lambda r: r['results'].pop(),
        'duplicate_visit': lambda r: r['results'].append(copy.deepcopy(r['results'][0])),
        'failed_oracle': lambda r: r['results'][0].update(valid=False),
        'changed_median': lambda r: r['results'][0].update(median_us=0.1),
        'mixed_input_bits': lambda r: r['results'][0].update(input_sha256=['bad']),
        'wrong_worker_count': lambda r: r['results'][0]['environment'].update(LUISA_SIMD_WORKER_COUNT='8'),
    }
    for name, change in mutations.items():
        modified = copy.deepcopy(captured)
        change(modified)
        try:
            audit_runtime(modified, True)
        except AssertionError:
            rejected.append(name)
        else:
            raise RuntimeError('audit accepted mutated evidence: ' + name)
    assert captured['binary_closure_sha256'] == matrix['binary_closure_sha256']
    native = []
    for rows, local, fusion in itertools.product((64, 1024), (1, 8), (0, 1)):
        columns = 256 if rows == 64 else 4096
        name = f'native-{rows}-l{local}-f{fusion}'
        report = read(HERE / name / 'results.json')
        assert report['runtime_excluded'] and report['cpu_threads'] == 1 and not report['hardware_cycles']
        assert report['samples'] == 7 and report['sample_target_ms'] == 30 and report['warmup_ms'] == 100
        values = {}
        for row in report['results']:
            key = (row['round'], row['variant'])
            assert key not in values and len(row['samples_us']) == 7
            assert all(math.isfinite(x) and x > 0 for x in row['samples_us'])
            assert row['median_us'] == stats.median(row['samples_us'])
            assert row['correctness']['elements'] == rows * columns and row['guard_elements'] == 68
            assert tuple(row['order']) == list(itertools.permutations(('baseline', 'candidate', 'inductor')))[row['round']]
            values[key] = row['median_us']
        assert set(values) == set(itertools.product(range(6), ('baseline', 'candidate', 'inductor')))
        for variant, packet in (('baseline', False), ('candidate', True)):
            capture = HERE / 'capture' / f'rmsnorm-{rows}x{columns}-r0-l{local}-f{fusion}-p{int(packet)}'
            a = report['artifacts'][variant]
            assert sha(gzip.decompress((capture / 'kernel.o.gz').read_bytes())) == a['object_sha256']
            assert sha(gzip.decompress((capture / 'kernel.ll.gz').read_bytes())) == a['llvm_sha256']
            assert sha(gzip.decompress((HERE / name / (variant + '.dylib.gz')).read_bytes())) == a['library_sha256']
            assert set(a['undefined_system_symbols']) <= {'_memcpy'}
            if a['undefined_system_symbols']:
                assert '/usr/lib/libSystem.B.dylib' in a['linked_libraries']
            assert report['input_sha256'] == capture_rows[('rmsnorm', rows, columns, local, bool(fusion), packet, 0)]['input_sha256']
        torch = report['artifacts']['inductor']
        assert sha((HERE / name / 'inductor.cpp').read_bytes()) == torch['source_sha256']
        assert sha(gzip.decompress((HERE / name / 'inductor.so.gz').read_bytes())) == torch['library_sha256']
        medians = {v: stats.median(values[r, v] for r in range(6)) for v in ('baseline', 'candidate', 'inductor')}
        assert medians == report['summary_us']
        native.append(dict(rows=rows, columns=columns, local_lanes=local, fusion=bool(fusion), median_us=medians,
                           packet_on_off=ratio([values[r, 'candidate'] / values[r, 'baseline'] for r in range(6)]),
                           candidate_inductor=ratio([values[r, 'candidate'] / values[r, 'inductor'] for r in range(6)])))
    e2e = []
    from measure import CASES
    for op, r, c in CASES:
        for local, fusion in itertools.product((1, 8), (False, True)):
            get = lambda packet, order: matrix_rows[(op, r, c, local, fusion, packet, order)]
            e2e.append(dict(operation=op, dimensions=[r, c], local_lanes=local, fusion=fusion,
                             off_us=stats.median(get(False, o)['median_us'] for o in range(2)),
                             on_us=stats.median(get(True, o)['median_us'] for o in range(2)),
                             packet_on_off=ratio([get(True, o)['median_us'] / get(False, o)['median_us'] for o in range(2)]),
                             specializations=[int(re.search(r'full_packet_specializations=(\d+)', get(True, o)['measurement']['realization'])[1]) for o in range(2)]))
    provenance = read(HERE / 'provenance.json')
    assert len(provenance['default_code_identity']) == 8
    assert all(r['object_byte_identical'] and r['llvm_byte_identical'] for r in provenance['default_code_identity'])
    ctest = (HERE / 'final-ctest.log').read_text()
    assert '100% tests passed out of 38' in ctest
    assert all('[OK] No issues found!' in p.read_text() for p in HERE.glob('syntax-*.log'))
    assert '[OK] No issues found!' in (HERE / 'final-syntax-test.log').read_text()
    raw_checked = 0
    if args.raw:
        import numpy as np
        sys.path.insert(0, str(HERE.parents[1]))
        from compare_llm import reference, validate_output
        for section, rows in (('capture', capture_rows), ('matrix', matrix_rows)):
            for row in rows.values():
                output = Path(row['command'][-1])
                assert output.is_relative_to(args.raw.resolve())
                arrays = [np.fromfile(str(output) + f'.input{i}.f32', np.float32).reshape(shape)
                          for i, shape in enumerate(row['measurement']['input_shapes'])]
                expected = reference(row['operation'], tuple(row['dimensions']), arrays)
                actual = np.fromfile(output, np.float32).reshape(expected.shape)
                assert validate_output(actual, expected) == row['correctness']
                raw_checked += 1
    result = dict(passed=True, native_visits=144, capture_visits=16, runtime_visits=240,
                  full_outputs_rechecked=raw_checked, default_actual_code_byte_identical=8,
                  rejected_mutated_evidence=rejected,
                  native=native, runtime=e2e,
                  assessment='Share with caveats: fixed mapping/one host/FP32; native ABI is RMSNorm only; no new Metal/MPS/BLAS or universal/default parity claim.')
    (HERE / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
