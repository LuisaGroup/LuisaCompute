"""Independently validate both complete diagnostic cohorts and frozen phases."""
import copy
import itertools
import json
from pathlib import Path
import statistics
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path('/Users/mike/CLionProjects/luisa')
sys.path.insert(0, str(REPO / 'scripts/benchmark/tile_torch'))
import native_rows as nr

def read(path):
    return json.loads(path.read_text())

def key(case):
    return case['operation'], tuple(case['dimensions'])

expected_keys = {(op, (m, n + (n % 2 if op == 'rope' else 0))) for op in nr.OPS for m, n in nr.SHAPES}
frozen = read(HERE / 'stage-a-frozen.json')
assert frozen['passed']
for item in frozen['binary_relocations'].values():
    assert nr.digest(item['path']) == item['sha256']
for filename, location in [('source-snapshot.json', 'frozen-stage-a-source'), ('projection-source-snapshot.json', 'source')]:
    for name, sha in read(HERE / filename)['source_sha256'].items():
        assert nr.digest(HERE / location / name) == sha, name
tests = read(HERE / 'projection-tests.json')
assert tests['passed'] and [r['tests'] for r in tests['runs']] == [80, 35]
assert all(r['exit_code'] == r['failures'] == r['skipped'] == 0 for r in tests['runs'])
report = dict(passed=False, performance_accepted=False, defaults_changed=False, cost_calibrated=False,
              source_commit='2cfc804932a014cd7a07eb3896166d53db32b3eb', phases=[], snapshots_reread=0,
              reason='Both full cohorts were prospectively diagnostic-only under desktop coactivity; no quiet-window acceptance or policy calibration.',
              guard_caveat='Guards were checked during native replay. This audit rereads retained outputs and verifies guard receipts, not the freed guards.',
              artifact_override='The user selected existing repository Markdown/Sphinx reports, not a new dashboard, Sites or standalone HTML app.',
              report_shape='Technical summary; measured boundary; per-shape evidence; code-form results; validation/limitations; next steps and open questions. Exact tables replace charts for per-case audit lookup.')

def check_visit(row, visit, samples):
    assert len(visit['samples_us']) == samples
    assert all(np.isfinite(v) and v > 0 for v in visit['samples_us'])
    assert statistics.median(visit['samples_us']) == visit['median_us']
    assert visit['repetitions'] > 0 and visit['inputs_unchanged'] and visit['workspace_guards_passed']
    assert visit['guard_elements'] >= 128
    assert visit['output_sha256'] == row['output_sha256'][visit['variant']]

for prefix, name in [('', 'packet_loop_inlining'), ('projection-', 'integer_lane_projection')]:
    capture = read(HERE / (prefix + 'capture/manifest.json'))
    assert capture['feature'] == name and capture['source_unchanged'] and capture['closure_unchanged']
    assert len(capture['cases']) == 24 and set(map(key, capture['cases'])) == expected_keys
    assert nr.digest(capture['helper']) == capture['helper_sha256']
    phase = dict(feature=name, capture_manifest_sha256=nr.digest(HERE / (prefix + 'capture/manifest.json')),
                 bitwise_equal_cases=0, smoke_visits=0, timed_visits=0, cases=[])
    cohorts = []
    for mode in ('verify', 'diagnostic'):
        host = read(HERE / (prefix + mode + '-host.json'))
        cohort = read(HERE / (prefix + mode + '/results.json'))
        assert host['exit_code'] == 0 and not host['performance_qualified']
        assert cohort['runner_unchanged'] and cohort.get('finished_unix')
        assert cohort['manifest_sha256'] == phase['capture_manifest_sha256']
        assert len(cohort['cases']) == 24 and set(map(key, cohort['cases'])) == expected_keys
        cohorts.append((mode, cohort, {key(c): c for c in cohort['cases']}))
    for case in capture['cases']:
        label = nr.case_name(case['operation'], case['dimensions'])
        inputs = nr.load_inputs(Path(case['inputs']), case['input_shapes'])
        assert [nr.array_digest(v) for v in inputs] == case['input_sha256']
        expected = nr.reference(case['operation'], case['dimensions'], inputs)
        for variant, entry in case['entries'].items():
            assert nr.digest(entry['library']) == entry['library_sha256']
            if variant != 'inductor':
                objects = list((Path(entry['capture']) / 'object').glob('*.o'))
                assert len(objects) == 1 and nr.digest(objects[0]) == entry['object_sha256']
                assert nr.digest(Path(entry['capture']) / 'kernel.ll') == entry['llvm_sha256']
                output = np.fromfile(Path(entry['capture']) / 'output.f32', np.float32).reshape(case['output_shape'])
                nr.validate_output(output, expected)
                report['snapshots_reread'] += 1
        for mode, cohort, rows in cohorts:
            row = rows[key(case)]
            orders = list(itertools.permutations(('off', 'on', 'inductor'))) if mode == 'diagnostic' else [('off', 'on', 'inductor')]
            assert len(row['results']) == 3 * len(orders)
            assert row['off_on_bitwise_equal'] and row['output_sha256']['off'] == row['output_sha256']['on']
            for ordinal, visit in enumerate(row['results']):
                round_id, position = divmod(ordinal, 3)
                assert visit['round'] == round_id and tuple(visit['order']) == orders[round_id]
                assert visit['variant'] == orders[round_id][position]
                check_visit(row, visit, 7 if mode == 'diagnostic' else 1)
            phase['timed_visits' if mode == 'diagnostic' else 'smoke_visits'] += len(row['results'])
            for variant in ('off', 'on', 'inductor'):
                output = np.fromfile(HERE / (prefix + mode) / label / (variant + '.f32'), np.float32).reshape(case['output_shape'])
                assert nr.array_digest(output) == row['output_sha256'][variant]
                nr.validate_output(output, expected)
                report['snapshots_reread'] += 1
            if mode == 'verify':
                phase['bitwise_equal_cases'] += 1
                continue
            for numerator, denominator in [('on', 'off'), ('on', 'inductor'), ('off', 'inductor')]:
                values = []
                for round_id in range(6):
                    pair = {v['variant']: v['median_us'] for v in row['results'] if v['round'] == round_id}
                    values.append(pair[numerator] / pair[denominator])
                reported = row['paired_ratios'][numerator + '/' + denominator]
                assert values == reported['rounds']
                assert statistics.median(values) == reported['median'] and min(values) == reported['minimum'] and max(values) == reported['maximum']
                assert sum(v < 1 for v in values) == reported['wins']
            phase['cases'].append(dict(operation=case['operation'], dimensions=case['dimensions'], paired_ratios=row['paired_ratios'],
                                       medians_us={v: statistics.median(x['median_us'] for x in row['results'] if x['variant'] == v) for v in ('off', 'on', 'inductor')}))
    assert (phase['smoke_visits'], phase['timed_visits'], phase['bitwise_equal_cases']) == (72, 432, 24)
    report['phases'].append(phase)

first = read(HERE / 'capture/manifest.json')
second = read(HERE / 'projection-capture/manifest.json')
report['phase_controls_identical_objects'] = sum(a['entries']['off']['object_sha256'] == b['entries']['off']['object_sha256'] for a, b in zip(first['cases'], second['cases']))
report['projection_preoptimization_llvm_identical'] = sum(c['entries']['off']['llvm_sha256'] == c['entries']['on']['llvm_sha256'] for c in second['cases'])
assert report['phase_controls_identical_objects'] == report['projection_preoptimization_llvm_identical'] == 24
mutations = []
for mutation in ('zero_sample', 'wrong_median', 'guard', 'input_mutation', 'output_hash'):
    value = copy.deepcopy(row['results'][0])
    if mutation == 'zero_sample': value['samples_us'][0] = 0.0
    elif mutation == 'wrong_median': value['median_us'] += 1.0
    elif mutation == 'guard': value['workspace_guards_passed'] = False
    elif mutation == 'input_mutation': value['inputs_unchanged'] = False
    else: value['output_sha256'] = '0' * 64
    try: check_visit(row, value, 7)
    except AssertionError: mutations.append(mutation)
for mutation in ('nan_output', 'changed_tail', 'truncated_output'):
    value = expected.astype(np.float32)
    if mutation == 'nan_output': value.flat[-1] = np.nan
    elif mutation == 'changed_tail': value.flat[-1] += 1.0
    else: value = value.reshape(-1)[:-1]
    try: nr.validate_output(value, expected)
    except ValueError: mutations.append(mutation)
assert len(mutations) == 8 and report['snapshots_reread'] == 384
report.update(passed=True, rejected_mutations=mutations)
nr.save(HERE / 'audit.json', report)
print(json.dumps({k: v for k, v in report.items() if k != 'phases'}, indent=2))
