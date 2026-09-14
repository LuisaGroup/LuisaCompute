"""Re-read native outputs and reject empty tests/contended performance claims."""
import copy
import json
from pathlib import Path
import re
import statistics
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path('/Users/mike/CLionProjects/luisa')
sys.path.insert(0, str(REPO / 'scripts/benchmark/tile_torch'))
import native_rows as nr


def check_visit(row, visit, samples):
    assert len(visit['samples_us']) == samples
    assert all(np.isfinite(value) and value > 0 for value in visit['samples_us'])
    assert statistics.median(visit['samples_us']) == visit['median_us']
    assert visit['repetitions'] > 0 and visit['inputs_unchanged'] and visit['workspace_guards_passed']
    assert visit['guard_elements'] >= 128
    assert visit['output_sha256'] == row['output_sha256'][visit['variant']]


capture = json.loads((HERE / 'capture/manifest.json').read_text())
snapshot = json.loads((HERE / 'source-snapshot.json').read_text())
for name, digest in snapshot['source_sha256'].items():
    assert nr.digest(HERE / 'source' / name) == digest, name
for name in snapshot['overlay']:
    assert nr.digest(REPO / name) == snapshot['source_sha256'][name], name
for name, digest in capture['binary_closure_sha256'].items():
    assert nr.digest(name) == digest, name
assert capture['source_unchanged'] and capture['closure_unchanged']
expected_keys = {(op, (m, n + (n % 2 if op == 'rope' else 0))) for op in nr.OPS for m, n in nr.SHAPES}
assert len(capture['cases']) == 24
assert {(case['operation'], tuple(case['dimensions'])) for case in capture['cases']} == expected_keys
verification = json.loads((HERE / 'verification.json').read_text())
widths = json.loads((HERE / 'verification-widths-v2.json').read_text())
assert verification['passed'] and widths['passed'] and len(widths['runs']) == 5
for run in verification['runs'][:3] + widths['runs']:
    assert run['exit_code'] == 0 and nr.digest(HERE / run['log']) == run['log_sha256']
assert [run['tests'] for run in verification['runs'][1:3]] == [35, 79]
assert all(run['assertions'] > 0 and run['selected_test_passed'] for run in widths['runs'])
smoke = json.loads((HERE / 'native-verify/results.json').read_text())
assert smoke['timing_not_comparative'] and smoke['metric'] == 'native_entry_correctness_smoke'
assert smoke['runner_unchanged'] and len(smoke['cases']) == 24
assert smoke['manifest_sha256'] == nr.digest(HERE / 'capture/manifest.json')
host = json.loads((HERE / 'host-replay-1.json').read_text())
partial = json.loads((HERE / 'replay-1/results.json').read_text())
assert host['replay_exit_code'] == 143 and not host['performance_qualified']
assert any(item['heavy'] for item in host['observations'])
assert partial.get('finished_unix') is None
audit = dict(status='passed', performance_accepted=False, defaults_changed=False, cost_calibrated=False,
             reason='Background CPU activity during preflight and foreign rendering during replay. Only the owned native benchmark was stopped with SIGTERM; the partial cohort is excluded in full.',
             capture_manifest_sha256=nr.digest(HERE / 'capture/manifest.json'),
             snapshot_sha256=nr.digest(HERE / 'source-snapshot.json'),
             snapshots_re_read=0, smoke_visits=0, excluded_timed_visits=0, bitwise_equal_cases=0,
             unchanged_llvm_cases=0, unchanged_object_cases=0, structure=[],
             guard_caveat='Ephemeral input/workspace guards were checked by the replay driver; this audit rereads retained output snapshots and verifies those receipts, not the freed guards.',
             test_caveat='Initial wildcard width launches executed zero assertions and are excluded; the exact-name rerun and nonzero-assertion receipts supersede them.')
partial_cases = {(case['operation'], tuple(case['dimensions'])): case for case in partial['cases']}
for case, row in zip(capture['cases'], smoke['cases']):
    op, dims = case['operation'], case['dimensions']
    assert (op, dims) == (row['operation'], row['dimensions'])
    name = nr.case_name(op, dims)
    inputs = nr.load_inputs(Path(case['inputs']), case['input_shapes'])
    assert [nr.array_digest(value) for value in inputs] == case['input_sha256']
    expected = nr.reference(op, dims, inputs)
    assert len(row['results']) == 3 and {visit['variant'] for visit in row['results']} == {'off', 'on', 'inductor'}
    for visit in row['results']:
        check_visit(row, visit, 1)
        audit['smoke_visits'] += 1
    for variant, entry in case['entries'].items():
        assert nr.digest(entry['library']) == entry['library_sha256']
        output = np.fromfile(HERE / 'native-verify' / name / (variant + '.f32'), np.float32).reshape(case['output_shape'])
        assert nr.array_digest(output) == row['output_sha256'][variant]
        nr.validate_output(output, expected)
        audit['snapshots_re_read'] += 1
        if variant != 'inductor':
            captured = np.fromfile(HERE / 'capture' / name / variant / 'output.f32', np.float32).reshape(case['output_shape'])
            nr.validate_output(captured, expected)
            assert nr.array_digest(captured) == nr.array_digest(output)
            audit['snapshots_re_read'] += 1
    assert row['off_on_bitwise_equal'] and row['output_sha256']['off'] == row['output_sha256']['on']
    audit['bitwise_equal_cases'] += 1
    off, on = case['entries']['off'], case['entries']['on']
    audit['unchanged_llvm_cases'] += off['llvm_sha256'] == on['llvm_sha256']
    audit['unchanged_object_cases'] += off['object_sha256'] == on['object_sha256']
    def field(entry, key):
        return int(re.search(re.escape(key) + r'=(\d+)', entry['realization'])[1])
    fused = field(on, 'fused_reduction_expressions')
    assert fused == (2 if op == 'masked_softmax' else 1 if op == 'layernorm' else 0)
    if fused:
        audit['structure'].append(dict(operation=op, dimensions=dims, fused_expressions=fused,
                                       schedule_blocks=[field(entry, 'Schedule blocks') for entry in (off, on)],
                                       cloned_instructions=[field(entry, 'full_packet_cloned_instructions') for entry in (off, on)],
                                       workspace_bytes=[entry['workspace_bytes'] for entry in (off, on)]))
    measured = partial_cases.get((op, tuple(dims)))
    if measured:
        for visit in measured['results']:
            check_visit(measured, visit, 7)
            audit['excluded_timed_visits'] += 1
        for variant, digest in measured['output_sha256'].items():
            output = np.fromfile(HERE / 'replay-1' / name / (variant + '.f32'), np.float32).reshape(case['output_shape'])
            assert nr.array_digest(output) == digest
            nr.validate_output(output, expected)
            audit['snapshots_re_read'] += 1
rejected = []
for mutation in ('zero_sample', 'wrong_median', 'guard', 'input_mutation', 'output_hash'):
    value = copy.deepcopy(row['results'][0])
    if mutation == 'zero_sample':
        value['samples_us'] = [0.0]
    elif mutation == 'wrong_median':
        value['median_us'] += 1.0
    elif mutation == 'guard':
        value['workspace_guards_passed'] = False
    elif mutation == 'input_mutation':
        value['inputs_unchanged'] = False
    else:
        value['output_sha256'] = '0' * 64
    try:
        check_visit(row, value, 1)
    except AssertionError:
        rejected.append(mutation)
for mutation in ('nonfinite_output', 'changed_tail', 'truncated_output'):
    value = expected.astype(np.float32)
    if mutation == 'nonfinite_output':
        value.flat[-1] = np.nan
    elif mutation == 'changed_tail':
        value.flat[-1] += 1.0
    else:
        value = value.reshape(-1)[:-1]
    try:
        nr.validate_output(value, expected)
    except ValueError:
        rejected.append(mutation)
assert len(rejected) == 8
audit['rejected_mutations'] = rejected
assert (audit['smoke_visits'], audit['bitwise_equal_cases'], audit['unchanged_llvm_cases']) == (72, 24, 16)
assert audit['excluded_timed_visits'] == 96 and audit['snapshots_re_read'] == 138
nr.save(HERE / 'audit.json', audit)
print(json.dumps({key: value for key, value in audit.items() if key != 'structure'}, indent=2))
