"""Validate both complete replays, without accepting their contended timings."""
import copy
import itertools
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np

ROOT = Path('/Users/mike/CLionProjects/luisa')
RAW = Path(__file__).resolve().parent
CAPTURE = Path('/tmp/luisa-pointwise.0imY7I/capture-final')
sys.path.insert(0, str(ROOT / 'scripts/benchmark/tile_torch'))
import native_rows as nr


def check_record(case):
    assert len(case['results']) == 18
    orders = set()
    for r in range(6):
        visits = [v for v in case['results'] if v['round'] == r]
        assert len(visits) == 3
        order = tuple(v['variant'] for v in visits)
        assert set(order) == {'off', 'on', 'inductor'}
        assert all(tuple(v['order']) == order for v in visits)
        orders.add(order)
        for visit in visits:
            samples = visit['samples_us']
            assert len(samples) == 7 and all(np.isfinite(x) and x > 0 for x in samples)
            assert statistics.median(samples) == visit['median_us']
            assert visit['repetitions'] > 0
            assert visit['inputs_unchanged'] and visit['workspace_guards_passed']
            assert visit['guard_elements'] >= 128
            assert visit['output_sha256'] == case['output_sha256'][visit['variant']]
    assert orders == set(itertools.permutations(('off', 'on', 'inductor')))
    for a, b in (('on', 'off'), ('on', 'inductor'), ('off', 'inductor')):
        ratios = []
        for r in range(6):
            pair = {v['variant']: v['median_us'] for v in case['results'] if v['round'] == r}
            ratios.append(pair[a] / pair[b])
        saved = case['paired_ratios'][a + '/' + b]
        assert saved == dict(median=statistics.median(ratios), minimum=min(ratios), maximum=max(ratios),
                             wins=sum(x < 1 for x in ratios), rounds=ratios)


manifest = json.loads((CAPTURE / 'manifest.json').read_text())
published = ROOT / 'scripts/benchmark/tile_torch/results/m1-max-20260909-xir-pointwise/provenance.json'
previous = json.loads(published.read_text())
assert nr.digest(CAPTURE / 'manifest.json') == previous['artifacts']['capture.json.gz']['source_sha256']
assert manifest['source_unchanged'] and manifest['closure_unchanged']
for name, sha in manifest['source_sha256'].items():
    committed = subprocess.check_output(['git', 'show', '67330a633ee7901034ab4a9360d9cd15be904b1f:' + name], cwd=ROOT)
    assert hashlib.sha256(committed).hexdigest() == sha, name
    assert nr.digest(Path('/tmp/luisa-reduction-checkpoint.gQUERp/source') / name) == sha, name
for path, sha in manifest['binary_closure_sha256'].items():
    assert nr.digest(path) == sha, path
assert nr.digest(manifest['helper']) == manifest['helper_sha256']
audit = dict(status='passed', capture_manifest_sha256=nr.digest(CAPTURE / 'manifest.json'),
             previous_checkpoint_sha256=nr.digest(published), runs=[], snapshots_re_read=0,
             timed_visits_checked=0, off_on_bitwise_equal_cases=0,
             performance_accepted=False,
             reason='Both complete attempts contain sampled concurrent foreign rendering. No performance ranking, default selection or model calibration is accepted.',
             guard_caveat='Input immutability and ephemeral guards were checked during replay, not retained for later inspection.')
expected_cases = {(op, (m, n + (n % 2 if op == 'rope' else 0))) for op in nr.OPS for m, n in nr.SHAPES}
for label, observation in (('replay-v1', 'host-observation.json'), ('replay-v2', 'host-observation-v2.json')):
    folder = RAW / label
    report = json.loads((folder / 'results.json').read_text())
    host = json.loads((RAW / observation).read_text())
    assert host['benchmark_exit_code'] == 0 and host['heavy_activity_seen']
    assert any(o['heavy'] for o in host['observations'])
    assert report['runner_unchanged'] and report['finished_unix'] >= report['started_unix']
    assert report['manifest_sha256'] == nr.digest(CAPTURE / 'manifest.json')
    assert report['metric'] == 'single_thread_native_entry_host_wall_us'
    assert (report['cpu_threads'], report['samples'], report['warmup_ms'], report['target_ms']) == (1, 7, 100, 30)
    assert nr.digest(folder / 'pilot.py') == report['pilot_sha256']
    assert nr.digest(folder / 'runner-sources/native_rows.py') == report['native_rows_sha256']
    assert {(c['operation'], tuple(c['dimensions'])) for c in report['cases']} == expected_cases
    assert len(manifest['cases']) == len(report['cases']) == 24
    for case, measured in zip(manifest['cases'], report['cases']):
        assert (case['operation'], case['dimensions']) == (measured['operation'], measured['dimensions'])
        check_record(measured)
        arrays = nr.load_inputs(Path(case['inputs']), case['input_shapes'])
        assert [nr.array_digest(a) for a in arrays] == case['input_sha256']
        expected = nr.reference(case['operation'], case['dimensions'], arrays)
        name = nr.case_name(case['operation'], case['dimensions'])
        for variant, entry in case['entries'].items():
            assert nr.digest(entry['library']) == entry['library_sha256']
            actual = np.fromfile(folder / name / (variant + '.f32'), np.float32).reshape(case['output_shape'])
            assert nr.array_digest(actual) == measured['output_sha256'][variant]
            nr.validate_output(actual, expected)
            audit['snapshots_re_read'] += 1
        assert measured['off_on_bitwise_equal']
        assert measured['output_sha256']['off'] == measured['output_sha256']['on']
        audit['off_on_bitwise_equal_cases'] += 1
        audit['timed_visits_checked'] += 18
    audit['runs'].append(dict(name=label, replay_sha256=nr.digest(folder / 'results.json'),
                             host_observation_sha256=nr.digest(RAW / observation),
                             heavy_samples=sum(bool(o['heavy']) for o in host['observations']),
                             total_observations=len(host['observations']), performance_accepted=False))
rejected = []
for mutation in ('sample', 'ratio', 'guard', 'output_hash', 'missing_visit'):
    value = copy.deepcopy(measured)
    if mutation == 'sample':
        value['results'][0]['samples_us'] = [0.0] * 7
    elif mutation == 'ratio':
        value['paired_ratios']['on/off']['median'] += 1
    elif mutation == 'guard':
        value['results'][0]['workspace_guards_passed'] = False
    elif mutation == 'output_hash':
        value['results'][0]['output_sha256'] = '0' * 64
    else:
        value['results'].pop()
    try:
        check_record(value)
    except AssertionError:
        rejected.append(mutation)
assert len(rejected) == 5
for mutation in ('changed_tail', 'nonfinite', 'truncated'):
    value = expected.astype(np.float32)
    if mutation == 'changed_tail':
        value.flat[-1] += 1.0
    elif mutation == 'nonfinite':
        value.flat[-1] = np.nan
    else:
        value = value.reshape(-1)[:-1]
    try:
        nr.validate_output(value, expected)
    except ValueError:
        rejected.append(mutation)
assert len(rejected) == 8
audit['rejected_in_memory_mutations'] = rejected
assert (audit['snapshots_re_read'], audit['timed_visits_checked'], audit['off_on_bitwise_equal_cases']) == (144, 864, 48)
(RAW / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps(audit, indent=2))
