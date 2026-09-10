#!/usr/bin/env python3
"""Re-read native pointwise outputs and independently recompute paired ratios."""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
from pathlib import Path
import statistics

import numpy as np

import native_rows as nr

VARIANTS = ('off', 'on', 'inductor')
ORDERS = list(itertools.permutations(VARIANTS))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def statistics_only(report, expected_cases):
    require(report.get('status') == 'passed', 'unfinished or failed replay')
    require(report['metric'] == 'single_thread_native_entry_host_wall_us', 'not native timing')
    require(not report.get('timing_not_comparative', False), 'verification-only timings')
    require((report['cpu_threads'], report['samples'], report['warmup_ms'], report['target_ms']) ==
            (1, 7, 100, 30), 'timing protocol changed')
    keys = [(c['operation'], tuple(c['dimensions'])) for c in report['cases']]
    require(keys == expected_cases and len(set(keys)) == len(keys), 'cohort mismatch')
    summaries = []
    for case in report['cases']:
        visits = case['results']
        require(len(visits) == 18, 'missing native visits')
        keys = [(v['round'], v['variant']) for v in visits]
        require(len(set(keys)) == 18 and set(keys) == set(itertools.product(range(6), VARIANTS)),
                'duplicate/missing visits')
        require(set(case['output_sha256']) == set(VARIANTS), 'missing output hashes')
        medians = {}
        for visit in visits:
            require(tuple(visit['order']) == ORDERS[visit['round']], 'unbalanced visit order')
            samples = visit['samples_us']
            require(len(samples) == 7 and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in samples), 'invalid samples')
            median = statistics.median(samples)
            require(median == visit['median_us'], 'wrong median')
            require(type(visit['repetitions']) is int and 1 <= visit['repetitions'] <= 1048576, 'bad repetitions')
            check = visit['correctness']
            require(check['elements'] == math.prod(case['dimensions']) and
                    check['atol'] == check['rtol'] == 5e-5, 'incomplete oracle')
            require(visit['inputs_unchanged'] is True and visit['workspace_guards_passed'] is True, 'input/workspace corruption')
            require(visit['guard_elements'] >= 512 and visit['guard_elements'] % 128 == 0, 'missing guards')
            require(visit['output_sha256'] == case['output_sha256'][visit['variant']], 'unstable output')
            medians[(visit['round'], visit['variant'])] = median
        ratios = {}
        for a, b in (('on', 'off'), ('on', 'inductor'), ('off', 'inductor')):
            values = [medians[(r, a)] / medians[(r, b)] for r in range(6)]
            ratio = dict(median=statistics.median(values), minimum=min(values), maximum=max(values),
                         wins=sum(x < 1 for x in values), rounds=values)
            require(case['paired_ratios'][a + '/' + b] == ratio, 'incorrect paired ratio')
            ratios[a + '/' + b] = ratio
        summary = {v: statistics.median(medians[(r, v)] for r in range(6)) for v in VARIANTS}
        require(case['summary_us'] == summary, 'incorrect case summary')
        summaries.append(dict(operation=case['operation'], dimensions=case['dimensions'],
                              summary_us=summary,
                              paired_ratios=ratios))
    return summaries


def mutation_checks(report, expected):
    changes = {
        'missing_visit': lambda x: x['cases'][0]['results'].pop(),
        'wrong_median': lambda x: x['cases'][0]['results'][0].__setitem__('median_us', -1),
        'wrong_ratio': lambda x: x['cases'][0]['paired_ratios']['on/off'].__setitem__('median', -1),
        'failed_guard': lambda x: x['cases'][0]['results'][0].__setitem__('workspace_guards_passed', False),
        'wrong_oracle': lambda x: x['cases'][0]['results'][0]['correctness'].__setitem__('elements', 1),
        'verification_only': lambda x: x.__setitem__('timing_not_comparative', True),
        'wrong_order': lambda x: x['cases'][0]['results'][0].__setitem__('order', list(reversed(VARIANTS))),
        'failed_status': lambda x: x.__setitem__('status', 'error'),
        'wrong_summary': lambda x: x['cases'][0]['summary_us'].__setitem__('on', -1),
        'duplicate_visit': lambda x: x['cases'][0]['results'].__setitem__(1, copy.deepcopy(x['cases'][0]['results'][0])),
        'boolean_sample': lambda x: x['cases'][0]['results'][0]['samples_us'].__setitem__(0, True),
    }
    for name, change in changes.items():
        altered = copy.deepcopy(report)
        change(altered)
        try:
            statistics_only(altered, expected)
        except ValueError:
            continue
        raise ValueError('accepted mutation: ' + name)
    return list(changes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'audit destination exists')
    manifest = json.loads((args.prepared / 'manifest.json').read_text())
    report = json.loads((args.replay / 'results.json').read_text())
    require(manifest.get('format') == 'native-pointwise-v1' and manifest.get('status') == 'captured', 'unfinished capture')
    require(report['manifest_sha256'] == nr.digest(args.prepared / 'manifest.json'), 'mixed experiments')
    require(report.get('runner_unchanged') is True and report.get('finished_unix'), 'unfinished replay')
    require(manifest.get('source_unchanged') is True and manifest.get('closure_unchanged') is True, 'unfinished capture')
    require(nr.digest(manifest['helper']) == manifest['helper_sha256'], 'changed helper')
    identity_checks = 0
    for document in (manifest, report):
        for group in ('artifact_sha256', 'runner_sha256'):
            require(document.get(group), 'missing frozen identity group: ' + group)
            for path, expected_sha in document[group].items():
                require(nr.digest(path) == expected_sha, 'changed artifact: ' + path)
                identity_checks += 1
    expected = [(c['operation'], tuple(c['dimensions'])) for c in manifest['cases']]
    summaries = statistics_only(report, expected)
    checks = []
    for case, result in zip(manifest['cases'], report['cases']):
        arrays = nr.load_inputs(Path(case['inputs']), case['input_shapes'])
        require([nr.array_digest(a) for a in arrays] == case['input_sha256'], 'changed inputs')
        oracle = nr.reference(case['operation'], case['dimensions'], arrays)
        for name in VARIANTS:
            entry = case['entries'][name]
            require(nr.digest(entry['library']) == entry['library_sha256'], 'changed native library')
            path = args.replay / nr.case_name(case['operation'], case['dimensions']) / (name + '.f32')
            require(nr.digest(path) == result['output_sha256'][name], 'changed native output')
            check = nr.validate_output(np.fromfile(path, np.float32).reshape(case['output_shape']), oracle)
            guards = 128 * (3 + len(entry['plan']['allocations'])) if entry['abi'] == 1 else 512
            require(all(v['guard_elements'] == guards for v in result['results'] if v['variant'] == name), 'wrong guard extent')
            checks.append(dict(case=nr.case_name(case['operation'], case['dimensions']), variant=name, correctness=check))
        require(result['off_on_bitwise_equal'] == (result['output_sha256']['off'] == result['output_sha256']['on']),
                'wrong bitwise equality claim')
    audit = dict(manifest_sha256=nr.digest(args.prepared / 'manifest.json'),
                 replay_sha256=nr.digest(args.replay / 'results.json'), auditor_sha256=nr.digest(__file__),
                 native_output_checks=checks, summaries=summaries, identity_checks=identity_checks,
                 rejected_mutations=mutation_checks(report, expected),
                 guard_caveat='Guards were checked during replay; not retained for post-hoc reinspection.',
                 status='passed')
    nr.save(args.output, audit)
    for item in summaries:
        print(nr.case_name(item['operation'], item['dimensions']), item['summary_us'],
              {k: round(v['median'], 4) for k, v in item['paired_ratios'].items()})


if __name__ == '__main__':
    main()
