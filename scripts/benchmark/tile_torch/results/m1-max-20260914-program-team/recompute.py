"""Recompute timing summaries directly from archived raw samples, without native code.

python recompute.py [archive-directory] prints JSON. It verifies archive/member
hashes, visit order, positive samples and original aggregates. It does not claim
to rerun the original in-memory guard checks or rebuild a historical toolchain.
"""
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics as st
import sys
import tarfile


def need(value, message):
    if not value:
        raise ValueError(message)


def med(row, count):
    values = row['samples_us']
    need(len(values) == count and all(math.isfinite(v) and v > 0 for v in values), 'invalid sample')
    value = st.median(values)
    need(value == row['median_us'], 'visit median mismatch')
    return value


def ratio(values):
    return dict(median=st.median(values), minimum=min(values), maximum=max(values),
                wins=sum(v < 1 for v in values), pairs=values)


def recompute(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    for name, entry in manifest['archives'].items():
        need(hashlib.sha256((directory / name).read_bytes()).hexdigest() == entry['sha256'], 'archive hash mismatch')
    with tarfile.open(directory / 'raw-evidence.tar.gz') as archive:
        members = manifest['archives']['raw-evidence.tar.gz']['members']
        need(set(archive.getnames()) == set(members), 'raw archive inventory mismatch')
        for entry in archive.getmembers():
            need(entry.isfile() and hashlib.sha256(archive.extractfile(entry).read()).hexdigest() == members[entry.name]['sha256'], 'raw member hash mismatch')

        def read(relative):
            return json.load(archive.extractfile('raw/' + relative))

        cases = read('cpu-capture-6-manifest.json')['cases']
        output = dict(status='passed', unit='microseconds', time_boundary='single_thread_native_entry_host_wall',
                      cpu_local1_vs_local8=[], cohort_private_off_vs_on=[], inductor_local1_local8=[],
                      no_cross_timer_ratio=True)
        for prefix, key, expected_variants in (
                ('cpu-replayed-', 'cpu_local1_vs_local8', ('local1', 'local8')),
                ('cohort-private-replayed-', 'cohort_private_off_vs_on', ('local8', 'local8-cohort-private'))):
            for name, op, dims, bq, bk in cases:
                report = read(prefix + name + '-6/results.json')
                need(report['status'] == 'passed' and report['artifacts_unchanged'] and len(report['visits']) == 12, 'incomplete native_tile cohort')
                need(report['options']['cycles'] == 3 and report['options']['samples'] == 5, 'changed native_tile protocol')
                values, groups = [], {variant: [] for variant in expected_variants}
                order = (expected_variants[0], expected_variants[1], expected_variants[1], expected_variants[0])
                for i, row in enumerate(report['visits']):
                    need(row['cycle'] == i // 4 and row['position'] == i % 4 and row['variant'] == order[i % 4] and
                         row['valid'] and row['returncode'] == 0 and row['all_guards_passed'] and row['inputs_unchanged'], 'invalid ABBA visit')
                    values.append(med(row, 5))
                    groups[row['variant']].append(values[-1])
                summary = {variant: st.median(rows) for variant, rows in groups.items()}
                pairs = [values[4 * c + b] / values[4 * c + a] for c in range(3) for a, b in ((0, 1), (3, 2))]
                need(summary == report['summary_us'] and pairs == report['candidate_over_baseline']['pairs'], 'native_tile summary mismatch')
                output[key].append(dict(case=name, operation=op, dimensions=list(map(int, dims.split(','))),
                                        attention_block=[int(bq), int(bk)], baseline=expected_variants[0], candidate=expected_variants[1],
                                        median_us=summary, candidate_over_baseline=ratio(pairs)))
        report = read('cpu-inductor.zGCsEf/replay/results.json')
        need(report['runner_unchanged'] and len(report['cases']) == 4 and report['samples'] == 7 and
             report['warmup_ms'] == 100 and report['target_ms'] == 30, 'incomplete/changed native_rows cohort')
        orders = list(itertools.permutations(('whole', 'local', 'inductor')))
        for case in report['cases']:
            need(len(case['results']) == 18, 'missing row visit')
            times = {}
            for i, row in enumerate(case['results']):
                need(row['round'] == i // 3 and row['order'] == list(orders[i // 3]) and row['variant'] == orders[i // 3][i % 3]
                     and row['workspace_guards_passed'] and row['inputs_unchanged'], 'invalid native_rows visit')
                times[i // 3, row['variant']] = med(row, 7)
            summary = {variant: st.median(times[r, variant] for r in range(6)) for variant in orders[0]}
            need(summary == case['summary_us'], 'native_rows summary mismatch')
            ratios = {}
            for numerator, denominator in (('local', 'inductor'), ('whole', 'inductor'), ('local', 'whole')):
                values = [times[r, numerator] / times[r, denominator] for r in range(6)]
                need(values == case['paired_ratios'][numerator + '/' + denominator]['rounds'], 'native_rows ratios mismatch')
                ratios[numerator + '/' + denominator] = ratio(values)
            output['inductor_local1_local8'].append(dict(operation=case['operation'], dimensions=case['dimensions'],
                                                        median_us=summary, paired_ratios=ratios))
        output['protocols'] = dict(native_tile=dict(cohorts=2, cases_per_cohort=6, visits_per_case=12, samples_per_visit=5,
                                                    warmup_ms=40, target_ms=20, order='ABBA x3'),
                                    native_rows=dict(cases=4, visits_per_case=18, samples_per_visit=7,
                                                     warmup_ms=100, target_ms=30, order='six permutations of three variants'))
        output['total_visits'], output['total_samples'] = 216, 1224
        return output


if __name__ == '__main__':
    print(json.dumps(recompute(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent), indent=2))
