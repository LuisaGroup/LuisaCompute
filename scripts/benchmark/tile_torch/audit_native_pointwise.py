#!/usr/bin/env python3
"""Audit native outputs, paired statistics and the preopt comparison contract."""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics

import numpy as np

import native_rows as nr

VARIANTS = ('off', 'on', 'inductor')
ORDERS = list(itertools.permutations(VARIANTS))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _semantics(comparison):
    require(type(comparison) is str and comparison in ('pointwise', 'outlined-tail'), 'unknown comparison')
    return {'off': {'pointwise_fusion': comparison == 'outlined-tail', 'outlined_packet_tail': False},
            'on': {'pointwise_fusion': True, 'outlined_packet_tail': comparison == 'outlined-tail'}}


def _flag(env, feature):
    enable, disable = (env.get('LUISA_SIMD_' + prefix + '_' + feature) for prefix in ('ENABLE', 'DISABLE'))
    require(all(value is None or (type(value) is str and value in ('0', '1')) for value in (enable, disable)),
            'noncanonical feature flag: ' + feature)
    return enable == '1' and disable != '1'


def _read_llvm(entry, manifest):
    path = Path(entry['capture']) / 'kernel.ll'
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    require(digest == entry['llvm_sha256'] == manifest['artifact_sha256'].get(str(path)),
            'LLVM payload/entry/inventory hash mismatch')
    return payload.decode('utf-8')


def _tail_attributes(source, enabled):
    # Independent bounded reader for the emitted preopt form. Quoted attribute
    # strings and comments are not LLVM attribute keywords. Unsupported forms
    # fail closed; this is not a general-purpose LLVM parser or machine-code audit.
    source = re.sub(r'"(?:\\.|[^"\\])*"|;[^\n]*', '', source)
    groups = {}
    for line in re.finditer(r'^attributes\s+#(\d+)\s*=\s*\{([^{}]*)\}\s*$', source, re.M):
        require(line[1] not in groups, 'duplicate attribute group')
        groups[line[1]] = line[2]

    def attrs(suffix):
        words = suffix
        for group in re.findall(r'#(\d+)', suffix):
            require(group in groups, 'missing attribute group')
            words += ' ' + groups[group]
        return set(re.findall(r'\b(?:noinline|alwaysinline)\b', words))

    calls = {'llm_rows': [], 'llm_rows.full_packet': []}
    definitions = {}
    for line in source.splitlines():
        call = re.fullmatch(r'\s*(?:(?:tail|musttail|notail)\s+)?call\s+void\s+'
                            r'@(llm_rows(?:\.full_packet)?)\(([^()]*)\)(.*)', line)
        if call:
            calls[call[1]].append((call[2], attrs(call[3])))
        definition = re.fullmatch(r'define\s+[^@]*@(llm_rows(?:\.full_packet)?)\(([^()]*)\)([^{}]*)\{\s*', line)
        if definition:
            require(definition[1] not in definitions, 'duplicate body definition')
            definitions[definition[1]] = attrs(definition[3])
    require(set(definitions) == set(calls), 'missing body definition')
    require(all('noinline' not in flags for flags in definitions.values()), 'function-level NoInline is forbidden')
    require(len(calls['llm_rows']) == 1, 'expected one ordinary-body tail call')
    operands, flags = calls['llm_rows'][0]
    require(operands.split(',')[-1].strip() == 'i32 %packet.tail.lane.count', 'ordinary call is not the narrow tail')
    require(('noinline' in flags) is enabled and 'alwaysinline' not in flags, 'incorrect tail callsite NoInline')
    full = calls['llm_rows.full_packet']
    require(full and all('noinline' not in flags for _, flags in full), 'missing full calls or full-call NoInline')
    return dict(stage='pre_optimization_llvm', tail_call_count=1, tail_call_noinline=enabled,
                full_packet_call_count=len(full), full_packet_noinline_calls=0,
                body_function_noinline=False, full_packet_function_noinline=False)


def comparison_checks(manifest, report, read_llvm=_read_llvm):
    explicit = 'comparison' in manifest or 'comparison' in report
    if explicit:
        require('comparison' in manifest and 'comparison' in report, 'comparison label missing on one side')
    comparison = manifest.get('comparison', 'pointwise')
    expected = _semantics(comparison)
    require(report.get('comparison', 'pointwise') == comparison, 'capture/replay comparison mismatch')
    for document in (manifest, report):
        if explicit:
            # JSON comparison also distinguishes true/false from integer 1/0.
            require(json.dumps(document.get('variant_semantics'), sort_keys=True) == json.dumps(expected, sort_keys=True),
                    'wrong or nonboolean comparison semantics')
        else:
            require('variant_semantics' not in document, 'legacy comparison has undeclared semantics')
    verified = []
    for case in manifest['cases']:
        require(set(case['entries']) == set(VARIANTS), 'missing comparison variant')
        environments, checks = {}, {}
        for variant in ('off', 'on'):
            entry = case['entries'][variant]
            env = entry['environment']
            require(all(env.get(k) == value for k, value in manifest['fixed_environment'].items()), 'fixed environment changed')
            for feature, key in (('POINTWISE_FUSION', 'pointwise_fusion'), ('OUTLINED_PACKET_TAIL', 'outlined_packet_tail')):
                require(_flag(env, feature) is expected[variant][key], 'variant feature mismatch: ' + feature)
            full = _flag(env, 'FULL_PACKET_SPECIALIZATION')
            require(comparison != 'outlined-tail' or full, 'outlined-tail requires full specialization')
            realization = entry['realization']
            require('W8, 32 workers/block, 1 CPU workers;' in realization, 'execution mapping changed')
            pairs = re.findall(r'\b([a-z_]+)=([^;]+)', realization)
            fields = dict(pairs)
            require(len(pairs) == len(fields), 'duplicate realization field')
            required = dict(local_lanes='8', blocks_per_task='0', max_unrolled_tile_elements='64',
                            unordered_reduction_partitions='4', load_reduction_fusion='false',
                            expression_reduction_fusion='false', map_fusion='false', fast_math='false',
                            custom_cost_policy='false', pointwise_fusion=str(expected[variant]['pointwise_fusion']).lower())
            require(all(fields.get(k) == value for k, value in required.items()), 'fixed realization changed')
            count = fields.get('full_packet_specializations', '')
            require(count.isdigit() and ((int(count) > 0) is full), 'full specialization metadata disagrees with environment')
            require(fields == entry['realization_fields'], 'realization fields disagree with text')
            ignored = {'LUISA_TILE_BENCH_DUMP_SOURCE', 'LUISA_SIMD_DUMP_ASSEMBLY_DIR'}
            if comparison == 'outlined-tail':
                require(env.get('LUISA_SIMD_ENABLE_OUTLINED_PACKET_TAIL') == '1' and
                        env.get('LUISA_SIMD_DISABLE_OUTLINED_PACKET_TAIL') == ('1' if variant == 'off' else None),
                        'outlined-tail must change only the disable-precedence control')
                ignored.add('LUISA_SIMD_DISABLE_OUTLINED_PACKET_TAIL')
                checks[variant] = _tail_attributes(read_llvm(entry, manifest), variant == 'on')
                require(json.dumps(checks[variant], sort_keys=True) == json.dumps(entry['codegen_checks'], sort_keys=True),
                        'recorded codegen checks disagree with actual LLVM')
            else:
                ignored.update(('LUISA_SIMD_ENABLE_POINTWISE_FUSION', 'LUISA_SIMD_DISABLE_POINTWISE_FUSION'))
            environments[variant] = {k: v for k, v in env.items() if k not in ignored}
        require(environments['off'] == environments['on'], 'uncontrolled off/on environment difference')
        verified.append(dict(operation=case['operation'], dimensions=case['dimensions'], codegen_checks=checks))
    return dict(comparison=comparison, variant_semantics=expected, comparison_checks=verified)


def comparison_mutation_checks(manifest, report, read_llvm=_read_llvm):
    comparison = manifest.get('comparison', 'pointwise')
    m, r = copy.deepcopy(manifest), copy.deepcopy(report)
    # Upgrade only in-memory fixtures; never rewrite a legacy evidence record.
    for document in (m, r):
        document.update(comparison=comparison, variant_semantics=_semantics(comparison))
    mutations = {
        'unknown_comparison': lambda a, b: b.__setitem__('comparison', 'unknown'),
        'mismatched_comparison': lambda a, b: b.__setitem__('comparison', 'outlined-tail' if comparison == 'pointwise' else 'pointwise'),
        'missing_comparison': lambda a, b: b.pop('comparison'),
        'nonboolean_semantics': lambda a, b: b['variant_semantics']['on'].__setitem__('pointwise_fusion', 1),
        'wrong_variant_feature': lambda a, b: a['cases'][0]['entries']['on']['environment'].__setitem__('LUISA_SIMD_DISABLE_POINTWISE_FUSION', '1'),
    }
    if comparison == 'outlined-tail':
        mutations.update({
            'disabled_full_specialization': lambda a, b: a['cases'][0]['entries']['on']['environment'].__setitem__('LUISA_SIMD_DISABLE_FULL_PACKET_SPECIALIZATION', '1'),
            'wrong_llvm_hash': lambda a, b: a['cases'][0]['entries']['off'].__setitem__('llvm_sha256', '0' * 64),
            'wrong_tail_noinline_claim': lambda a, b: a['cases'][0]['entries']['off']['codegen_checks'].__setitem__('tail_call_noinline', True),
        })
    for name, change in mutations.items():
        a, b = copy.deepcopy(m), copy.deepcopy(r)
        change(a, b)
        try:
            comparison_checks(a, b, read_llvm)
        except ValueError:
            continue
        raise ValueError('accepted comparison mutation: ' + name)
    return list(mutations)


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
    comparison = comparison_checks(manifest, report)
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
                 rejected_comparison_mutations=comparison_mutation_checks(manifest, report),
                 guard_caveat='Guards were checked during replay; not retained for post-hoc reinspection.',
                 status='passed', **comparison)
    nr.save(args.output, audit)
    for item in summaries:
        print(nr.case_name(item['operation'], item['dimensions']), item['summary_us'],
              {k: round(v['median'], 4) for k, v in item['paired_ratios'].items()})


if __name__ == '__main__':
    main()
