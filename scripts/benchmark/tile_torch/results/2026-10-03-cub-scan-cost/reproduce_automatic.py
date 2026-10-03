"""Standard-library replay of compact automatic-policy evidence; never fits or runs GPU work."""
from pathlib import Path
import hashlib
import json
import math
import statistics

HERE = Path(__file__).resolve().parent
PROFILE_ID = '67127e819f80a395aeecae55cd999c8e3f8b1bcf894fdf0672c58611cca87f66'
PROFILE_SHA = 'd38f317cb43b1e646fcdb959e30ea89675b79722508d973ee037609db24d9186'
TILE = (0.7284119403590213, 0.0, 0.00026641302734655293)
CUB = (0.6027021529393383, 12.227593513162688, 0.027854484240835357, 0.022923736019806327)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, tolerance=2e-12):
    require(type(a) in (float, int) and type(b) in (float, int) and math.isfinite(a) and math.isfinite(b) and
            math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance), f'numeric mismatch: {a}, {b}')


def equivalent(a, b):
    if type(a) is float or type(b) is float:
        close(a, b)
    elif isinstance(a, dict):
        require(isinstance(b, dict) and a.keys() == b.keys(), 'mapping mismatch')
        for key in a:
            equivalent(a[key], b[key])
    elif isinstance(a, list):
        require(isinstance(b, list) and len(a) == len(b), 'sequence mismatch')
        for x, y in zip(a, b):
            equivalent(x, y)
    else:
        require(a == b, f'value mismatch: {a}, {b}')


def series(values):
    require(isinstance(values, list) and len(values) == 7 and all(type(x) in (float, int) and math.isfinite(x) and x > 0 for x in values),
            'all seven positive timing samples required')
    return statistics.median(values)


def route_check(route, native):
    values = route['event_us']['samples']
    close(route['event_us']['p50'], series(values))
    sampling, warmup = route['sampling'], route['warmup']
    require(sampling['protocol'] == 'adaptive_replay_span_v2', 'wrong graph protocol')
    replays, operations = sampling['replays_per_sample'], sampling['operations_per_sample']
    require(type(replays) is int and 1 <= replays <= 65536 and operations == replays * 100 and
            sampling['replay_cap'] == 65536 and sampling['operation_cap'] == 10000000 and sampling['sample_target_ms'] == 100,
            'graph replay count/cap/normalization mismatch')
    spans = sampling['event_span_ms']['samples']
    close(sampling['event_span_ms']['p50'], series(spans))
    close(sampling['host_span_ms']['p50'], series(sampling['host_span_ms']['samples']))
    for span, value in zip(spans, values):
        close(span * 1000 / operations, value, 2e-7)
    calibration = sampling['calibration']
    require(isinstance(calibration, list) and 1 <= len(calibration) <= 4, 'missing graph calibration')
    wanted = 1
    for index, attempt in enumerate(calibration):
        count, span, wall = (attempt[k] for k in ('replays', 'event_span_ms', 'host_wall_ms'))
        require(count == wanted and span > 0 and wall > 0 and math.isfinite(span) and math.isfinite(wall), 'invalid calibration attempt')
        stop = span >= 80 or count == 65536 or index == 3
        if index + 1 == len(calibration):
            require(stop and count == replays and sampling['calibration_target_reached'] is (span >= 80), 'unmeasured formal replay count')
        else:
            require(not stop, 'calibration continued past stopping rule')
            wanted = min(65536, max(count + 1, math.ceil(count * 100 / span)))
    require(warmup['target_ms'] == 500 and warmup['actual_ms'] >= 500 and warmup['replays'] >= 1 and
            sampling['prime_event_ms'] > 0 and sampling['prime_host_ms'] > 0, 'warmup/priming missing')
    if native:
        require(sampling['kernel_stage_dispatches_per_sample'] == operations, 'native stage count mismatch')
        correctness = route['recorded_correctness']
        require(correctness['errors'] == 0 and correctness['inputs_unchanged'] and correctness['guards_unchanged'] and
                correctness['all_outputs_finite'], 'recorded native correctness failed')
    else:
        require(all(c['failed_elements'] == 0 for c in route['recorded_correctness'].values()), 'recorded Torch correctness failed')
    require(route['saved_output_recheck']['failed_elements'] == 0, 'upstream full saved-output oracle failed')


def cost_check(case, stage):
    cost = stage['cost_receipts'][0]
    require(cost['fit'] == PROFILE_ID and cost['profile_file_sha256'] == PROFILE_SHA, 'frozen cost profile mismatch')
    selected = cost['selected_threads']
    scored = cost['original_score'] is not None
    expected_original = None
    rows, width = case['dimensions']
    if scored:
        require(case['operation'] == 'scan' and case['precision'] in ('fp16', 'bf16') and not case['fast_math'] and
                case['tile'] == [1, width, 1], 'scored fixture contract mismatch')
        require(cost['device_query_ok'] and cost['device'] == dict(sm=89, processors=24, warp=32,
                **{'resident-threads': 1536, 'driver-api': 13040}, toolkit=13040, nvrtc=130400), 'cost device changed')
        features = [1, rows * width * 4 / (24 * 2**20), math.ceil(rows / 24) * width]
        equivalent(cost['original_features'], features)
        expected_original = sum(a * b for a, b in zip(TILE, features))
        close(cost['original_score'], expected_original)
    else:
        require(cost['status'] == 'ineligible' and selected == 0 and cost['compiler_call_count'] == cost['search_count'] == 0,
                'ineligible profile performed a search')
    require([c['threads'] for c in cost['candidates']] == [128, 256, 512, 1024], 'candidate inventory/order changed')
    best_threads, best_score, calls = 0, expected_original, 0
    for candidate in cost['candidates']:
        threads, resources, score = candidate['threads'], candidate['resources'], candidate['score']
        attempted = candidate['compile_status'] != 'not-attempted'
        calls += attempted
        require(candidate['compile_key_known'] is attempted, 'compilation identity mismatch')
        name = f'cub-cost-source-stage0-t{threads}.cu'
        source = stage['cost_source_files'].get(name)
        require(bool(source) == attempted, 'candidate source availability mismatch')
        if source:
            require(source['compile_key'] == candidate['compile_key'] and source['source_key'] == candidate['source_key'] and
                    source['installed'] is candidate['installed'], 'candidate source/key binding mismatch')
        if score is not None:
            require(scored and candidate['query_scope'] == 'loaded-candidate' and candidate['compile_status'] == candidate['load_status'] == candidate['entry_status'] == 'ok' and
                    resources['status'] == resources['capacity_status'] == 'ok' and resources['local-bytes'] == 0 and
                    resources['max-threads'] >= threads and resources['capacity'] > 0 and resources['capacity_threads'] == threads and
                    resources['dynamic_shared_bytes'] == 0 and width % (8 * threads) == 0, 'scored unknown/ineligible resources')
            h, j = math.ceil(rows / (24 * resources['capacity'])), width // (8 * threads)
            features = [1, cost['original_features'][1], h * j * 8, h * j * math.ceil(threads / 32)]
            equivalent(candidate['features'], features)
            expected = sum(a * b for a, b in zip(CUB, features))
            close(score, expected)
            if expected < best_score:
                best_threads, best_score = threads, expected
        require(candidate['installed'] is (threads == selected and selected != 0), 'installed winner mismatch')
        if candidate['installed']:
            require(candidate['cleanup'] == 'shader-owned', 'winner ownership mismatch')
            final = stage['cub_source_files']['cub-source-stage0.cu']
            require(all(final[key] == source[key] for key in ('sha256', 'bytes', 'compile_key')), 'winner source identity mismatch')
    require(calls == cost['compiler_call_count'] and 0 <= calls <= cost['search_count'] <= 4, 'compiler method count mismatch')
    predicted = best_threads if scored and best_threads and best_score / expected_original < .95 else 0
    require(cost['predicted_threads'] == predicted, 'frozen prediction mismatch')
    if selected:
        require(selected == predicted and cost['status'] == 'selected' and cost['reason'] == 'predicted-saving-installed', 'score/installed selection mismatch')
        close(cost['selected_score'], best_score)
    elif scored:
        require(cost['status'] == 'retained' and cost['reason'] in ('predicted-original', 'candidate-install-failed', 'candidate-cleanup-failed'), 'retention reason mismatch')
        require(cost['reason'] != 'predicted-original' or predicted == 0, 'ignored predicted saving')
        close(cost['selected_score'], expected_original)
    selection = stage['cub_selection'][0]
    require(selection['threads_requested'] == selected and selection['available'] is bool(selected), 'final legacy winner mismatch')
    guarded = selection['available'] and selection['static_ranges_disjoint'] and selection['final_pointers_aligned16']
    require(selection['selected'] is guarded and selection['expected_selected_entry'] == ('luisa_tile_cub_scan' if guarded else 'luisa_tile_main') and
            selection['expected_selected_block'] == [selected if guarded else 1, 1, 1], 'final pointer-guard prediction mismatch')


def compare_case(row):
    stages = row['stages']
    require(set(stages) == {'default', 'automatic', 'recheck'}, 'incomplete case stages')
    native = {}
    for name, stage in stages.items():
        require(stage['status'] == 'validated' and stage['original_case_status'] == 'passed', 'case validation did not pass')
        require(stage['fixture_sha256'] == stages['default']['fixture_sha256'] and stage['original_source_sha256'] == stages['default']['original_source_sha256'],
                'original fixture/source changed across stages')
        route_check(stage['routes']['native'], True)
        native[name] = stage['routes']['native']['event_us']
        if name == 'default':
            route_check(stage['routes']['torch'], False)
        else:
            require(stage['routes']['torch']['status'] == 'not_requested', 'secondary stage claims a Torch denominator')
    cost_check(row['case'], stages['automatic'])
    a, b, c = (native[name]['p50'] for name in ('default', 'automatic', 'recheck'))
    t = stages['default']['routes']['torch']['event_us']['p50']
    expected = dict(native_ratio_over_initial=b/a, native_ratio_over_recheck=b/c, automatic_over_fresh_torch=b/t,
                    default_over_fresh_torch=a/t, recheck_over_fresh_torch=c/t, control_drift=c/a-1)
    for key, value in expected.items():
        close(row['comparisons'][key], value)
    intervals = {name: [min(native['automatic']['samples'])/max(native[name]['samples']), max(native['automatic']['samples'])/min(native[name]['samples'])]
                 for name in ('default', 'recheck')}
    equivalent(row['comparisons']['sample_extrema_ratio_intervals'], intervals)
    selected = stages['automatic']['cub_selection'][0]['selected']
    require(row['comparisons']['runtime_selected'] is selected and row['comparisons']['regression_either_control'] is (max(b/a, b/c) > 1) and
            row['comparisons']['selected_regression_over_three_percent'] is (selected and max(b/a, b/c) > 1.03) and
            row['comparisons']['threshold_overlap'] is (selected and any(lo <= 1.03 <= hi for lo, hi in intervals.values())), 'regression/overlap flags differ')
    extra = row['comparisons']['extra_search']
    cost = stages['automatic']['cost_receipts'][0]
    close(extra['search_ms'], cost['search_ms'])
    require(extra['compiler_call_count'] == cost['compiler_call_count'], 'search count changed')
    for candidate in cost['candidates']:
        close(extra['candidate_compile_ms'][str(candidate['threads'])], candidate['compile_ms'])
    for name, stage in stages.items():
        close(extra['measured_total_compile_ms'][name], stage['routes']['native']['cold_ms']['compile_ms'])


def check_metrics(dataset):
    rows = dataset['cases']
    group_keys = sorted({tuple(row['case']['dimensions']) for row in rows})
    def gm(key):
        return math.exp(statistics.mean(statistics.mean(math.log(row['comparisons'][key]) for row in rows if tuple(row['case']['dimensions']) == group) for group in group_keys))
    m = dataset['metrics']
    require(m['cases'] == len(rows) and m['geometries'] == len(group_keys) and m['runtime_selected'] == sum(row['comparisons']['runtime_selected'] for row in rows), 'aggregate coverage mismatch')
    for target, key in (('geomean_over_initial', 'native_ratio_over_initial'), ('geomean_over_recheck', 'native_ratio_over_recheck'), ('automatic_over_fresh_torch', 'automatic_over_fresh_torch')):
        close(m[target], gm(key))
    predicates = dict(all_regressions=lambda r: r['comparisons']['regression_either_control'],
        selected_regressions_over_three_percent=lambda r: r['comparisons']['selected_regression_over_three_percent'],
        drift_over_three_percent=lambda r: abs(r['comparisons']['control_drift']) > .03,
        unresolved_threshold_overlap=lambda r: r['comparisons']['threshold_overlap'],
        parity_failures=lambda r: r.get('historical_parity', {}).get('status') == 'failed')
    for key, predicate in predicates.items():
        require(m[key] == [row['case']['id'] for row in rows if predicate(row)], 'negative evidence list changed: ' + key)
    gates = m['engineering_acceptance']
    if dataset['cohort'] == 'eligibility':
        require(gates['status'] == 'not_applicable_eligibility_no_improvement_gate' and not gates['failures'] and not gates['unresolved'], 'eligibility acquired an improvement filter')
    else:
        failures, unresolved = [], []
        if max(m['geomean_over_initial'], m['geomean_over_recheck']) > .95: failures.append('geometry_equal_gain_below_five_percent')
        if m['selected_regressions_over_three_percent']: failures.append('selected_regression_exceeds_three_percent')
        if m['parity_failures']: failures.append('frozen_independent_policy_or_candidate_parity_failed')
        if m['drift_over_three_percent']: unresolved.append('control_drift_exceeds_three_percent')
        if m['unresolved_threshold_overlap']: unresolved.append('sample_extrema_overlap_three_percent_threshold')
        require(gates['failures'] == failures and gates['unresolved'] == unresolved and
                gates['status'] == ('inconclusive' if unresolved else 'failed' if failures else 'passed_measured_engineering_gates'), 'engineering gates changed')


def validate(document):
    require(document['schema'] == 'automatic-scan-cost-public-v1' and document['source_profile_id'] == PROFILE_ID and
            document['source_profile_sha256'] == PROFILE_SHA and tuple(document['tile_coefficients']) == TILE and tuple(document['cub_coefficients']) == CUB,
            'wrong frozen profile/schema')
    counts = dict(native_observations=0, torch_observations=0, timing_samples=0, candidates=0)
    seen = set()
    for dataset in document['datasets']:
        require(dataset['cohort'] in ('heldout', 'eligibility') and dataset['cohort'] not in seen and dataset['status'] == 'completed_validated', 'duplicate/unvalidated dataset')
        seen.add(dataset['cohort'])
        require(len(dataset['cases']) == (8 if dataset['cohort'] == 'heldout' else 6), 'incomplete fixed inventory')
        require(len({r['case']['id'] for r in dataset['cases']}) == len(dataset['cases']), 'duplicate case')
        for row in dataset['cases']:
            compare_case(row)
            counts['native_observations'] += 3
            counts['torch_observations'] += 1
            counts['timing_samples'] += 28
            counts['candidates'] += 4
        check_metrics(dataset)
    return counts


def main():
    manifest = json.loads((HERE / 'manifest-automatic.json').read_text(encoding='utf-8'))
    for name, wanted in manifest['files'].items():
        require(Path(name).name == name, 'manifest path escapes package')
        raw = (HERE / name).read_bytes()
        require(len(raw) == wanted['bytes'] and hashlib.sha256(raw).hexdigest() == wanted['sha256'], 'public bytes changed: ' + name)
    document = json.loads((HERE / 'automatic.json').read_text(encoding='utf-8'))
    print(json.dumps(dict(status='passed', **validate(document))))


if __name__ == '__main__':
    main()
