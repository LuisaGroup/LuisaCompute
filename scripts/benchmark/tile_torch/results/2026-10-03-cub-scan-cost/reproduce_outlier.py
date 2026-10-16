"""Replay a separate fixed/automatic diagnostic; never pool it with the formal queue."""
from pathlib import Path
import hashlib
import json

import reproduce_automatic as replay

HERE = Path(__file__).resolve().parent
NAMES = ('fixed1', 'auto1', 'fixed2', 'auto2')


def validate(document, formal):
    replay.require(document['schema'] == 'automatic-scan-outlier-public-v1' and
                   document['status'] == 'completed_validated' and len(document['cases']) == 2,
                   'wrong or incomplete diagnostic')
    base = next(d for d in formal['datasets'] if d['cohort'] == 'heldout')
    replay.require(document['original_formal_summary']['sha256'] == base['original_summary_bytes']['sha256'],
                   'formal byte identity differs')
    for row in document['cases']:
        case, stages = row['case'], row['stages']
        replay.require(case['dimensions'] == [256, 16384] and case['precision'] in ('fp16', 'bf16') and
                       set(stages) == set(NAMES), 'diagnostic geometry/stages changed')
        original = next(r for r in base['cases'] if r['case']['id'] == case['id'])
        replay.equivalent(original['case'], case)
        initial = stages['fixed1']
        identity = None
        for name in NAMES:
            stage = stages[name]
            replay.require(stage['status'] == 'validated' and stage['original_case_status'] == 'passed' and
                           stage['fixture_sha256'] == initial['fixture_sha256'] and
                           stage['manifest_canonical_sha256'] == initial['manifest_canonical_sha256'],
                           'fixture or validation differs')
            replay.route_check(stage['routes']['native'], True)
            replay.require(stage['routes']['torch']['status'] == 'not_requested', 'diagnostic invented a Torch denominator')
            if name.startswith('auto'):
                replay.cost_check(case, stage)
            else:
                replay.require(not stage['cost_receipts'], 'fixed recipe gained a search')
            selector = stage['cub_selection'][0]
            replay.require(selector['threads_requested'] == 128 and selector['selected'] and selector['available'] and
                           selector['static_ranges_disjoint'] and selector['final_pointers_aligned16'] and
                           selector['expected_selected_entry'] == 'luisa_tile_cub_scan' and
                           selector['expected_selected_grid'] == [256, 1, 1] and
                           selector['expected_selected_block'] == [128, 1, 1], 'different guarded launch')
            proof = row['same_selected_entry_proof'][name]
            replay.require(proof['final_argument_mod16'][0] == proof['final_argument_mod16'][3] == 0,
                           'missing final-pointer alignment')
            source = stage['cub_source_files']['cub-source-stage0.cu']
            for key in ('sha256', 'bytes', 'compile_key'):
                replay.require(source[key] == proof['source'][key], 'selected source mismatch')
            replay.require(stage['resource_facts'] == proof['resources'] and
                           stage['original_source_sha256'] == proof['original_source_sha256'], 'resource/source proof differs')
            if identity is None:
                identity = proof
            replay.equivalent(identity, proof)
        replay.require(len({s['routes']['native']['result_receipt']['sha256'] for s in stages.values()}) == 4,
                       'reused diagnostic result')
        old = row['original_formal_automatic']
        replay.equivalent(old['event_us'], original['stages']['automatic']['routes']['native']['event_us'])
        replay.equivalent(old['result_receipt'], original['stages']['automatic']['routes']['native']['result_receipt'])
        replay.require(old['same_selected_entry_and_fixture'] and
                       initial['fixture_sha256'] == original['stages']['automatic']['fixture_sha256'],
                       'original observation identity changed')
        expected_events = {n: stages[n]['routes']['native']['event_us'] for n in NAMES}
        replay.equivalent(row['comparisons']['event_us'], expected_events)
        for a in ('auto1', 'auto2'):
            for f in ('fixed1', 'fixed2'):
                replay.close(row['comparisons']['automatic_over_fixed'][a][f], expected_events[a]['p50']/expected_events[f]['p50'])
                replay.equivalent(row['comparisons']['sample_extrema_ratio_intervals'][a][f],
                                  [min(expected_events[a]['samples'])/max(expected_events[f]['samples']),
                                   max(expected_events[a]['samples'])/min(expected_events[f]['samples'])])
        replay.close(row['comparisons']['fixed2_over_fixed1'], expected_events['fixed2']['p50']/expected_events['fixed1']['p50'])
        replay.close(row['comparisons']['auto2_over_auto1'], expected_events['auto2']['p50']/expected_events['auto1']['p50'])
    return dict(native_observations=8, timing_samples=56, pooled_with_formal=False, causal_conclusion=False)


def main():
    manifest = json.loads((HERE / 'manifest-outlier.json').read_text(encoding='utf-8'))
    for name, r in manifest['files'].items():
        replay.require(Path(name).name == name, 'manifest path escapes package')
        raw = (HERE / name).read_bytes()
        replay.require(hashlib.sha256(raw).hexdigest() == r['sha256'] and len(raw) == r['bytes'], 'changed bytes: '+name)
    formal = json.loads((HERE / 'automatic.json').read_text(encoding='utf-8'))
    replay.validate(formal)
    document = json.loads((HERE / 'automatic-outlier.json').read_text(encoding='utf-8'))
    print(json.dumps(dict(status='passed', **validate(document, formal))))


if __name__ == '__main__':
    main()
