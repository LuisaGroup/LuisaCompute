"""Recompute every published median/ratio; stdlib only, no GPU or original packets."""
import hashlib
import json
import math
from pathlib import Path
import statistics

def main():
    path = Path(__file__).with_name('aligned-view-sm89.json')
    data = json.loads(path.read_text(encoding='utf-8'))
    inventory = json.loads(path.with_name('aligned-view-sm89-cases.json').read_text(encoding='utf-8'))
    assert data['status'] == 'completed_validated' and len(data['cases']) == 16
    assert [row['case'] for row in data['cases']] == inventory['cases']
    count = 0
    output = []
    for row in data['cases']:
        medians = {}
        for label, route in row['routes'].items():
            samples = route['event_us']['samples']
            assert len(samples) == 7 and all(math.isfinite(x) and x > 0 for x in samples)
            assert statistics.median(samples) == route['event_us']['p50']
            assert route['saved_output_recheck']['failed_elements'] == 0
            medians[label] = statistics.median(samples)
            count += len(samples)
        ratios = {'aligned_over_default': medians['aligned16']/medians['default'], 'aligned_over_recheck': medians['aligned16']/medians['recheck'], 'default_over_torch': medians['default']/medians['torch'], 'aligned_over_torch': medians['aligned16']/medians['torch'], 'recheck_over_default': medians['recheck']/medians['default']}
        assert ratios == row['ratios']
        src = row['source']
        assert src['default']['source.txt']['sha256'] == src['recheck']['source.txt']['sha256'] == src['aligned16']['source.txt']['original_source_sha256']
        output.append({'id': row['case']['id'], 'medians_us': medians, 'ratios': ratios, 'fallback': row['ineligible_fallback']})
    assert count == 448 and sum(row['ineligible_fallback'] for row in data['cases']) == 3
    print(json.dumps({'status': 'reproduced', 'public_json_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'scope': 'Arithmetic and evidence identity only; no tensor revalidation, source-file reread, or rerun of measurements.', 'raw_samples': count, 'cases': output}, indent=2, allow_nan=False))

if __name__ == '__main__':
    main()
