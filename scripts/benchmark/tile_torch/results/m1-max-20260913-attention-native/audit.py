"""Read-only audit: python audit.py EXTRACTED_EVIDENCE_DIRECTORY."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_output(path, expected):
    actual = np.fromfile(path, dtype=np.float32)
    assert actual.size == expected.size and np.isfinite(actual).all(), path
    assert np.all(np.abs(actual.astype(np.float64) - expected.ravel()) <= 5e-5 + 5e-5 * np.abs(expected.ravel())), path


def audit(root):
    captures = json.loads((root / 'captures.json').read_text())
    assert len(captures) == 8 and all(c['status'] == 'captured' for c in captures)
    for case in captures:
        name = '-'.join((case['case'], case['qk'], case['pv']))
        folder = root / name
        m = case['metadata']
        assert (m['attention_qk'], m['attention_pv']) == (case['qk'], case['pv'])
        x, k, v = [np.fromfile(folder / f'output.f32.input{i}.f32', dtype=np.float32).reshape(s).astype(np.float64)
                   for i, s in enumerate(m['input_shapes'])]
        _, h, kh, q, keys, d, dv = case['dimensions']
        k, v = np.repeat(k, h // kh, axis=1), np.repeat(v, h // kh, axis=1)
        score = (x @ k.swapaxes(-1, -2)) * float(np.float32(1) / np.sqrt(np.float32(d)))
        valid = np.arange(keys)[None, :] <= np.arange(q)[:, None] + keys - q
        score = np.where(valid, score, -1e30)
        p = np.where(valid, np.exp(score - score.max(axis=-1, keepdims=True)), 0)
        oracle = (p / p.sum(axis=-1, keepdims=True)) @ v
        expected = np.fromfile(folder / 'output.f32.expected.f64', dtype=np.float64).reshape(m['output_shape'])
        assert np.allclose(expected, oracle, atol=1e-12, rtol=1e-12)
        check_output(folder / 'output.f32', expected)
        prepared = root / ('prepared-' + name)
        manifest = json.loads((prepared / 'prepared.json').read_text())
        assert manifest['status'] == 'prepared'
        for relative, sha in manifest['files'].items():
            assert digest(prepared / relative) == sha, relative
    results = []
    visits = 0
    controls = ('operation', 'dimensions', 'attention_block', 'precision', 'fast_math', 'relaxed_precision',
                'requested_group_threads', 'requested_input_views', 'reduction_tree', 'source_reduction_policy',
                'input_shapes', 'output_shape')
    for experiment in json.loads((root / 'replays.json').read_text())['experiments']:
        name, label = experiment['name'], experiment['label']
        path = root / ('replay-' + name + '-' + label)
        r = json.loads((path / 'results.json').read_text())
        assert r['status'] == 'passed' and len(r['visits']) == 12
        ratio = r['candidate_over_baseline']
        a, b = ratio['baseline'], ratio['candidate']
        manifests = [json.loads((root / f'prepared-{name}-{variant}' / 'prepared.json').read_text()) for variant in (a, b)]
        left, right = [m['metadata'] for m in manifests]
        assert all(left[key] == right[key] for key in controls)
        changes = [key for key in ('attention_qk', 'attention_pv') if left[key] != right[key]]
        assert changes == (['attention_pv'] if label.startswith('pv-') else ['attention_qk'])
        assert all(manifests[0]['files'][f] == manifests[1]['files'][f]
                   for f in ('input0.f32', 'input1.f32', 'input2.f32', 'expected.f64'))
        expected = np.fromfile(root / f'prepared-{name}-{a}' / 'expected.f64', dtype=np.float64)
        pairs = []
        for cycle in range(3):
            rows = r['visits'][cycle * 4:cycle * 4 + 4]
            assert [row['variant'] for row in rows] == [a, b, b, a]
            pairs += [rows[1]['median_us'] / rows[0]['median_us'], rows[2]['median_us'] / rows[3]['median_us']]
            for row in rows:
                assert row['valid'] and row['all_guards_passed'] and row['inputs_unchanged'] and row['returncode'] == 0
                assert len(row['samples_us']) == 7 and all(np.isfinite(x) and x > 0 for x in row['samples_us'])
                assert row['median_us'] == statistics.median(row['samples_us'])
                assert digest(path / row['output']) == row['output_sha256']
                check_output(path / row['output'], expected)
                visits += 1
        assert pairs == ratio['pairs'] and statistics.median(pairs) == ratio['median']
        medians = {variant: statistics.median(v['median_us'] for v in r['visits'] if v['variant'] == variant) for variant in (a,b)}
        assert medians == r['summary_us']
        results.append(dict(case=name, experiment=label, controls_equal=list(controls), changed=changes,
                            medians_us=medians, paired_ratio=ratio['median'], pair_range=[min(pairs),max(pairs)]))
    assert visits == 72 and len(results) == 6
    return dict(status='passed', independent_numpy_oracles=8, replay_output_checks=visits,
                guard_scope='Numerical outputs replayed independently; buffer/workspace/input guards are C++ execution receipts, not saved guard payloads.',
                experiments=results)


if __name__ == '__main__':
    print(json.dumps(audit(Path(sys.argv[1])), indent=2, allow_nan=False))
