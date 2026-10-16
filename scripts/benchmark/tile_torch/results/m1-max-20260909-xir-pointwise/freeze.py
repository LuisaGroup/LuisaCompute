#!/usr/bin/env python3
"""Freeze correctness evidence; busy-host smoke timings are not performance data."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT / 'scripts/benchmark/tile_torch'))
import native_rows as nr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    args = parser.parse_args()
    raw = args.raw.resolve()
    if (HERE / 'provenance.json').exists():
        raise ValueError('immutable checkpoint: choose a new destination')
    capture = json.loads((raw / 'capture-final/manifest.json').read_text())
    smoke = json.loads((raw / 'native-verify/results.json').read_text())
    assert capture['source_unchanged'] and capture['closure_unchanged']
    assert smoke['runner_unchanged'] and smoke['timing_not_comparative']
    assert smoke['metric'] == 'native_entry_correctness_smoke'
    assert smoke['manifest_sha256'] == nr.digest(raw / 'capture-final/manifest.json')
    assert nr.digest(capture['previous']) == capture['previous_sha256']
    previous = json.loads(Path(capture['previous']).read_text())
    previous_cases = {(c['operation'], tuple(c['dimensions'])): c for c in previous['cases']}
    for folder, record in ((raw / 'capture-final', capture), (raw / 'native-verify', smoke)):
        assert nr.digest(folder / 'pilot.py') == record['pilot_sha256']
        assert nr.digest(folder / 'runner-sources/native_rows.py') == record['native_rows_sha256']
    assert len(capture['cases']) == len(smoke['cases']) == 24
    assert sum(len(c['results']) for c in smoke['cases']) == 72
    expected_keys = {(op, (m, n + (n % 2 if op == 'rope' else 0))) for op in nr.OPS for m, n in nr.SHAPES}
    assert {(c['operation'], tuple(c['dimensions'])) for c in capture['cases']} == expected_keys
    for name, sha in capture['source_sha256'].items():
        assert nr.digest(ROOT / name) == sha, name
    for name, sha in capture['binary_closure_sha256'].items():
        assert nr.digest(name) == sha, name
    assert nr.digest(capture['helper']) == capture['helper_sha256']
    artifacts = {}

    def store(name, data, origin):
        target = HERE / name
        if target.exists():
            raise ValueError('refusing to overwrite evidence: ' + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        encoded = gzip.compress(data, mtime=0) if name.endswith('.gz') else data
        target.write_bytes(encoded)
        artifacts[name] = dict(origin=str(origin), source_sha256=hashlib.sha256(data).hexdigest(),
                               archived_sha256=hashlib.sha256(encoded).hexdigest(), source_bytes=len(data), archived_bytes=len(encoded))

    def archive(name, source):
        store(name, Path(source).read_bytes(), source)

    captures_checked = snapshots_checked = 0
    for case, result in zip(capture['cases'], smoke['cases']):
        assert (case['operation'], case['dimensions']) == (result['operation'], result['dimensions'])
        assert case['off_matches_previous_llvm'] and case['off_matches_previous_object']
        assert result['off_on_bitwise_equal']
        assert {v['variant'] for v in result['results']} == {'off', 'on', 'inductor'}
        assert len(result['results']) == 3
        op, dims = case['operation'], case['dimensions']
        name = nr.case_name(op, dims)
        previous_entry = previous_cases[op, tuple(dims)]['entries']['local']
        assert case['entries']['off']['object_sha256'] == previous_entry['object_sha256']
        assert case['entries']['off']['llvm_sha256'] == previous_entry['llvm_sha256']
        arrays = nr.load_inputs(Path(case['inputs']), case['input_shapes'])
        assert [nr.array_digest(a) for a in arrays] == case['input_sha256']
        expected = nr.reference(op, dims, arrays)
        for variant, entry in case['entries'].items():
            assert nr.digest(entry['library']) == entry['library_sha256']
            if variant != 'inductor':
                folder = Path(entry['capture'])
                values = nr.load_inputs(folder, case['input_shapes'])
                assert [nr.array_digest(a) for a in values] == case['input_sha256']
                nr.validate_output(np.fromfile(folder / 'output.f32', np.float32).reshape(case['output_shape']), expected)
                captures_checked += 1
                objects = list((folder / 'object').glob('*.o'))
                assert len(objects) == 1 and nr.digest(objects[0]) == entry['object_sha256']
                assert nr.digest(folder / 'kernel.ll') == entry['llvm_sha256']
                prefix = 'native/' + name + '/' + variant + '/'
                for saved, source in (('kernel.o.gz', objects[0]), ('kernel.ll.gz', folder / 'kernel.ll'),
                                      ('entry.dylib.gz', Path(entry['library'])), ('assembly.txt.gz', folder / 'assembly.stdout.log'),
                                      ('measurement.json', folder / 'measurement.json')):
                    archive(prefix + saved, source)
            actual = np.fromfile(raw / 'native-verify' / name / (variant + '.f32'), np.float32).reshape(case['output_shape'])
            assert nr.array_digest(actual) == result['output_sha256'][variant]
            nr.validate_output(actual, expected)
            visit = next(v for v in result['results'] if v['variant'] == variant)
            assert visit['inputs_unchanged'] and visit['workspace_guards_passed'] and visit['guard_elements'] > 0
            assert visit['output_sha256'] == nr.array_digest(actual)
            snapshots_checked += 1
        off = np.fromfile(raw / 'native-verify' / name / 'off.f32', np.uint32)
        on = np.fromfile(raw / 'native-verify' / name / 'on.f32', np.uint32)
        assert np.array_equal(off, on)
    # Verify the full-output gate rejects corruption, nonfinite data and size
    # mismatch. Mutations are in-memory only; saved evidence remains unchanged.
    rejected = 0
    for mutation in ('changed_tail', 'nan', 'truncated'):
        value = expected.astype(np.float32)
        if mutation == 'changed_tail':
            value.flat[-1] += np.float32(1.0)
        elif mutation == 'nan':
            value.flat[-1] = np.nan
        else:
            value = value.reshape(-1)[:-1]
        try:
            nr.validate_output(value, expected)
        except ValueError:
            rejected += 1
    assert (captures_checked, snapshots_checked, rejected) == (48, 72, 3)
    audit = dict(status='passed', capture_outputs_re_read=captures_checked, native_outputs_re_read=snapshots_checked,
                 off_on_bitwise_equal_cases=24, previous_off_object_identical_cases=24, rejected_output_mutations=rejected,
                 performance_conclusion='None: smoke samples deliberately excluded due to concurrent builds/high load.',
                 guard_caveat='Guards and input immutability were checked at each invocation; ephemeral guard allocations are not retained for re-reading.')
    store('audit.json', (json.dumps(audit, indent=2) + '\n').encode(), 'independent artifact re-read and mutation checks')
    for name, source in (('capture.json.gz', raw / 'capture-final/manifest.json'), ('native-smoke.json.gz', raw / 'native-verify/results.json'),
                         ('baseline.json.gz', raw / 'baseline.json'), ('pilot-capture.py', raw / 'capture-final/pilot.py'),
                         ('pilot-verify.py', raw / 'native-verify/pilot.py'), ('qa-docs.cjs', raw / 'qa-docs.cjs')):
        archive(name, source)
    for source in sorted((raw / 'native-verify/runner-sources').iterdir()):
        archive('runner-sources/' + source.name, source)
    for name in ('build-v1.log', 'build-v2.log', 'build-v3.log', 'ctest-v1.log', 'ctest-v2.log', 'ctest-llm-fused-v2.log',
                 'runtime-w2-v2.log', 'runtime-w16-v2.log', 'runtime-perf-options-v2.log', 'bounds-v3.log',
                 'clangd-lower-final.log', 'capture-final.log', 'native-verify.log', 'doxygen.log', 'sphinx-final.log',
                 'docs-check-final.log', 'docs-qa-final.log', 'pilot-capture-v2.log', 'pilot-capture-v3.log',
                 'classic-probe.log', 'docs-qa.log'):
        archive('logs/' + name + '.gz', raw / name)
    qa = json.loads((raw / 'docs-qa-final/receipt.json').read_text())
    assert qa['passed'] and len(qa['receipts']) == 4
    archive('docs-qa.json', raw / 'docs-qa-final/receipt.json')
    provenance = dict(frozen_unix=time.time(), parent_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                      fetched_next=subprocess.check_output(['git', 'rev-parse', 'origin/next'], cwd=ROOT, text=True).strip(), next_merged=False,
                      raw=str(raw), source_sha256=capture['source_sha256'], binary_closure_sha256=capture['binary_closure_sha256'],
                      previous_native_checkpoint='../m1-max-20260909-native-rows/provenance.json',
                      previous_native_checkpoint_sha256=nr.digest(HERE.parent / 'm1-max-20260909-native-rows/provenance.json'),
                      scope='Guarded same-domain pointwise realization, default off; no new cost calibration or selected default.',
                      source_caveat=capture['source_caveat'], artifacts=artifacts,
                      timing='Correctness smoke only. No speedup, parity, model-fit or E2E claim; formal paired native replay remains pending.',
                      failures='Transient missing uv executable; interrupted busy-host capture; link timeout; system Node missing uvwasi. Retained logs; none contributes comparative timing.',
                      payload_retention='Input/output payloads remain in raw; Git keeps hashes, source, actual code and validation. Inductor/helper artifacts are reused by hash from the previous native checkpoint.')
    (HERE / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    for name, item in artifacts.items():
        data = (HERE / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == item['archived_sha256']
        decoded = gzip.decompress(data) if name.endswith('.gz') else data
        assert hashlib.sha256(decoded).hexdigest() == item['source_sha256']
    print('PASS:', captures_checked, 'captures,', snapshots_checked, 'native snapshots,', rejected, 'rejected mutations;', len(artifacts), 'verified artifacts')


if __name__ == '__main__':
    main()
