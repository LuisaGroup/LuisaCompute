#!/usr/bin/env python3
"""Freeze tested source identities and final-binary smoke separately from timing cohorts."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

from audit import verify_visits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
FILES = [
    'include/luisa/tile/bridge/xir/planner.h', 'src/tile/bridge/xir/planner.cpp',
    'src/backends/simd/runtime/simd_shader.h', 'src/backends/simd/runtime/simd_shader.cpp',
    'src/backends/simd/runtime/simd_tile.cpp', 'src/tests/benchmark/benchmark_tile_xir.cpp',
    'src/tests/unit/tile/bridge/test_xir.cpp', 'src/tests/unit/tile/bridge/test_xir_runtime.cpp',
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--isolated', type=Path, required=True)
    args = parser.parse_args()
    args.raw = args.raw.resolve()
    if (HERE / 'provenance.json').exists():
        raise ValueError('provenance is immutable; choose a new evidence directory')
    smoke = json.loads((args.raw / 'final-smoke/results.json').read_text())
    pilot = json.loads((args.raw / 'pilot/results.json').read_text())
    verify_visits(smoke, 'pilot', 2)
    assert smoke['closure_unchanged']
    before = {(r['operation'], tuple(r['dimensions']), r['round'], r['variant']): r for r in pilot['results']}
    for row in smoke['results']:
        original = before[(row['operation'], tuple(row['dimensions']), row['round'], row['variant'])]
        assert row['input_sha256'] == original['input_sha256']
        for field in ('selected_local', 'selected_block', 'selected_grain', 'output_sha256'):
            assert row[field] == original[field]
        assert row['measurement']['realization'] == original['measurement']['realization']
    final_sources = {}
    for file in FILES:
        data = (ROOT / file).read_bytes()
        assert data == (args.isolated / 'source' / file).read_bytes(), file
        final_sources[file] = sha(data)
    measured_sources = {}
    measured_metadata = {}
    with tarfile.open(args.raw / 'measured-sources.tar.gz') as archive:
        # macOS bsdtar hides these AppleDouble sidecars in its own listing,
        # while Python tarfile exposes them. Keep and fingerprint the original
        # snapshot, accepting only the observed sidecar for each explicit file.
        sidecars = {str(Path(file).with_name('._' + Path(file).name)) for file in FILES}
        members = archive.getmembers()
        assert len(members) == len(FILES) + len(sidecars)
        assert {member.name for member in members} == set(FILES) | sidecars
        assert all(member.isfile() for member in members)
        for file in sorted(sidecars):
            data = archive.extractfile(file).read()
            assert data[:4] == b'\x00\x05\x16\x07', file
            measured_metadata[file] = sha(data)
        for file in FILES:
            measured_sources[file] = sha(archive.extractfile(file).read())
    changes = [file for file in FILES if final_sources[file] != measured_sources[file]]
    assert changes == ['src/tile/bridge/xir/planner.cpp', 'src/tests/unit/tile/bridge/test_xir.cpp']
    for file, value in smoke['binary_closure_sha256'].items():
        assert sha(Path(file).read_bytes()) == value, file
    logs = [*args.raw.glob('*.log'), args.isolated / 'build/Testing/Temporary/LastTest.log']
    for file in logs:
        (HERE / ('ctest-details.log' if file.name == 'LastTest.log' else file.name)).write_bytes(file.read_bytes())
    (HERE / 'measured-sources.tar.gz').write_bytes((args.raw / 'measured-sources.tar.gz').read_bytes())
    patch = subprocess.check_output(['git', 'diff', '--', *FILES], cwd=ROOT)
    (HERE / 'source-overlay.patch.gz').write_bytes(gzip.compress(patch, mtime=0))
    (HERE / 'final-smoke.json.gz').write_bytes(gzip.compress((json.dumps(smoke, indent=2) + '\n').encode(), mtime=0))
    report = dict(parent_commit='9105ad98159e6b87c954b7ba67fe2ca0fea014b9',
                  isolated_baseline_context='../m1-max-20260908-full-packet/provenance.json',
                  measured_source_sha256=measured_sources, final_source_sha256=final_sources,
                  measured_appledouble_sha256=measured_metadata,
                  post_measurement_changes=changes,
                  change_scope='single-CPU-worker whole-range callback accounting plus its unit oracle; all comparisons request eight workers',
                  final_smoke_visits=40, final_smoke_same_plans_realization_inputs_outputs=True,
                  final_smoke_binary_closure_sha256=smoke['binary_closure_sha256'],
                  cmake_cache_sha256=sha((args.isolated / 'build/CMakeCache.txt').read_bytes()),
                  raw=str(args.raw), isolated=str(args.isolated),
                  script_sha256={name: sha((HERE / name).read_bytes()) for name in ('measure.py', 'audit.py', 'archive.py')})
    (HERE / 'provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS: measured/final source identities, 40 final smoke outputs and identical plan/realization metadata, current final binary closure')


if __name__ == '__main__':
    main()
