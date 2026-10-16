"""Small full10 diagnostic supplement; never reads current ROOT or SELECTED."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

HERE = Path(__file__).resolve().parent
RAW = Path('/tmp/luisa-metal-program-team.TGZv8f')
FREEZE = 'source-freeze-collective-epoch-counterexample.json'
SOURCE = 'source-collective-epoch-counterexample.tar.gz'
FREEZE_SHA = 'f37f46ba0a668e9966d74168f037b9e4e18f1b17be597521aad080c5e0b3b55f'


def sha(value):
    return hashlib.sha256(value).hexdigest()


def validate(files):
    freeze = json.loads(files[FREEZE])
    assert sha(files[FREEZE]) == FREEZE_SHA
    assert len(freeze['source_sha256']) == 23
    assert sha(files['owned-through-full10.json']) == freeze['owned_sha256']
    assert set(json.loads(files['owned-through-full10.json'])) == set(freeze['source_sha256'])
    with tarfile.open(fileobj=io.BytesIO(files[SOURCE])) as archive:
        members = archive.getmembers()
        assert len(members) == 23 and all(m.isfile() for m in members)
        assert {m.name for m in members} == set(freeze['source_sha256'])
        for member in members:
            assert sha(archive.extractfile(member).read()) == freeze['source_sha256'][member.name]
    prior = json.loads(files['full9-freeze-reference.json'])
    changed = sorted(k for k in freeze['source_sha256']
                     if freeze['source_sha256'][k] != prior['source_sha256'][k])
    assert changed == ['src/tests/unit/simd/test_llvm_schedule_codegen.cpp',
                       'src/tests/unit/simd/test_xir_to_schedule.cpp']
    for folder, status, code in [('build-full-10', 'passed', 0),
                                  ('test-full10-epoch-counterexample', 'failed', 8)]:
        receipt = json.loads(files[folder + '/result.json'])
        assert receipt['status'] == status and receipt['exit_code'] == code
        assert not receipt.get('timed_out', False)
        assert receipt['source_freeze_sha256'] == FREEZE_SHA
        assert receipt['source_sha256'] == freeze['source_sha256']
        assert receipt['owned_sha256'] == freeze['owned_sha256']
        assert freeze['finished'] < receipt['started'] < receipt['finished']
        for stream in ('stdout', 'stderr'):
            assert sha(files[f'{folder}/{stream}.txt']) == receipt[stream + '_sha256']
    build = json.loads(files['build-full-10/result.json'])
    test = json.loads(files['test-full10-epoch-counterexample/result.json'])
    assert build['finished'] < test['started']
    log = files['test-full10-epoch-counterexample/stdout.txt'].decode()
    assert '0% tests passed, 2 tests failed out of 2' in log
    records = sorted(set(re.findall(
        r'cohort exit oracle: stage=(second loop sum|escaped population), width=(\d+), '
        r'enabled=([01]), active=(\d+), lane=(\d+), actual=(\d+), expected=(\d+), direct=(\d+)', log)))
    expected = sorted((stage, str(width), str(enabled), '2', lane, actual, want, '0')
                      for width in (2, 4, 8, 16) for enabled in (0, 1)
                      for stage, lane, actual, want in (
                          ('second loop sum', '0', '121', '111'),
                          ('escaped population', '1', '2', '1')))
    assert records == expected
    return dict(status='verified_failed_counterexample', producer='full10 diagnostic after full9 performance',
                freeze_sha256=FREEZE_SHA, source_files=23, changed_from_full9=changed,
                build_status='passed', test_status='failed', test_exit_code=8, timed_out=False,
                unique_numeric_failure_records=len(records), numeric_failure_records=[list(row) for row in records],
                native_execution_by_this_verifier=False,
                scope='Actual native failure in both enabled=0/1 modes. Not a full9 admission gate or performance sample.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', action='store_true')
    args = parser.parse_args()
    if args.archive:
        paths = {name: RAW / name for name in (FREEZE, SOURCE, 'owned-through-full10.json')}
        for folder in ('build-full-10', 'test-full10-epoch-counterexample'):
            for path in sorted((RAW / folder).iterdir()):
                assert path.is_file() and not path.is_symlink()
                paths[folder + '/' + path.name] = path
        paths['full9-freeze-reference.json'] = (
            RAW / 'cpu-ablation-full7.ojSyZd/full9-immutable/raw/source-freeze-cohort-recurrences.json')
        files = {name: path.read_bytes() for name, path in paths.items()}
        result = validate(files)
        target = HERE / 'evidence.tar.gz'
        with target.open('xb') as output, gzip.GzipFile(fileobj=output, mode='wb', mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w') as archive:
                for name, value in sorted(files.items()):
                    entry = tarfile.TarInfo(name)
                    entry.size, entry.mode = len(value), 0o644
                    archive.addfile(entry, io.BytesIO(value))
        assert target.stat().st_size < 45 * 1024 * 1024
        manifest = dict(archive_sha256=sha(target.read_bytes()), archive_bytes=target.stat().st_size,
                        files={name: dict(sha256=sha(value), size=len(value), original=str(paths[name]))
                               for name, value in sorted(files.items())}, verification=result)
        with (HERE / 'manifest.json').open('x') as output:
            json.dump(manifest, output, indent=2)
            output.write('\n')
        assert all(path.read_bytes() == files[name] for name, path in paths.items())
        with (HERE / 'SHA256SUMS').open('x') as output:
            for path in sorted(HERE.iterdir()):
                if path.is_file() and path.name != 'SHA256SUMS':
                    output.write(f'{sha(path.read_bytes())}  {path.name}\n')
    else:
        manifest = json.loads((HERE / 'manifest.json').read_text())
        assert sha((HERE / 'evidence.tar.gz').read_bytes()) == manifest['archive_sha256']
        with tarfile.open(HERE / 'evidence.tar.gz') as archive:
            members = archive.getmembers()
            assert all(m.isfile() for m in members)
            assert len(members) == len(manifest['files'])
            files = {m.name: archive.extractfile(m).read() for m in members}
        assert set(files) == set(manifest['files'])
        for name, value in files.items():
            assert sha(value) == manifest['files'][name]['sha256']
            assert len(value) == manifest['files'][name]['size']
        result = validate(files)
        assert result == manifest['verification']
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
