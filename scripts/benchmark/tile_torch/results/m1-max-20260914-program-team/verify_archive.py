"""Verify every archived byte and independently regenerate all timing summaries."""
import hashlib
import json
from pathlib import Path
import tarfile
import time

from recompute import recompute

HERE = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    started = time.time()
    manifest = json.loads((HERE / 'manifest.json').read_text())
    count = 0
    for name, expected in manifest['archives'].items():
        assert sha((HERE / name).read_bytes()) == expected['sha256']
        with tarfile.open(HERE / name) as archive:
            assert set(archive.getnames()) == set(expected['members'])
            for member in archive.getmembers():
                assert member.isfile()
                data = archive.extractfile(member).read()
                assert len(data) == expected['members'][member.name]['size']
                assert sha(data) == expected['members'][member.name]['sha256']
                count += 1
    for name, expected in manifest['scripts'].items():
        assert sha((HERE / name).read_bytes()) == expected
    with tarfile.open(HERE / 'source-default-predication.tar.gz') as archive:
        actual = {member.name: sha(archive.extractfile(member).read()) for member in archive.getmembers()}
        assert actual == manifest['full6_source_files'] and len(actual) == 17
    summary = recompute(HERE)
    assert summary == json.loads((HERE / 'summary.json').read_text())
    result = dict(status='passed', started=started, finished=time.time(), archive_members=count,
                  manifest_sha256=sha((HERE / 'manifest.json').read_bytes()),
                  summary_sha256=sha((HERE / 'summary.json').read_bytes()),
                  verification_source_sha256=sha(Path(__file__).read_bytes()),
                  visits=summary['total_visits'], samples=summary['total_samples'],
                  archived_full6_source_files=17, no_native_execution=True)
    with (HERE / 'verification.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    with (HERE / 'SHA256SUMS').open('x') as stream:
        for path in sorted(HERE.iterdir()):
            if path.is_file() and path.name != 'SHA256SUMS':
                stream.write(sha(path.read_bytes()) + '  ' + path.name + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
