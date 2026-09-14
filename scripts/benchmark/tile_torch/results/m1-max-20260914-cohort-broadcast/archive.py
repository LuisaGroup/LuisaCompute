"""Content-addressed full9 cohort archive. Reads only raw evidence, never ROOT/SELECTED."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time

HERE = Path(__file__).resolve().parent
RAW = Path('/tmp/luisa-metal-program-team.TGZv8f').resolve()
COHORT = RAW / 'cpu-ablation-full7.ojSyZd'
SHARD_LIMIT = 32 * 1024 * 1024


def sha(data):
    return hashlib.sha256(data).hexdigest()


def normalize(path):
    value = str(path)
    return '/private' + value if value.startswith('/tmp/') else value


def main():
    started = time.time()
    admission = json.loads((COHORT / 'admission.json').read_text())
    frozen = json.loads((COHORT / 'full9-immutable/manifest.json').read_text())
    assert frozen['status'] == 'passed' and frozen['full9_source_files'] == 23
    files = {}
    for path in sorted(COHORT.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            files['cohort/' + str(path.relative_to(COHORT))] = path
    receipts = [Path(p).parent for p in admission['gate_sha256']]
    receipts += [RAW / name for name in ('tidy-full9-xir-schedule', 'tidy-full9-test-xir-schedule', 'tidy-full9-test-schedule-codegen')]
    for folder in receipts:
        data = json.loads((folder / 'result.json').read_text())
        assert data['status'] == 'passed' and data['exit_code'] == 0 and 'finished' in data
        assert data['source_freeze_sha256'] == admission['freeze_sha256']
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                files['full9-receipts/' + str(path.relative_to(RAW))] = path
    for folder in (RAW / 'build-full-7', RAW / 'test-full7-unit-simd'):
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                files['history/' + str(path.relative_to(RAW))] = path
    files['history/source-freeze-uniform-read-lane.json'] = RAW / 'source-freeze-uniform-read-lane.json'
    entries, unique, aliases = {}, {}, {}
    for logical, path in sorted(files.items()):
        assert not path.is_symlink()
        data = path.read_bytes()
        digest = sha(data)
        entries[logical] = dict(sha256=digest, size=len(data), original_path=str(path))
        unique.setdefault(digest, path)
        aliases[normalize(path)] = logical
    for original, record in frozen['saved'].items():
        logical = 'cohort/full9-immutable/' + record['path']
        assert entries[logical]['sha256'] == record['sha256']
        aliases[normalize(original)] = logical
    for original, expected in admission['critical_files_sha256'].items():
        assert entries[aliases[normalize(original)]]['sha256'] == expected
    for original, expected in admission['gate_sha256'].items():
        assert entries[aliases[normalize(original)]]['sha256'] == expected
    shards, objects = {}, {}
    current, size = [], 0

    def flush():
        nonlocal current, size
        if not current:
            return
        name = f'blobs-{len(shards) + 1:03d}.tar.gz'
        path = HERE / name
        with path.open('xb') as stream, gzip.GzipFile(fileobj=stream, mode='wb', mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w') as archive:
                for digest in current:
                    data = unique[digest].read_bytes()
                    assert sha(data) == digest
                    entry = tarfile.TarInfo(digest)
                    entry.size, entry.mode = len(data), 0o644
                    archive.addfile(entry, io.BytesIO(data))
                    objects[digest] = dict(shard=name, size=len(data))
        assert path.stat().st_size < 45 * 1024 * 1024
        shards[name] = dict(sha256=sha(path.read_bytes()), bytes=path.stat().st_size, objects=len(current))
        current, size = [], 0

    for digest, path in sorted(unique.items()):
        length = path.stat().st_size
        assert length < SHARD_LIMIT
        if size + length + 1024 > SHARD_LIMIT:
            flush()
        current.append(digest)
        size += length + 1024
    flush()
    assert all(sha(path.read_bytes()) == entries[name]['sha256'] for name, path in files.items())
    manifest = dict(status='archived', started=started, finished=time.time(), producer='full9 / cohort-recurrences',
                    freeze_sha256=admission['freeze_sha256'], files=entries, aliases=aliases, objects=objects, shards=shards,
                    original_bytes=sum(e['size'] for e in entries.values()), unique_bytes=sum(e['size'] for e in objects.values()),
                    compressed_bytes=sum(e['bytes'] for e in shards.values()),
                    content_addressing='Each logical path maps to one SHA256 blob; identical bytes deduplicated without dropping logical files.',
                    source_scope='23 frozen full9 source files from independent raw snapshot; never current ROOT/SELECTED.',
                    binary_scope='Four admitted critical binary artifacts and actual per-capture ORC/replay libraries; not complete historical system/Torch/LLVM closure.',
                    exclusions=['pycache', 'live or later producer revisions', 'machine caches'],
                    archive_script_sha256=sha(Path(__file__).read_bytes()))
    with (HERE / 'manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: manifest[k] for k in ('status', 'original_bytes', 'unique_bytes', 'compressed_bytes') } |
                     dict(files=len(entries), objects=len(objects), shards=len(shards)), indent=2))


if __name__ == '__main__':
    main()
