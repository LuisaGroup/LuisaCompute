"""Archive terminal full11 native evidence; never read mutable ROOT/SELECTED."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time

HERE = Path(__file__).resolve().parent
RAW = Path('/tmp/luisa-metal-program-team.TGZv8f').resolve()
COHORT = RAW / 'cpu-full11-vs-full9.H7yzJk'
OLD = RAW / 'cpu-ablation-full7.ojSyZd'
TORCH = RAW / 'cpu-inductor.zGCsEf'
LIMIT = 32 * 1024 * 1024


def sha(data):
    return hashlib.sha256(data).hexdigest()


def norm(path):
    name = str(path)
    return '/private' + name if name.startswith('/tmp/') else name


def main():
    started = time.time()
    joined = json.loads((COHORT / 'joined-admission.json').read_text())
    admission = json.loads((COHORT / 'admission.json').read_text())
    old_snapshot = json.loads((OLD / 'full9-immutable/manifest.json').read_text())
    for mode, count in (('capture', 22), ('prepare', 22), ('replay', 22), ('torch', 8)):
        data = json.loads((COHORT / f'{mode}-results.json').read_text())
        assert data['status'] == 'passed' and len(data['jobs']) == count
        assert all(row['status'] == 'passed' for row in data['jobs'])
    paths, alias_overrides, full9_aliases = {}, {}, {}
    def tree(folder, prefix):
        for path in sorted(folder.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                assert not path.is_symlink()
                paths[prefix + '/' + str(path.relative_to(folder))] = path
    tree(COHORT, 'cohort')
    tree(OLD / 'full9-immutable', 'full9/immutable')
    for row in joined['baseline']:
        label = f"{row['case']}-l{row['local_lanes']}-default"
        for stage in ('capture', 'prepare', 'prepared'):
            tree(OLD / (stage + '-' + label), 'full9/' + stage + '-' + label)
    for name in ('admission.json', 'capture-results.json', 'prepare-results.json', 'gates-full9.json'):
        paths['full9/' + name] = OLD / name
    for original, record in old_snapshot['saved'].items():
        full9_aliases[norm(original)] = 'full9/immutable/' + record['path']
    # Reconstruct original aliases from the independently saved full11 tree.
    snapshot = COHORT / 'full11-immutable'
    selected = Path(json.loads((snapshot / 'raw/source-freeze-collective-epoch-storage.json').read_text())['selected_source'])
    for original, expected in joined['full11_snapshot'].items():
        path = Path(original)
        assert path.is_relative_to(snapshot) and sha(path.read_bytes()) == expected
        relative = path.relative_to(snapshot)
        kind, tail = relative.parts[0], Path(*relative.parts[1:])
        old_path = selected / tail if kind == 'selected' else selected.parent / 'build' / tail if kind == 'build' else RAW / tail
        alias_overrides[norm(old_path)] = 'cohort/full11-immutable/' + str(relative)
    for original, expected in joined['torch']['files_sha256'].items():
        path = Path(original)
        if path.is_relative_to(selected):
            if path.name in ('native_rows.py', 'native_rows_replay.cpp'):
                path = TORCH / 'replay/runner-sources' / path.name
            else:
                path = snapshot / 'selected' / path.relative_to(selected)
        assert path.is_relative_to(RAW) and sha(path.read_bytes()) == expected
        logical = 'torch/' + str(path.relative_to(TORCH)) if path.is_relative_to(TORCH) else 'torch/abi/' + path.name
        paths[logical] = path
        alias_overrides[norm(original)] = logical
    files, unique, aliases = {}, {}, {}
    for logical, path in sorted(paths.items()):
        assert not path.is_symlink() and path.is_relative_to(RAW)
        data = path.read_bytes()
        digest = sha(data)
        files[logical] = dict(sha256=digest, size=len(data), original_path=str(path))
        unique.setdefault(digest, path)
        aliases[norm(path)] = logical
    aliases.update(alias_overrides)
    for original, expected in admission['critical_files_sha256'].items() | admission['gate_sha256'].items():
        assert files[aliases[norm(original)]]['sha256'] == expected
    for original, record in old_snapshot['saved'].items():
        assert files[full9_aliases[norm(original)]]['sha256'] == record['sha256']
    for path, expected in joined['historical_pins'].items():
        assert files[aliases[norm(path)]]['sha256'] == expected
    shards, objects, current = {}, {}, []
    size = 0
    def flush():
        nonlocal current, size
        if not current:
            return
        name = f'blobs-{len(shards) + 1:03d}.tar.gz'
        target = HERE / name
        with target.open('xb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0, compresslevel=6) as compressed:
            with tarfile.open(fileobj=compressed, mode='w') as archive:
                for digest in current:
                    data = unique[digest].read_bytes()
                    assert sha(data) == digest
                    entry = tarfile.TarInfo(digest)
                    entry.size, entry.mode = len(data), 0o644
                    archive.addfile(entry, io.BytesIO(data))
                    objects[digest] = dict(shard=name, size=len(data))
        shards[name] = dict(sha256=sha(target.read_bytes()), bytes=target.stat().st_size, objects=len(current))
        assert target.stat().st_size < 45 * 1024 * 1024
        current, size = [], 0
    for digest, path in sorted(unique.items()):
        length = path.stat().st_size
        assert length < LIMIT
        if size + length + 1024 > LIMIT:
            flush()
        current.append(digest)
        size += length + 1024
    flush()
    assert all(sha(path.read_bytes()) == files[name]['sha256'] for name, path in paths.items())
    manifest = dict(status='archived', started=started, finished=time.time(), files=files, aliases=aliases,
                    full9_aliases=full9_aliases, objects=objects, shards=shards,
                    original_bytes=sum(row['size'] for row in files.values()), unique_bytes=sum(row['size'] for row in objects.values()),
                    compressed_bytes=sum(row['bytes'] for row in shards.values()),
                    producer='full11 collective-epoch-storage; fresh 22 full9 pairs plus separate 8 Torch pairs',
                    source_scope='26 independently frozen full11 overlay files and retained 23-file full9 overlay; never current source.',
                    binary_scope='Actual native ORC objects/helpers, four critical full11/full9 binaries, retained Inductor source/ABI/library; not complete dynamic dependency closure.',
                    exclusions=['pycache', 'unrelated historical ablation captures', 'machine caches', 'unpersisted guard and scratch bytes'],
                    archive_script_sha256=sha(Path(__file__).read_bytes()))
    with (HERE / 'manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    print(json.dumps({key: manifest[key] for key in ('status', 'original_bytes', 'unique_bytes', 'compressed_bytes')} |
                     dict(files=len(files), objects=len(objects), shards=len(shards)), indent=2))


if __name__ == '__main__':
    main()
