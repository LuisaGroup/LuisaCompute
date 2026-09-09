"""Materialize committed source and pinned submodules without changing user checkouts."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path('/Users/mike/CLionProjects/luisa')
HERE = Path(__file__).resolve().parent
TARGET = HERE / 'source'


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args])


revision = git(ROOT, 'rev-parse', 'HEAD').decode().strip()
assert revision.startswith('360d9791e')
assert not TARGET.exists()
records = []


def archive(repo, destination, commit, relative):
    actual_root = Path(git(repo, 'rev-parse', '--show-toplevel').decode().strip()).resolve()
    if actual_root != repo.resolve():
        raise ValueError('Missing local submodule object store: ' + str(repo))
    git(repo, 'cat-file', '-e', commit + '^{commit}')
    destination.mkdir(parents=True, exist_ok=True)
    producer = subprocess.Popen(['git', '-C', str(repo), 'archive', '--format=tar', commit], stdout=subprocess.PIPE)
    consumer = subprocess.run(['tar', '-xf', '-', '-C', str(destination)], stdin=producer.stdout)
    producer.stdout.close()
    if producer.wait() or consumer.returncode:
        raise ValueError('Archive failed: ' + relative)
    records.append(dict(path=relative, pinned_commit=commit,
                        checkout_commit=git(repo, 'rev-parse', 'HEAD').decode().strip(),
                        working_status=git(repo, 'status', '--porcelain', '--untracked-files=no').decode()))
    print('Archived', relative, commit, flush=True)
    for record in git(repo, 'ls-tree', '-rz', commit).split(b'\0'):
        if not record:
            continue
        metadata, path = record.split(b'\t', 1)
        mode, kind, sha = metadata.split()
        if mode == b'160000':
            child = path.decode()
            archive(repo / child, destination / child, sha.decode(), relative + '/' + child)


archive(ROOT, TARGET, revision, '.')
hashes = {str(p.relative_to(TARGET)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(TARGET.rglob('*')) if p.is_file()}
report = dict(created_unix=time.time(), source_commit=revision, source=str(TARGET),
              submodules=records, source_sha256=hashes,
              contract='Git archives of the merge commit and recursively pinned submodule commits; uncommitted source and dependency checkouts excluded.')
(HERE / 'snapshot.json').write_text(json.dumps(report, indent=2) + '\n')
print('PASS:', len(records), 'pinned repositories;', len(hashes), 'source files', flush=True)
