"""Pin compiler inputs independently of dirty user/dependency checkouts."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/Users/mike/CLionProjects/luisa')
HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'source'
FILES = [
    'include/luisa/tile/bridge/xir/lower.h',
    'include/luisa/tile/bridge/xir/planner.h',
    'src/tile/bridge/xir/representation.h',
    'src/tile/bridge/xir/lower.cpp',
    'src/tile/bridge/xir/planner.cpp',
    'src/backends/simd/runtime/simd_tile.cpp',
    'src/tests/unit/tile/bridge/test_xir.cpp',
    'src/tests/unit/tile/bridge/test_xir_runtime.cpp',
]


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive(repo, dest, commit, relative, records):
    assert Path(git(repo, 'rev-parse', '--show-toplevel').decode().strip()).resolve() == repo.resolve()
    git(repo, 'cat-file', '-e', commit + '^{commit}')
    dest.mkdir(parents=True, exist_ok=True)
    producer = subprocess.Popen(['git', '-C', str(repo), 'archive', '--format=tar', commit], stdout=subprocess.PIPE)
    consumer = subprocess.run(['tar', '-xf', '-', '-C', str(dest)], stdin=producer.stdout)
    producer.stdout.close()
    assert producer.wait() == 0 and consumer.returncode == 0
    records.append(dict(path=relative, pinned_commit=commit, checkout_commit=git(repo, 'rev-parse', 'HEAD').decode().strip()))
    for entry in git(repo, 'ls-tree', '-rz', commit).split(b'\0'):
        if not entry:
            continue
        metadata, path = entry.split(b'\t', 1)
        mode, kind, revision = metadata.split()
        if mode == b'160000':
            child = path.decode()
            archive(repo / child, dest / child, revision.decode(), relative + '/' + child, records)


assert git(ROOT, 'rev-parse', 'HEAD').decode().strip() == '2634be45d21bc8b2678104dadea99a5a0595d7c4'
if sys.argv[1] == 'snapshot':
    assert not SOURCE.exists()
    records = []
    archive(ROOT, SOURCE, '2634be45d21bc8b2678104dadea99a5a0595d7c4', '.', records)
    (HERE / 'pinned-repositories.json').write_text(json.dumps(records, indent=2) + '\n')
if sys.argv[1] in ('snapshot', 'refresh'):
    overlay = git(ROOT, 'diff', '--binary', 'HEAD', '--', *FILES)
    for name in FILES:
        # Explicit candidate overlay only, never any user-owned dirty file.
        (SOURCE / name).write_bytes((ROOT / name).read_bytes())
    (HERE / 'candidate.patch').write_bytes(overlay)
    manifest = dict(time=time.time(), source_commit='2634be45d21bc8b2678104dadea99a5a0595d7c4',
                    overlay=FILES, source_sha256={str(p.relative_to(SOURCE)): sha(p) for p in sorted(SOURCE.rglob('*')) if p.is_file()})
    (HERE / 'source-snapshot.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Pinned source:', len(manifest['source_sha256']), 'files; overlay', len(FILES), flush=True)
elif sys.argv[1] == 'configure':
    previous = json.loads(Path('/tmp/luisa-next-checkpoint.J7yODi/configure-command.json').read_text())
    argv = [arg.replace('/tmp/luisa-next-checkpoint.J7yODi', str(HERE)) for arg in previous['argv']]
    argv.append('-DCMAKE_EXPORT_COMPILE_COMMANDS=ON')
    (HERE / 'configure-command.json').write_text(json.dumps(dict(argv=argv), indent=2) + '\n')
    with (HERE / 'configure.log').open('w') as log:
        result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
    print('Configure exit', result.returncode, flush=True)
    sys.exit(result.returncode)
