"""Validate a committed next integration without dirty checkout inputs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

ROOT = Path('/Users/mike/CLionProjects/luisa')
HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'source'
BUILD = HERE / 'build'
COMMIT = '2cfc804932a014cd7a07eb3896166d53db32b3eb'
NEXT = '03a0f5158b53768abefea555f480f87ee5bc5a1e'
ENV = {k: v for k, v in os.environ.items()
       if not k.startswith(('LUISA_', 'DYLD_'))}
REPORT = dict(source_commit=COMMIT, next_commit=NEXT,
              source_kind='committed Git archives, recursively pinned dependencies, no overlay',
              started_unix=time.time(), performance_measurement=False, runs=[])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args])


def save(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2) + '\n')


def archive(repo, dest, commit, relative, records):
    assert Path(git(repo, 'rev-parse', '--show-toplevel').decode().strip()).resolve() == repo.resolve()
    git(repo, 'cat-file', '-e', commit + '^{commit}')
    dest.mkdir(parents=True, exist_ok=True)
    producer = subprocess.Popen(['git', '-C', str(repo), 'archive', '--format=tar', commit], stdout=subprocess.PIPE)
    consumer = subprocess.run(['tar', '-xf', '-', '-C', str(dest)], stdin=producer.stdout)
    producer.stdout.close()
    assert producer.wait() == 0 and consumer.returncode == 0
    records.append(dict(path=relative, pinned_commit=commit,
                        checkout_commit=git(repo, 'rev-parse', 'HEAD').decode().strip()))
    for entry in git(repo, 'ls-tree', '-rz', commit).split(b'\0'):
        if not entry:
            continue
        metadata, path = entry.split(b'\t', 1)
        mode, kind, revision = metadata.split()
        if mode == b'160000':
            child = path.decode()
            archive(repo / child, dest / child, revision.decode(), relative + '/' + child, records)


def run(name, argv, junit=None, expected=None):
    argv = list(map(str, argv))
    item = dict(name=name, command=argv, started_unix=time.time(), log=name + '.log')
    REPORT['runs'].append(item)
    save('verification.json', REPORT)
    with (HERE / item['log']).open('x') as log:
        result = subprocess.run(argv, cwd=HERE, env=ENV, stdout=log, stderr=subprocess.STDOUT)
    item.update(exit_code=result.returncode, finished_unix=time.time(), log_sha256=sha(HERE / item['log']))
    if junit:
        cases = list(ET.parse(HERE / junit).getroot().iter('testcase'))
        item.update(tests=len(cases),
                    failures=sum(c.find('failure') is not None or c.find('error') is not None for c in cases),
                    skipped=sum(c.find('skipped') is not None for c in cases),
                    junit=junit, junit_sha256=sha(HERE / junit))
        assert cases and item['failures'] == 0 and item['skipped'] == 0, item
        assert expected is None or len(cases) == expected, item
    save('verification.json', REPORT)
    print(name, 'exit', result.returncode, 'tests', item.get('tests'), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


assert not SOURCE.exists()
assert git(ROOT, 'rev-parse', 'HEAD').decode().strip() == COMMIT
records = []
archive(ROOT, SOURCE, COMMIT, '.', records)
save('pinned-repositories.json', records)
hashes = {str(p.relative_to(SOURCE)): sha(p) for p in sorted(SOURCE.rglob('*')) if p.is_file()}
save('baseline-source.json', dict(source_commit=COMMIT, source_sha256=hashes))
print('Source pinned:', len(hashes), 'files in', len(records), 'repositories', flush=True)
raise SystemExit(0)

argv = ['cmake', '-S', str(SOURCE), '-B', str(BUILD), '-G', 'Ninja',
        '-DCMAKE_BUILD_TYPE=RelWithDebInfo', '-DCMAKE_C_COMPILER=/usr/bin/cc',
        '-DCMAKE_CXX_COMPILER=/usr/bin/c++',
        '-DLLVM_DIR=/opt/homebrew/opt/llvm@21/lib/cmake/llvm',
        '-DLUISA_COMPUTE_BUILD_TESTS=ON', '-DLUISA_COMPUTE_ENABLE_DSL=ON',
        '-DLUISA_COMPUTE_ENABLE_SIMD=ON', '-DLUISA_COMPUTE_ENABLE_METAL=ON',
        '-DLUISA_COMPUTE_ENABLE_TILE_TIRX_BRIDGE=ON',
        '-DLUISA_COMPUTE_ENABLE_METAL4=OFF', '-DLUISA_COMPUTE_ENABLE_GUI=OFF',
        '-DLUISA_COMPUTE_ENABLE_CUDA=OFF', '-DLUISA_COMPUTE_ENABLE_HIP=OFF',
        '-DLUISA_COMPUTE_ENABLE_DX=OFF', '-DLUISA_COMPUTE_ENABLE_VULKAN=OFF',
        '-DLUISA_COMPUTE_ENABLE_FALLBACK=OFF', '-DLUISA_COMPUTE_ENABLE_REMOTE=OFF',
        '-DLUISA_COMPUTE_DOWNLOAD_OIDN=OFF',
        '-DLUISA_COMPUTE_TVM_INCLUDE_DIR=/Users/mike/.cache/luisa-tile/tvm-pinned.9TMqn1/include',
        '-DLUISA_COMPUTE_TVM_FFI_INCLUDE_DIR=/Users/mike/.cache/luisa-tile/tvm-pinned.9TMqn1/3rdparty/tvm-ffi/include',
        '-DLUISA_COMPUTE_TVM_LIBRARY_DIR=/Users/mike/.cache/luisa-tile/tvm-fragment-build.0ToXYr/lib',
        '-DLUISA_COMPUTE_TVM_FFI_LIBRARY_DIR=/Users/mike/.cache/luisa-tile/tvm-fragment-build.0ToXYr/lib',
        '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON']
save('configure-command.json', dict(argv=argv))
run('configure', argv)
run('full-build', ['cmake', '--build', BUILD, '--parallel', '8'])
assert all(sha(SOURCE / name) == digest for name, digest in hashes.items())
run('full-build-gate', ['cmake', '--build', BUILD, '--parallel', '8'])
ctest = ['ctest', '--test-dir', BUILD, '--output-on-failure', '--no-tests=error',
         '--timeout', '240', '-j', '1']
run('ctest-xir-simd', [*ctest, '-L', 'unit_xir|unit_simd', '--output-junit', HERE / 'xir-simd.xml'],
    junit='xir-simd.xml', expected=80)
run('ctest-tile', [*ctest, '-R', '^test_tile_', '--output-junit', HERE / 'tile.xml'],
    junit='tile.xml', expected=35)
run('ctest-metal-codegen', [*ctest, '-R', '^test_metal_codegen_regressions$',
                          '--output-junit', HERE / 'metal-codegen.xml'],
    junit='metal-codegen.xml', expected=1)
REPORT['source_unchanged'] = all(sha(SOURCE / name) == digest for name, digest in hashes.items())
REPORT['dependency_checkouts_unchanged'] = all(
    git(ROOT / r['path'], 'rev-parse', 'HEAD').decode().strip() == r['checkout_commit']
    for r in records if r['path'] != '.')
assert REPORT['source_unchanged'] and REPORT['dependency_checkouts_unchanged']
REPORT.update(finished_unix=time.time(), passed=True)
save('verification.json', REPORT)
