"""Finalize the integration gates with the registered Metal CTest name."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = Path('/Users/mike/CLionProjects/luisa')
SOURCE = HERE / 'source'
BUILD = HERE / 'build'
ENV = {k: v for k, v in os.environ.items() if not k.startswith(('LUISA_', 'DYLD_'))}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def junit_counts(name):
    cases = list(ET.parse(HERE / name).getroot().iter('testcase'))
    return dict(tests=len(cases), failures=sum(
        c.find('failure') is not None or c.find('error') is not None for c in cases),
        skipped=sum(c.find('skipped') is not None for c in cases))


initial = json.loads((HERE / 'verification.json').read_text())
accepted = []
for name, expected in [('configure', None), ('full-build', None), ('full-build-gate', None),
                       ('ctest-xir-simd', 80), ('ctest-tile', 35)]:
    run = next(r for r in initial['runs'] if r['name'] == name)
    assert run['exit_code'] == 0 and sha(HERE / run['log']) == run['log_sha256'], run
    if expected:
        assert junit_counts(run['junit']) == dict(tests=expected, failures=0, skipped=0), run
        assert sha(HERE / run['junit']) == run['junit_sha256'], run
    accepted.append(run)
assert 'No tests were found' in (HERE / 'ctest-metal-codegen.log').read_text()
report = dict(source_commit=initial['source_commit'], next_commit=initial['next_commit'],
              source_snapshot_sha256=initial['source_snapshot_sha256'],
              started_unix=time.time(), runs=accepted, performance_measurement=False,
              excluded_invocation=dict(record='verification.json', record_sha256=sha(HERE / 'verification.json'),
                  log='ctest-metal-codegen.log', log_sha256=sha(HERE / 'ctest-metal-codegen.log'),
                  reason='The executable name was used as the CTest name; --no-tests=error rejects this empty selection.'))


def save():
    (HERE / 'verification-final.json').write_text(json.dumps(report, indent=2) + '\n')


def run(name, argv, junit=None, expected=None, allowed=(0,)):
    argv = list(map(str, argv))
    item = dict(name=name, command=argv, started_unix=time.time(), log=name + '.log')
    report['runs'].append(item)
    save()
    with (HERE / item['log']).open('x') as output:
        result = subprocess.run(argv, cwd=ROOT, env=ENV, stdout=output, stderr=subprocess.STDOUT)
    item.update(exit_code=result.returncode, finished_unix=time.time(), log_sha256=sha(HERE / item['log']))
    if junit and (HERE / junit).exists():
        item.update(junit_counts(junit), junit=junit, junit_sha256=sha(HERE / junit))
    save()
    assert result.returncode in allowed, item
    if junit:
        assert item['tests'] == expected and item['failures'] == 0 and item['skipped'] == 0, item
    print(name, 'exit', result.returncode, 'tests', item.get('tests'), flush=True)


run('full-build-recheck', ['cmake', '--build', BUILD, '--parallel', '8'])
run('ctest-metal-local-codegen', ['ctest', '--test-dir', BUILD, '--output-on-failure', '--no-tests=error',
    '-R', '^test_metal_local_codegen$', '--output-junit', HERE / 'metal-local-codegen.xml'],
    junit='metal-local-codegen.xml', expected=1)
fixture = 'src/tests/unit/runtime/test_metal_codegen_regressions.cpp'
run('syntax-metal-fixture', ['uv', 'run', '--offline', '--no-project', '--python', '3.13', '--with', 'orjson',
    'python', 'scripts/check_cpp_syntax.py', '--project-root', SOURCE, '--compile-commands-dir', BUILD,
    '--clangd', '/opt/homebrew/opt/llvm@21/bin/clangd', '--clang-tidy', SOURCE / fixture])
run('format-metal-fixture', ['/opt/homebrew/opt/llvm@21/bin/clang-format', '--dry-run', '--Werror', SOURCE / fixture],
    allowed=(1,))
assert (HERE / 'format-metal-fixture.log').read_text().count('[-Wclang-format-violations]') == 7
assert (SOURCE / fixture).read_bytes() == subprocess.check_output(
    ['git', 'show', report['next_commit'] + ':' + fixture], cwd=ROOT)
run('format-merged-lines', ['/opt/homebrew/opt/llvm@21/bin/clang-format', '--dry-run', '--Werror',
    '--lines=370:375', SOURCE / fixture])
run('preserved-worktree', ['shasum', '-a', '256', '--check',
    '/tmp/luisa-pointwise-native.1P87uU/premerge-user-sha256.txt'])
snapshot = json.loads((HERE / 'source-snapshot.json').read_text())
assert sha(HERE / 'source-snapshot.json') == report['source_snapshot_sha256']
report['source_unchanged'] = all(sha(SOURCE / p) == h for p, h in snapshot['source_sha256'].items())
pins = json.loads((HERE / 'pinned-repositories.json').read_text())
report['dependency_checkouts_unchanged'] = all(subprocess.check_output(
    ['git', '-C', str(ROOT / p['path']), 'rev-parse', 'HEAD']).decode().strip() == p['checkout_commit']
    for p in pins if p['path'] != '.')
assert report['source_unchanged'] and report['dependency_checkouts_unchanged']
report.update(passed=True, finished_unix=time.time(),
              quality_caveat='No syntax errors; one inherited unused-include warning and seven inherited full-file format diagnostics. Changed lines pass.')
save()
