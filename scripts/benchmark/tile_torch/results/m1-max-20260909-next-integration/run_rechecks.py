"""Retain the initial failures and verify the fixture/configuration corrections."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
REPO = Path('/Users/mike/CLionProjects/luisa')
BUILD = HERE / 'build'
initial = json.loads((HERE / 'verification.json').read_text())
name = 'src/tests/unit/runtime/test_metal_codegen_regressions.cpp'
assert (REPO / name).read_bytes() == (HERE / 'source' / name).read_bytes()
report = dict(started_unix=time.time(), source_commit=initial['source_commit'],
              source_overlays={name: hashlib.sha256((HERE / 'source' / name).read_bytes()).hexdigest()},
              overlay_scope='One test-only argv guard; no compiler/library/Tile source changes.',
              initial_verification_sha256=hashlib.sha256((HERE / 'verification.json').read_bytes()).hexdigest(),
              full_build_log='full-build-fixed.log',
              full_build_log_sha256=hashlib.sha256((HERE / 'full-build-fixed.log').read_bytes()).hexdigest(),
              retained_initial_failures=[r for r in initial['runs'] if r['exit_code']],
              runs=[r | {'source_phase': 'before test-only Metal argv correction; tested source unchanged'} for r in initial['runs']
                    if r['name'] in ('ctest-tile', 'ctest-xir-simd')])
assert len(report['runs']) == 2 and all(r['exit_code'] == 0 for r in report['runs'])


def save():
    (HERE / 'verification-final.json').write_text(json.dumps(report, indent=2) + '\n')


def run(label, args, env=None, junit=None):
    path = HERE / (label + '.log')
    item = dict(name=label, command=list(map(str, args)), log=path.name, started_unix=time.time())
    clean = {k: v for k, v in os.environ.items() if not k.startswith('LUISA_SIMD_')}
    item['environment'] = env or {}
    item['environment_reset'] = 'Inherited LUISA_SIMD_* removed before applying the recorded test configuration.'
    report['runs'].append(item)
    save()
    with path.open('x') as log:
        result = subprocess.run(args, cwd=HERE, env=clean | (env or {}), stdout=log, stderr=subprocess.STDOUT)
    item.update(exit_code=result.returncode, finished_unix=time.time(), log_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if junit and (HERE / junit).is_file():
        cases = list(ET.parse(HERE / junit).getroot().iter('testcase'))
        item.update(junit=junit, tests=len(cases), failures=sum(c.find('failure') is not None or c.find('error') is not None for c in cases),
                    skipped=sum(c.find('skipped') is not None for c in cases))
    save()
    print(label, 'exit', result.returncode, 'tests', item.get('tests'), flush=True)
    return result.returncode


report['full_build_exit_code'] = run('full-build-final-gate', ['cmake', '--build', str(BUILD), '--parallel', '8'])
save()
if report['full_build_exit_code']:
    raise SystemExit(report['full_build_exit_code'])
common = ['ctest', '--test-dir', str(BUILD), '--output-on-failure', '--timeout', '180', '-j', '1']
run('ctest-metal-fixed', [*common, '-R', '^test_metal_(local_codegen|shared_ptr)$', '--output-junit', str(HERE / 'metal-fixed.xml')], junit='metal-fixed.xml')
run('metal-local-only', [str(BUILD / 'bin/test_metal_codegen_regressions'), 'metal', '--local-only'])
perf = {'LUISA_SIMD_WARP_WIDTH': '8', 'LUISA_SIMD_ENABLE_FULL_PACKET_SPECIALIZATION': '1',
        'LUISA_SIMD_ENABLE_PREDICATED_MEMORY_EFFECTS': '1', 'LUISA_SIMD_ENABLE_COHORT_PRIVATE_ACCESS': '1',
        'LUISA_SIMD_DISABLE_LOAD_REDUCTION_FUSION': '1'}
# Runtime explicitly compiles off and on within the alias fixture. A global
# force-on override would invalidate that fixture's metadata assertions.
run('ctest-runtime-ab', [*common, '-R', '^test_tile_xir_runtime$', '--output-junit', str(HERE / 'runtime-ab.xml')],
    env=perf, junit='runtime-ab.xml')
run('ctest-llm-fused', [*common, '-R', '^test_tile_xir_llm$', '--output-junit', str(HERE / 'llm-fused.xml')],
    env=perf | {'LUISA_SIMD_ENABLE_POINTWISE_FUSION': '1'}, junit='llm-fused.xml')
run('format-final', ['/opt/homebrew/opt/llvm/bin/clang-format', '--dry-run', '--Werror', str(REPO / name)])
run('syntax-final', ['uv', 'run', '--offline', '--no-project', '--python', '3.13', '--with', 'orjson', 'python',
                    str(REPO / 'scripts/check_cpp_syntax.py'), str(HERE / 'source' / name),
                    '--project-root', str(HERE / 'source'), '--compile-commands-dir', str(BUILD),
                    '--clangd', '/opt/homebrew/opt/llvm@21/bin/clangd', '--clang-tidy'])
report['finished_unix'] = time.time()
report['passed'] = all(r['exit_code'] == 0 for r in report['runs'])
save()
raise SystemExit(0 if report['passed'] else 1)
