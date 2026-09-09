"""Qualify integration checks while retaining the failed earlier invocations."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
REPO = Path('/Users/mike/CLionProjects/luisa')
BUILD = HERE / 'build'
prior = json.loads((HERE / 'verification-final.json').read_text())
name = 'src/tests/unit/runtime/test_metal_codegen_regressions.cpp'
assert (REPO / name).read_bytes() == (HERE / 'source' / name).read_bytes()
assert hashlib.sha256((REPO / name).read_bytes()).hexdigest() == prior['source_overlays'][name]
report = dict(started_unix=time.time(), source_commit=prior['source_commit'],
              source_overlays=prior['source_overlays'], overlay_scope=prior['overlay_scope'],
              prior_receipts={n: hashlib.sha256((HERE / n).read_bytes()).hexdigest()
                              for n in ('verification.json', 'verification-final.json')},
              full_build_log=prior['full_build_log'], full_build_log_sha256=prior['full_build_log_sha256'],
              full_build_exit_code=prior['full_build_exit_code'],
              retained_failed_checks=prior['retained_initial_failures'] + [r for r in prior['runs'] if r['exit_code']],
              runs=[r for r in prior['runs'] if r['exit_code'] == 0])


def save():
    (HERE / 'verification-qualified.json').write_text(json.dumps(report, indent=2) + '\n')


def run(label, args, env=None, junit=None):
    path = HERE / (label + '.log')
    item = dict(name=label, command=list(map(str, args)), log=path.name, started_unix=time.time())
    clean = {k: v for k, v in os.environ.items() if not k.startswith('LUISA_SIMD_')}
    item['environment'] = env or {}
    item['environment_reset'] = 'Inherited LUISA_SIMD_* removed; no forced Tile fusion override for Runtime A/B fixtures.'
    report['runs'].append(item)
    save()
    with path.open('x') as log:
        result = subprocess.run(args, cwd=HERE, env=clean | (env or {}), stdout=log, stderr=subprocess.STDOUT)
    item.update(exit_code=result.returncode, finished_unix=time.time(), log_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if junit:
        cases = list(ET.parse(HERE / junit).getroot().iter('testcase'))
        item.update(junit=junit, tests=len(cases), failures=sum(c.find('failure') is not None or c.find('error') is not None for c in cases),
                    skipped=sum(c.find('skipped') is not None for c in cases))
    save()
    print(label, 'exit', result.returncode, 'tests', item.get('tests'), flush=True)


common = ['ctest', '--test-dir', str(BUILD), '--output-on-failure', '--timeout', '180', '-j', '1']
perf = {'LUISA_SIMD_WARP_WIDTH': '8', 'LUISA_SIMD_ENABLE_FULL_PACKET_SPECIALIZATION': '1',
        'LUISA_SIMD_ENABLE_PREDICATED_MEMORY_EFFECTS': '1', 'LUISA_SIMD_ENABLE_COHORT_PRIVATE_ACCESS': '1'}
# Both pointwise and load/reduction fusion are explicitly toggled by fixtures.
# The earlier force-on/force-off environment invalidated metadata A/B checks.
run('ctest-runtime-clean', [*common, '-R', '^test_tile_xir_runtime$', '--output-junit', str(HERE / 'runtime-clean.xml')],
    env=perf, junit='runtime-clean.xml')
run('format-changed', ['/opt/homebrew/opt/llvm/bin/clang-format', '--dry-run', '--Werror', '--lines=369:380', str(REPO / name)])

# Do not reformat unrelated upstream lines to hide the inherited full-file gate.
# Compare its exact diagnostic locations to the pristine committed file.
original = subprocess.check_output(['git', 'show', report['source_commit'] + ':' + name], cwd=REPO)
args = ['/opt/homebrew/opt/llvm/bin/clang-format', '--dry-run', '--Werror', '--assume-filename=' + str(REPO / name)]
baseline = subprocess.run(args, input=original, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
with (HERE / 'format-pristine.log').open('xb') as f:
    f.write(baseline.stdout)
pattern = r':(\d+):(\d+): error: code should be clang-formatted'
positions = re.findall(pattern, baseline.stdout.decode())
previous = re.findall(pattern, (HERE / 'format-final.log').read_text())
report['inherited_formatting'] = dict(command=args, input_commit=report['source_commit'], input_path=name,
    input_sha256=hashlib.sha256(original).hexdigest(), exit_code=baseline.returncode,
    log='format-pristine.log', log_sha256=hashlib.sha256(baseline.stdout).hexdigest(),
    diagnostic_positions=positions, diagnostics_identical_to_changed_file=positions == previous,
    scoped_gate='Only changed lines are required to pass; full-file failures are retained, not called passing.')
report['finished_unix'] = time.time()
report['passed'] = (all(r['exit_code'] == 0 for r in report['runs']) and
                    all(r.get('failures', 0) == r.get('skipped', 0) == 0 for r in report['runs']) and
                    baseline.returncode == 1 and len(positions) == 7 and positions == previous)
save()
print('qualified', report['passed'], 'inherited formatting diagnostics', len(positions), flush=True)
raise SystemExit(0 if report['passed'] else 1)
