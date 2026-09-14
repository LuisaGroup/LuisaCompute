"""Record a full-build gate followed by sequential integration checks."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / 'build'
snapshot = json.loads((ROOT / 'snapshot.json').read_text())
report = dict(started_unix=time.time(), source_commit=snapshot['source_commit'],
              full_build_log_sha256=hashlib.sha256((ROOT / 'full-build.log').read_bytes()).hexdigest(), runs=[])


def save():
    (ROOT / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')


def run(label, args, env=None, junit=None):
    path = ROOT / (label + '.log')
    item = dict(name=label, command=list(map(str, args)), log=path.name, started_unix=time.time())
    if env:
        item['environment'] = env
    report['runs'].append(item)
    save()
    with path.open('x') as log:
        completed = subprocess.run(args, cwd=ROOT, env=os.environ | (env or {}), stdout=log, stderr=subprocess.STDOUT)
    item.update(exit_code=completed.returncode, finished_unix=time.time(), log_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if junit and (ROOT / junit).is_file():
        result = ET.parse(ROOT / junit).getroot()
        cases = list(result.iter('testcase'))
        item.update(junit=junit, tests=len(cases),
                    failures=sum(c.find('failure') is not None or c.find('error') is not None for c in cases),
                    skipped=sum(c.find('skipped') is not None for c in cases))
    save()
    print(label, 'exit', completed.returncode, 'tests', item.get('tests'), flush=True)
    return completed.returncode


report['full_build_exit_code'] = run('full-build-gate', ['cmake', '--build', str(BUILD), '--parallel', '8'])
save()
if report['full_build_exit_code']:
    raise SystemExit(report['full_build_exit_code'])
common = ['ctest', '--test-dir', str(BUILD), '--output-on-failure', '--timeout', '180', '-j', '1']
run('ctest-tile', [*common, '-R', '^test_tile_', '--output-junit', str(ROOT / 'tile.xml')], junit='tile.xml')
run('ctest-xir-simd', [*common, '-L', 'unit_xir|unit_simd', '--output-junit', str(ROOT / 'xir-simd.xml')], junit='xir-simd.xml')
run('ctest-metal', [*common, '-R', '^test_metal_(local_codegen|shared_ptr)$', '--output-junit', str(ROOT / 'metal.xml')], junit='metal.xml')
run('ctest-pointwise', [*common, '-R', '^test_tile_xir_(runtime|llm)$', '--output-junit', str(ROOT / 'pointwise.xml')],
    env={'LUISA_SIMD_WARP_WIDTH': '8', 'LUISA_SIMD_ENABLE_POINTWISE_FUSION': '1',
         'LUISA_SIMD_ENABLE_FULL_PACKET_SPECIALIZATION': '1', 'LUISA_SIMD_ENABLE_PREDICATED_MEMORY_EFFECTS': '1',
         'LUISA_SIMD_ENABLE_COHORT_PRIVATE_ACCESS': '1', 'LUISA_SIMD_DISABLE_LOAD_REDUCTION_FUSION': '1'},
    junit='pointwise.xml')
report['finished_unix'] = time.time()
report['passed'] = all(r['exit_code'] == 0 for r in report['runs'])
save()
raise SystemExit(0 if report['passed'] else 1)
