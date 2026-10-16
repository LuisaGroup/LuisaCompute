"""Record a complete build gate and sequential, non-comparative regressions."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
BUILD = HERE / 'build'
report = dict(started_unix=time.time(), source_snapshot_sha256=hashlib.sha256((HERE / 'source-snapshot.json').read_bytes()).hexdigest(),
              source_commit='2634be45d21bc8b2678104dadea99a5a0595d7c4', runs=[], timing_not_comparative=True)
environment = {k: v for k, v in os.environ.items() if not k.startswith(('LUISA_SIMD_', 'LUISA_TILE_BENCH_', 'DYLD_'))}


def save():
    (HERE / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')


def run(label, argv, env=None, junit=None):
    log = HERE / (label + '.log')
    item = dict(name=label, command=list(map(str, argv)), environment=env or {}, started_unix=time.time())
    report['runs'].append(item)
    save()
    with log.open('x') as output:
        result = subprocess.run(argv, cwd=HERE, env=environment | (env or {}), stdout=output, stderr=subprocess.STDOUT)
    item.update(exit_code=result.returncode, finished_unix=time.time(), log=log.name, log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
    if junit and (HERE / junit).exists():
        cases = list(ET.parse(HERE / junit).getroot().iter('testcase'))
        item.update(tests=len(cases), failures=sum(c.find('failure') is not None or c.find('error') is not None for c in cases),
                    skipped=sum(c.find('skipped') is not None for c in cases), junit=junit)
    save()
    print(label, 'exit', result.returncode, 'tests', item.get('tests'), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


run('full-build-gate', ['cmake', '--build', BUILD, '--parallel', '8'])
ctest = ['ctest', '--test-dir', BUILD, '--output-on-failure', '--timeout', '240', '-j', '1']
run('ctest-tile', [*ctest, '-R', '^test_tile_', '--output-junit', HERE / 'tile.xml'], junit='tile.xml')
run('ctest-xir-simd', [*ctest, '-L', 'unit_xir|unit_simd', '--output-junit', HERE / 'xir-simd.xml'], junit='xir-simd.xml')
for width in (1, 2, 4, 8, 16):
    run('expressions-w' + str(width), [BUILD / 'bin/test_tile_xir_runtime', 'simd', 'tile_xir_runtime_fused_expressions*'],
        env=dict(LUISA_SIMD_WARP_WIDTH=str(width), LUISA_SIMD_WORKER_COUNT='1',
                 LUISA_SIMD_ENABLE_FULL_PACKET_SPECIALIZATION='1', LUISA_SIMD_ENABLE_PREDICATED_MEMORY_EFFECTS='1',
                 LUISA_SIMD_ENABLE_COHORT_PRIVATE_ACCESS='1'))
report.update(finished_unix=time.time(), passed=True)
save()
