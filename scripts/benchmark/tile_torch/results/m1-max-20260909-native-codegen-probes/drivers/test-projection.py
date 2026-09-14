"""Force the diagnostic projection through the existing broad regression set."""
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
assert json.loads((HERE / 'projection-build.json').read_text())['passed']
env = {k: v for k, v in os.environ.items() if not k.startswith(('LUISA_', 'DYLD_'))}
env['LUISA_SIMD_EXPERIMENT_PROJECT_INTEGER_LANES'] = '1'
report = dict(started_unix=time.time(), experimental_projection_forced=True, runs=[])
def save():
    (HERE / 'projection-tests.json').write_text(json.dumps(report, indent=2) + '\n')
for name, selector, expected in [('xir-simd', ['-L', 'unit_xir|unit_simd'], 80), ('tile', ['-R', '^test_tile_'], 35)]:
    xml = HERE / ('projection-' + name + '.xml')
    command = ['ctest', '--test-dir', str(HERE / 'build'), '--output-on-failure', '--no-tests=error',
               '--timeout', '240', '-j', '1', *selector, '--output-junit', str(xml)]
    row = dict(name=name, command=command, started_unix=time.time())
    report['runs'].append(row)
    save()
    with (HERE / ('projection-' + name + '.log')).open('x') as log:
        result = subprocess.run(command, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
    cases = list(ET.parse(xml).getroot().iter('testcase'))
    row.update(finished_unix=time.time(), exit_code=result.returncode, tests=len(cases),
               failures=sum(c.find('failure') is not None or c.find('error') is not None for c in cases),
               skipped=sum(c.find('skipped') is not None for c in cases))
    save()
    print(name, row, flush=True)
    assert result.returncode == 0 and len(cases) == expected and row['failures'] == 0 and row['skipped'] == 0
report.update(finished_unix=time.time(), passed=True)
save()
