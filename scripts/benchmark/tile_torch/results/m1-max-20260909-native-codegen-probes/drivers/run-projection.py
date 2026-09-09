"""Observe diagnostic replay conditions without controlling unrelated apps."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
mode = sys.argv[1]
assert mode in ('capture', 'verify', 'diagnostic')
ENV = {k: v for k, v in os.environ.items() if not k.startswith(('LUISA_', 'DYLD_'))}
python = ['uv', 'run', '--offline', '--no-project', '--python', '3.13', '--with', 'numpy', '--with', 'torch==2.14.0', 'python']
if mode == 'capture':
    command = [*python, str(HERE / 'capture-projection.py')]
else:
    command = [*python, '/tmp/luisa-pointwise.0imY7I/pilot.py', 'verify' if mode == 'verify' else 'replay',
               '--prepared', str(HERE / 'projection-capture'), '--output', str(HERE / ('projection-' + mode))]
report = dict(started_unix=time.time(), command=command, observations=[],
              performance_qualified=False,
              scope='Diagnostic only: concurrent desktop activity; no accepted speedup, parity claim, default promotion or cost calibration from these timings.')
for name, cmd in [('power', ['pmset', '-g', 'batt']), ('thermal', ['pmset', '-g', 'therm'])]:
    p = subprocess.run(cmd, text=True, capture_output=True)
    report[name] = dict(exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr)
def save():
    (HERE / ('projection-' + mode + '-host.json')).write_text(json.dumps(report, indent=2) + '\n')
assert json.loads((HERE / 'projection-build.json').read_text())['passed']
with (HERE / ('projection-' + mode + '.log')).open('x') as log:
    process = subprocess.Popen(command, cwd=HERE, env=ENV, stdout=log, stderr=subprocess.STDOUT)
    while True:
        snapshot = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,pcpu=,comm='], text=True)
        report['observations'].append(dict(unix=time.time(), processes=snapshot))
        save()
        code = process.poll()
        if code is not None:
            break
        time.sleep(5)
report.update(finished_unix=time.time(), exit_code=code)
save()
print(mode, 'exit', code, 'performance-qualified', False, flush=True)
raise SystemExit(code)
