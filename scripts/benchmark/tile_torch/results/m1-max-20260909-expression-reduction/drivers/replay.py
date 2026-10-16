"""Observe a bounded quiet window and a fixed native-entry comparison cohort."""
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
COHORT = int(sys.argv[1])
assert COHORT in (1, 2)
assert json.loads((HERE / 'verification.json').read_text()).get('passed')
assert json.loads((HERE / 'verification-widths-v2.json').read_text()).get('passed')
OUTPUT = HERE / ('replay-' + str(COHORT))
OBSERVATION = HERE / ('host-replay-' + str(COHORT) + '.json')
assert not OUTPUT.exists() and not OBSERVATION.exists()
HEAVY = ('psycles_render', '/ninja', '/clang', '/cc1', '/test_', 'benchmark_tile', 'doxygen', 'sphinx-build', 'xcodebuild')
report = dict(started_unix=time.time(), cohort=COHORT, quiet_seconds=60, maximum_preflight_seconds=180,
              sample_seconds=2, preflight=[], observations=[],
              qualification_caveat='Read-only process samples cannot establish exclusive hardware use or catch every short task. No observed render/build/test activity is admitted.',
              heavy_patterns=HEAVY, performance_qualified=False)


def save():
    OBSERVATION.write_text(json.dumps(report, indent=2) + '\n')


def observe():
    text = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,pcpu=,comm='], text=True)
    heavy, active = [], []
    for line in text.splitlines():
        parts = line.split(None, 3)
        if len(parts) != 4:
            continue
        if any(pattern in parts[3] for pattern in HEAVY):
            heavy.append(line.strip())
        if float(parts[2]) >= 10:
            active.append(line.strip())
    return dict(unix=time.time(), heavy=heavy, active_cpu=active)


for label, argv in [('power', ['pmset', '-g', 'batt']), ('thermal', ['pmset', '-g', 'therm'])]:
    result = subprocess.run(argv, text=True, capture_output=True)
    report[label] = dict(argv=argv, exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
quiet = time.monotonic()
deadline = quiet + report['maximum_preflight_seconds']
while True:
    item = observe()
    report['preflight'].append(item)
    if item['heavy']:
        quiet = time.monotonic()
    save()
    if time.monotonic() - quiet >= report['quiet_seconds']:
        break
    if time.monotonic() >= deadline:
        report.update(finished_unix=time.time(), exclusion='No eligible preflight window; replay not started.')
        save()
        raise SystemExit(2)
    time.sleep(2)
argv = ['caffeinate', '-i', 'uv', 'run', '--offline', '--no-project', '--python', '3.13',
        '--with', 'numpy', '--with', 'torch==2.14.0', 'python', '/tmp/luisa-pointwise.0imY7I/pilot.py',
        'replay', '--prepared', str(HERE / 'capture'), '--output', str(OUTPUT)]
report.update(command=argv, replay_started_unix=time.time())
with (HERE / ('replay-' + str(COHORT) + '.log')).open('x') as log:
    process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
    while True:
        report['observations'].append(observe())
        save()
        code = process.poll()
        if code is not None:
            break
        time.sleep(2)
report.update(finished_unix=time.time(), replay_exit_code=code,
              performance_qualified=code == 0 and not any(item['heavy'] for item in report['observations']))
if not report['performance_qualified']:
    report['exclusion'] = 'Replay failure or observed concurrent render/build/test activity; do not update rankings.'
save()
print('Cohort', COHORT, 'exit', code, 'uncontended-process-screen', report['performance_qualified'], flush=True)
raise SystemExit(code)
