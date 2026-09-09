"""Fixed replay after a bounded quiet-window check; preserve all coactivity."""
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = Path('/Users/mike/CLionProjects/luisa')
HEAVY = ('psycles_render', '/ninja', '/cc1', '/clang', 'benchmark_tile', 'test_tile_xir')


def command(*args):
    return subprocess.check_output(args, cwd=REPO, text=True).strip()


report = {
    'started_unix': time.time(),
    'head': command('git', 'rev-parse', 'HEAD'),
    'origin_next': command('git', 'rev-parse', 'origin/next'),
    'sampling_seconds': 5,
    'sampling_caveat': 'Read-only process samples, not proof of exclusive CPU/GPU use; short activity between samples can be missed.',
    'heavy_patterns': HEAVY,
    'quiet_preflight_seconds': 60,
    'maximum_preflight_seconds': 600,
    'preflight': [],
    'observations': [],
}


def observe():
    snapshot = command('ps', '-axo', 'pid=,ppid=,pcpu=,comm=')
    heavy, active = [], []
    for line in snapshot.splitlines():
        fields = line.split(None, 3)
        if len(fields) != 4:
            continue
        if any(pattern in fields[3] for pattern in HEAVY):
            heavy.append(line.strip())
        if float(fields[2]) >= 10:
            active.append(line.strip())
    return {'unix': time.time(), 'heavy': heavy, 'active_cpu': active}


def save():
    (ROOT / 'host-observation-v2.json').write_text(json.dumps(report, indent=2) + '\n')


quiet_since = time.monotonic()
deadline = quiet_since + report['maximum_preflight_seconds']
while True:
    item = observe()
    report['preflight'].append(item)
    if item['heavy']:
        quiet_since = time.monotonic()
    save()
    if time.monotonic() - quiet_since >= report['quiet_preflight_seconds']:
        break
    if time.monotonic() >= deadline:
        raise RuntimeError('No eligible preflight window; native replay was not started.')
    time.sleep(5)

args = ['caffeinate', '-i', 'uv', 'run', '--offline', '--no-project', '--python', '3.13',
        '--with', 'numpy', '--with', 'torch', 'python', '/tmp/luisa-pointwise.0imY7I/pilot.py',
        'replay', '--prepared', '/tmp/luisa-pointwise.0imY7I/capture-final',
        '--output', str(ROOT / 'replay-v2')]
report['command'] = args
report['replay_started_unix'] = time.time()
with (ROOT / 'replay-v2.log').open('x') as log:
    process = subprocess.Popen(args, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    while True:
        report['observations'].append(observe())
        save()
        code = process.poll()
        if code is not None:
            break
        time.sleep(5)
report['finished_unix'] = time.time()
report['benchmark_exit_code'] = code
report['heavy_activity_seen'] = any(x['heavy'] for x in report['observations'])
save()
print(json.dumps({k: v for k, v in report.items() if k not in ('preflight', 'observations')}, indent=2), flush=True)
raise SystemExit(code)
