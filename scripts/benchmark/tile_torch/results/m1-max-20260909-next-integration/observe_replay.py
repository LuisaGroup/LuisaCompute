"""Observe process coactivity without controlling any unrelated workload."""
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
    'observations': [],
}
args = ['caffeinate', '-i', 'uv', 'run', '--offline', '--no-project', '--python', '3.13',
        '--with', 'numpy', '--with', 'torch', 'python', '/tmp/luisa-pointwise.0imY7I/pilot.py',
        'replay', '--prepared', '/tmp/luisa-pointwise.0imY7I/capture-final',
        '--output', str(ROOT / 'replay-v1')]
report['command'] = args
with (ROOT / 'replay-v1.log').open('x') as log:
    process = subprocess.Popen(args, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    while True:
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
        report['observations'].append({'unix': time.time(), 'heavy': heavy, 'active_cpu': active})
        (ROOT / 'host-observation.json').write_text(json.dumps(report, indent=2) + '\n')
        code = process.poll()
        if code is not None:
            break
        time.sleep(5)
report['finished_unix'] = time.time()
report['benchmark_exit_code'] = code
report['heavy_activity_seen'] = any(x['heavy'] for x in report['observations'])
(ROOT / 'host-observation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'observations'}, indent=2), flush=True)
raise SystemExit(code)
