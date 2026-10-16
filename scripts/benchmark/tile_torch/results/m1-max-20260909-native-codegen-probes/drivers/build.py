"""Build a pinned diagnostic overlay; this is not a production policy."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'source'
BUILD = HERE / 'build'
BASE = Path('/tmp/luisa-next-xir-checkpoint.pgfKld')
ENV = {k: v for k, v in os.environ.items() if not k.startswith(('LUISA_', 'DYLD_'))}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2) + '\n')

baseline = json.loads((HERE / 'baseline-source.json').read_text())
hashes = {str(p.relative_to(SOURCE)): sha(p) for p in sorted(SOURCE.rglob('*')) if p.is_file()}
changed = [name for name, digest in hashes.items() if digest != baseline['source_sha256'].get(name)]
assert set(hashes) == set(baseline['source_sha256'])
assert changed == ['src/backends/simd/simd_compiler.cpp'], changed
save('source-snapshot.json', dict(source_commit=baseline['source_commit'], source_sha256=hashes,
                                 overlay=changed, experimental_only=True))
argv = json.loads((BASE / 'configure-command.json').read_text())['argv']
argv[argv.index('-S') + 1] = str(SOURCE)
argv[argv.index('-B') + 1] = str(BUILD)
report = dict(started_unix=time.time(), source_snapshot_sha256=sha(HERE / 'source-snapshot.json'),
              runs=[], performance_measurement=False)
save('configure-command.json', dict(argv=argv))
for name, command in [('configure', argv), ('full-build', ['cmake', '--build', str(BUILD), '--parallel', '8'])]:
    item = dict(name=name, command=command, started_unix=time.time())
    report['runs'].append(item)
    save('build.json', report)
    with (HERE / (name + '.log')).open('x') as log:
        result = subprocess.run(command, cwd=HERE, env=ENV, stdout=log, stderr=subprocess.STDOUT)
    item.update(exit_code=result.returncode, finished_unix=time.time(), log_sha256=sha(HERE / (name + '.log')))
    save('build.json', report)
    print(name, result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
report['source_unchanged'] = all(sha(SOURCE / name) == digest for name, digest in hashes.items())
assert report['source_unchanged']
report.update(finished_unix=time.time(), passed=True)
save('build.json', report)
