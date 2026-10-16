"""Build the existing documentation hierarchy with strict checks."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path('/Users/mike/CLionProjects/luisa')
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / 'docs-html-v3'
assert not OUTPUT.exists()
previous = json.loads((HERE / 'documentation.json').read_text())
assert previous['runs'][-1]['exit_code'] == -10
report = dict(started_unix=time.time(), code_commit='2cfc804932a014cd7a07eb3896166d53db32b3eb', runs=[],
              previous_attempts=[previous],
              retry_scope='Unchanged Doxyfile and input. Previous Doxygen 1.18.0 run crashed with SIGBUS in commentscanYYlex; retained here, not counted as success. No input exclusions or warning relaxation.')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run(name, argv):
    argv = list(map(str, argv))
    item = dict(name=name, command=argv, log='docs-v3-' + name + '.log', started_unix=time.time())
    report['runs'].append(item)
    with (HERE / item['log']).open('x') as log:
        result = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    item.update(exit_code=result.returncode, finished_unix=time.time(), log_sha256=sha(HERE / item['log']))
    (HERE / 'documentation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(name, 'exit', result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)

archive = 'scripts/benchmark/tile_torch/results/m1-max-20260909-native-codegen-probes/'
files = ['docs/Doxyfile', 'docs/conf.py', 'docs/source/performance/tile/index.md',
         'docs/source/performance/tile/results.md', 'docs/source/performance/tile/validation.md',
         archive + 'notes.md', archive + 'tables.md', archive + 'freeze.py', archive + 'provenance.json']
report['source_sha256'] = {p: sha(ROOT / p) for p in files}
run('doxygen', ['doxygen', 'docs/Doxyfile'])
uv = ['uv', 'run', '--offline', '--no-project', '--python', '3.13', '--with', 'sphinx',
      '--with', 'sphinx-rtd-theme', '--with', 'myst-parser', '--with', 'breathe']
run('sphinx', [*uv, 'sphinx-build', '-b', 'html', '-W', '--keep-going', 'docs', OUTPUT])
run('links', [*uv, 'python', 'scripts/check_docs.py', OUTPUT])
report['source_unchanged'] = all(sha(ROOT / p) == digest for p, digest in report['source_sha256'].items())
assert report['source_unchanged']
report.update(passed=True, finished_unix=time.time())
(HERE / 'documentation.json').write_text(json.dumps(report, indent=2) + '\n')
