"""Strictly build the integrated documentation and check local references."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path('/Users/mike/CLionProjects/luisa')
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / 'docs-html'
assert not OUTPUT.exists()
report = dict(started_unix=time.time(), code_commit='bc7b1df1f4e8785d50736169a2a7bde3023b231e', runs=[])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(name, argv):
    argv = list(map(str, argv))
    item = dict(name=name, command=argv, log=name + '.log', started_unix=time.time())
    report['runs'].append(item)
    with (HERE / item['log']).open('x') as log:
        result = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    item.update(exit_code=result.returncode, finished_unix=time.time(), log_sha256=sha(HERE / item['log']))
    (HERE / 'documentation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(name, 'exit', result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


files = ['docs/Doxyfile', 'docs/conf.py', 'docs/source/performance/validation.md',
         'docs/source/performance/tile/validation.md',
         'scripts/benchmark/tile_torch/results/m1-max-20260909-next-integration/xir-followup/notes.md']
report['source_sha256'] = {p: sha(ROOT / p) for p in files}
run('doxygen', ['doxygen', 'docs/Doxyfile'])
uv = ['uv', 'run', '--offline', '--no-project', '--python', '3.13', '--with', 'sphinx',
      '--with', 'sphinx-rtd-theme', '--with', 'myst-parser', '--with', 'breathe']
run('sphinx', [*uv, 'sphinx-build', '-b', 'html', '-W', '--keep-going', 'docs', OUTPUT])
run('docs-links', [*uv, 'python', 'scripts/check_docs.py', OUTPUT])
report['source_unchanged'] = all(sha(ROOT / p) == h for p, h in report['source_sha256'].items())
assert report['source_unchanged']
report.update(passed=True, finished_unix=time.time())
(HERE / 'documentation.json').write_text(json.dumps(report, indent=2) + '\n')
