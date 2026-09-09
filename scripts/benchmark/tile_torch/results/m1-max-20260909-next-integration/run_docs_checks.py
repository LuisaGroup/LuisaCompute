"""Record observed documentation builds and execute link/render checks."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPO = Path('/Users/mike/CLionProjects/luisa')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--doxygen-exit-code', type=int, required=True)
parser.add_argument('--initial-sphinx-exit-code', type=int, required=True)
parser.add_argument('--sphinx-exit-code', type=int, required=True)
args = parser.parse_args()
assert (args.doxygen_exit_code, args.initial_sphinx_exit_code, args.sphinx_exit_code) == (0, 2, 0)
report = dict(started_unix=time.time(),
              source_contract='Documentation is rendered from the original worktree, including its preserved user-edited matrix reference; this is separate from the isolated C++ build.',
              runs=[], retained_failures=[])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save():
    (HERE / 'documentation.json').write_text(json.dumps(report, indent=2) + '\n')


base = ['uv', 'run', '--offline', '--no-project', '--python', '3.13', '--with', 'sphinx']
build = [*base, '--with', 'sphinx-rtd-theme', '--with', 'myst-parser', '--with', 'breathe',
         'sphinx-build', '-b', 'html', '-W', '--keep-going', 'docs']
for name, command, code, destination in (
    ('doxygen.log', ['doxygen', 'docs/Doxyfile'], args.doxygen_exit_code, 'runs'),
    ('sphinx.log', [*build, str(HERE / 'docs-html')], args.initial_sphinx_exit_code, 'retained_failures'),
    ('sphinx-v2.log', [*build, str(HERE / 'docs-html-v2')], args.sphinx_exit_code, 'runs'),
):
    report[destination].append(dict(name=name.removesuffix('.log'), command=command,
        exit_code=code, log=name, log_sha256=sha(HERE / name),
        exit_observation='Completed exec tool terminal status supplied explicitly; commands ran before this supervisor.'))


def run(name, command):
    log = HERE / (name + '.log')
    item = dict(name=name, command=command, log=log.name, started_unix=time.time())
    report['runs'].append(item)
    save()
    with log.open('x') as f:
        result = subprocess.run(command, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    item.update(exit_code=result.returncode, finished_unix=time.time(), log_sha256=sha(log))
    save()
    print(name, 'exit', result.returncode, flush=True)


run('docs-links', [*base, 'python', 'scripts/check_docs.py', str(HERE / 'docs-html-v2')])
run('docs-render', ['/Users/mike/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node',
                    str(HERE / 'qa-docs.cjs'), str(HERE / 'docs-html-v2'), str(HERE / 'docs-qa')])
names = set(subprocess.check_output(['git', 'ls-files', 'docs'], cwd=REPO, text=True).splitlines())
names.add('docs/source/performance/validation.md')
names.update(('scripts/check_docs.py', 'scripts/benchmark/tile_torch/results/m1-max-20260909-next-integration/notes.md'))
report['source_sha256'] = {name: sha(REPO / name) for name in sorted(names) if (REPO / name).is_file()}
report['doxygen_index_sha256'] = sha(REPO / 'docs/output/xml/index.xml')
report['finished_unix'] = time.time()
report['passed'] = all(r['exit_code'] == 0 for r in report['runs'])
save()
raise SystemExit(0 if report['passed'] else 1)
