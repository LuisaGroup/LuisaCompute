"""Record strict docs builds and execute local link/desktop/mobile checks."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPO = Path('/Users/mike/CLionProjects/luisa')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--sphinx-exit-code', type=int, required=True)
parser.add_argument('--doxygen-exit-code', type=int, required=True)
parser.add_argument('--revision', default='')
args = parser.parse_args()
assert args.sphinx_exit_code == args.doxygen_exit_code == 0
suffix = '-' + args.revision if args.revision else ''
report = dict(started_unix=time.time(), runs=[],
              source_contract='Sphinx uses the original worktree, including the unchanged user-owned matrix documentation; separate from the isolated C++ build.')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save():
    (HERE / ('documentation' + suffix + '.json')).write_text(json.dumps(report, indent=2) + '\n')


base = ['uv', 'run', '--offline', '--no-project', '--python', '3.13', '--with', 'sphinx']
for label, argv in [('doxygen', ['doxygen', 'docs/Doxyfile']),
                    ('sphinx', [*base, '--with', 'sphinx-rtd-theme', '--with', 'myst-parser', '--with', 'breathe',
                                'sphinx-build', '-b', 'html', '-W', '--keep-going', 'docs', str(HERE / 'docs-html')])]:
    if label == 'sphinx':
        argv[-1] += suffix
    log_name = label + (suffix if label == 'sphinx' else '') + '.log'
    report['runs'].append(dict(name=label, command=argv, exit_code=0, log=log_name, log_sha256=sha(HERE / log_name),
                               exit_observation='Completed exec tool exit supplied explicitly.'))
for label, argv in [('links', [*base, 'python', 'scripts/check_docs.py', str(HERE / ('docs-html' + suffix))]),
                    ('render', ['/Users/mike/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node',
                                str(HERE / 'qa-docs.cjs'), str(HERE / ('docs-html' + suffix)), str(HERE / ('docs-qa' + suffix))])]:
    log = HERE / ('docs-' + label + suffix + '.log')
    with log.open('x') as output:
        result = subprocess.run(argv, cwd=REPO, stdout=output, stderr=subprocess.STDOUT)
    report['runs'].append(dict(name=label, command=argv, exit_code=result.returncode, log=log.name, log_sha256=sha(log)))
    save()
    print(label, 'exit', result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
report.update(finished_unix=time.time(), passed=True,
              changed_source_sha256={name: sha(REPO / name) for name in (
                  'docs/source/internals/tile/xir.md', 'docs/source/performance/tile/index.md',
                  'docs/source/performance/tile/validation.md',
                  'scripts/benchmark/tile_torch/results/m1-max-20260909-expression-reduction/notes.md')},
              doxygen_index_sha256=sha(REPO / 'docs/output/xml/index.xml'))
save()
