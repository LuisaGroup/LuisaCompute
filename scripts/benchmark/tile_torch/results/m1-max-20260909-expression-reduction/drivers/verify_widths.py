"""Rerun exact-name width checks; zero assertions cannot satisfy this gate."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

HERE = Path(__file__).resolve().parent
report = dict(started_unix=time.time(), runs=[], passed=False,
              supersedes='verification.json expressions-w1/w2/w4/w8/w16: wildcard query selected zero tests; these runs are excluded.')
env = {k: v for k, v in os.environ.items() if not k.startswith(('LUISA_SIMD_', 'LUISA_TILE_BENCH_', 'DYLD_'))}
target = 'tile_xir_runtime_fused_expressions_preserve_tree_and_snapshots'


def save():
    (HERE / 'verification-widths-v2.json').write_text(json.dumps(report, indent=2) + '\n')


for width in (1, 2, 4, 8, 16):
    label = 'expressions-verified-w' + str(width)
    argv = [str(HERE / 'build/bin/test_tile_xir_runtime'), 'simd', target]
    options = dict(LUISA_SIMD_WARP_WIDTH=str(width), LUISA_SIMD_WORKER_COUNT='1',
                   LUISA_SIMD_ENABLE_FULL_PACKET_SPECIALIZATION='1', LUISA_SIMD_ENABLE_PREDICATED_MEMORY_EFFECTS='1',
                   LUISA_SIMD_ENABLE_COHORT_PRIVATE_ACCESS='1')
    log = HERE / (label + '.log')
    item = dict(name=label, argv=argv, environment=options, started_unix=time.time())
    report['runs'].append(item)
    save()
    with log.open('x') as output:
        result = subprocess.run(argv, cwd=HERE, env=env | options, stdout=output, stderr=subprocess.STDOUT)
    text = re.sub(r'\x1b\[[0-9;]*m', '', log.read_text())
    count = re.search(r'\((\d+) asserts in (\d+) tests\)', text)
    # Successful cases are silent in this UT reporter. It prints only skips
    # and the suite aggregate; require exactly one executed case, not a
    # nonexistent per-case PASSED banner.
    skipped = len(re.findall(r'Running "[^"]+"\.\.\. SKIPPED', text))
    nonempty = count is not None and int(count[1]) > 0 and int(count[2]) - skipped == 1 and 'all tests passed' in text
    item.update(exit_code=result.returncode, finished_unix=time.time(), log=log.name,
                assertions=int(count[1]) if count else 0, skipped=skipped, selected_test_passed=nonempty,
                log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
    save()
    print(label, 'exit', result.returncode, 'assertions', item['assertions'], flush=True)
    if result.returncode or not nonempty:
        raise SystemExit(1)
report.update(finished_unix=time.time(), passed=True)
save()
