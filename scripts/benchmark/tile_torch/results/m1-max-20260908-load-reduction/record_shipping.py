#!/usr/bin/env python3
"""Record the default-off policy and compare actual shipping captures exactly."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--isolated', type=Path, required=True)
    args = parser.parse_args()
    measured = json.loads((HERE / 'provenance.json').read_text())
    sources = {}
    for file in measured['tested_source_sha256']:
        data = (ROOT / file).read_bytes()
        if data != (args.isolated / 'source' / file).read_bytes():
            raise ValueError('shipping source differs from tested overlay: ' + file)
        sources[file] = digest(data)
    for name in ('lower.cpp', 'planner.cpp', 'representation.h'):
        file = 'src/tile/bridge/xir/' + name
        if sources[file] != measured['tested_source_sha256'][file]:
            raise ValueError('optimization implementation changed after measurement')
    for file in ('lower.h', 'planner.h'):
        if 'bool enable_load_reduction_fusion{false};' not in (ROOT / 'include/luisa/tile/bridge/xir' / file).read_text():
            raise ValueError('shipping default is not disabled')
    report = json.loads((args.capture / 'results.json').read_text())
    if not report['capture_only'] or not report['closure_unchanged'] or len(report['results']) != 8:
        raise ValueError('incomplete shipping capture')
    comparisons = []
    for row in report['results']:
        rows, columns = row['dimensions']
        name = f'rmsnorm-{rows}x{columns}-r0-l{row["local_lanes"]}-v{int(row["load_reduction_fusion"])}'
        current = args.capture / name
        previous = HERE / 'capture-final' / name
        objects = list((current / 'object').glob('*.o'))
        if not row['valid'] or len(objects) != 1:
            raise ValueError('invalid shipping capture: ' + name)
        llvm = (current / 'kernel.ll').read_bytes()
        obj = objects[0].read_bytes()
        if llvm != gzip.decompress((previous / 'kernel.ll.gz').read_bytes()) or obj != gzip.decompress((previous / 'kernel.o.gz').read_bytes()):
            raise ValueError('actual shipping code differs from measured code: ' + name)
        comparisons.append(dict(case=name, llvm_sha256=digest(llvm), object_sha256=digest(obj), byte_identical=True))
    ctest = (args.isolated / 'build/Testing/Temporary/LastTest.log').read_bytes()
    if len(re.findall(rb'^Test Passed\.$', ctest, re.M)) != 38 or b'Test Failed.' in ctest:
        raise ValueError('shipping tests incomplete')
    (HERE / 'shipping-ctest.log').write_bytes(ctest)
    (HERE / 'shipping-capture.json').write_text(json.dumps(report, indent=2) + '\n')
    patch = subprocess.check_output(['git', 'diff', 'ba8bc18cc', '--', *sources], cwd=ROOT)
    (HERE / 'shipping.patch.gz').write_bytes(gzip.compress(patch, mtime=0))
    result = dict(default_load_reduction_fusion=False, tested_source_sha256=sources,
                  previous_measured_source='provenance.json', optimization_implementation_unchanged=True,
                  exact_actual_code_comparisons=comparisons, shipping_ctest_passed=38,
                  timing_inclusion='shipping smoke is excluded from performance statistics')
    (HERE / 'shipping.json').write_text(json.dumps(result, indent=2) + '\n')
    print('PASS: 38 shipping CTests and 8 byte-identical shipping LLVM/ORC captures')


if __name__ == '__main__':
    main()
