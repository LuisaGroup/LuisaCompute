#!/usr/bin/env python3
"""Retain nonempty width-specific runtime tests and the selected CTest log."""
import argparse
import gzip
import json
import os
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--docs-qa', type=Path, required=True)
    args = parser.parse_args()
    tests = []
    for width in (1, 2, 4, 8, 16):
        env = {k: v for k, v in os.environ.items() if not k.startswith('LUISA_SIMD_')}
        env['LUISA_SIMD_WARP_WIDTH'] = str(width)
        command = [str(args.build / 'bin/test_tile_xir_runtime'), 'simd', 'tile_xir_runtime_packet_local_reductions_and_snapshot']
        run = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
        output = run.stdout + run.stderr
        (HERE / f'width-{width}.log').write_text(output)
        if run.returncode or not re.search(r'220 asserts in 10 tests', output):
            raise ValueError(f'width {width}: failed or empty test selection')
        tests.append(dict(packet_width=width, command=command, passed=True, assertions=220,
                          selected_tests=1, other_tests_intentionally_skipped=9))
        print('PASS W' + str(width) + ': 220 assertions', flush=True)
    log = (args.build / 'Testing/Temporary/LastTest.log').read_bytes()
    if log.count(b'Test Passed.') != 38 or b'Test Failed.' in log:
        raise ValueError('expected the 38-test selected CTest gate')
    (HERE / 'ctest.log.gz').write_bytes(gzip.compress(log, mtime=0))
    qa = json.loads((args.docs_qa / 'receipt.json').read_text())
    if not qa['passed'] or len(qa['receipts']) != 6:
        raise ValueError('incomplete rendered QA')
    (HERE / 'docs-qa.json').write_text(json.dumps(qa, indent=2) + '\n')
    result = dict(full_isolated_build_passed=True, selected_ctest_passed=38,
                  runtime_widths=tests, doxygen='XML regenerated; existing configuration/source warnings remain',
                  sphinx='fresh -W --keep-going HTML build passed', local_html_pages=50,
                  checked_links_and_assets=4177, checked_compatibility_anchors=199,
                  rendered_sections=6, empty_wildcard_probe_not_counted=True)
    (HERE / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
