#!/usr/bin/env python3
"""Fixed-mapping on/off trial of use-site private index equality.

Runtime timing includes dispatch. Actual ORC captures feed a separate native
entry replay; no JIT or Python execution is included in that replay's timer.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from compare_llm import reference, validate_output

OPERATIONS = ('rmsnorm', 'layernorm', 'masked_softmax', 'swiglu', 'gelu_residual', 'rope')
SHAPES = ((3, 63), (17, 65), (64, 256), (129, 769), (257, 1538), (1024, 4097), (17, 16384), (137, 1023), (513, 2051))
NATIVE_SHAPES = ((17, 65), (257, 1538), (1024, 4097), (129, 768), (137, 1023), (513, 2051))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cases(suite):
    if suite == 'capture':
        return [('rmsnorm', s) for s in NATIVE_SHAPES]
    rows = [(op, (m, n + (n % 2 if op == 'rope' else 0))) for op in OPERATIONS for m, n in SHAPES]
    return rows + [('attention', s) for s in ((1, 4, 2, 16, 32, 16, 16), (1, 4, 1, 64, 128, 32, 32))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('broad', 'capture'), required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    closure = [binary] + sorted(p for p in binary.parent.glob('libluisa-*') if p.is_file())
    hashes = {str(p): digest(p) for p in closure}
    capture = args.suite == 'capture'
    report = dict(suite=args.suite, capture_only=capture, metric='runtime_e2e_synchronized_host_wall_us',
                  platform=platform.platform(), started_unix=time.time(),
                  source_runner_sha256=digest(Path(__file__)), binary_closure_sha256=hashes, results=[])
    for op, dims in cases(args.suite):
        variants = list(itertools.product((8,) if capture else ((1,) if op == 'attention' else (1, 8)), (False, True)))
        orders = [variants] if capture else [variants, list(reversed(variants))]
        for round_id, order in enumerate(orders):
            for local, enabled in order:
                directory = args.output / (op + '-' + 'x'.join(map(str, dims)) + f'-r{round_id}-l{local}-p{int(enabled)}')
                directory.mkdir()
                output = directory / 'output.f32'
                env = {k: v for k, v in os.environ.items() if not k.startswith(('LUISA_SIMD_', 'LUISA_TILE_BENCH_', 'DYLD_'))}
                env.update(LUISA_SIMD_WARP_WIDTH='8', LUISA_SIMD_WORKER_COUNT='1' if capture else '8',
                           LUISA_TILE_BENCH_XIR_LOCAL_LANES=str(local), LUISA_TILE_BENCH_XIR_BLOCK_SIZE='32',
                           LUISA_TILE_BENCH_XIR_BLOCKS_PER_TASK='0', LUISA_SIMD_DISABLE_LOAD_REDUCTION_FUSION='1',
                           LUISA_SIMD_ENABLE_FULL_PACKET_SPECIALIZATION='1',
                           LUISA_SIMD_ENABLE_PREDICATED_MEMORY_EFFECTS='1',
                           LUISA_SIMD_ENABLE_COHORT_PRIVATE_ACCESS='1')
                if not enabled:
                    env['LUISA_SIMD_DISABLE_COHORT_PRIVATE_ACCESS'] = '1'
                if capture:
                    env.update(LUISA_TILE_BENCH_DUMP_SOURCE=str(directory / 'kernel.ll'),
                               LUISA_SIMD_DUMP_ASSEMBLY_DIR=str(directory / 'object'))
                command = [str(binary), 'llm', op, ','.join(map(str, dims)), '1', '1', '7', '20', '75', str(output)]
                result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=180)
                (directory / 'stdout.log').write_text(result.stdout)
                (directory / 'stderr.log').write_text(result.stderr)
                row = dict(operation=op, dimensions=list(dims), round=round_id, order=order,
                           local_lanes=local, cohort_private_access=enabled,
                           command=command, environment={k: v for k, v in env.items() if k.startswith('LUISA_')},
                           returncode=result.returncode, directory=str(directory), valid=False)
                try:
                    if result.returncode:
                        raise ValueError(result.stderr + result.stdout[-1500:])
                    measurement = json.loads(result.stdout)
                    inputs = [Path(str(output) + f'.input{i}.f32') for i in range(len(measurement['input_shapes']))]
                    arrays = [np.fromfile(p, dtype=np.float32).reshape(s) for p, s in zip(inputs, measurement['input_shapes'])]
                    expected = reference(op, tuple(dims), arrays)
                    check = validate_output(np.fromfile(output, dtype=np.float32).reshape(expected.shape), expected)
                    row.update(valid=True, measurement=measurement, correctness=check,
                               input_sha256=[digest(p) for p in inputs], output_sha256=digest(output),
                               median_us=statistics.median(measurement['throughput_us']))
                    (directory / 'measurement.json').write_text(json.dumps(measurement, indent=2) + '\n')
                    print(directory.name, row['median_us'], 'us', 'direct CFG=true' in measurement['realization'], flush=True)
                except Exception as error:
                    row['error'] = str(error)
                    print(directory.name, 'FAILED:', error, flush=True)
                report['results'].append(row)
                (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    report.update(closure_unchanged=hashes == {str(p): digest(p) for p in closure}, finished_unix=time.time())
    (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    if not report['closure_unchanged'] or not all(row['valid'] for row in report['results']):
        raise RuntimeError('failed visits retained')


if __name__ == '__main__':
    main()
