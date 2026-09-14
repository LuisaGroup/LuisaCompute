#!/usr/bin/env python3
"""CPU task grain and execution mapping: fixed, auditable Runtime controls.

Full-packet specialization is enabled and load/reduction fusion is disabled
for every visit. The provisional activation coefficient is a relative prior,
not a fitted nanosecond model and not a benchmark-time schedule oracle.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from compare_llm import reference, validate_output

PILOT = [('rmsnorm', (17, 65)), ('rmsnorm', (64, 256)),
         ('rmsnorm', (1024, 4096)), ('gelu_residual', (64, 256)),
         ('swiglu', (64, 256))]
VALIDATION = [(op, dims) for op in ('rmsnorm', 'layernorm', 'masked_softmax', 'swiglu', 'gelu_residual', 'rope')
              for dims in ((3, 1024), (129, 768), (257, 1538), (4096, 1024))]
VALIDATION += [('attention', (1, 4, 2, 16, 32, 16, 16)),
               ('attention', (1, 4, 1, 64, 128, 32, 32))]
VARIANTS = {
    'whole': dict(local=1, search=0, activation=0, task=0),
    'local': dict(local=8, search=0, activation=0, task=0),
    'joint_legacy': dict(local=0, search=0, activation=0, task=0),
    'joint_tasks': dict(local=0, search=1, activation=1000000, task=128),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--suite', choices=['pilot', 'validation', 'fixed-grain'], required=True)
    parser.add_argument('--rounds', type=int, default=2)
    args = parser.parse_args()
    if args.rounds not in (2, 4):
        raise ValueError('use two or four prespecified orders')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve()
    closure = [binary] + sorted(p for p in binary.parent.glob('libluisa-*') if p.is_file())
    hashes = {str(p): digest(p) for p in closure}
    cases = PILOT if args.suite == 'pilot' else VALIDATION
    if args.suite == 'fixed-grain':
        cases = [(op, dims) for op in ('rmsnorm', 'layernorm', 'masked_softmax', 'swiglu', 'gelu_residual', 'rope')
                 for dims in ((17, 66), (129, 768), (1024, 4096))]
    report = dict(metric='runtime_e2e_synchronized_host_wall_us', suite=args.suite,
                  runner_sha256=digest(Path(__file__)), started_unix=time.time(), platform=platform.platform(),
                  full_packet_specialization=True, load_reduction_fusion=False,
                  cpu_workers_requested=8, packet_width=8, provisional_activation=1000000,
                  policy_fit='none; prespecified relative-work activation prior',
                  binary_closure_sha256=hashes, results=[])
    for op, dims in cases:
        variants = VARIANTS.copy()
        if op == 'attention':
            variants.pop('local')  # local redistribution is not implemented; not a measured candidate
        if args.suite == 'fixed-grain':
            variants = {f'grain_{g}': dict(local=8, search=0, activation=0, task=0, block=32, grain=g)
                        for g in (0, 1, 3, 4294967295)}
        names = list(variants)
        orders = [names, list(reversed(names)), names[2:] + names[:2], list(reversed(names[2:] + names[:2]))]
        for round_id, order in enumerate(orders[:args.rounds]):
            for name in order:
                variant = variants[name]
                directory = args.output / f'{op}-{"x".join(map(str, dims))}-r{round_id}-{name}'
                directory.mkdir()
                output = directory / 'output.f32'
                env = {k: v for k, v in os.environ.items() if not k.startswith(('LUISA_SIMD_', 'LUISA_TILE_BENCH_', 'DYLD_'))}
                env.update(LUISA_SIMD_WARP_WIDTH='8', LUISA_SIMD_WORKER_COUNT='8',
                           LUISA_SIMD_ENABLE_FULL_PACKET_SPECIALIZATION='1',
                           LUISA_SIMD_DISABLE_LOAD_REDUCTION_FUSION='1',
                           LUISA_TILE_BENCH_XIR_LOCAL_LANES=str(variant['local']),
                           LUISA_TILE_BENCH_XIR_SEARCH_TASK_GRAIN=str(variant['search']),
                           LUISA_TILE_BENCH_XIR_WORKER_ACTIVATION=str(variant['activation']),
                           LUISA_TILE_BENCH_XIR_TASK_DISPATCH=str(variant['task']))
                if args.suite == 'fixed-grain':
                    env.update(LUISA_TILE_BENCH_XIR_BLOCK_SIZE=str(variant['block']),
                               LUISA_TILE_BENCH_XIR_BLOCKS_PER_TASK=str(variant['grain']),
                               LUISA_TILE_BENCH_DUMP_SOURCE=str(directory / 'kernel.ll'),
                               LUISA_SIMD_DUMP_ASSEMBLY_DIR=str(directory / 'object'))
                command = [str(binary), 'llm', op, ','.join(map(str, dims)), '1', '1', '7', '20', '75', str(output)]
                result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=360)
                (directory / 'stdout.log').write_text(result.stdout)
                (directory / 'stderr.log').write_text(result.stderr)
                row = dict(operation=op, dimensions=dims, round=round_id, order=order, variant=name,
                           candidate=variant, command=command, environment={k: v for k, v in env.items() if k.startswith('LUISA_')},
                           returncode=result.returncode, valid=False)
                try:
                    if result.returncode:
                        raise ValueError(result.stderr)
                    m = json.loads(result.stdout)
                    paths = [Path(str(output) + f'.input{i}.f32') for i in range(len(m['input_shapes']))]
                    arrays = [np.fromfile(p, dtype=np.float32).reshape(s) for p, s in zip(paths, m['input_shapes'])]
                    expected = reference(op, dims, arrays)
                    check = validate_output(np.fromfile(output, dtype=np.float32).reshape(expected.shape), expected)
                    realization = m['realization']
                    local = int(re.search(r'local_lanes=(\d+)', realization)[1])
                    block = int(re.search(r'W8, (\d+) workers/block', realization)[1])
                    grain = int(re.search(r'blocks_per_task=(\d+)', realization)[1])
                    row.update(valid=True, measurement=m, correctness=check,
                               input_sha256=[digest(p) for p in paths], output_sha256=digest(output),
                               selected_local=local, selected_block=block, selected_grain=grain,
                               median_us=statistics.median(m['throughput_us']))
                    if args.suite == 'fixed-grain':
                        objects = list((directory / 'object').glob('*.o'))
                        if len(objects) != 1:
                            raise ValueError('expected exactly one ORC object')
                        row.update(llvm_sha256=digest(directory / 'kernel.ll'), object_sha256=digest(objects[0]))
                    (directory / 'measurement.json').write_text(json.dumps(m, indent=2) + '\n')
                    print(directory.name, row['median_us'], 'us', f'L{local} B{block} G{grain}', flush=True)
                except Exception as error:
                    row.update(valid=False, error=str(error))
                    print(directory.name, 'FAILED:', error, flush=True)
                report['results'].append(row)
                (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    report.update(closure_unchanged=hashes == {str(p): digest(p) for p in closure}, finished_unix=time.time())
    (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    if not report['closure_unchanged'] or not all(row['valid'] for row in report['results']):
        raise RuntimeError('failed evidence is retained')


if __name__ == '__main__':
    main()
