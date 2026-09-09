#!/usr/bin/env python3
"""Fixed A/B-B/A diagnostic; no timing-based candidate selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from compare_llm import reference, validate_output


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('capture', 'matrix', 'factorial'), required=True)
    parser.add_argument('--disable-interleaving', action='store_true')
    args = parser.parse_args()
    binary = args.binary.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    closure = [binary] + sorted(binary.parent.glob('libluisa-*'))
    closure = [p for p in closure if p.is_file()]
    hashes = {str(p): digest(p) for p in closure}
    report = dict(metric='runtime_e2e_synchronized_host_wall_us', mode=args.mode,
                  private_array_interleaving=not args.disable_interleaving,
                  platform=platform.platform(), binary_closure_sha256=hashes,
                  source_runner_sha256=digest(Path(__file__)), results=[])
    cases = [('rmsnorm', 64, 256), ('rmsnorm', 1024, 4096)] if args.mode == 'capture' else [
        ('rmsnorm', 64, 256), ('rmsnorm', 17, 65), ('rmsnorm', 1024, 4096),
        ('rmsnorm', 17, 16384), ('layernorm', 17, 16384), ('layernorm', 1024, 4096),
        ('masked_softmax', 64, 4096), ('swiglu', 17, 65), ('swiglu', 1024, 4096),
        ('gelu_residual', 17, 65), ('gelu_residual', 1024, 4096),
        ('rope', 64, 128), ('rope', 17, 66)]
    variants = [(1, not args.disable_interleaving), (8, not args.disable_interleaving)]
    if args.mode == 'factorial':
        if args.disable_interleaving:
            raise ValueError('factorial already controls both layouts')
        variants = [(1, False), (1, True), (8, True), (8, False)]
    orders = [variants] if args.mode == 'capture' else [variants, list(reversed(variants))]
    for op, rows, columns in cases:
        for round_id, order in enumerate(orders):
            for local, interleave in order:
                suffix = f'-soa{int(interleave)}' if args.mode == 'factorial' else ''
                directory = args.output / f'{op}-{rows}x{columns}-r{round_id}-l{local}{suffix}'
                directory.mkdir()
                output = directory / 'output.f32'
                env = {k: v for k, v in os.environ.items() if not k.startswith(('LUISA_SIMD_', 'LUISA_TILE_BENCH_', 'DYLD_'))}
                env.update(LUISA_SIMD_WARP_WIDTH='8', LUISA_SIMD_WORKER_COUNT='1' if args.mode == 'capture' else '8',
                           LUISA_TILE_BENCH_XIR_LOCAL_LANES=str(local))
                if not interleave:
                    env['LUISA_SIMD_DISABLE_INTERLEAVED_PRIVATE_ARRAYS'] = '1'
                if args.mode == 'capture':
                    env.update(LUISA_TILE_BENCH_DUMP_SOURCE=str(directory / 'kernel.ll'),
                               LUISA_SIMD_DUMP_ASSEMBLY_DIR=str(directory / 'object'))
                command = [str(binary), 'llm', op, f'{rows},{columns}', '1', '1', '5', '15', '50', str(output)]
                result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=180)
                (directory / 'stdout.log').write_text(result.stdout)
                (directory / 'stderr.log').write_text(result.stderr)
                row = dict(operation=op, dimensions=[rows, columns], round=round_id, order=order, local_lanes=local, private_array_interleaving=interleave,
                           command=command, environment={k: v for k, v in env.items() if k.startswith('LUISA_')},
                           returncode=result.returncode, valid=False)
                if result.returncode == 0:
                    m = json.loads(result.stdout)
                    inputs = [np.fromfile(str(output) + f'.input{i}.f32', dtype=np.float32).reshape(s) for i, s in enumerate(m['input_shapes'])]
                    row['input_sha256'] = [digest(Path(str(output) + f'.input{i}.f32')) for i in range(len(inputs))]
                    expected = reference(op, (rows, columns), inputs)
                    row['correctness'] = validate_output(np.fromfile(output, dtype=np.float32).reshape(expected.shape), expected)
                    row.update(valid=True, measurement=m, median_us=statistics.median(m['throughput_us']))
                    (directory / 'measurement.json').write_text(json.dumps(m, indent=2) + '\n')
                    print(directory.name, row['median_us'], 'us', flush=True)
                else:
                    row['error'] = result.stderr
                    print(directory.name, 'FAILED', flush=True)
                report['results'].append(row)
                (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    report['closure_unchanged'] = hashes == {str(p): digest(p) for p in closure}
    (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    if not report['closure_unchanged'] or not all(row['valid'] for row in report['results']):
        raise RuntimeError('measurement validation failed; retain the failed rows')


if __name__ == '__main__':
    main()
