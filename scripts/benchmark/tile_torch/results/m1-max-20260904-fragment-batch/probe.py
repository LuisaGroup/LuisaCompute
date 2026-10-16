"""Bounded same-binary TIRx fragment batching experiment, no search."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile

ROOT = Path('/Users/mike/CLionProjects/luisa')
sys.path.insert(0, str(ROOT / 'scripts/benchmark/tile_torch'))
from compare_lowerings import implementation_order, validate_times
from compare_mpp import oracle, validate_output
from repeat import load_plan
from run import summarize
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--shape', action='append', default=[])
    args = parser.parse_args()
    if args.rounds <= 0 or args.rounds % 6:
        parser.error('rounds must be positive and divisible by six')
    binary = ROOT / 'cmake-build-tirx/bin/benchmark_tile_tirx'
    plan_path = ROOT / 'scripts/benchmark/tile_torch/results/m1-max-20260904-joint-search/results.json'
    frozen = list(load_plan(plan_path, {'gemm'}).values())
    if args.shape:
        frozen = [p for p in frozen if ','.join(str(p['case'][k]) for k in ('m', 'n', 'k')) in args.shape]
    assert frozen
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'sources').mkdir()
    files = [binary] + sorted(p for p in binary.parent.iterdir() if p.is_file() and p.suffix in ('.dylib', '.so'))
    fingerprint = lambda: {str(p): digest(p) for p in files}
    report = {'metadata': {
        'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'platform': platform.platform(), 'rounds': args.rounds, 'variants': [1, 2, 4],
        'balanced': True, 'samples': 7, 'sample_ms': 30, 'warmup_ms': 200,
        'timing': 'synchronized host-wall batched throughput including dispatch; not GPU time',
        'frozen_plan_sha256': digest(plan_path), 'frozen_plans': frozen,
        'artifacts_before': fingerprint(), 'runner_sha256': digest(Path(__file__)),
        'scope': 'Independent TVM runtime for every variant; TVMx requests Metal fast math on. No MPP/MPS library calls.',
    }, 'results': []}
    references = {tuple(p['case'][k] for k in ('m', 'n', 'k')): oracle(np, tuple(p['case'][k] for k in ('m', 'n', 'k'))) for p in frozen}
    failed = False
    for round_index in range(args.rounds):
        shift = round_index % len(frozen)
        for config in frozen[shift:] + frozen[:shift]:
            shape = tuple(config['case'][k] for k in ('m', 'n', 'k'))
            case_index = frozen.index(config)
            for batch in implementation_order(round_index, case_index, (1, 2, 4)):
                row = dict(shape=shape, round=round_index, batch=batch, valid=False)
                try:
                    with tempfile.TemporaryDirectory(prefix='tile-fragment-batch-') as folder:
                        output = Path(folder) / 'output.f32'
                        source = Path(folder) / 'source.metal'
                        vector = 'auto-vectorize' if config['auto_vectorize'] else 'no-vectorize' if config['no_vectorize'] else 'vectorize'
                        command = [str(binary), 'metal', 'gemm', *map(str, shape), *map(str, config['gemm_block']),
                                   '7', '30', '200', str(output), config['execution_scope'], str(config['pipeline_window']),
                                   'matrix', vector, str(config['group_threads']) if config['group_threads'] else 'auto', str(config['copy_batch']), 'tvm', str(batch)]
                        env = os.environ.copy()
                        env['LUISA_TILE_BENCH_DUMP_SOURCE'] = str(source)
                        row['command'] = command
                        proc = subprocess.run(command, env=env, text=True, capture_output=True, timeout=120)
                        row['stderr'] = proc.stderr
                        if proc.returncode:
                            raise RuntimeError(f'{proc.returncode}: {proc.stderr[-4000:]} {proc.stdout[-2000:]}')
                        lines = [s for s in proc.stdout.splitlines() if s.startswith('{')]
                        assert len(lines) == 1
                        result = json.loads(lines[0])
                        assert result['matrix_load_batch'] == batch and result['matrix_intrinsics'] > 0
                        assert result['planner_threads'] == config['group_threads'] and result['copy_batch'] == config['copy_batch']
                        plans = result['execution_plans']
                        assert len(plans) == 1 and plans[0]['threads'] == config['group_threads']
                        expected_batch = max(i for i in range(1, batch + 1) if (config['gemm_block'][2] // 8) % i == 0)
                        assert len(plans[0]['matrices']) == 1 and plans[0]['matrices'][0]['k_atom_batch'] == expected_batch
                        row['source_sha256'] = digest(source)
                        destination = args.output / 'sources' / (row['source_sha256'] + '.metal')
                        if not destination.exists():
                            destination.write_bytes(source.read_bytes())
                        validate_times(result, 7)
                        actual = np.fromfile(output, dtype='<f4').reshape(shape[:2])
                        row['correctness'] = validate_output(np, actual, references[shape])
                        summarize(result)
                        row.update(measurement=result, valid=True)
                        print(f'round {round_index + 1}: {shape} batch {batch}: {result["throughput_us_p50"]:.3f} us', flush=True)
                except Exception as error:
                    failed = True
                    row['error'] = str(error)
                    print(f'FAILED {shape} batch {batch}: {error}', flush=True)
                report['results'].append(row)
                (args.output / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    report['metadata']['artifacts_after'] = fingerprint()
    report['metadata']['artifacts_unchanged'] = report['metadata']['artifacts_before'] == report['metadata']['artifacts_after']
    assert report['metadata']['artifacts_unchanged']
    (args.output / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for shape in references:
        print(shape, {b: statistics.median(r['measurement']['throughput_us_p50'] for r in report['results'] if tuple(r['shape']) == shape and r['batch'] == b and r['valid']) for b in (1, 2, 4)}, flush=True)
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
