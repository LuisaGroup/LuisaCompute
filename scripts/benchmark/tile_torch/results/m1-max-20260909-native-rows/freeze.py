#!/usr/bin/env python3
"""Archive an audited native cohort without copying tensor payloads into Git."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    args = parser.parse_args()
    raw = args.raw.resolve()
    if (HERE / 'provenance.json').exists():
        raise ValueError('immutable evidence; choose a new checkpoint')
    capture, prepared, replay = (raw / p for p in ('capture-v3', 'prepared-v2', 'replay-v2'))
    captured = json.loads((capture / 'results.json').read_text())
    manifest = json.loads((prepared / 'manifest.json').read_text())
    measured = json.loads((replay / 'results.json').read_text())
    audit = json.loads((raw / 'audit.json').read_text())
    assert audit['status'] == 'passed'
    assert (audit['capture_outputs_re_read'], audit['native_unique_outputs_re_read'], audit['native_timed_visits']) == (48, 72, 432)
    assert audit['replay_sha256'] == digest((replay / 'results.json').read_bytes())
    assert len(measured['cases']) == 24
    for folder, record in ((capture, captured), (prepared, manifest), (replay, measured)):
        assert record['runner_unchanged']
        assert digest((folder / 'runner-sources/native_rows.py').read_bytes()) == record['source_sha256']
    assert captured['source_sha256'] == manifest['source_sha256'] == measured['source_sha256']
    artifacts = {}

    def store(name, data, origin):
        path = HERE / name
        if path.exists():
            raise ValueError('refusing to replace artifact: ' + name)
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = gzip.compress(data, mtime=0) if name.endswith('.gz') else data
        path.write_bytes(encoded)
        artifacts[name] = dict(origin=str(origin), source_sha256=digest(data), archived_sha256=digest(encoded),
                               source_bytes=len(data), archived_bytes=len(encoded))

    def archive(name, source):
        store(name, Path(source).read_bytes(), source)

    for name, source in (('capture.json.gz', capture / 'results.json'), ('manifest.json.gz', prepared / 'manifest.json'),
                         ('replay.json.gz', replay / 'results.json'), ('replay.log.gz', raw / 'replay-v2.log'),
                         ('audit.json', raw / 'audit.json'), ('native_rows_replay.dylib.gz', Path(manifest['helper']))):
        archive(name, source)
    for source in sorted((replay / 'runner-sources').iterdir()):
        archive('runner-sources/' + source.name, source)
    archive('runner-sources/audit_native_rows.py', ROOT / 'scripts/benchmark/tile_torch/audit_native_rows.py')
    archive('runner-sources/test_native_rows.py', ROOT / 'scripts/benchmark/tile_torch/test_native_rows.py')
    for row in captured['results']:
        source = Path(row['directory'])
        prefix = 'captures/' + source.name + '/'
        archive(prefix + 'measurement.json', source / 'measurement.json')
        archive(prefix + 'kernel.ll.gz', source / 'kernel.ll')
        objects = list((source / 'object').glob('*.o'))
        assert len(objects) == 1
        archive(prefix + 'kernel.o.gz', objects[0])
    for case in manifest['cases']:
        for variant, entry in case['entries'].items():
            source = Path(entry['library'])
            prefix = 'native/' + source.parent.name + '/'
            assert digest(source.read_bytes()) == entry['library_sha256']
            archive(prefix + source.name + '.gz', source)
            archive(prefix + variant + '.asm.gz', source.parent / (variant + '-assembly.stdout.log'))
            if variant == 'inductor':
                for name in ('inductor.cpp', 'inductor-graph.json', 'inductor-wrapper.py'):
                    archive(prefix + name, source.parent / name)
    logs = {}
    for folder in (capture, prepared):
        for source in sorted(folder.rglob('*')):
            if 'inductor-cache' in source.parts or not source.is_file():
                continue
            if source.name.endswith(('.log', '.command.json')) and '-assembly.' not in source.name:
                logs[str(source.relative_to(raw))] = source.read_text()
    store('build-capture-prepare-logs.json.gz', (json.dumps(logs, indent=2) + '\n').encode(), 'capture/prepare subprocess logs')
    failures = {str(source.relative_to(raw)): source.read_text() for source in
                (raw / 'capture/results.json', raw / 'replay/results.json', raw / 'replay.log')}
    store('preflight-failures.json.gz', (json.dumps(failures, indent=2) + '\n').encode(), 'failed runs; excluded from comparison')
    lines = ['# 完整 native 对照', '',
             'M1 Max / FP32 / 单线程；单位 µs。Whole/local 分别固定 local_lanes=1/8，其余编译开关相同。',
             '时间是六轮 p50 的中位数；比值是同轮比值的中位数，不是表内时间相除。',
             '范围是六轮观测范围，不是置信区间。所有 24 个尺寸均保留；更小的比值更好。', '',
             '| 算子 | 尺寸 | Whole | Local | Inductor | Local/Inductor | 六轮范围 | 胜出轮数 | Local/Whole |',
             '|---|---|---:|---:|---:|---:|---|---:|---:|']
    for case in measured['cases']:
        timing = case['summary_us']
        ratio = case['paired_ratios']['local/inductor']
        shape = '×'.join(map(str, case['dimensions']))
        lines.append(f"| {case['operation']} | {shape} | {timing['whole']:.3f} | {timing['local']:.3f} | {timing['inductor']:.3f} | {ratio['median']:.3f} | {ratio['minimum']:.3f}–{ratio['maximum']:.3f} | {ratio['wins']}/6 | {case['paired_ratios']['local/whole']['median']:.3f} |")
    lines.extend(['', '源记录：`replay.json.gz`；逐次输出/guard 校验和独立复核：`audit.json`。',
                  '计时包含实际入口、共同 C++ callback、必要的 block 遍历/launch reset，以及生成代码内部的 libc/分配调用；',
                  '不包含 Runtime/Python/JIT 或 caller 的分配。不是硬件周期计数，也不是 E2E 延迟。', ''])
    store('tables.md', '\n'.join(lines).encode(), 'derived from replay.json.gz')
    predecessor_path = Path(captured['predecessor'])
    predecessor = json.loads(predecessor_path.read_text())
    assert digest(predecessor_path.read_bytes()) == captured['predecessor_sha256']
    for path, sha in captured['binary_closure_sha256'].items():
        assert digest(Path(path).read_bytes()) == sha
    source_hashes = predecessor['inherited_source_sha256'] | predecessor['tested_source_sha256']
    for path, sha in source_hashes.items():
        assert digest((ROOT / path).read_bytes()) == sha, path
    provenance = dict(frozen_unix=time.time(), raw=str(raw),
                      parent_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                      fetched_next=subprocess.check_output(['git', 'rev-parse', 'origin/next'], cwd=ROOT, text=True).strip(),
                      next_merged=False, compiler_implementation_changes_this_checkpoint=False,
                      source_chain='../m1-max-20260909-cohort-private/provenance.json',
                      source_chain_sha256=captured['predecessor_sha256'], source_sha256=source_hashes,
                      binary_closure_sha256=captured['binary_closure_sha256'],
                      source_caveat='Recorded isolated archive + predecessor overlays; not a clean checkout assertion. Current unrelated worktree changes excluded.',
                      selection='24 preselected cases; fixed whole/local mappings, no timed selection or model fitting; no new holdout claim.',
                      timing_boundary=measured['boundary'],
                      artifacts=artifacts,
                      tensor_retention='Raw capture/replay tensor files remain in raw. Git stores their hashes and complete checks, not the tensor payloads. Guard arrays were checked at each visit but not retained.',
                      failures='Initial metadata-label rejection and first-replay JSON serialization failure retained separately; neither supplies comparative timings.')
    (HERE / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    for name, item in artifacts.items():
        encoded = (HERE / name).read_bytes()
        assert digest(encoded) == item['archived_sha256']
        assert digest(gzip.decompress(encoded) if name.endswith('.gz') else encoded) == item['source_sha256']
    print(f'PASS: {len(artifacts)} archived artifacts, all compressed/uncompressed hashes checked')


if __name__ == '__main__':
    main()
