#!/usr/bin/env python3
"""Freeze two complete diagnostic cohorts without promoting their timings."""
import argparse
import difflib
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--documentation', action='store_true')
    args = parser.parse_args()
    raw = args.raw.resolve()
    destination = HERE / 'documentation' if args.documentation else HERE
    assert not (destination / 'provenance.json').exists(), 'Checkpoint is immutable; use a new destination.'
    artifacts = {}

    def store(name, data, origin, role):
        target = destination / name
        assert not target.exists(), name
        target.parent.mkdir(parents=True, exist_ok=True)
        encoded = gzip.compress(data, mtime=0) if name.endswith('.gz') else data
        target.write_bytes(encoded)
        artifacts[name] = dict(origin=str(origin), role=role, source_sha256=sha(data), archived_sha256=sha(encoded),
                               source_bytes=len(data), archived_bytes=len(encoded))

    def archive(name, path, role):
        store(name, Path(path).read_bytes(), path, role)

    if args.documentation:
        receipt = read(raw / 'documentation.json')
        assert receipt['passed'] and receipt['source_unchanged']
        assert all(r['exit_code'] == 0 for r in receipt['runs'])
        for path, digest in receipt['source_sha256'].items():
            assert sha((REPO / path).read_bytes()) == digest, path
        qa = read(raw / 'docs-qa/receipt.json')
        assert qa['passed']
        for name in ('documentation.py', 'documentation.json', 'qa-docs.cjs'):
            archive(name, raw / name, 'Strict existing-Sphinx build or rendered QA source/receipt.')
        for run in receipt['runs']:
            archive(run['log'] + '.gz', raw / run['log'], 'Executed documentation command output.')
        archive('rendered-qa.json', raw / 'docs-qa/receipt.json', 'Desktop/mobile DOM and screenshot receipts; screenshots remain local.')
        metadata = dict(scope='Documentation validation only; no performance acceptance.',
                        source_sha256=receipt['source_sha256'])
    else:
        audit = read(raw / 'audit.json')
        assert audit['passed'] and not audit['performance_accepted']
        assert not audit['defaults_changed'] and not audit['cost_calibrated']
        assert audit['snapshots_reread'] == 384 and len(audit['rejected_mutations']) == 8
        assert audit['phase_controls_identical_objects'] == audit['projection_preoptimization_llvm_identical'] == 24
        assert all((p['smoke_visits'], p['timed_visits'], p['bitwise_equal_cases']) == (72, 432, 24) for p in audit['phases'])
        tests = read(raw / 'projection-tests.json')
        assert tests['passed'] and tests['experimental_projection_forced']
        assert [r['tests'] for r in tests['runs']] == [80, 35]
        assert all(r['exit_code'] == r['failures'] == r['skipped'] == 0 for r in tests['runs'])
        baseline = read(raw / 'baseline-source.json')
        frozen = read(raw / 'stage-a-frozen.json')
        assert frozen['passed'] and frozen['source_files'] == 19177
        for item in frozen['binary_relocations'].values():
            assert sha(Path(item['path']).read_bytes()) == item['sha256']
        for prefix, source_root, build_file, snapshot_file in (
                ('inline', raw / 'frozen-stage-a-source', 'build.json', 'source-snapshot.json'),
                ('projection', raw / 'source', 'projection-build.json', 'projection-source-snapshot.json')):
            build = read(raw / build_file)
            runs = build['runs'] if 'runs' in build else [build]
            assert build['passed'] and build['source_unchanged'] and all(r['exit_code'] == 0 for r in runs)
            snapshot = read(raw / snapshot_file)
            assert snapshot['source_commit'] == audit['source_commit'] == baseline['source_commit']
            for path, digest in snapshot['source_sha256'].items():
                assert sha((source_root / path).read_bytes()) == digest, path
            assert sorted(snapshot['overlay']) == sorted(path for path, digest in snapshot['source_sha256'].items()
                                                        if digest != baseline['source_sha256'][path])
            patch = []
            for path in snapshot['overlay']:
                original = subprocess.check_output(['git', 'show', audit['source_commit'] + ':' + path], cwd=REPO)
                # These are source-export probes, not edits to the working compiler.
                assert (REPO / path).read_bytes() == original, path
                patch.extend(difflib.unified_diff(original.decode().splitlines(keepends=True),
                                                  (source_root / path).read_text().splitlines(keepends=True),
                                                  fromfile='a/' + path, tofile='b/' + path))
            store(prefix + '/experimental.patch.gz', ''.join(patch).encode(), source_root,
                  'Diagnostic-only overlay from the recorded commit; not production implementation.')
            archive(prefix + '/source-snapshot.json.gz', raw / snapshot_file, 'Exact pinned inputs and separately identified overlay.')
            archive(prefix + '/build.json', raw / build_file, 'Successful full configured-build gate before execution.')

        preserved = {}
        for line in Path('/tmp/luisa-pointwise-native.1P87uU/premerge-user-sha256.txt').read_text().splitlines():
            digest, path = line.split(None, 1)
            assert sha((REPO / path).read_bytes()) == digest, path
            preserved[path] = digest
        dependencies = {}
        for item in read(raw / 'pinned-repositories.json'):
            if item['path'] == '.':
                continue
            revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO / item['path'], text=True).strip()
            assert revision == item['checkout_commit']
            dependencies[item['path']] = revision

        for name in ('audit.json', 'projection-tests.json', 'stage-a-frozen.json', 'configure-command.json',
                     'pinned-repositories.json', 'roundeven-portability.json'):
            archive(name, raw / name, 'Observed validation, preparation or compiler-only probe receipt.')
        archive('baseline-source.json.gz', raw / 'baseline-source.json', 'Recursive pinned Git archive inputs, before overlays.')
        for name in ('setup.py', 'build.py', 'capture.py', 'run.py', 'freeze-stage-a.py', 'build-projection.py',
                     'capture-projection.py', 'run-projection.py', 'test-projection.py', 'audit.py', 'roundeven-portability.py'):
            archive('drivers/' + name, raw / name, 'Executed preparation, native capture/replay or audit source; paths need relocation.')
        archive('drivers/pilot.py', raw / 'verify/pilot.py', 'Unmodified native replay driver; only verify/replay modes were used here.')
        for path in sorted((raw / 'verify/runner-sources').iterdir()):
            assert path.read_bytes() == (raw / 'projection-verify/runner-sources' / path.name).read_bytes()
            archive('drivers/' + path.name, path, 'Shared replay ABI/helper and numerical-reference source, identical in both phases.')
        for name in ('full-build.log', 'projection-build.log', 'configure.log', 'capture.log', 'verify.log', 'diagnostic.log',
                     'projection-capture.log', 'projection-verify.log', 'projection-diagnostic.log',
                     'projection-xir-simd.log', 'projection-tile.log'):
            archive('logs/' + name + '.gz', raw / name, 'Executed command output; diagnostic timings are not accepted performance evidence.')
        for name in ('projection-xir-simd.xml', 'projection-tile.xml'):
            archive(name + '.gz', raw / name, 'Case-level CTest receipt with the projection environment flag forced on.')

        for prefix, raw_prefix in [('inline', ''), ('projection', 'projection-')]:
            capture = read(raw / (raw_prefix + 'capture/manifest.json'))
            assert capture['source_unchanged'] and capture['closure_unchanged']
            if prefix == 'projection':
                for path, digest in capture['binary_closure_sha256'].items():
                    assert sha(Path(path).read_bytes()) == digest
            archive(prefix + '/capture.json.gz', raw / (raw_prefix + 'capture/manifest.json'), 'Actual ORC entries, input/ABI/source/binary identities and controls.')
            for mode in ('verify', 'diagnostic'):
                archive(prefix + '/' + mode + '.json.gz', raw / (raw_prefix + mode + '/results.json'),
                        'Complete native outputs/check receipts and samples; diagnostic only, not a promoted ranking.')
            for mode in ('capture', 'verify', 'diagnostic'):
                path = raw / (raw_prefix + mode + '-host.json')
                host = read(path)
                assert host['exit_code'] == 0 and not host['performance_qualified']
                sanitized = {k: v for k, v in host.items() if k != 'observations'}
                sanitized.update(original_sha256=sha(path.read_bytes()), privacy='Process identifiers, names and paths omitted. All-process CPU includes this experiment and is not foreign-only.')
                sanitized['observations'] = []
                for item in host['observations']:
                    percentages = [float(line.split(None, 3)[2]) for line in item['processes'].splitlines() if line.strip()]
                    sanitized['observations'].append(dict(unix=item['unix'], process_count=len(percentages),
                                                         all_process_cpu_percent=sum(percentages),
                                                         over_10_percent=sorted((v for v in percentages if v > 10), reverse=True)))
                store(prefix + '/' + mode + '-coactivity.json.gz', (json.dumps(sanitized, indent=2) + '\n').encode(), path,
                      'Privacy-minimized prospective diagnostic qualification; not proof of exclusivity or clock stability.')
            for case in capture['cases']:
                label = case['operation'] + '-' + 'x'.join(map(str, case['dimensions']))
                for variant in ('off', 'on'):
                    entry = case['entries'][variant]
                    source = Path(entry['capture'])
                    objects = list((source / 'object').glob('*.o'))
                    assert len(objects) == 1 and sha(objects[0].read_bytes()) == entry['object_sha256']
                    assert sha((source / 'kernel.ll').read_bytes()) == entry['llvm_sha256']
                    base = prefix + '/native/' + label + '/' + variant
                    archive(base + '.o.gz', objects[0], 'Actual intercepted ORC object, not reconstructed code.')
                    archive(base + '.ll.gz', source / 'kernel.ll', 'Actual pre-O2 LLVM; not the final optimized IR.')
                    archive(base + '.s.gz', source / 'assembly.stdout.log', 'Disassembly of the actual ORC object.')
                    commands = {p.name: read(p) for p in sorted(source.glob('*.command.json'))}
                    store(base + '-commands.json', (json.dumps(commands, indent=2) + '\n').encode(), source, 'Exact native capture, import inspection, link and disassembly commands.')
                if prefix == 'inline':
                    entry = case['entries']['inductor']
                    assert sha(Path(entry['source']).read_bytes()) == entry['source_sha256']
                    archive('inductor/' + label + '.cpp', entry['source'], 'Frozen TorchInductor source reused in both phases; native library identity is in capture manifests.')
        for name in ('roundeven-probe.ll', 'roundeven-x86-baseline.s', 'roundeven-apple-m1.s',
                     'rope-no-slp.ll', 'rope-with-slp.ll', 'rope-no-slp.s', 'rope-late-cleanup.ll'):
            archive('compiler-only/' + name + '.gz', raw / name, 'Offline compiler diagnostic; never relinked or timed as a benchmark entry.')
        archive('compiler-only/rejected-roundeven.patch.gz', raw / 'rejected-roundeven.patch', 'Discarded draft, including proposed tests that were NOT compiled or run.')

        lines = ['# 两组 native codegen 诊断：完整数据表', '',
                 '**全部计时仅供诊断，不纳入性能达标、默认策略或模型校准。** 两组实验分开执行，不能横向拼接绝对时间。', '',
                 'FP32、单 CPU 线程；单位 µs。每个时间是六个轮次 p50 的中位数；每个轮次含七个样本。',
                 '配对比值是六个同轮次时间比的中位数，不是显示时间的商；范围为六轮最小–最大值，不是置信区间。',
                 '`On<Off` 只是观测轮次数，不是统计显著性或已接受的胜场。完整样本与输出校验在各阶段 JSON 中。', '']
        for phase in audit['phases']:
            lines += ['## ' + phase['feature'], '',
                      '| 算子 | 尺寸 | Off µs | On µs | Inductor µs | On/Off（范围） | On<Off | On/Inductor（范围） |',
                      '|---|---|---:|---:|---:|---:|---:|---:|']
            for case in phase['cases']:
                ratios = case['paired_ratios']
                ratio_text = lambda key: f"{ratios[key]['median']:.6f} ({ratios[key]['minimum']:.6f}–{ratios[key]['maximum']:.6f})"
                medians = case['medians_us']
                lines.append(f"| {case['operation']} | {'×'.join(map(str, case['dimensions']))} | "
                             f"{medians['off']:.6f} | {medians['on']:.6f} | {medians['inductor']:.6f} | "
                             f"{ratio_text('on/off')} | {ratios['on/off']['wins']}/6 | {ratio_text('on/inductor')} |")
            lines.append('')
        store('tables.md', ('\n'.join(lines).rstrip() + '\n').encode(), raw / 'audit.json', 'Mechanical table projection of independently recomputed diagnostics; no acceptance.')
        metadata = dict(source_commit=audit['source_commit'], merged_next='03a0f5158b53768abefea555f480f87ee5bc5a1e',
                        scope='Two independent diagnostic-only overlays in recursive pinned Git exports; working compiler unchanged.',
                        performance_accepted=False, defaults_changed=False, cost_calibrated=False,
                        preserved_user_file_sha256=preserved, preserved_dependency_checkouts=dependencies,
                        host=dict(cpu='Apple M1 Max', os='macOS 26.6.2', os_build='25G83', llvm='21.1.8'),
                        large_payload_retention='Input/output arrays, linked dylibs, external dependencies and complete exported trees remain at the recorded local paths. Exact code objects, generated source, commands, hashes and receipts are archived here.')
    metadata.update(frozen_unix=time.time(), raw=str(raw), artifacts=artifacts, freeze_source_sha256=sha(Path(__file__).read_bytes()))
    (destination / 'provenance.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('Archived', len(artifacts), 'artifacts;', sum(v['archived_bytes'] for v in artifacts.values()), 'bytes; no performance acceptance.')


if __name__ == '__main__':
    main()
