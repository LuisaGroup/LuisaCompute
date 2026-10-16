#!/usr/bin/env python3
"""Freeze existing evidence only; never build, capture, replay or run an audit.

The destination must be fresh except for independently authored notes/tables.
Original JSON bytes and absolute source identities are never rewritten.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--repository', type=Path, required=True)
    args = parser.parse_args()
    raw, repository = args.raw.resolve(), args.repository.resolve()
    destination = Path(__file__).resolve().parent
    runners = repository / 'scripts/benchmark/tile_torch'
    artifacts, original_hashes, expected_hashes = [], {}, {}
    reports = {}
    for phase in ('pilot', 'matrix'):
        captured = json.loads((raw / phase / 'manifest.json').read_text())
        replayed = json.loads((raw / (phase + '-replay') / 'results.json').read_text())
        audited = json.loads((raw / ('pilot-audit-v2.json' if phase == 'pilot' else 'matrix-audit.json')).read_text())
        if captured['status'] != 'captured' or replayed['status'] != 'passed' or audited['status'] != 'passed':
            raise ValueError('unfinished phase: ' + phase)
        if len(audited['rejected_mutations']) != 11:
            raise ValueError('expected final strengthened audit: ' + phase)
        if audited['manifest_sha256'] != digest((raw / phase / 'manifest.json').read_bytes()) or \
                audited['replay_sha256'] != digest((raw / (phase + '-replay') / 'results.json').read_bytes()):
            raise ValueError('audit points to another experiment')
        reports[phase] = (captured, replayed, audited)
        for report in (captured, replayed):
            for name, sha in report['artifact_sha256'].items():
                path = str(Path(name).resolve())
                if path in expected_hashes and expected_hashes[path] != sha:
                    raise ValueError('conflicting recorded artifact identity')
                expected_hashes[path] = sha

    def read_source(path):
        path = path.resolve()
        data = path.read_bytes()
        sha = digest(data)
        recorded = expected_hashes.get(str(path))
        if recorded is not None and recorded != sha:
            raise ValueError('source differs from its recorded artifact identity: ' + str(path))
        if str(path) in original_hashes and original_hashes[str(path)] != sha:
            raise ValueError('source changed during archive: ' + str(path))
        original_hashes[str(path)] = sha
        return data

    def output(name, data):
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(data)
        return dict(path=name, bytes=len(data), sha256=digest(data))

    def copy_source(name, source, compressed=False, role='evidence'):
        data = read_source(source)
        encoded = gzip.compress(data, compresslevel=9, mtime=0) if compressed else data
        record = output(name, encoded)
        record.update(role=role, source=str(source.resolve()), source_bytes=len(data), source_sha256=digest(data),
                      encoding='gzip' if compressed else 'identity')
        artifacts.append(record)

    # The six sources implementing this experiment. Additional sources needed
    # by the combined unit receipt live in validation.tar.gz, not a fake audit
    # test file that was never executed.
    primary_runners = ('native_pointwise.py', 'native_rows.py', 'compare_llm.py',
                       'native_rows_replay.cpp', 'audit_native_pointwise.py', 'test_native_pointwise.py')
    for name in primary_runners:
        copy_source('runners/' + name, runners / name, role='frozen runner/helper/auditor/unit source')
    auditor_sha = digest((destination / 'runners/audit_native_pointwise.py').read_bytes())
    if any(report[2]['auditor_sha256'] != auditor_sha for report in reports.values()):
        raise ValueError('current auditor is not the one used by the final audits')
    print('Primary runner source snapshot complete (six files).', flush=True)

    for phase in ('pilot', 'matrix'):
        copy_source(phase + '-capture.json.gz', raw / phase / 'manifest.json', True, 'unmodified raw capture manifest')
        copy_source(phase + '-replay.json.gz', raw / (phase + '-replay') / 'results.json', True, 'unmodified raw native replay result')
    for name in ('pilot-audit.json', 'pilot-audit-v2.json', 'matrix-audit.json'):
        copy_source(name, raw / name, role='superseded pilot audit' if name == 'pilot-audit.json' else 'final independent audit')

    def archive(name, members):
        entries, data = [], io.BytesIO()
        with gzip.GzipFile(fileobj=data, mode='wb', compresslevel=9, mtime=0, filename='') as zipped:
            with tarfile.open(fileobj=zipped, mode='w|', format=tarfile.PAX_FORMAT) as tar:
                seen = set()
                for member, source in sorted(members):
                    if member.startswith('/') or '..' in Path(member).parts or member in seen:
                        raise ValueError('unsafe or duplicate archive member: ' + member)
                    seen.add(member)
                    content = read_source(source)
                    info = tarfile.TarInfo(member)
                    info.size = len(content)
                    info.mode = source.stat().st_mode & 0o777
                    info.mtime = info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    tar.addfile(info, io.BytesIO(content))
                    entries.append(dict(path=member, source=str(source.resolve()), bytes=len(content),
                                        sha256=digest(content), mode=info.mode,
                                        checked_against_recorded_artifact=str(source.resolve()) in expected_hashes))
        record = output(name, data.getvalue())
        record.update(role='compressed evidence bundle', encoding='tar.gz', members=entries,
                      uncompressed_member_bytes=sum(entry['bytes'] for entry in entries))
        artifacts.append(record)

    for phase in ('pilot', 'matrix'):
        members = []
        for folder in (phase, phase + '-replay'):
            for path in (raw / folder).rglob('*'):
                if not path.is_file() or path.suffix in ('.f32', '.f64'):
                    continue
                if path == raw / phase / 'manifest.json' or path == raw / (phase + '-replay') / 'results.json':
                    continue
                members.append((str(path.relative_to(raw)), path))
        archive(phase + '-native.tar.gz', members)

    validation = [('logs/' + path.name, path) for path in raw.glob('*.log')]
    for name in ('test_native_rows.py', 'test_native_tile_replay.py', 'native_tile.py', 'native_tile_replay.cpp'):
        validation.append(('unit-support/' + name, runners / name))
    source_root = Path(reports['matrix'][0]['source_root'])
    for name in ('test_xir_runtime.cpp', 'test_xir_llm.cpp'):
        path = source_root / 'src/tests/unit/tile/bridge' / name
        recorded = reports['matrix'][0]['source_sha256'].get(str(path.resolve()))
        if recorded is None or digest(path.read_bytes()) != recorded:
            raise ValueError('validation source differs from captured source freeze: ' + str(path))
        validation.append(('tested-source/' + name, path))
    archive('validation.tar.gz', validation)

    # Verify every compressed byte stream/member independently after writing.
    verified_members = verified_gzip = 0
    for record in artifacts:
        data = (destination / record['path']).read_bytes()
        if len(data) != record['bytes'] or digest(data) != record['sha256']:
            raise ValueError('archive file identity mismatch')
        if record['encoding'] == 'gzip':
            decoded = gzip.decompress(data)
            if digest(decoded) != record['source_sha256'] or len(decoded) != record['source_bytes']:
                raise ValueError('raw JSON gzip mismatch')
            verified_gzip += 1
        elif record['encoding'] == 'tar.gz':
            with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tar:
                if tar.getnames() != [entry['path'] for entry in record['members']]:
                    raise ValueError('tar member inventory mismatch')
                for entry in record['members']:
                    decoded = tar.extractfile(entry['path']).read()
                    if len(decoded) != entry['bytes'] or digest(decoded) != entry['sha256']:
                        raise ValueError('tar member byte mismatch: ' + entry['path'])
                    verified_members += 1
    for path, sha in original_hashes.items():
        if digest(Path(path).read_bytes()) != sha:
            raise ValueError('original changed during archive: ' + path)
    stored_bytes = sum(entry['bytes'] for entry in artifacts)
    if stored_bytes > 12 * 1024 * 1024:
        raise ValueError('archive exceeded the bounded 12 MiB storage plan')

    inventory = dict(format='native-pointwise-archive-inventory-v1', artifacts=artifacts,
                     inventory_scope='Evidence owned by freeze.py; notes.md/tables.md are independently authored and excluded.',
                     member_paths='Relative tar paths only; raw absolute identities retained in inventory and original JSON.',
                     verification=dict(gzip_raw_json_streams=verified_gzip, tar_members=verified_members,
                                       source_files_rechecked=len(original_hashes), byte_identity_passed=True),
                     stored_artifact_bytes=stored_bytes)
    output('inventory.json', (json.dumps(inventory, indent=2) + '\n').encode())
    tensor_paths = [p for p in raw.rglob('*') if p.is_file() and p.suffix in ('.f32', '.f64')]
    phase_records = {}
    for phase, (captured, replayed, audited) in reports.items():
        phase_records[phase] = dict(capture_raw=str(raw / phase), replay_raw=str(raw / (phase + '-replay')),
                                   captured_cases=len(captured['cases']), native_timed_visits=sum(len(c['results']) for c in replayed['cases']),
                                   audit_status=audited['status'], native_output_checks=len(audited['native_output_checks']),
                                   identity_checks=audited['identity_checks'], rejected_mutations=audited['rejected_mutations'],
                                   source_root=captured['source_root'], binary=captured['binary'],
                                   baseline_manifest=captured['baseline_manifest'], full_build_gate=captured['full_build_gate'])
    provenance = dict(format='native-pointwise-archive-provenance-v1', frozen_unix=time.time(), raw_root=str(raw),
                      raw_user_spelling=str(args.raw), inventory='inventory.json', inventory_sha256=digest((destination / 'inventory.json').read_bytes()),
                      phases=phase_records, raw_json_policy='Original manifest/results bytes preserved with gzip, no path/hash/source rewrites.',
                      superseded_audit='pilot-audit.json is retained history. Cite pilot-audit-v2.json and matrix-audit.json for final 11-mutation/identity validation.',
                      primary_runner_files=list(primary_runners), unit_support='validation.tar.gz records all remaining source files required by the combined 29-unit-test receipt.',
                      helper_scope='Actual linked native entries, captured ORC objects, emitted LLVM, assembly, Inductor generated source/wrapper/library and the C++ replay helper are archived. No LLVM source is recompiled by this archiver.',
                      excluded=dict(tensor_files=len(tensor_paths), tensor_bytes=sum(p.stat().st_size for p in tensor_paths),
                                    tensor_payloads='All f32/f64 inputs, expected values and emitted outputs remain only in the raw local directories. Their exact hashes remain in the unmodified manifests/replay records.',
                                    other=['whole build binary closure', 'full compiler source tree', 'Doxygen XML', 'external LLVM/Torch/TVM installations']),
                      replayability=dict(self_contained=False, tensor_payloads_persistently_archived=False,
                                         requirements='The original raw tensor directories and baseline dependency records, compatible Darwin arm64 loader/toolchain and the exact Torch build are still required. The archived folder alone cannot run replay or full tensor audit.',
                                         regeneration='Captured commands and frozen fixture/oracle/helper source identify how to regenerate inputs, but regenerated bytes must be checked against recorded hashes. A fresh build/capture/replay/audit is a new experiment, never a substitute for this run\'s discarded or missing output bytes.',
                                         validation='The final audits actually re-read the original outputs at audit time. This archive rechecks compressed/member byte identity; it does not repeat tensor validation. Runtime/workspace guard arrays were checked during execution and were not retained.'),
                      timing_scope='Replay contains fixed-candidate native-entry host-wall comparisons under recorded desktop background load. Capture Runtime timings are not comparative. No quiet-machine, default-planner or global performance-parity claim is implied.',
                      archiver_sha256=digest(Path(__file__).read_bytes()), verification=inventory['verification'])
    output('provenance.json', (json.dumps(provenance, indent=2) + '\n').encode())
    names = sorted([record['path'] for record in artifacts] + ['inventory.json', 'provenance.json', 'freeze.py'])
    sums = ''.join(digest((destination / name).read_bytes()) + '  ' + name + '\n' for name in names)
    output('SHA256SUMS', sums.encode())
    print(json.dumps(dict(stored_artifact_bytes=stored_bytes, tar_members=verified_members,
                          raw_json_streams=verified_gzip, source_files_rechecked=len(original_hashes),
                          excluded_tensor_files=len(tensor_paths), artifacts=[record['path'] for record in artifacts]), indent=2))


if __name__ == '__main__':
    main()
