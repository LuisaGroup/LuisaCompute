#!/usr/bin/env python3
"""Freeze the completed outlined-tail experiment, without executing kernels.

Run against the original raw directories and unchanged source snapshots. The
destination must be fresh except for this script and independently authored
notes.md/tables.md. --destination supports deterministic regeneration elsewhere;
the candidate patch in this script's directory is reused after its rejection.
No manifests, source identities, timing values or original paths are rewritten.
"""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


BASE_COMMIT = '147700bd431eb054694ba649817b7f50a335b2b1'
CANDIDATE_SOURCES = (
    'src/backends/simd/llvm/llvm_schedule_codegen.cpp',
    'src/tests/unit/simd/test_llvm_schedule_codegen.cpp',
)
VALIDATION_SOURCES = CANDIDATE_SOURCES + (
    'src/tests/common/tile_llm_test_utils.h',
    'src/tests/unit/tile/bridge/test_xir_runtime.cpp',
    'src/tests/unit/tile/bridge/test_xir_llm.cpp',
)
# Capture/replay sources also have independently recorded manifest identities.
# Supplemental tests were not hashed in the historical unit receipt. These
# pinned hashes identify the snapshot available when this archive was made;
# they must not be presented as a contemporaneous source attestation for it.
SUPPLEMENTAL_RUNNERS = {
    'audit_native_pointwise.py': 'd93c047c8a6ee7de013bbbc6394f937aaee9ffe48468d0bf6d337defb69d5bf3',
    'test_audit_native_pointwise.py': 'c43f898ac113efbb75e83bbfa72d1f1fa9f2638a2e50f66589c84128128320ea',
    'test_native_pointwise.py': '85dc2069a99c6d38035753131c24c0508f89158b002a5afb7240b12d6d556807',
    'native_pointwise_no_specialization.py': '1a1e41536717106958af08b92c5b37bc15323ff9999aa1092a5524481918d4ee',
    'test_native_pointwise_no_specialization.py': '07f6729dfcf1ab97a00a79ee310e209ca19111d045a51ca1c5546e4facede0a3',
    'test_native_rows.py': '551bbce5f8007a828fb01d4d2221d8091dcbe290eb452e4b5889a3dd055a4b20',
    'native_tile.py': 'baa483ef0ead9243d53e650fed4ac30380492b92c9e60d4226eb5acf8f8545b7',
    'test_native_tile_replay.py': 'bf846997c70ced12793cd40567902c81271ae72c9f2edd73118c55ada34c28fb',
    'native_tile_replay.cpp': '2be9010d56ec9ca1e0572fa58e24bb15dfd324b6eae38075baabdfabde6b1504',
}
TENSOR_SUFFIXES = {'.f32', '.f64'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode_json(value):
    return (json.dumps(value, indent=2) + '\n').encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    raw, repository = args.raw.resolve(), args.repository.resolve()
    script = Path(__file__).resolve()
    destination = args.destination.resolve() if args.destination else script.parent
    destination.mkdir(parents=True, exist_ok=True)
    if any(p.name not in {'freeze.py', 'notes.md', 'tables.md'} for p in destination.iterdir()):
        raise ValueError('destination is not fresh')

    paths = {
        'capture': raw / 'capture/manifest.json',
        'failed_replay': raw / 'replay/results.json',
        'replay': raw / 'replay-v2/results.json',
        'audit': raw / 'audit.json',
    }
    reports = {name: json.loads(path.read_bytes()) for name, path in paths.items()}
    capture, failed, replay, audit = (reports[key] for key in paths)
    semantics = {
        'off': {'pointwise_fusion': True, 'outlined_packet_tail': False},
        'on': {'pointwise_fusion': True, 'outlined_packet_tail': True},
    }
    if capture['status'] != 'captured' or replay['status'] != 'passed' or audit['status'] != 'passed':
        raise ValueError('the final experiment is not complete')
    if len(capture['cases']) != 24 or len(replay['cases']) != 24 or \
            sum(len(case['results']) for case in replay['cases']) != 432:
        raise ValueError('unexpected final cohort size')
    if failed['status'] != 'error' or failed['cases'] or \
            failed['error'] != "ModuleNotFoundError: No module named 'torch'":
        raise ValueError('the retained failed attempt does not match its recorded history')
    if any(report.get('comparison') != 'outlined-tail' or report.get('variant_semantics') != semantics
           for report in reports.values()):
        raise ValueError('comparison identity differs between reports')
    if audit['identity_checks'] != 1413 or len(audit['native_output_checks']) != 72 or \
            len(audit['comparison_checks']) != 24 or len(audit['rejected_mutations']) != 11 or \
            len(audit['rejected_comparison_mutations']) != 8:
        raise ValueError('unexpected final audit coverage')
    for name, field in [('capture', 'manifest_sha256'), ('replay', 'replay_sha256')]:
        if digest(paths[name].read_bytes()) != audit[field]:
            raise ValueError('audit refers to a different raw report')
    for report in (failed, replay):
        if report['manifest_sha256'] != digest(paths['capture'].read_bytes()):
            raise ValueError('replay attempt refers to a different capture')

    expected_hashes = {}
    for report in (capture, failed, replay):
        for group in ('artifact_sha256', 'runner_sha256', 'source_sha256'):
            for name, sha in report.get(group, {}).items():
                path = str(Path(name).resolve())
                if path in expected_hashes and expected_hashes[path] != sha:
                    raise ValueError('conflicting recorded identity: ' + path)
                expected_hashes[path] = sha
    expected_hashes[str(Path(capture['baseline_manifest']).resolve())] = capture['baseline_manifest_sha256']
    runners = repository / 'scripts/benchmark/tile_torch'
    for name, sha in SUPPLEMENTAL_RUNNERS.items():
        expected_hashes[str((runners / name).resolve())] = sha
    if audit['auditor_sha256'] != SUPPLEMENTAL_RUNNERS['audit_native_pointwise.py']:
        raise ValueError('auditor snapshot is not the one used by the final audit')
    original_hashes = {}

    def read_source(path, required_identity=False):
        path = path.resolve()
        data = path.read_bytes()
        sha = digest(data)
        expected = expected_hashes.get(str(path))
        if required_identity and expected is None:
            raise ValueError('missing recorded identity: ' + str(path))
        if expected is not None and sha != expected:
            raise ValueError('source differs from recorded identity: ' + str(path))
        if str(path) in original_hashes and original_hashes[str(path)] != sha:
            raise ValueError('source changed while archiving: ' + str(path))
        original_hashes[str(path)] = sha
        return data

    def output(name, data):
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(data)
        return {'path': name, 'bytes': len(data), 'sha256': digest(data)}

    if destination != script.parent:
        output('freeze.py', script.read_bytes())
    source_root = Path(capture['source_root'])
    tested_sources = {}
    for name in VALIDATION_SOURCES:
        path = source_root / name
        tested_sources[name] = digest(read_source(path, required_identity=True))

    # Preserve only the two experimental C++ changes, not the unrelated WIP.
    # A later repeat reuses the saved patch because the candidate is rejected.
    saved_patch = script.parent / 'candidate.patch'
    if saved_patch.exists():
        patch = saved_patch.read_bytes()
    else:
        if subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip() != BASE_COMMIT:
            raise ValueError('initial patch snapshot requires the recorded base commit')
        for name in CANDIDATE_SOURCES:
            if digest((repository / name).read_bytes()) != tested_sources[name]:
                raise ValueError('working candidate differs from tested source: ' + name)
        patch = subprocess.check_output([
            'git', '-C', str(repository), 'diff', '--binary', '--full-index', BASE_COMMIT, '--', *CANDIDATE_SOURCES])
    headers = [line for line in patch.decode().splitlines() if line.startswith('diff --git ')]
    if headers != [f'diff --git a/{name} b/{name}' for name in CANDIDATE_SOURCES]:
        raise ValueError('candidate patch contains missing or unrelated paths')
    base_sources = {}
    for name in CANDIDATE_SOURCES:
        base = subprocess.check_output(['git', '-C', str(repository), 'show', f'{BASE_COMMIT}:{name}'])
        base_sources[name] = digest(base)

    artifacts = []
    record = output('candidate.patch', patch)
    record.update(encoding='identity', role='rejected experimental candidate only; scoped two-file git diff',
                  base_commit=BASE_COMMIT, base_source_sha256=base_sources, tested_source_sha256={
                      name: tested_sources[name] for name in CANDIDATE_SOURCES})
    artifacts.append(record)

    def copy_source(name, path, compressed=False, role='evidence'):
        data = read_source(path)
        record = output(name, gzip.compress(data, compresslevel=9, mtime=0) if compressed else data)
        record.update(encoding='gzip' if compressed else 'identity', role=role,
                      source=str(path.resolve()), source_bytes=len(data), source_sha256=digest(data))
        artifacts.append(record)

    copy_source('capture.json.gz', paths['capture'], True, 'original final capture manifest')
    copy_source('replay-failed.json.gz', paths['failed_replay'], True, 'original failed attempt; zero visits; not performance data')
    copy_source('replay.json.gz', paths['replay'], True, 'original final native replay; 432 timed visits')
    copy_source('audit.json', paths['audit'], role='original final independent numerical/identity/comparison audit')
    copy_source('baseline-manifest.json.gz', Path(capture['baseline_manifest']), True, 'original inherited baseline manifest')

    def archive(name, members):
        encoded, entries = io.BytesIO(), []
        with gzip.GzipFile(fileobj=encoded, mode='wb', compresslevel=9, mtime=0, filename='') as zipped:
            with tarfile.open(fileobj=zipped, mode='w|', format=tarfile.PAX_FORMAT) as tar:
                seen = set()
                for member, path in sorted(members):
                    if member.startswith('/') or '..' in Path(member).parts or member in seen:
                        raise ValueError('unsafe or duplicate tar member: ' + member)
                    seen.add(member)
                    content = read_source(path)
                    info = tarfile.TarInfo(member)
                    info.size = len(content)
                    info.mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
                    info.mtime = info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    tar.addfile(info, io.BytesIO(content))
                    entries.append(dict(path=member, source=str(path.resolve()), bytes=len(content),
                                        sha256=digest(content), mode=info.mode,
                                        checked_against_recorded_or_pinned_identity=str(path.resolve()) in expected_hashes))
        record = output(name, encoded.getvalue())
        record.update(encoding='tar.gz', role='compressed non-tensor evidence bundle', members=entries,
                      uncompressed_member_bytes=sum(entry['bytes'] for entry in entries))
        artifacts.append(record)

    individually_copied = set(paths.values())
    evidence, tensors = [], []
    for path in raw.rglob('*'):
        if not path.is_file():
            continue
        if path.suffix in TENSOR_SUFFIXES:
            tensors.append(path)
        elif path not in individually_copied:
            evidence.append((str(path.relative_to(raw)), path))
    archive('evidence.tar.gz', evidence)
    validation = [('tested-source/' + name, source_root / name) for name in VALIDATION_SOURCES]
    validation += [('runners/' + name, runners / name) for name in SUPPLEMENTAL_RUNNERS]
    # All four capture/replay runner snapshots are already in evidence.tar.gz;
    # additionally validate their original identities and each frozen copy.
    for report, directory in [(capture, 'capture'), (failed, 'replay'), (replay, 'replay-v2')]:
        for path, sha in report['runner_sha256'].items():
            source = Path(path)
            frozen = raw / directory / 'runner-sources' / source.name
            if digest(read_source(frozen)) != sha or digest(read_source(source, required_identity=True)) != sha:
                raise ValueError('frozen runner differs from recorded original: ' + path)
    archive('validation.tar.gz', validation)

    verified_members = verified_gzip = 0
    for record in artifacts:
        data = (destination / record['path']).read_bytes()
        if len(data) != record['bytes'] or digest(data) != record['sha256']:
            raise ValueError('written artifact identity mismatch')
        if record['encoding'] == 'gzip':
            decoded = gzip.decompress(data)
            if len(decoded) != record['source_bytes'] or digest(decoded) != record['source_sha256']:
                raise ValueError('compressed raw JSON byte mismatch')
            verified_gzip += 1
        elif record['encoding'] == 'tar.gz':
            with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tar:
                if tar.getnames() != [entry['path'] for entry in record['members']]:
                    raise ValueError('tar member list differs from inventory')
                for entry in record['members']:
                    decoded = tar.extractfile(entry['path']).read()
                    if len(decoded) != entry['bytes'] or digest(decoded) != entry['sha256']:
                        raise ValueError('tar member byte mismatch: ' + entry['path'])
                    verified_members += 1
    for path, sha in original_hashes.items():
        if digest(Path(path).read_bytes()) != sha:
            raise ValueError('source changed during archive: ' + path)

    stored_bytes = sum(record['bytes'] for record in artifacts)
    verification = dict(compressed_json_streams=verified_gzip, tar_members=verified_members,
                        source_files_rechecked=len(original_hashes), byte_identity_passed=True)
    inventory = dict(format='native-outlined-tail-archive-inventory-v1', artifacts=artifacts,
                     ownership='Only files produced by freeze.py; root-authored notes.md/tables.md are excluded.',
                     raw_identity_policy='Raw reports are byte-preserved. Tar paths are relative; original absolute identities remain in member records.',
                     verification=verification, stored_artifact_bytes=stored_bytes)
    output('inventory.json', encode_json(inventory))
    provenance = dict(
        format='native-outlined-tail-archive-provenance-v1',
        raw_root=str(raw), raw_user_spelling=str(args.raw), inventory='inventory.json',
        inventory_sha256=digest((destination / 'inventory.json').read_bytes()),
        raw_reports={name: str(path) for name, path in paths.items()},
        disposition='Rejected experimental candidate; not a deployed compiler or planner optimization. The scoped patch is evidence only.',
        comparison='Both variants keep pointwise fusion and full-packet specialization enabled. Only the real narrow-tail call-site NoInline policy differs.',
        variant_semantics=semantics, captured_cases=24, native_timed_visits=432,
        failed_attempt=dict(status=failed['status'], timed_visits=0, error=failed['error'],
                            policy='Preserved unchanged; never pooled with successful timings or counted as a kernel failure.'),
        audit=dict(status=audit['status'], identity_checks=audit['identity_checks'],
                   native_output_checks=len(audit['native_output_checks']), comparison_checks=len(audit['comparison_checks']),
                   rejected_mutations=audit['rejected_mutations'], rejected_comparison_mutations=audit['rejected_comparison_mutations'],
                   auditor_sha256=audit['auditor_sha256']),
        source_root=capture['source_root'], binary=capture['binary'], baseline_manifest=capture['baseline_manifest'],
        full_build_gate=capture['full_build_gate'], fixed_environment=capture['fixed_environment'],
        candidate=dict(base_commit=BASE_COMMIT, patch='candidate.patch', patch_sha256=digest(patch),
                       base_source_sha256=base_sources, tested_source_sha256=tested_sources,
                       scope='Only the two C++ candidate files are included in the patch. The five tested source/fixture snapshots match capture source hashes.'),
        raw_json_policy='Original JSON bytes are gzip-preserved; no path, hash, source or timing metadata is rewritten.',
        runner_freeze='The four original runners match capture/replay hashes, as do each attempt\'s frozen runner-sources copies. The current auditor matches the final audit hash.',
        supplemental_source_caveat='The supplemental unit tests and support sources have pinned archival hashes. The original 52-test unit receipt did not hash those files; this archive does not invent a contemporaneous unit-source attestation or claim the auditor tests ran with that earlier receipt.',
        included='Actual emitted LLVM, ORC object and assembly, linked entry dylibs, Inductor sources/wrappers/libraries, ABI header, C++ helper dylib, capture/export/link/replay command metadata, all raw logs including both replay attempts, frozen runners, tested sources and rejected patch.',
        excluded=dict(tensor_files=len(tensors), tensor_bytes=sum(path.stat().st_size for path in tensors),
                      tensor_hash_policy='Original input/expected/output hashes remain in unmodified reports. Tensor bytes are not re-read or revalidated by this archiver.',
                      other=['whole build binary closure', 'full compiler source tree', 'external LLVM/Torch/TVM installations', 'Doxygen XML']),
        replayability=dict(self_contained=False, tensor_payloads_persistently_archived=False,
                           requirements='Original raw tensors, the recorded baseline dependencies and a compatible Darwin arm64 loader/toolchain/Torch are required. This folder alone cannot replay kernels or repeat the full numerical audit.',
                           regeneration='Frozen commands and fixture/oracle sources describe regeneration, but regenerated inputs must match recorded hashes. A new capture/replay/audit is a new experiment, not a replacement for missing original output bytes.',
                           validation='The final audit re-read original outputs at audit time. This archival step only verifies file/compression identity; runtime/workspace guards were execution-time checks, not retained payloads.'),
        timing_scope='Fixed-candidate one-host-thread native-entry host-wall replay under recorded desktop background load. Capture Runtime times are not comparative. No quiet-machine, default-planner or global parity claim is implied.',
        archiver_sha256=digest(script.read_bytes()), verification=verification)
    output('provenance.json', encode_json(provenance))
    names = sorted([record['path'] for record in artifacts] + ['inventory.json', 'provenance.json', 'freeze.py'])
    output('SHA256SUMS', ''.join(digest((destination / name).read_bytes()) + '  ' + name + '\n' for name in names).encode())
    total = sum((destination / name).stat().st_size for name in names + ['SHA256SUMS'])
    if total > 12 * 1024 * 1024:
        raise ValueError('non-tensor checkpoint exceeds the 12 MiB storage budget')
    print(json.dumps(dict(status='archived', checkpoint_bytes=total, verification=verification,
                          excluded_tensor_files=len(tensors), candidate_source_sha256={
                              name: tested_sources[name] for name in CANDIDATE_SOURCES}), indent=2))


if __name__ == '__main__':
    main()
