#!/usr/bin/env python3
"""Archive expression-fusion correctness/code evidence, never the excluded ranking."""
import argparse
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    args = parser.parse_args()
    raw = args.raw.resolve()
    assert not (HERE / 'provenance.json').exists(), 'Use a new immutable checkpoint destination.'
    audit = json.loads((raw / 'audit.json').read_text())
    assert audit['status'] == 'passed' and not audit['performance_accepted']
    assert (audit['snapshots_re_read'], audit['smoke_visits'], audit['bitwise_equal_cases']) == (138, 72, 24)
    assert audit['excluded_timed_visits'] == 96 and len(audit['rejected_mutations']) == 8
    docs = json.loads((raw / 'documentation-v2.json').read_text())
    assert docs['passed'] and all(run['exit_code'] == 0 for run in docs['runs'])
    for name, digest in docs['changed_source_sha256'].items():
        assert sha((REPO / name).read_bytes()) == digest, name
    qa = json.loads((raw / 'docs-qa-v2/receipt.json').read_text())
    assert qa['passed'] and len(qa['receipts']) == 6
    widths = json.loads((raw / 'verification-widths-v2.json').read_text())
    assert widths['passed'] and all(run['assertions'] and run['selected_test_passed'] for run in widths['runs'])
    snapshot = json.loads((raw / 'source-snapshot.json').read_text())
    for name in snapshot['overlay']:
        assert sha((REPO / name).read_bytes()) == snapshot['source_sha256'][name]
    records = json.loads((raw / 'pinned-repositories.json').read_text())
    dependencies = {}
    for record in records:
        if record['path'] != '.':
            revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO / record['path'], text=True).strip()
            assert revision == record['checkout_commit']
            dependencies[record['path']] = revision
    preserved = {}
    for line in Path('/tmp/luisa-pointwise-native.1P87uU/premerge-user-sha256.txt').read_text().splitlines():
        digest, name = line.split(None, 1)
        assert sha((REPO / name).read_bytes()) == digest, name
        preserved[name] = digest
    artifacts = {}

    def store(name, data, origin, role):
        target = HERE / name
        assert not target.exists(), name
        target.parent.mkdir(parents=True, exist_ok=True)
        encoded = gzip.compress(data, mtime=0) if name.endswith('.gz') else data
        target.write_bytes(encoded)
        artifacts[name] = dict(origin=str(origin), role=role, source_sha256=sha(data), archived_sha256=sha(encoded),
                               source_bytes=len(data), archived_bytes=len(encoded))

    def archive(name, origin, role):
        store(name, Path(origin).read_bytes(), origin, role)

    for name in ('audit.json', 'verification.json', 'verification-widths.json', 'verification-widths-v2.json',
                 'configure-command.json', 'pinned-repositories.json', 'documentation.json', 'documentation-v2.json'):
        archive(name, raw / name, 'Executed evidence; initial empty-width and audit mistakes are explicitly superseded.')
    for name in ('prepare.py', 'capture.py', 'verify.py', 'verify_widths.py', 'replay.py', 'audit.py', 'check_docs.py', 'qa-docs.cjs'):
        archive('drivers/' + name, raw / name, 'Checkpoint preparation, execution or validation source; see retained harness caveats.')
    archive('source-snapshot.json.gz', raw / 'source-snapshot.json', 'Pinned source plus exactly the recorded eight-file overlay.')
    archive('candidate.patch.gz', raw / 'candidate.patch', 'Explicit candidate source changes from 2634be45d.')
    archive('capture.json.gz', raw / 'capture/manifest.json', 'Actual ORC entries, fixed mapping, controls and source/binary hashes.')
    archive('native-verify.json', raw / 'native-verify/results.json', 'Full native-entry correctness smoke; timings are non-comparative.')
    archive('excluded-replay.json.gz', raw / 'replay-1/results.json', 'Interrupted, contended partial cohort; excluded in full.')
    archive('drivers/pilot.py', raw / 'native-verify/pilot.py', 'Unmodified common native replay driver reused with the new capture manifest.')
    for path in sorted((raw / 'native-verify/runner-sources').iterdir()):
        archive('drivers/' + path.name, path, 'Exact replay helper and numerical reference sources.')
    for path in sorted(raw.glob('*.log')):
        archive('logs/' + path.name + '.gz', path, 'Observed command output, including failed/empty/terminated invocations.')
    for name in ('tile.xml', 'xir-simd.xml'):
        archive(name + '.gz', raw / name, 'Case-level CTest receipt.')
    archive('docs-qa.json', raw / 'docs-qa-v2/receipt.json', 'Final desktop/mobile DOM and screenshot receipts; images retained locally.')
    host_path = raw / 'host-replay-1.json'
    host = json.loads(host_path.read_text())
    assert not host['performance_qualified'] and host['replay_exit_code'] == 143
    sanitized = {key: value for key, value in host.items() if key not in ('preflight', 'observations')}
    sanitized.update(original_sha256=sha(host_path.read_bytes()),
                     privacy='Foreign process identifiers, application names and paths are omitted. Percentages include the benchmark process where active.')
    for label in ('preflight', 'observations'):
        sanitized[label] = [dict(unix=item['unix'], heavy_count=len(item['heavy']),
                                 active_cpu_percent=[float(line.split(None, 3)[2]) for line in item['active_cpu']]) for item in host[label]]
    store('coactivity.json', (json.dumps(sanitized, indent=2) + '\n').encode(), host_path, 'Privacy-minimized exclusion evidence, not a hardware-exclusivity claim.')
    capture = json.loads((raw / 'capture/manifest.json').read_text())
    for case in capture['cases']:
        name = case['operation'] + '-' + 'x'.join(map(str, case['dimensions']))
        for variant in ('off', 'on'):
            source = raw / 'capture' / name / variant
            archive('native/' + name + '/' + variant + '.ll.gz', source / 'kernel.ll', 'Actual captured LLVM, not retimed reconstructed source.')
            archive('native/' + name + '/' + variant + '.s.gz', source / 'assembly.stdout.log', 'Disassembly of the actual ORC object.')
            objects = list((source / 'object').glob('*.o'))
            assert len(objects) == 1 and sha(objects[0].read_bytes()) == case['entries'][variant]['object_sha256']
            archive('native/' + name + '/' + variant + '.o.gz', objects[0], 'Actual ORC object; entry dylib can be relinked with the recorded command.')
        inductor = case['entries']['inductor']
        assert sha(Path(inductor['source']).read_bytes()) == inductor['source_sha256']
        archive('native/' + name + '/inductor.cpp', inductor['source'], 'Generated C++ for the reused, fingerprinted Inductor native entry.')
    provenance = dict(frozen_unix=time.time(), raw=str(raw), source_commit=snapshot['source_commit'],
                      merged_next='8911828eb234a37d1059bb9b01e82d2915624f80', source_overlay=snapshot['overlay'],
                      source_contract='Recursive pinned Git archives, followed by the explicit candidate overlay. Not the dirty source checkout.',
                      performance_accepted=False, defaults_changed=False, cost_calibrated=False,
                      preserved_user_file_sha256=preserved, preserved_dependency_checkouts=dependencies,
                      host=dict(cpu='Apple M1 Max', cpu_cores=10, memory_gib=64, os='macOS 26.6.2', build='25G83'),
                      artifacts=artifacts,
                      payload_retention='Large input/output arrays, linked dylibs and screenshots remain local at the recorded paths; source, ORC objects, disassembly, output hashes, validation and exact commands are retained here.')
    (HERE / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('Archived', len(artifacts), 'artifacts;', sum(item['archived_bytes'] for item in artifacts.values()), 'bytes; no performance claim')


if __name__ == '__main__':
    main()
