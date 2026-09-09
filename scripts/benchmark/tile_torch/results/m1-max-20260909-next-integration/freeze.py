#!/usr/bin/env python3
"""Freeze next integration and explicitly excluded native timing evidence."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-raw', type=Path, required=True)
    parser.add_argument('--integration-raw', type=Path, required=True)
    args = parser.parse_args()
    native, integration = args.native_raw.resolve(), args.integration_raw.resolve()
    if (HERE / 'provenance.json').exists():
        raise ValueError('Immutable checkpoint: choose a new destination.')
    audit = json.loads((native / 'audit.json').read_text())
    assert audit['status'] == 'passed' and not audit['performance_accepted']
    assert (audit['snapshots_re_read'], audit['timed_visits_checked'], audit['off_on_bitwise_equal_cases']) == (144, 864, 48)
    assert len(audit['rejected_in_memory_mutations']) == 8
    snapshot = json.loads((integration / 'snapshot.json').read_text())
    assert snapshot['source_commit'] == '360d9791e344da06bcbde4dc871526aa336900b9'
    user_hashes = {}
    for line in (native / 'premerge-user-sha256.txt').read_text().splitlines():
        digest, name = line.split(None, 1)
        user_hashes[name] = digest
        assert sha((ROOT / name).read_bytes()) == digest, name
    assert len(user_hashes) == 11
    dependency_checkouts = {}
    for entry in snapshot['submodules']:
        if entry['path'] != '.':
            current = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT / entry['path'], text=True).strip()
            assert current == entry['checkout_commit'], entry['path']
            dependency_checkouts[entry['path']] = current
    verification = json.loads((integration / 'verification-qualified.json').read_text())
    overlay = verification['source_overlays']
    assert set(overlay) == {'src/tests/unit/runtime/test_metal_codegen_regressions.cpp'}
    for name, digest in overlay.items():
        assert sha((ROOT / name).read_bytes()) == digest
    for name, digest in snapshot['source_sha256'].items():
        assert sha((Path(snapshot['source']) / name).read_bytes()) == overlay.get(name, digest), name
    assert verification['full_build_exit_code'] == 0
    assert verification['source_commit'] == snapshot['source_commit']
    assert verification['full_build_log_sha256'] == sha((integration / verification['full_build_log']).read_bytes())
    assert verification['passed'] and verification['finished_unix'] >= verification['started_unix']
    assert {r['name'] for r in verification['runs']} == {
        'ctest-tile', 'ctest-xir-simd', 'full-build-final-gate', 'ctest-metal-fixed', 'metal-local-only',
        'ctest-llm-fused', 'syntax-final', 'ctest-runtime-clean', 'format-changed'}
    assert all(r['exit_code'] == 0 for r in verification['runs'])
    expected_tests = {'ctest-tile': 35, 'ctest-xir-simd': 79, 'ctest-metal-fixed': 2,
                      'ctest-runtime-clean': 1, 'ctest-llm-fused': 1}
    for run in verification['runs']:
        assert sha((integration / run['log']).read_bytes()) == run['log_sha256']
        if run['name'] in expected_tests:
            cases = list(ET.parse(integration / run['junit']).getroot().iter('testcase'))
            assert run['tests'] == len(cases) == expected_tests[run['name']]
            assert run['failures'] == run['skipped'] == 0
            assert all(c.find('failure') is None and c.find('error') is None and c.find('skipped') is None for c in cases)
    receipts = [verification]
    for name, digest in verification['prior_receipts'].items():
        assert sha((integration / name).read_bytes()) == digest
        prior = json.loads((integration / name).read_text())
        assert not prior['passed'] and prior['finished_unix'] >= prior['started_unix']
        assert any(r['exit_code'] for r in prior['runs'])
        receipts.append(prior)
    inherited = verification['inherited_formatting']
    assert inherited['exit_code'] == 1 and inherited['diagnostics_identical_to_changed_file']
    assert len(inherited['diagnostic_positions']) == 7
    assert sha((integration / inherited['log']).read_bytes()) == inherited['log_sha256']
    docs = json.loads((integration / 'documentation.json').read_text())
    assert docs['passed'] and all(r['exit_code'] == 0 for r in docs['runs'])
    for name, digest in docs['source_sha256'].items():
        assert sha((ROOT / name).read_bytes()) == digest, name
    qa = json.loads((integration / 'docs-qa/receipt.json').read_text())
    assert qa['passed'] and len(qa['receipts']) == 6
    artifacts = {}

    def store(name, data, origin, role):
        target = HERE / name
        if target.exists():
            raise ValueError('Refusing to overwrite evidence: ' + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        encoded = gzip.compress(data, mtime=0) if name.endswith('.gz') else data
        target.write_bytes(encoded)
        artifacts[name] = dict(origin=str(origin), role=role, source_sha256=sha(data), archived_sha256=sha(encoded),
                               source_bytes=len(data), archived_bytes=len(encoded))

    def archive(name, source, role):
        store(name, Path(source).read_bytes(), source, role)

    archive('audit.json', native / 'audit.json', 'native correctness and exclusion audit')
    archive('verification.json', integration / 'verification-qualified.json', 'qualified integration checks with explicit test-only overlay')
    archive('verification-initial.json', integration / 'verification.json', 'retained initial failed invocations')
    archive('verification-recheck.json', integration / 'verification-final.json', 'retained second failed invocations')
    archive('source-snapshot.json.gz', integration / 'snapshot.json', 'initial committed source and recursively pinned dependencies; overlay is recorded separately')
    for name in overlay:
        archive('overlay/' + name, ROOT / name, 'only source change from the initial archive; test-only argument guard')
    archive('snapshot.py', integration / 'snapshot.py', 'snapshot authoring source')
    archive('audit_replays.py', native / 'audit_replays.py', 'independent native audit source')
    archive('observe_replay.py', native / 'observe_replay.py', 'first process observer source')
    archive('observe_replay_v2.py', native / 'observe_replay_v2.py', 'second process observer with quiet preflight')
    archive('pilot.py', native / 'replay-v2/pilot.py', 'unchanged native replay source')
    archive('run_tests.py', integration / 'run_tests.py', 'integration test supervisor')
    archive('run_rechecks.py', integration / 'run_rechecks.py', 'second supervisor; failed A/B environment preserved')
    archive('finish_verification.py', integration / 'finish_verification.py', 'final supervisor; policy switches left to Runtime fixtures')
    archive('documentation.json', integration / 'documentation.json', 'strict docs build and rendered QA')
    archive('run_docs_checks.py', integration / 'run_docs_checks.py', 'documentation check supervisor')
    archive('qa-docs.cjs', integration / 'qa-docs.cjs', 'rendered desktop/mobile QA source')
    archive('docs-qa.json', integration / 'docs-qa/receipt.json', 'rendered desktop/mobile QA receipt; screenshots stay local')
    archive('configure-command.json', integration / 'configure-command.json', 'exact build configuration')
    for name in ('snapshot.log', 'snapshot-v2.log', 'spirv-tools-fetch.log', 'configure.log', 'full-build.log',
                 'full-build-fixed.log', 'metal-stack.log', 'metal-lldb.log', 'metal-full-lldb.log',
                 'pointwise-ab-recheck.log', 'test-supervisor.log', 'recheck-supervisor.log',
                 'finish-supervisor.log', 'format-pristine.log'):
        archive('logs/' + name + '.gz', integration / name, 'integration preparation; first snapshot failure retained')
    for receipt in receipts + [docs]:
        for run in receipt['runs'] + receipt.get('retained_failures', []):
            assert sha((integration / run['log']).read_bytes()) == run['log_sha256']
            target = 'logs/' + run['log'] + '.gz'
            if target not in artifacts:
                archive(target, integration / run['log'], 'executed check; both successes and failures retained')
            if run.get('junit') and run['junit'] not in artifacts:
                archive(run['junit'], integration / run['junit'], 'CTest case-level result')
    for name in ('audit-final.log', 'full-build.log', 'source-main.log', 'source-isolated.log',
                 'binaries-before.log', 'binaries-after.log', 'next-merge.log', 'premerge-user-sha256.txt',
                 'premerge-submodule-heads.txt', 'postmerge-submodule-heads.txt'):
        archive('logs/native-' + name + '.gz', native / name, 'premerge identity, preservation or audit evidence')
    for index, run in enumerate(audit['runs'], 1):
        folder = native / run['name']
        original = folder / 'results.json'
        assert sha(original.read_bytes()) == run['replay_sha256']
        archive(f'excluded-native-replay-v{index}.json.gz', original, 'EXCLUDED performance; additional correctness only')
        archive(f'logs/excluded-native-replay-v{index}.log.gz', native / (run['name'] + '.log'), 'EXCLUDED performance')
        host_path = native / ('host-observation.json' if index == 1 else 'host-observation-v2.json')
        assert sha(host_path.read_bytes()) == run['host_observation_sha256']
        host = json.loads(host_path.read_text())
        assert host['heavy_activity_seen'] and host['benchmark_exit_code'] == 0
        # Keep timing and matched interference evidence, not an inventory of
        # the user's unrelated desktop apps or another workspace's paths.
        for key in ('preflight', 'observations'):
            for item in host.get(key, []):
                item['active_cpu_processes'] = len(item.pop('active_cpu'))
                matched = []
                for line in item['heavy']:
                    pid, parent, cpu, executable = line.split(None, 3)
                    matched.append(dict(pid=int(pid), parent=int(parent), cpu_percent=float(cpu), executable=Path(executable).name))
                item['heavy'] = matched
        host['raw_sha256'] = run['host_observation_sha256']
        host['redaction'] = 'Matched executable basenames and numeric observations retained; unrelated app inventory and external workspace paths omitted. Original remains local.'
        store(f'host-observation-v{index}.json.gz', (json.dumps(host, indent=2) + '\n').encode(), host_path, 'privacy-minimized coactivity evidence')
    for name in ('libtvm_compiler.dylib', 'libtvm_runtime.dylib', 'libtvm_ffi.dylib'):
        path = Path('/Users/mike/.cache/luisa-tile/tvm-fragment-build.0ToXYr/lib') / name
        assert path.is_file()
    binaries = [p for p in (integration / 'build/bin').glob('libluisa-*') if p.is_file()]
    binaries += [Path('/Users/mike/.cache/luisa-tile/tvm-fragment-build.0ToXYr/lib') / n for n in
                 ('libtvm_compiler.dylib', 'libtvm_runtime.dylib', 'libtvm_ffi.dylib')]
    provenance = dict(frozen_unix=time.time(), parent_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                      merged_next='8911828eb234a37d1059bb9b01e82d2915624f80', merge_commit=snapshot['source_commit'],
                      native_compiler_commit='67330a633ee7901034ab4a9360d9cd15be904b1f',
                      native_raw=str(native), integration_raw=str(integration),
                      source_contract=snapshot['contract'] + ' Final build additionally contains the separately fingerprinted Metal test-only argv guard overlay.',
                      source_overlays=overlay, pinned_repositories=len(snapshot['submodules']),
                      preserved_user_file_sha256=user_hashes, preserved_dependency_checkouts=dependency_checkouts,
                      integration_binary_sha256={str(p): sha(p.read_bytes()) for p in binaries},
                      native_code_checkpoint='../m1-max-20260909-xir-pointwise/provenance.json',
                      native_code_checkpoint_sha256=audit['previous_checkpoint_sha256'],
                      new_performance_claim=False, new_cost_calibration=False, pointwise_default_changed=False,
                      selection='Both premerge fixed 24-case attempts excluded as complete cohorts after observed foreign render interference. No cherry-picked clean cases, performance ratios or defaults are promoted.',
                      payload_retention='Full tensor outputs stay in the native raw directory; Git keeps hashes and independent checks. Ephemeral guards were checked during replay.',
                      artifacts=artifacts)
    (HERE / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    for name, item in artifacts.items():
        encoded = (HERE / name).read_bytes()
        assert sha(encoded) == item['archived_sha256']
        assert sha(gzip.decompress(encoded) if name.endswith('.gz') else encoded) == item['source_sha256']
    print('PASS:', len(artifacts), 'archives checked; integration separated from excluded premerge timings')


if __name__ == '__main__':
    main()
