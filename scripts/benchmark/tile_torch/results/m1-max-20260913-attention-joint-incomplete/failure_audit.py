"""Audit an INCOMPLETE capture cohort without executing any native artifact.

This is deliberately not the frozen complete-cohort audit.py. Successful exit
means the incomplete/error evidence is internally consistent, not that the
performance experiment passed. Uses only the supplied archive/extracted files.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile

MARKER = 'luisa-attention-joint-copy-mma.h94FRk'
FAILED = 'decode-mha-d64-m0-c4'
ARMS = ('m0-c0', 'm0-c4', 'm4-c0', 'm4-c4', 'm4-c4-simplified')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: ' + key)
        result[key] = value
    return result


def load(path):
    return json.loads(path.read_text(), object_pairs_hook=unique,
                      parse_constant=lambda value: require(False, 'nonfinite JSON: ' + value))


def audit(raw, source_archive):
    raw = raw.resolve(strict=True)

    def file(relative):
        name = PurePosixPath(relative)
        require(relative and not name.is_absolute() and '..' not in name.parts, 'unsafe path')
        path = raw
        for component in name.parts:
            path /= component
            require(not path.is_symlink(), 'symlinked evidence: ' + relative)
        require(path.is_file(), 'missing evidence: ' + relative)
        return path

    plan = load(file('plan.json'))
    provenance = load(file('provenance.json'))
    declared = load(file('predeclared.json'))
    captures = load(file('captures.json'))
    require(plan['controls'] == declared, 'predeclared controls changed')
    require(digest(file('plan.json')) == captures['plan_sha256'] == provenance['plan_sha256'], 'plan identity')
    require(captures['status'] == 'incomplete', 'failure audit must not certify a complete cohort')
    require([v['name'] for v in declared['variants']] == list(ARMS), 'variant inventory')
    require([declared[k] for k in ('captures', 'pair_replays', 'visits', 'samples')] == [30, 30, 360, 2520], 'declared target counts')
    require(not (raw / 'replays.json').exists() and not list(raw.glob('replay-*')), 'replay evidence exists: incomplete-only audit is not applicable')
    identities = plan['identities']
    require(identities['owned_source_sha256'] == identities['selected_source_sha256'], 'owned selected-source receipt mismatch')
    for historical, expected in identities['files_sha256'].items():
        parts = PurePosixPath(historical).parts
        if MARKER in parts:
            relative = PurePosixPath(*parts[parts.index(MARKER) + 1:]).as_posix()
            require(digest(file(relative)) == expected, 'frozen local harness/gate changed: ' + relative)
    require(digest(source_archive) == provenance['source_archive_sha256'], 'source archive changed')
    source_members = {}
    with tarfile.open(source_archive, 'r:gz') as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            require(member.isfile() and not name.is_absolute() and '..' not in name.parts,
                    'unsafe/nonregular source archive member')
            require(member.name not in source_members, 'duplicate source archive member')
            stream = archive.extractfile(member)
            source_members[member.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
            require(provenance['source_inventory'][member.name]['bytes'] == member.size, 'source member size')
    require(source_members == provenance['source_sha256'], 'source archive inventory/hash mismatch')
    require(all(source_members[k] == v for k, v in identities['owned_source_sha256'].items()), 'gate/owned source differs from archived source')
    build, tests = [load(file(plan[key])) for key in ('build_gate', 'test_gate')]
    for key, gate in zip(('build_gate', 'test_gate'), (build, tests)):
        require(gate['returncode'] == 0 and gate['source_unchanged'] is True, 'final selected gate failed')
        require(gate['source_sha256'] == identities['owned_source_sha256'], 'gate source mismatch')
        require(digest(file(str(Path(plan[key]).with_suffix('.log')))) == gate['log_sha256'], 'gate log changed')
    require(build['command'] == ['cmake', '--build', build['build'], '-j', '6'], 'not a full selected build')
    require(tests['command'][:4] == ['ctest', '--test-dir', build['build'], '--output-on-failure'], 'CTest command changed')
    test_log = file(str(Path(plan['test_gate']).with_suffix('.log'))).read_text()
    require('100% tests passed out of 11' in test_log, 'eleven-test gate incomplete')
    require(build['finished'] <= tests['started'] <= tests['finished'] <= plan['frozen_unix'] <=
            provenance['frozen_unix'] <= captures['started'] <= captures['updated'], 'freeze/gate/capture chronology')
    require(provenance['observed_capture_markers_at_freeze'] is False, 'source was not frozen before captures')
    cases = declared['cases']
    require([r['case'] for r in captures['cases']] == [c['case'] for c in cases], 'case order mismatch')
    require([r['status'] for r in captures['cases']] == ['Error', 'OK', 'OK', 'OK', 'OK', 'OK'], 'case status mismatch')
    planned = [case['case'] + '-' + arm for case in cases for arm in ARMS]
    attempted = {p.parent.name for p in raw.glob('*/capture.command.json')}
    missing = set(planned) - attempted
    require(missing == {'decode-mha-d64-' + a for a in ARMS[2:]}, 'unexpected NotRun candidates')
    require(len(attempted) == 27, 'capture attempt count differs')
    successful = []
    commands = set()
    failure = None
    for case, row in zip(cases, captures['cases']):
        named_receipts = {r['label']: r for r in row['variants']}
        expected_success = [case['case'] + '-' + a for a in (ARMS[:1] if row['status'] == 'Error' else ARMS)]
        require(list(named_receipts) == expected_success, 'successful capture receipts differ')
        for arm in ARMS:
            label = case['case'] + '-' + arm
            if label in missing:
                require(not (raw / label).exists() and not (raw / ('prepared-' + label)).exists(), 'NotRun has artifacts')
                continue
            initial = load(file(label + '/capture.command.json'))
            final = load(file(label + '/capture.result.json'))
            commands.add(label + '/capture.command.json')
            require(all(final.get(k) == v for k, v in initial.items()), 'capture receipt mutation')
            require(initial['timeout_s'] == 180 and initial['process_group_isolated'] is True, 'capture supervision changed')
            require(initial['started'] >= provenance['frozen_unix'] and final['finished'] >= initial['started'], 'capture chronology')
            require(initial['command'][0] == provenance['selected_build'] + '/bin/benchmark_tile_xir', 'wrong capture producer')
            require(initial['command'][1:-1] == ['llm', 'attention', ','.join(map(str, case['dimensions'])),
                                               *map(str, case['block']), '3', '1', '1'], 'capture geometry/argv')
            env = initial['environment']
            require(env['LUISA_SIMD_NATIVE_MMA_VECTOR_WIDTH'] == arm[1] and env['LUISA_SIMD_NATIVE_COPY_VECTOR_WIDTH'] == arm[4], 'MMA/copy flag mismatch')
            require(env['LUISA_SIMD_ENABLE_SIMPLIFIED_FULL_PACKET_SPECIALIZATION'] == str(int(arm.endswith('-simplified'))), 'simplified flag mismatch')
            if label == FAILED:
                elapsed = final['finished'] - initial['started']
                require(final['returncode'] == -9 and 'error' in final and 180 <= elapsed <= 210, 'expected supervised timeout missing')
                require(not file(label + '/capture.stdout').read_bytes() and not file(label + '/capture.stderr').read_bytes(), 'unexpected failed-capture output')
                require(not (raw / ('prepared-' + label)).exists() and not (raw / label / 'output.f32').exists(), 'failed candidate unexpectedly produced output/preparation')
                for suffix in ('.input0.f32', '.input1.f32', '.input2.f32', '.expected.f64'):
                    require(digest(file(label + '/output.f32' + suffix)) == digest(file('decode-mha-d64-m0-c0/output.f32' + suffix)), 'failed capture input/oracle drift')
                failure = dict(label=label, status='Error', returncode=-9, timeout_s=180,
                               observed_elapsed_s=elapsed, source_ir_written=False, output_written=False)
                continue
            require(final['returncode'] == 0 and 'error' not in final and file(label + '/capture.stderr').stat().st_size == 0, 'unexpected unsuccessful capture')
            metadata = load(file(label + '/capture.stdout'))
            require(metadata['implementation'] == 'tile_xir_simd' and metadata['precision'] == 'fp32' and metadata['fast_math'] is False,
                    'capture implementation/math drift')
            require(metadata['dimensions'] == case['dimensions'] and metadata['correctness']['checks'] == 2, 'capture correctness/shape receipt')
            prepared = 'prepared-' + label
            manifest = load(file(prepared + '/prepared.json'))
            require(manifest['status'] == 'prepared' and manifest['metadata'] == metadata, 'prepared capture mismatch')
            require(digest(file(prepared + '/prepared.json')) == named_receipts[label]['prepared_manifest_sha256'], 'prepared receipt hash')
            members = {p.relative_to(raw / prepared).as_posix() for p in (raw / prepared).rglob('*') if p.is_file() and p.name != 'prepared.json'}
            require(members == set(manifest['files']), 'prepared complete file inventory')
            for relative, expected in manifest['files'].items():
                require(digest(file(prepared + '/' + relative)) == expected, 'prepared artifact changed')
            objects = list((raw / label / 'objects').glob('*.o'))
            require(len(objects) == 1 and digest(objects[0]) == manifest['files']['kernel.o'], 'actual ORC object identity')
            for suffix, target in (('', 'captured.f32'), ('.source.txt', 'kernel.ll'), ('.expected.f64', 'expected.f64'),
                                   ('.input0.f32', 'input0.f32'), ('.input1.f32', 'input1.f32'), ('.input2.f32', 'input2.f32')):
                require(digest(file(label + '/output.f32' + suffix)) == manifest['files'][target], 'raw/prepared payload identity')
            pc, pr = [load(file(label + '/prepare.' + suffix + '.json')) for suffix in ('command', 'result')]
            require(all(pr.get(k) == v for k, v in pc.items()) and pr['returncode'] == 0 and 'error' not in pr, 'prepare failed')
            require(file(label + '/prepare.stderr').stat().st_size == 0, 'prepare stderr')
            commands.add(label + '/prepare.command.json')
            for stem in ('imports', 'exports', 'compiler', 'link', 'helper'):
                receipt = load(file(prepared + '/' + stem + '.command.json'))
                require(receipt['returncode'] == 0 and file(prepared + '/' + stem + '.stderr').stat().st_size == 0, 'internal preparation command failed')
                commands.add(prepared + '/' + stem + '.command.json')
            successful.append(label)
    require(len(successful) == 26 and failure is not None, 'success/error totals differ')
    actual_commands = {p.relative_to(raw).as_posix() for p in raw.rglob('*.command.json')}
    require(actual_commands == commands and len(commands) == 183, 'complete command receipt coverage')
    sample = file('m0-c4-compile-sample.txt').read_text()
    require('Process:         benchmark_tile_xir' in sample and 'LLVMJIT::lookup' in sample and
            'llvm::LiveVariables::analyze' in sample and 'llvm::LiveVariables::runOnBlock' in sample, 'compiler sample evidence absent')
    require('727 Thread_' in sample and '726 llvm::LiveVariables::analyze' in sample, 'sample count changed')
    return dict(audit_status='incomplete_evidence_consistent', experiment_status='incomplete',
                performance_comparison_status='NotRun', successful_captures=26, failed_captures=1,
                not_run_captures=3, completed_replays=0, competitive_visits=0, competitive_samples=0,
                capture_receipts_checked=27, all_command_receipts_checked=183,
                failure=failure, successful_labels=successful, not_run_labels=sorted(missing),
                source_members_checked=len(source_members), source_archive_sha256=digest(source_archive),
                plan_sha256=digest(file('plan.json')), provenance_sha256=digest(file('provenance.json')),
                sample_sha256=digest(file('m0-c4-compile-sample.txt')),
                selected_gates=dict(build=plan['build_gate'], tests=plan['test_gate'], tests_passed=11),
                limitations=['No native execution, benchmark rerun, new FP64 numerical recomputation, or timing comparison in this failure audit.',
                             'Successful capture correctness summaries are preserved producer/driver receipts, not a complete-cohort audit pass.',
                             'Capture logs contain diagnostic synchronized Runtime timings; no paired native replay was run.',
                             'The 180-second limit covers assembly-copy codegen followed by ORC codegen; it does not establish a single production JIT duration.',
                             'Source archive is the selected export, not all HEAD files or a loader-closure/reproducible-build attestation.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw', type=Path)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.raw, args.sources)
    content = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        require(not args.output.resolve().is_relative_to(args.raw.resolve()), 'refusing to write original/extracted raw evidence')
        with args.output.open('x') as stream:
            stream.write(content)
    else:
        print(content, end='')
