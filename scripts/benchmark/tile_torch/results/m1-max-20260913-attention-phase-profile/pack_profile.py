"""Package an explicit profiling evidence whitelist; never execute native code.

Usage: python3 -B pack_profile.py EXISTING_REPORT_DIR
Run only after all profile/parser/report writers stop and the main task reviews
the XML privacy/path inventory. The report must contain only notes.md. Outputs
are exclusive; a failed attempt is retained, never deleted or overwritten.
Whole traces, TOCs, raw sample reports and inherited environments are excluded.
"""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tarfile
import time

RAW = Path(__file__).resolve().parent
NATIVE = Path('/tmp/luisa-attention-native-mma.DImNTO').resolve()
REPORT = Path('/Users/mike/CLionProjects/luisa/scripts/benchmark/tile_torch/results/m1-max-20260913-attention-phase-profile').resolve()
SOURCE_SHA256 = 'b4547fe068beb9d1ad5903f40b2690cb25022144118e42819eadf4f1c35227e3'
PLAN_SHA256 = 'c47e04da6ce70cde319c6481eaab1162a0d1e0c630a008e3e71a182ce51375ae'
NATIVE_EVIDENCE_SHA256 = '9c110f4eb61f723a7d7f4453c1383925b22f386d6258556ddb563fe893595a7e'
LABELS = tuple(case + '-' + arm for case in ('decode-mha-d64', 'decode-long-kv', 'batch-gqa-q4') for arm in ('off', 'on'))
PROFILE_SCRIPTS = ('run.py', 'profile_child.py', 'trace.py', 'parse_profile.py', 'audit-notes.md', 'pack_profile.py')
TRACE_FILES = ('plan.json', 'result.json', 'trace.command.json', 'trace.result.json',
               'run/ready.json', 'run/profile-begin.json', 'run/result.json',
               'run/preflight.f32', 'run/profiled.f32', 'time-sample.xml', 'time-profile.xml')
NATIVE_FILES = ('prepared.json', 'kernel.o', 'kernel.ll', 'kernel.dylib', 'replay.dylib',
                'native_tile.py', 'native_tile_replay.cpp', 'backends/simd/llvm/llvm_schedule_codegen.h',
                'captured.f32', 'expected.f64')
# The parser's final fixed public names and the reviewed machine-analysis list
# are supplied by a small allowlist receipt, not discovered recursively.
SELECTION = 'public-selection.json'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_name(name):
    require(type(name) is str and bool(name), 'invalid archive name')
    path = PurePosixPath(name)
    require(not path.is_absolute() and '..' not in path.parts and '\\' not in name and '\x00' not in name,
            'unsafe archive name: ' + name)
    return name


def regular(root, name):
    safe_name(name)
    path = root / name
    for index in range(1, len(PurePosixPath(name).parts) + 1):
        require(not (root / Path(*PurePosixPath(name).parts[:index])).is_symlink(), 'symlink selection: ' + name)
    require(stat.S_ISREG(path.stat().st_mode) and path.resolve().is_relative_to(root), 'nonregular/escaped selection: ' + name)
    return path


def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def load(root, name):
    return json.loads(regular(root, name).read_bytes())


def current(path):
    require(not path.is_symlink() and stat.S_ISREG(path.stat().st_mode), 'selection changed type')
    return dict(bytes=path.stat().st_size, sha256=sha(path))


def main():
    require(len(sys.argv) == 2, __doc__)
    report = Path(sys.argv[1]).resolve()
    require(RAW.name == 'luisa-attention-phase-profile.TAsOL6' and report == REPORT and report.is_dir(), 'unexpected raw/report roots')
    require({p.name for p in report.iterdir()} == {'notes.md'}, 'report must contain only notes.md; refusing overwrite')
    notes = regular(report, 'notes.md')
    notes_hash = sha(notes)
    plan = load(RAW, 'plan.json')
    require(plan['source_plan_sha256'] == PLAN_SHA256 and plan['source_archive_sha256'] == SOURCE_SHA256, 'original cohort identity')
    require(sha(regular(NATIVE, 'plan.json')) == PLAN_SHA256, 'original native plan changed')
    prior_audit = load(NATIVE, 'audit.json')
    require(prior_audit['status'] == 'passed' and prior_audit['source_archive_sha256'] == SOURCE_SHA256 and
            prior_audit['plan_sha256'] == PLAN_SHA256, 'original native audit identity')
    selected = load(RAW, SELECTION)
    require(selected['format'] == 'profile-public-selection-v1' and selected['labels'] == list(LABELS), 'six-trace public selection')
    require(selected['xml_privacy_reviewed'] is True, 'main task XML/path review not acknowledged')
    require(set(selected['traces']) == set(LABELS), 'missing trace selection')
    files = {}

    def add(root, relative, destination):
        safe_name(destination)
        require(destination not in files and not any(part.endswith('.trace') for part in PurePosixPath(relative).parts) and
                PurePosixPath(relative).name not in ('toc.xml', 'sample.txt', 'child.stdout', 'child.stderr', 'trace.stdout', 'trace.stderr'),
                'duplicate/forbidden public artifact: ' + relative)
        files[destination] = regular(root, relative)

    for name in (*PROFILE_SCRIPTS, 'plan.json', SELECTION):
        add(RAW, name, 'profile/' + name)
    require(len(set(selected['extra_files'])) == len(selected['extra_files']), 'duplicate root extra selection')
    for name in selected['extra_files']:
        require(PurePosixPath(safe_name(name)).name == name, 'extra file must be root-level')
        add(RAW, name, 'profile/' + name)
    for name, expected in plan['scripts'].items():
        require(name in ('run.py', 'profile_child.py') and sha(regular(RAW, name)) == expected, 'frozen sampler source changed')
    for label in LABELS:
        folder = 'trace-' + label
        trace_plan = load(RAW, folder + '/plan.json')
        parent = load(RAW, folder + '/result.json')
        ready = load(RAW, folder + '/run/ready.json')
        result = load(RAW, folder + '/run/result.json')
        native_folder = 'prepared-' + label
        manifest = load(NATIVE, native_folder + '/prepared.json')
        manifest_hash = sha(regular(NATIVE, native_folder + '/prepared.json'))
        require(manifest_hash == plan['prepared'][label] == trace_plan['manifest_sha256'] == ready['prepared_sha256'] == result['prepared_sha256'],
                'prepared/trace/cohort identity: ' + label)
        require(manifest['status'] == 'prepared' and parent['status'] == 'OK' and parent['returncode'] == 0 and
                parent['ready'] == ready and parent['result'] == result and result['status'] == 'passed' and
                result['artifacts_unchanged'] is True and result['performance_comparison_accepted'] is False, 'profile completion/validation')
        require(all(parent[k] == value for k, value in trace_plan.items() if k != 'status') and
                ready['event'] == 'READY' and result['event'] == 'RESULT', 'supervisor plan/result drift')
        trace_command = load(RAW, folder + '/trace.command.json')
        trace_result = load(RAW, folder + '/trace.result.json')
        require(all(trace_result[k] == value for k, value in trace_command.items()) and trace_result['returncode'] == 0 and
                trace_command['command'][:8] == ['xcrun', 'xctrace', 'record', '--template', 'Time Profiler', '--attach', str(ready['pid']), '--time-limit'] and
                trace_command['command'][8] == '5s' and trace_result['finished'] >= trace_result['started'], 'owned target profiler command')
        require(trace_plan['driver_sha256'] == sha(regular(RAW, 'trace.py')) and trace_plan['child_sha256'] ==
                plan['scripts']['profile_child.py'] == ready['driver_sha256'] == result['driver_sha256'], 'profile driver identity')
        for key in ('pid', 'thread_id', 'images', 'abi', 'dispatch', 'block', 'packet_width', 'workspace_bytes', 'object_sha256'):
            require(ready[key] == result[key], 'ready/result identity drift: ' + key)
        for key in ('abi', 'dispatch', 'block', 'packet_width', 'workspace_bytes'):
            require(result[key] == manifest[key], 'actual native launch drift')
        require(result['object_sha256'] == manifest['files']['kernel.o'], 'actual object identity')
        for key, image in (('kernel', 'kernel.dylib'), ('helper', 'replay.dylib')):
            require(result['images'][key]['sha256'] == manifest['files'][image], 'loaded image hash differs')
        for phase in ('preflight', 'profiled'):
            row = result[phase]
            require(row['returncode'] == 0 and all(row[k] is True for k in ('own_capture_bitwise_equal', 'all_guards_passed', 'inputs_unchanged', 'launch_metadata_unchanged')),
                    'profile phase failed')
            require(sha(regular(RAW, folder + '/run/' + phase + '.f32')) == row['output_sha256'] == manifest['files']['captured.f32'],
                    'profile output differs from original capture')
            require(row['correctness']['elements'] == manifest['output_elements'] and
                    row['correctness']['atol'] == manifest['atol'] and row['correctness']['rtol'] == manifest['rtol'], 'profile numerical extent/tolerance')
        for name in TRACE_FILES:
            add(RAW, folder + '/' + name, 'profile/' + folder + '/' + name)
        # Public analysis may only come from explicitly reviewed files under
        # this trace; exact XML source identities must match that approval.
        approval = selected['traces'][label]
        require(set(approval['xml_sha256']) == {'time-sample.xml', 'time-profile.xml'}, 'XML approval coverage')
        analysis = load(RAW, folder + '/analysis-v3/profile-summary.json')
        privacy = load(RAW, folder + '/analysis-v3/xml-privacy-audit.json')
        require(analysis['status'] == 'checked' and analysis['performance_comparison_accepted'] is False and
                analysis['parser_sha256'] == sha(regular(RAW, 'parse_profile.py')), 'final parser identity/status')
        for table in ('time_sample', 'time_profile'):
            require(privacy[table]['status'] == 'target_only_fields_checked' and privacy[table]['environment_fields_present'] is False and
                    privacy[table]['pid'] == ready['pid'] and privacy[table]['tid'] == ready['thread_id'], 'XML structural/privacy admission')
            for image in privacy[table]['loaded_image_path_inventory']:
                path = image['path']
                require(path in (ready['images']['kernel']['resolved_path'], ready['images']['helper']['resolved_path']) or
                        path.startswith(('/usr/lib/', '/opt/homebrew/Caskroom/miniconda/base/')), 'unapproved XML image path')
        for name, expected in approval['xml_sha256'].items():
            require(sha(regular(RAW, folder + '/' + name)) == expected == analysis['inputs_sha256'][name], 'reviewed XML changed')
        for name in ('run/ready.json', 'run/result.json', 'run/profile-begin.json'):
            require(sha(regular(RAW, folder + '/' + name)) == analysis['inputs_sha256'][name], 'parser receipt input changed')
        require(approval['public_files'] and len(set(approval['public_files'])) == len(approval['public_files']), 'empty/duplicate analysis allowlist')
        for name in approval['public_files']:
            add(RAW, folder + '/' + name, 'profile/' + folder + '/' + name)
        for name in NATIVE_FILES:
            path = regular(NATIVE, native_folder + '/' + name)
            if name != 'prepared.json':
                require(sha(path) == manifest['files'][name], 'selected original bundle file changed')
            add(NATIVE, native_folder + '/' + name, 'native/' + native_folder + '/' + name)
    # The collapsed sampler's exact owned receipts and validation outputs are
    # retained, but not sample.txt or its full process/image listing.
    for name in ('child.command.json', 'child.result.json', 'sample.command.json', 'sample.result.json',
                 'run/ready.json', 'run/profile-begin.json', 'run/result.json', 'run/preflight.f32', 'run/profiled.f32'):
        add(RAW, 'decode-mha-d64-off/' + name, 'profile/decode-mha-d64-off/' + name)
    require(len(set(selected['machine_analysis_files'])) == len(selected['machine_analysis_files']), 'duplicate machine analysis selection')
    for name in selected['machine_analysis_files']:
        require(name.startswith('linked-analysis.pb43ED/'), 'machine analysis outside reviewed subtree')
        add(RAW, name, 'profile/' + name)
    add(NATIVE, 'audit.py', 'dependency/native-audit.py')
    add(NATIVE, 'audit.json', 'dependency/native-audit.json')
    add(NATIVE, 'plan.json', 'dependency/native-plan.json')
    dependency = dict(format='profile-native-dependency-v1', report='../m1-max-20260913-attention-native-mma',
                      source_archive_sha256=SOURCE_SHA256, evidence_archive_sha256=NATIVE_EVIDENCE_SHA256,
                      native_plan_sha256=PLAN_SHA256,
                      omitted_from_selected_bundle='Full compiler source archive and attention input tensors; use the identified prior committed artifact for independent mathematical-oracle reconstruction/full native replay.',
                      available_offline='Exact target-only profile XML, clock anchor, raw-PC extraction, linked-image classification and summary arithmetic; profile outputs bind bitwise to the already audited original capture.')
    generated = json.dumps(dependency, indent=2, allow_nan=False).encode() + b'\n'
    before = {name: current(path) for name, path in sorted(files.items())}
    require(sum(x['bytes'] for x in before.values()) <= 512 * 1024**2, 'selected evidence exceeds the bounded 512 MiB raw budget')
    started = time.time()
    archive_path = report / 'evidence.tar.xz'
    with archive_path.open('xb') as output:
        with tarfile.open(fileobj=output, mode='w:xz', preset=9) as archive:
            for name, row in before.items():
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = row['bytes'], 0o644, int(started)
                with files[name].open('rb') as input_file:
                    archive.addfile(info, input_file)
            info = tarfile.TarInfo('dependency/native-artifact.json')
            info.size, info.mode, info.mtime = len(generated), 0o644, int(started)
            archive.addfile(info, io.BytesIO(generated))
    require(before == {name: current(path) for name, path in sorted(files.items())}, 'selected sources changed while packaging')
    require(sha(notes) == notes_hash, 'report changed while packaging')
    inventory = dict(format='native-profile-package-v1', started_unix=started, finished_unix=time.time(),
                     files=before, generated_member=dict(name='dependency/native-artifact.json', bytes=len(generated), sha256=hashlib.sha256(generated).hexdigest()),
                     archive_sha256=sha(archive_path), archive_bytes=archive_path.stat().st_size,
                     notes_sha256=notes_hash, public_selection_sha256=sha(regular(RAW, SELECTION)),
                     excluded=['recording.trace/**', 'toc.xml', 'raw sample.txt', 'unreviewed child/profiler stdout/stderr', '__pycache__/**',
                               'inherited environments and unrelated process/image metadata', 'full native inputs/compiler archive (explicit prior-artifact dependency)'],
                     boundary='Exact own process IDs, ASLR/image addresses, temporary/source paths and reviewed target-only XML are retained by explicit scope; no whole-trace/TOC environment archive. No native execution or performance comparison is performed by this packager.')
    with (report / 'package-inventory.json').open('x') as output:
        json.dump(inventory, output, indent=2, allow_nan=False)
        output.write('\n')
    for name in ('pack_profile.py', 'verify_profile.py'):
        with (report / name).open('xb') as output, regular(RAW, name).open('rb') as input_file:
            shutil.copyfileobj(input_file, output)
    expected = {'notes.md', 'evidence.tar.xz', 'package-inventory.json', 'pack_profile.py', 'verify_profile.py'}
    require({p.name for p in report.iterdir()} == expected, 'unexpected report writer')
    hashes = {name: sha(regular(report, name)) for name in sorted(expected)}
    with (report / 'SHA256SUMS').open('x') as output:
        for name, value in hashes.items():
            output.write(value + '  ' + name + '\n')
    print(json.dumps(dict(status='packaged', regular_members=len(before) + 1, archive_bytes=archive_path.stat().st_size,
                          archive_sha256=hashes['evidence.tar.xz'], checksums=len(hashes)), indent=2))


if __name__ == '__main__':
    main()
