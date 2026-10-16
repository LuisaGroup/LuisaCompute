"""Full12 correctness/source evidence: stdlib only, no native execution.

--archive creates a new archive once; normal use reads this directory only.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile

HERE = Path(__file__).resolve().parent
RAW = Path('/tmp/luisa-metal-program-team.TGZv8f')
CANDIDATE = RAW / 'byte-private-candidate.Oe9AVl'
FREEZE = 'source-freeze-byte-private.json'
FREEZE_SHA = '71573837811263abe91d86a4c0b33bb425479e2bd5c2638e92d899a1afb56720'
PARENT_FREEZE = 'source-freeze-collective-epoch-storage.json'
PARENT_SHA = '29ef908f51beadddedaac6cc85735f2c260cf86978c075bd0fb875707230099c'
GATES = ('build-full-12', 'test-full12-unit-simd', 'test-full12-tile-simd', 'test-full12-host-plan')
STATIC = ('tidy-byte-private-memory', 'tidy-byte-private-codegen-test')
PARENT_GATES = ('build-full-11', 'test-full11-unit-simd', 'test-full11-tile-simd', 'test-full11-host-plan')
CHANNELS = ('command.json', 'result.json', 'stdout.txt', 'stderr.txt')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    require(path.is_file() and not path.is_symlink(), f'not a regular file: {path}')
    return path.read_bytes()


def unpack(data):
    files = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            require(member.isfile() and not path.is_absolute() and '..' not in path.parts and
                    str(path) == member.name and member.name not in files, f'unsafe member: {member.name}')
            files[member.name] = archive.extractfile(member).read()
            require(len(files[member.name]) == member.size, f'short member: {member.name}')
    return files


def validate(files):
    def js(name):
        return json.loads(files[name])

    def receipt(folder):
        result, command = js(folder + '/result.json'), js(folder + '/command.json')
        require(result['status'] == 'passed' and result['exit_code'] == 0 and
                not result.get('timed_out', False), f'nonpassing receipt: {folder}')
        require(all(result.get(k) == v for k, v in command.items()), f'command/result mismatch: {folder}')
        require(result['started'] <= result['finished'], f'wrong chronology: {folder}')
        for channel in ('stdout', 'stderr'):
            require(sha(files[folder + '/' + channel + '.txt']) == result[channel + '_sha256'], f'log mismatch: {folder}')
        return result

    freeze, plan, selfcheck = (js('full12/' + n) for n in (FREEZE, 'plan.json', 'self-check.json'))
    parent = js('parent/' + PARENT_FREEZE)
    require(sha(files['full12/' + FREEZE]) == FREEZE_SHA, 'wrong full12 freeze')
    require(sha(files['parent/' + PARENT_FREEZE]) == PARENT_SHA, 'wrong parent freeze')
    require(freeze['status'] == selfcheck['status'] == 'passed', 'freeze/selfcheck failed')
    identity = freeze['identity']
    require(identity == selfcheck['identity'], 'selfcheck identity mismatch')
    for name, key in (('run.py', 'runner_sha256'), ('plan.json', 'plan_sha256'), ('owned.json', 'owned_sha256')):
        require(sha(files['full12/' + name]) == identity[key], f'identity mismatch: {name}')
    require(identity['parent_pins'] == plan['parent_pins'], 'parent pin map mismatch')
    for name, digest in plan['parent_pins'].items():
        require(sha(files['parent/' + name]) == digest, f'parent pin mismatch: {name}')
    require(identity['original_harness_sha256'] == sha(files['parent/run.py']), 'wrong parent runner')
    require(identity['toolchain_sha256'] == sha(files['parent/toolchain.json']), 'wrong toolchain')
    require(parent['owned_sha256'] == sha(files['parent/owned.json']) and
            parent['baseline_clone_sha256'] == sha(files['parent/source-clone.json']), 'wrong parent metadata')
    sources = {name.removeprefix('full12/frozen-source/'): sha(data) for name, data in files.items()
               if name.startswith('full12/frozen-source/')}
    require(sources == freeze['source_sha256'] and len(sources) == 27, 'wrong frozen source inventory')
    require(set(js('full12/owned.json')) == set(sources), 'wrong owned inventory')
    baseline = freeze['baseline_source_sha256']
    require(len(parent['source_sha256']) == 26 and all(baseline[n] == h for n, h in parent['source_sha256'].items()),
            'baseline does not inherit full11')
    require(baseline == selfcheck['baseline_source_sha256'], 'selfcheck baseline mismatch')
    require(set(plan['candidate']) == set(freeze['overlaid_files']) and len(plan['candidate']) == 2,
            'wrong overlay inventory')
    for name, pins in plan['candidate'].items():
        require(sha(files['full12/baseline-source/' + name]) == baseline[name] == pins['baseline_sha256'], 'wrong baseline copy')
        require(sha(files['full12/candidate-source/' + name]) == sources[name] == pins['candidate_sha256'] ==
                selfcheck['candidate_sha256'][name], 'wrong candidate copy')
    require(set(freeze['inherited_files']) == set(sources) - set(plan['candidate']) and
            selfcheck['inherited_file_count'] == 25, 'wrong inheritance inventory')
    require(all(sources[n] == baseline[n] for n in freeze['inherited_files']), 'inherited source changed')
    require(selfcheck['selected_writes'] == 0 and selfcheck['build_executed'] is False and
            selfcheck['native_executed'] is False and selfcheck['finished'] <= freeze['finished'], 'bad selfcheck boundary')
    records = {}
    for folder in GATES + STATIC:
        result = receipt('full12/' + folder)
        require(result['identity'] == identity and result['source_freeze_sha256'] == FREEZE_SHA and
                result['source_sha256'] == sources and result['source_unchanged'] and result['child_exited'],
                f'full12 source/terminal binding mismatch: {folder}')
        require(result['started'] >= freeze['finished'], f'full12 receipt predates freeze: {folder}')
        records[folder] = result
    build = records[GATES[0]]
    require(build['kind'] == 'build' and '--build' in build['command'] and '--target' not in build['command'], 'not full build')
    for folder in GATES[1:] + STATIC:
        require(records[folder]['started'] >= build['finished'], f'gate predates build: {folder}')
    for folder, count in zip(GATES[1:], (13, 1, 2)):
        require(records[folder]['kind'] == 'test' and records[folder]['native'] is True, f'not native gate: {folder}')
        require(f'100% tests passed out of {count}' in files[f'full12/{folder}/stdout.txt'].decode(), 'missing CTest summary')
    expected_commands = {
        GATES[1]: ['-L', 'unit_simd', '--output-on-failure', '-j1', '-V'],
        GATES[2]: ['-R', '^test_tile_xir_runtime$', '--output-on-failure', '-V'],
        GATES[3]: ['-R', '^(test_tile_xir_target_info|test_tile_xir_program_team)$', '--output-on-failure', '-V'],
    }
    for folder, suffix in expected_commands.items():
        require(records[folder]['command'][3:] == suffix, f'wrong test selection: {folder}')
    unit = files[f'full12/{GATES[1]}/stdout.txt'].decode()
    for name in ('private bool byte storage, masks and loop exit snapshots',
                 'private int8 storage, masks and loop exit snapshots',
                 'private uint8 storage, masks and loop exit snapshots'):
        require('[pass] ' + name in unit, f'missing byte regression: {name}')
    require('5207697 asserts in 31 tests' in files[f'full12/{GATES[2]}/stdout.txt'].decode(), 'wrong Tile suite')
    host = files[f'full12/{GATES[3]}/stdout.txt'].decode()
    require('24580 asserts in 21 tests' in host and '339562 asserts in 13 tests' in host, 'wrong host suites')
    warnings = {}
    for folder in STATIC:
        result, log = records[folder], files[f'full12/{folder}/stdout.txt'].decode()
        require(result['kind'] == 'diagnostic' and result['native'] is False and '--clang-tidy' in result['command'], 'wrong static mode')
        match = re.search(r'Total: (\d+) error\(s\), (\d+) warning\(s\)', log)
        require((match is not None and match[1] == '0') or '[OK] No issues found!' in log, 'nonzero syntax errors')
        warnings[folder] = int(match[2]) if match else 0
    require(list(warnings.values()) == [0, 21], 'wrong static warning counts')
    require("Uninitialized record type: 'cells'" in files[f'full12/{STATIC[1]}/stdout.txt'].decode(), 'missing preserved cells warning')
    test_source = files['full12/frozen-source/src/tests/unit/simd/test_llvm_schedule_codegen.cpp'].decode()
    require(re.search(r'std::array<uint8_t, 17u> cells;\s*cells\.fill\(0xa5u\);', test_source), 'cells fill context missing')
    for folder in PARENT_GATES:
        result = receipt('parent/' + folder)
        require(result['source_freeze_sha256'] == PARENT_SHA and result['source_sha256'] == parent['source_sha256'] and
                result['harness_sha256'] == sha(files['parent/run.py']) and
                result['toolchain_sha256'] == sha(files['parent/toolchain.json']), 'parent gate binding mismatch')
        require(parent['finished'] <= result['started'] <= result['finished'] <= selfcheck['finished'], 'wrong parent chronology')
    toolchain = js('parent/toolchain.json')
    for tool in toolchain['tools'].values():
        folder = 'parent/' + Path(tool['receipt']).parent.name
        result = receipt(folder)
        require(result['command'] == tool['command'] and files[folder + '/stdout.txt'].decode().strip() == tool['output'], 'tool receipt mismatch')
    clone = js('parent/source-clone.json')
    require(sha(files['parent/previous-source-receipt.json']) == clone['previous_source_receipt_sha256'], 'wrong predecessor receipt')
    return dict(status='verified', full12_freeze_sha256=FREEZE_SHA, parent_full11_freeze_sha256=PARENT_SHA,
                full12_source_files=27, inherited_unchanged_files=25, baseline_copies=2, candidate_copies=2,
                ctest_executables=[13, 1, 2], tile_tests=31, tile_assertions=5207697, host_tests=34, host_assertions=364142,
                build_wall_seconds=build['finished'] - build['started'],
                gate_wall_seconds={n: records[n]['finished'] - records[n]['started'] for n in GATES[1:]},
                static_errors=0, static_warnings=warnings, preserved_cells_warning=True,
                native_execution_by_this_tool=False, performance_samples=0, binary_closure_claim=False)


def collect():
    paths = {'full12/' + str(p.relative_to(CANDIDATE)): p for p in CANDIDATE.rglob('*')
             if p.is_file() and '__pycache__' not in p.parts}
    plan = json.loads(read(CANDIDATE / 'plan.json'))
    for name in plan['parent_pins']:
        paths['parent/' + name] = RAW / name
    for folder in PARENT_GATES:
        for name in CHANNELS:
            paths[f'parent/{folder}/{name}'] = RAW / folder / name
    for tool in json.loads(read(RAW / 'toolchain.json'))['tools'].values():
        folder = Path(tool['receipt']).parent
        for name in CHANNELS:
            paths[f'parent/{folder.name}/{name}'] = folder / name
    clone = json.loads(read(RAW / 'source-clone.json'))
    paths['parent/previous-source-receipt.json'] = Path(clone['previous_source_receipt'])
    files = {name: read(path) for name, path in paths.items()}
    validate(files)
    return paths, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', action='store_true')
    args = parser.parse_args()
    target = HERE / 'evidence.tar.gz'
    if args.archive:
        paths, files = collect()
        with target.open('xb') as output, gzip.GzipFile(filename='', fileobj=output, mode='wb', compresslevel=6, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w') as archive:
                for name, data in sorted(files.items()):
                    member = tarfile.TarInfo(name)
                    member.size, member.mode = len(data), 0o644
                    archive.addfile(member, io.BytesIO(data))
        require(all(read(paths[name]) == data for name, data in files.items()), 'source changed while archiving')
        manifest = dict(schema_version=1, archive_sha256=sha(read(target)),
                        scope='Full12 frozen overlay and correctness receipts; no performance or binaries.',
                        files={n: dict(sha256=sha(data), size=len(data), original_path=str(paths[n])) for n, data in sorted(files.items())})
        with (HERE / 'manifest.json').open('x') as stream:
            json.dump(manifest, stream, indent=2)
            stream.write('\n')
    else:
        manifest = json.loads(read(HERE / 'manifest.json'))
        require(sha(read(target)) == manifest['archive_sha256'], 'archive hash mismatch')
        files = unpack(read(target))
        require(set(files) == set(manifest['files']), 'archive inventory mismatch')
        for name, data in files.items():
            row = manifest['files'][name]
            require(sha(data) == row['sha256'] and len(data) == row['size'], f'member hash mismatch: {name}')
    result = validate(files)
    result['archived_files'] = len(files)
    if args.archive:
        with (HERE / 'verification.json').open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
        with (HERE / 'SHA256SUMS').open('x') as stream:
            for path in sorted(HERE.iterdir()):
                if path.is_file() and path.name != 'SHA256SUMS':
                    stream.write(sha(read(path)) + '  ' + path.name + '\n')
    else:
        require(result == json.loads(read(HERE / 'verification.json')), 'derived verification mismatch')
        checksum_names = set()
        for line in read(HERE / 'SHA256SUMS').decode().splitlines():
            digest, name = line.split('  ', 1)
            require('/' not in name and name not in checksum_names and sha(read(HERE / name)) == digest, 'top-level checksum mismatch')
            checksum_names.add(name)
        require(checksum_names == {p.name for p in HERE.iterdir() if p.is_file() and p.name != 'SHA256SUMS'}, 'checksum inventory mismatch')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
