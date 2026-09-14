"""Independent full11 correction evidence; stdlib only, no native execution."""
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
SELECTED = Path('/Users/mike/.cache/luisa-tile/metal-program-team.xIQgCS/source')
FREEZE = 'source-freeze-collective-epoch-storage.json'
FREEZE_SHA = '29ef908f51beadddedaac6cc85735f2c260cf86978c075bd0fb875707230099c'
RED_FREEZE = 'source-freeze-collective-epoch-counterexample.json'
RED_SHA = 'f37f46ba0a668e9966d74168f037b9e4e18f1b17be597521aad080c5e0b3b55f'
GREEN = ('build-full-11', 'test-full11-unit-simd', 'test-full11-tile-simd', 'test-full11-host-plan')
STATIC = tuple('tidy-full11-' + name for name in (
    'warp-uniformity', 'xir-schedule', 'schedule-collectives',
    'test-warp-uniformity', 'test-xir-schedule', 'test-schedule-codegen'))
RED = ('build-full-10', 'test-full10-epoch-counterexample')
LIMIT = 45 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    require(path.is_file() and not path.is_symlink(), f'not a regular source file: {path}')
    return path.read_bytes()


def safe_name(name):
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and '..' not in path.parts and str(path) == name


def unpack(data):
    files = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
        for member in archive:
            require(member.isfile() and safe_name(member.name), f'unsafe member: {member.name}')
            require(member.name not in files, f'duplicate member: {member.name}')
            files[member.name] = archive.extractfile(member).read()
            require(len(files[member.name]) == member.size, f'short member: {member.name}')
    return files


def validate(files):
    freeze = json.loads(files['raw/' + FREEZE])
    red = json.loads(files['raw/' + RED_FREEZE])
    require(sha(files['raw/' + FREEZE]) == FREEZE_SHA, 'wrong full11 freeze')
    require(sha(files['raw/' + RED_FREEZE]) == RED_SHA, 'wrong full10 freeze')
    require(len(freeze['source_sha256']) == 26, 'full11 must contain 26 source files')
    source_files = {name[7:]: data for name, data in files.items() if name.startswith('source/')}
    require({name: sha(data) for name, data in source_files.items()} == freeze['source_sha256'], 'full11 source mismatch')
    for metadata, key in (('owned.json', 'owned_sha256'), ('toolchain.json', 'toolchain_sha256'),
                          ('source-clone.json', 'baseline_clone_sha256')):
        require(sha(files['raw/' + metadata]) == freeze[key], f'wrong {metadata}')
    require(set(json.loads(files['raw/owned.json'])) == set(source_files), 'wrong ownership list')
    require(sha(files['raw/owned-through-full10.json']) == red['owned_sha256'], 'wrong full10 ownership')
    red_sources = unpack(files['raw/source-collective-epoch-counterexample.tar.gz'])
    require(len(red_sources) == 23 and {name: sha(data) for name, data in red_sources.items()} == red['source_sha256'], 'full10 source mismatch')
    receipts = {}
    for folder in GREEN + STATIC + RED:
        receipt = json.loads(files[f'raw/{folder}/result.json'])
        command = json.loads(files[f'raw/{folder}/command.json'])
        expected_freeze = red if folder in RED else freeze
        expected_sha = RED_SHA if folder in RED else FREEZE_SHA
        failed = folder == RED[1]
        require(receipt['status'] == ('failed' if failed else 'passed') and
                receipt['exit_code'] == (8 if failed else 0) and not receipt.get('timed_out', False), f'wrong terminal status: {folder}')
        require(all(receipt.get(key) == value for key, value in command.items()), f'command/result mismatch: {folder}')
        require(receipt['source_freeze_sha256'] == expected_sha and
                receipt['source_sha256'] == expected_freeze['source_sha256'] and
                receipt['owned_sha256'] == expected_freeze['owned_sha256'], f'wrong source binding: {folder}')
        require(expected_freeze['finished'] <= receipt['started'] <= receipt['finished'], f'bad chronology: {folder}')
        require(receipt['harness_sha256'] == sha(files['raw/run.py']) and
                receipt['toolchain_sha256'] == sha(files['raw/toolchain.json']), f'wrong harness/toolchain: {folder}')
        for channel in ('stdout', 'stderr'):
            require(sha(files[f'raw/{folder}/{channel}.txt']) == receipt[channel + '_sha256'], f'wrong {channel}: {folder}')
        receipts[folder] = receipt
    for folder in GREEN[1:] + STATIC:
        require(receipts[GREEN[0]]['finished'] <= receipts[folder]['started'], f'gate precedes build: {folder}')
    require(receipts[RED[0]]['finished'] <= receipts[RED[1]]['started'], 'red precedes its build')
    red_log = files[f'raw/{RED[1]}/stdout.txt'].decode()
    require('0% tests passed, 2 tests failed out of 2' in red_log, 'missing actual red summary')
    failures = sorted(set(re.findall(
        r'cohort exit oracle: stage=(second loop sum|escaped population), width=(\d+), '
        r'enabled=([01]), active=(\d+), lane=(\d+), actual=(\d+), expected=(\d+), direct=(\d+)', red_log)))
    expected = sorted((stage, str(width), str(enabled), '2', lane, actual, want, '0')
                      for width in (2, 4, 8, 16) for enabled in (0, 1)
                      for stage, lane, actual, want in (('second loop sum', '0', '121', '111'),
                                                        ('escaped population', '1', '2', '1')))
    require(failures == expected, 'missing full10 numeric failures')
    for folder, count in zip(GREEN[1:], (13, 1, 2)):
        require(f'100% tests passed out of {count}' in files[f'raw/{folder}/stdout.txt'].decode(), f'missing green summary: {folder}')
    green_log = files['raw/test-full11-unit-simd/stdout.txt'].decode()
    for name in ('nested cohort uint4 mask and read-first exit snapshots',
                 'cohort collective exit values and subsequent recurrence',
                 'uniform read-lane source indices and loop epochs'):
        require('[pass] ' + name in green_log, f'missing green counterexample: {name}')
    warnings = {}
    for folder in STATIC:
        log = files[f'raw/{folder}/stdout.txt'].decode()
        match = re.search(r'Total: (\d+) error\(s\), (\d+) warning\(s\)', log)
        require((match is not None and match[1] == '0') or '[OK] No issues found!' in log, f'bad static result: {folder}')
        warnings[folder] = int(match[2]) if match else 0
    return dict(status='verified', full11_freeze_sha256=FREEZE_SHA, full11_source_files=26,
                full10_freeze_sha256=RED_SHA, full10_source_files=23,
                full10_unique_numeric_failures=len(failures), full11_green_ctest_counts=[13, 1, 2],
                static_warnings=warnings, native_execution_by_this_tool=False, performance_claim=False)


def collect():
    freeze_data = read(RAW / FREEZE)
    require(sha(freeze_data) == FREEZE_SHA, 'live freeze changed')
    freeze = json.loads(freeze_data)
    paths = {'source/' + name: SELECTED / name for name in freeze['source_sha256']}
    for name in (FREEZE, RED_FREEZE, 'owned.json', 'owned-through-full10.json', 'toolchain.json',
                 'source-clone.json', 'run.py', 'source-collective-epoch-counterexample.tar.gz'):
        paths['raw/' + name] = RAW / name
    for folder in GREEN + STATIC + RED:
        for name in ('command.json', 'result.json', 'stdout.txt', 'stderr.txt'):
            paths[f'raw/{folder}/{name}'] = RAW / folder / name
    for name in ('.clangd', '.clang-tidy', '.clang-format', 'scripts/check_cpp_syntax.py'):
        paths['tools/' + name] = SELECTED / name
    paths['tools/compile_commands.json'] = SELECTED.parent / 'build/compile_commands.json'
    files = {name: read(path) for name, path in paths.items()}
    validate(files)
    return paths, files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', action='store_true')
    args = parser.parse_args()
    target = HERE / 'evidence.tar.gz'
    if args.archive:
        paths, files = collect()
        with target.open('xb') as output, gzip.GzipFile(filename='', fileobj=output, mode='wb', compresslevel=6, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w') as archive:
                for name, data in sorted(files.items()):
                    require(safe_name(name), 'unsafe archive path')
                    entry = tarfile.TarInfo(name)
                    entry.size, entry.mode = len(data), 0o644
                    archive.addfile(entry, io.BytesIO(data))
        require(target.stat().st_size < LIMIT, 'archive exceeds size limit')
        require(all(read(path) == files[name] for name, path in paths.items()), 'source changed during archive')
        manifest = dict(schema_version=1, archive_sha256=sha(read(target)), archive_bytes=target.stat().st_size,
                        local_commit='e20d24d524908076e978b1752b05112262f925b1',
                        live_sources_unchanged_before_after=True,
                        files={name: dict(sha256=sha(data), size=len(data), original=str(paths[name]))
                               for name, data in sorted(files.items())}, verification=validate(files))
        with (HERE / 'manifest.json').open('x') as output:
            json.dump(manifest, output, indent=2, ensure_ascii=False)
            output.write('\n')
        with (HERE / 'SHA256SUMS').open('x') as output:
            for name in ('evidence.py', 'notes.md', 'evidence.tar.gz', 'manifest.json'):
                output.write(f'{sha(read(HERE / name))}  {name}\n')
    manifest = json.loads(read(HERE / 'manifest.json'))
    data = read(target)
    require(len(data) == manifest['archive_bytes'] and len(data) < LIMIT and sha(data) == manifest['archive_sha256'], 'archive mismatch')
    files = unpack(data)
    require(set(files) == set(manifest['files']), 'wrong archive members')
    for name, data in files.items():
        require(sha(data) == manifest['files'][name]['sha256'] and len(data) == manifest['files'][name]['size'], f'wrong member: {name}')
    result = validate(files)
    require(result == manifest['verification'], 'verification result changed')
    for line in read(HERE / 'SHA256SUMS').decode().splitlines():
        checksum, name = line.split('  ', 1)
        require(safe_name(name) and '/' not in name and sha(read(HERE / name)) == checksum, f'checksum mismatch: {name}')
    print(json.dumps(dict(result, archive_bytes=manifest['archive_bytes'], archive_sha256=manifest['archive_sha256'], members=len(files)), ensure_ascii=False))


if __name__ == '__main__':
    main()
