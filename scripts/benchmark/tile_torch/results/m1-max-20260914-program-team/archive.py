"""Mechanically archive this bounded cohort; never compile or execute kernels."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time

HERE = Path(__file__).resolve().parent
RAW = Path('/tmp/luisa-metal-program-team.TGZv8f').resolve()
SELECTED = Path('/Users/mike/.cache/luisa-tile/metal-program-team.xIQgCS/source')
BUILD = SELECTED.parent / 'build'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def need(value, message):
    if not value:
        raise ValueError(message)


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def pack(name, files):
    members = {}
    path = HERE / name
    with path.open('xb') as stream, gzip.GzipFile(fileobj=stream, mode='wb', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as archive:
            for relative, source in sorted(files.items()):
                need(not source.is_symlink() and source.is_file(), 'not a regular source: ' + str(source))
                data = source.read_bytes()
                info = tarfile.TarInfo(relative)
                info.size, info.mode = len(data), 0o644
                archive.addfile(info, io.BytesIO(data))
                members[relative] = dict(sha256=digest(data), size=len(data), original_path=str(source))
    with tarfile.open(path) as archive:
        need(set(archive.getnames()) == set(members), 'archive inventory differs')
        for member in archive.getmembers():
            need(member.isfile() and digest(archive.extractfile(member).read()) == members[member.name]['sha256'], 'archive member hash differs')
    need(all(digest(source.read_bytes()) == members[relative]['sha256'] for relative, source in files.items()), 'source changed during archive')
    return dict(sha256=digest(path.read_bytes()), size=path.stat().st_size, members=members)


def main():
    started = time.time()
    freeze_path = RAW / 'source-freeze-default-predication.json'
    freeze = read(freeze_path)
    need(len(freeze['source_sha256']) == 17 and freeze['selected_source'] == str(SELECTED), 'unexpected full6 source scope')
    full6 = {relative: SELECTED / relative for relative in freeze['source_sha256']}
    need(all(digest(path.read_bytes()) == freeze['source_sha256'][relative] for relative, path in full6.items()), 'selected full6 frozen source changed')
    historical = {}
    for tag in ('owner-slot', 'composed-predicates'):
        source = RAW / f'source-{tag}.tar.gz'
        receipt = read(RAW / f'source-freeze-{tag}.json')
        with tarfile.open(source) as archive:
            contents = {member.name: archive.extractfile(member).read() for member in archive.getmembers() if member.isfile()}
        sidecars = {name: digest(data) for name, data in contents.items() if Path(name).name.startswith('._')}
        for name in sidecars:
            companion = str(Path(name).with_name(Path(name).name[2:]))
            need(companion in receipt['source_sha256'] and contents[name][:4] == b'\x00\x05\x16\x07', 'unexpected historical tar extra')
        actual = {name: digest(data) for name, data in contents.items() if name not in sidecars}
        need(actual == receipt['source_sha256'], 'historical overlay tar mismatch: ' + tag)
        historical[tag] = dict(sha256=digest(source.read_bytes()), files=actual, preserved_appledouble_metadata=sidecars)
    exact = {'run.py', 'owned.json', 'owned-through-full4.json', 'owned-through-full5.json',
             'toolchain.json', 'source-clone.json', 'isolated_tests.py', 'cpu_probe.py',
             'cohort_private_probe.py', 'probe.py', 'clone', 'configure',
             'source-owner-slot.tar.gz', 'source-composed-predicates.tar.gz',
             'metal4_copy_only_control_v1.cpp', 'metal4_copy_only_control_v1.py', 'metal4_copy_only_control_v2.py',
             'capture_copy_control_offline_prepare.py', 'metal4-copy-only-control-v1-offline-prepare',
             'metal4-copy-only-control-v2-offline-prepare', 'metal4-copy-only-control-v2'}
    prefixes = ('source-freeze-', 'build-full-', 'host-', 'metal-isolated-', 'simd-',
                'syntax-', 'tidy-', 'tool-', 'cpu-', 'cohort-private-')
    raw_files = {}
    excluded = []
    for top in sorted(RAW.iterdir()):
        if top.name not in exact and not top.name.startswith(prefixes):
            excluded.append(top.name)
            continue
        for path in ([top] if top.is_file() else sorted(top.rglob('*'))):
            if not path.is_file() or any(part in ('__pycache__', 'inductor-cache') for part in path.relative_to(RAW).parts):
                continue
            raw_files['raw/' + str(path.relative_to(RAW))] = path
    need('raw/cpu-inductor.zGCsEf/audit-results.json' in raw_files, 'missing numerical audit')
    capture = read(RAW / 'cpu-capture-6-manifest.json')
    binaries = {Path(path).name: Path(path) for path in capture['binary_sha256']}
    need(len(binaries) == 4 and all(digest(Path(path).read_bytes()) == expected for path, expected in capture['binary_sha256'].items()), 'critical binary drift')
    helper_names = ['scripts/benchmark/tile_torch/' + name for name in
                    ('native_tile.py', 'native_tile_replay.cpp', 'native_rows.py', 'native_rows_replay.cpp',
                     'compare_llm.py', 'repeat.py', 'run.py')]
    helper_names += ['src/backends/simd/llvm/llvm_schedule_codegen.h',
                     'src/tests/benchmark/benchmark_tile_xir.cpp', 'src/tests/common/tile_llm_benchmark.h',
                     'src/tests/common/tile_llm_test_utils.h']
    helpers = {'selected/' + relative: SELECTED / relative for relative in helper_names}
    for path, expected in capture['helper_sha256'].items():
        need(digest(Path(path).read_bytes()) == expected, 'capture helper drift')
    validated = read(RAW / 'cpu-inductor.zGCsEf/validated.json')
    for path, expected in validated['source_hashes'].items():
        need(digest(Path(path).read_bytes()) == expected, 'Inductor helper drift')
    helpers.update({'build/' + name: BUILD / name for name in ('CMakeCache.txt', 'compile_commands.json')})
    archives = {
        'raw-evidence.tar.gz': pack('raw-evidence.tar.gz', raw_files),
        'source-default-predication.tar.gz': pack('source-default-predication.tar.gz', full6),
        'critical-binaries-full6.tar.gz': pack('critical-binaries-full6.tar.gz', binaries),
        'supporting-sources.tar.gz': pack('supporting-sources.tar.gz', helpers),
    }
    gates = {}
    for version in range(1, 7):
        row = read(RAW / f'build-full-{version}/result.json')
        gates[f'build-full-{version}'] = {key: row.get(key) for key in ('status', 'exit_code', 'timed_out', 'source_freeze_sha256', 'started', 'finished')}
    output = dict(status='passed', started=started, finished=time.time(), original_raw=str(RAW),
                  selected_source=str(SELECTED), repository_head_label=freeze['repository_head'],
                  source_freeze_sha256=digest(freeze_path.read_bytes()), full6_source_files=freeze['source_sha256'],
                  historical_overlay_tars=historical, build_diagnostics=gates, archives=archives,
                  explicit_excluded_top_level=excluded, recursive_exclusions=['__pycache__', 'inductor-cache'],
                  binary_scope='Four critical full6 capture artifacts only, NOT a complete historical runtime/library closure.',
                  source_scope='Explicit frozen full4/full5/full6 overlays and selected helper/build configuration files, NOT a complete Git/source snapshot.',
                  scripts={name: digest((HERE / name).read_bytes()) for name in ('archive.py', 'recompute.py', 'notes.md')})
    write(HERE / 'manifest.json', output)
    print(json.dumps(dict(status='passed', archived_files=sum(len(v['members']) for v in archives.values()),
                          total_archive_bytes=sum(v['size'] for v in archives.values()),
                          archives={key: {k: v for k, v in value.items() if k != 'members'} for key, value in archives.items()}), indent=2))


if __name__ == '__main__':
    main()
