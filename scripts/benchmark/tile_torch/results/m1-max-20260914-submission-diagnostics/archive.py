"""Archive only terminal submission/compiler diagnostics; no native execution."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time

HERE = Path(__file__).resolve().parent
RAW = Path('/tmp/luisa-metal-program-team.TGZv8f').resolve()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def need(value, message):
    if not value:
        raise ValueError(message)


def main():
    started = time.time()
    names = {'metal4-copy-only-control-v3', 'metal4-copy-only-control-v4',
             'metal4-copy-only-control-v3-offline-prepare', 'capture_copy_control_v3_offline_prepare.py',
             'metal4_copy_only_control_v3.cpp', 'metal4_copy_only_control_v3.py',
             'metal4_copy_only_control_v4.cpp', 'metal4_copy_only_control_v4.py',
             'source-uniform-read-lane.tar.gz', 'source-uniform-read-lane-diagnostic.tar.gz',
             'source-freeze-uniform-read-lane.json', 'source-freeze-uniform-read-lane-diagnostic.json',
             'build-full-7', 'test-full7-unit-simd', 'build-full-8', 'test-full8-codegen-diagnostic',
             'owned.json', 'owned-through-full6.json', 'run.py', 'toolchain.json', 'source-clone.json', 'clone'}
    for prefix in ('tidy-full7-', 'format-full7-'):
        matched = sorted(path.name for path in RAW.iterdir() if path.name.startswith(prefix))
        need(len(matched) == 8, 'expected eight terminal ' + prefix + 'receipts')
        names.update(matched)
    freezes, overlays = {}, {}
    for tag in ('uniform-read-lane', 'uniform-read-lane-diagnostic'):
        freeze_path = RAW / ('source-freeze-' + tag + '.json')
        frozen = read(freeze_path)
        freezes[tag] = frozen
        need(len(frozen['source_sha256']) == 23 and sha((RAW / 'owned.json').read_bytes()) == frozen['owned_sha256'], 'wrong frozen ownership')
        need(sha((RAW / 'toolchain.json').read_bytes()) == frozen['toolchain_sha256'] and
             sha((RAW / 'source-clone.json').read_bytes()) == frozen['baseline_clone_sha256'], 'toolchain/clone receipt drift')
        path = RAW / ('source-' + tag + '.tar.gz')
        with tarfile.open(path) as archive:
            contents = {entry.name: archive.extractfile(entry).read() for entry in archive.getmembers() if entry.isfile()}
        sidecars = {name: sha(data) for name, data in contents.items() if Path(name).name.startswith('._')}
        for name in sidecars:
            companion = str(Path(name).with_name(Path(name).name[2:]))
            need(companion in frozen['source_sha256'] and contents[name][:4] == b'\x00\x05\x16\x07', 'unrecognized tar metadata')
        actual = {name: sha(data) for name, data in contents.items() if name not in sidecars}
        need(actual == frozen['source_sha256'], 'frozen tar source mismatch')
        overlays[tag] = dict(source_sha256=actual, preserved_appledouble_metadata=sidecars, original_tar_sha256=sha(path.read_bytes()),
                             freeze_sha256=sha(freeze_path.read_bytes()))
    need(len(read(RAW / 'owned-through-full6.json')) == 17, 'wrong previous ownership inventory')
    statuses = {}
    for name in sorted(names):
        folder = RAW / name
        if not folder.is_dir():
            continue
        result_path = folder / ('native/result.json' if name in ('metal4-copy-only-control-v3', 'metal4-copy-only-control-v4') else 'result.json')
        if not result_path.is_file():
            continue
        result = read(result_path)
        need('finished' in result and result.get('status') in ('passed', 'failed', 'Error'), 'nonterminal receipt: ' + name)
        statuses[name] = {key: result.get(key) for key in ('status', 'exit_code', 'timed_out', 'termination_reason', 'started', 'finished')}
        for channel in ('stdout', 'stderr'):
            key = channel + '_sha256'
            if key in result:
                need(sha((result_path.parent / (channel + '.txt')).read_bytes()) == result[key], 'changed diagnostic log')
        if name in ('metal4-copy-only-control-v3', 'metal4-copy-only-control-v4'):
            need(result['child_exited'] and result['status'] == 'failed', 'Metal diagnostic is not terminal Error')
            for path, expected in result['artifacts'].items():
                need(sha(Path(path).read_bytes()) == expected['sha256'], 'Metal retained artifact drift')
        if name.startswith(('build-full-7', 'test-full7-', 'tidy-full7-', 'format-full7-')):
            need(result['source_sha256'] == freezes['uniform-read-lane']['source_sha256'] and
                 result['source_freeze_sha256'] == overlays['uniform-read-lane']['freeze_sha256'], 'full7 receipt/freeze mismatch')
        if name.startswith(('build-full-8', 'test-full8-')):
            need(result['source_sha256'] == freezes['uniform-read-lane-diagnostic']['source_sha256'] and
                 result['source_freeze_sha256'] == overlays['uniform-read-lane-diagnostic']['freeze_sha256'], 'full8 receipt/freeze mismatch')
    need(statuses['test-full7-unit-simd']['status'] == statuses['test-full8-codegen-diagnostic']['status'] == 'failed', 'lost original test failures')
    files = {}
    for name in sorted(names):
        path = RAW / name
        need(path.exists(), 'missing explicit archive target: ' + name)
        for source in ([path] if path.is_file() else sorted(path.rglob('*'))):
            if source.is_file() and '__pycache__' not in source.parts:
                need(not source.is_symlink(), 'unexpected link')
                files['raw/' + str(source.relative_to(RAW))] = source
    members = {}
    output = HERE / 'diagnostics.tar.gz'
    with output.open('xb') as stream, gzip.GzipFile(fileobj=stream, mode='wb', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as archive:
            for name, source in sorted(files.items()):
                data = source.read_bytes()
                entry = tarfile.TarInfo(name)
                entry.size, entry.mode = len(data), 0o644
                archive.addfile(entry, io.BytesIO(data))
                members[name] = dict(sha256=sha(data), size=len(data), original_path=str(source))
    with tarfile.open(output) as archive:
        need(set(archive.getnames()) == set(members), 'archive inventory mismatch')
        for entry in archive.getmembers():
            need(sha(archive.extractfile(entry).read()) == members[entry.name]['sha256'], 'archive byte mismatch')
    need(all(sha(path.read_bytes()) == members[name]['sha256'] for name, path in files.items()), 'source changed while archiving')
    result = dict(status='passed', interpretation='Archive integrity passed; retained native/compiler diagnostics remain Error.',
                  started=started, finished=time.time(), archive='diagnostics.tar.gz', archive_sha256=sha(output.read_bytes()),
                  archive_bytes=output.stat().st_size, original_raw=str(RAW), source_overlays=overlays,
                  terminal_diagnostics=statuses, included_top_level=sorted(names), members=members,
                  excluded='All other raw entries, live full9/CPU ablation work, pycache, old full6 large archive and machine caches.',
                  source_scope='Two explicit 23-file source overlays verified against their own freeze; not a full Git snapshot.',
                  predecessor='../m1-max-20260914-program-team/',
                  predecessor_manifest_sha256=sha((HERE.parent / 'm1-max-20260914-program-team/manifest.json').read_bytes()),
                  no_native_execution=True, no_selected_or_root_source_reads=True,
                  archive_script_sha256=sha(Path(__file__).read_bytes()), notes_sha256=sha((HERE / 'notes.md').read_bytes()))
    with (HERE / 'manifest.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    with (HERE / 'SHA256SUMS').open('x') as stream:
        for path in sorted(HERE.iterdir()):
            if path.is_file() and path.name != 'SHA256SUMS':
                stream.write(sha(path.read_bytes()) + '  ' + path.name + '\n')
    print(json.dumps(dict(status='passed', files=len(files), bytes=output.stat().st_size,
                          archive_sha256=result['archive_sha256'], manifest_sha256=sha((HERE / 'manifest.json').read_bytes())), indent=2))


if __name__ == '__main__':
    main()
