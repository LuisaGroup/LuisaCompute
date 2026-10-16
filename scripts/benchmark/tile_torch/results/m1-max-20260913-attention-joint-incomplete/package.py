"""Package frozen incomplete evidence; never run its native objects or audit.py."""
import argparse
import hashlib
import json
import lzma
from pathlib import Path
import shutil
import tarfile

OUT = Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inventory(root):
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('refusing symlink: ' + str(path))
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError('refusing nonregular evidence: ' + str(path))
        result[path.relative_to(root).as_posix()] = dict(bytes=path.stat().st_size, sha256=sha(path))
    return result


def write_json(name, value):
    with (OUT / name).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw', type=Path)
    args = parser.parse_args()
    raw = args.raw.resolve(strict=True)
    if raw == OUT or OUT.is_relative_to(raw) or raw.is_relative_to(OUT):
        raise ValueError('source and package must be disjoint')
    before = inventory(raw)
    archive_names = ('evidence.tar.xz', 'sources.tar.gz', 'provenance.json', 'package-inventory.json', 'SHA256SUMS')
    if any((OUT / name).exists() for name in archive_names):
        raise ValueError('package outputs already exist; preserve the prior attempt')
    source = before['sources.tar.gz']
    evidence = {name: info for name, info in before.items() if name != 'sources.tar.gz'}
    with (OUT / 'evidence.tar.xz').open('xb') as destination:
        # Fast hash-chain search, with enough history to deduplicate complete
        # raw/prepared tensors across the five captured variants. Preset 0's
        # default 256 KiB dictionary produced a 229 MB first attempt despite
        # these exact duplicates; increase history, not search complexity.
        filters = [{'id': lzma.FILTER_LZMA2, 'preset': 0, 'dict_size': 128 * 1024 * 1024}]
        with lzma.LZMAFile(destination, 'w', filters=filters) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as archive:
                for name in evidence:
                    path = raw / name
                    info = tarfile.TarInfo(name)
                    info.size = evidence[name]['bytes']
                    info.mode = 0o644
                    with path.open('rb') as stream:
                        archive.addfile(info, stream)
    shutil.copyfile(raw / 'sources.tar.gz', OUT / 'sources.tar.gz')
    shutil.copyfile(raw / 'provenance.json', OUT / 'provenance.json')
    if inventory(raw) != before:
        raise ValueError('raw evidence changed during packaging; preserve failed output')
    # Read the resulting compressed stream independently; do not extract or
    # execute archived helpers. This also catches accidental duplicate paths.
    verified = {}
    with tarfile.open(OUT / 'evidence.tar.xz', 'r:xz') as archive:
        for member in archive:
            if not member.isfile() or member.name in verified:
                raise ValueError('nonregular/duplicate evidence member')
            verified[member.name] = dict(bytes=member.size, sha256=hashlib.file_digest(archive.extractfile(member), 'sha256').hexdigest())
    if verified != evidence or sha(OUT / 'sources.tar.gz') != source['sha256']:
        raise ValueError('packaged contents differ from frozen raw')
    write_json('package-inventory.json', dict(format='incomplete-attention-evidence-v1',
                original_raw=str(raw), raw_unchanged_after_packaging=True,
                evidence_compression=dict(format='xz', codec='lzma2', preset=0,
                                          dictionary_bytes=128 * 1024 * 1024, threads=1), evidence_files=evidence,
                source_archive=source, excluded_from_evidence=['sources.tar.gz'],
                evidence_file_count=len(evidence), evidence_uncompressed_bytes=sum(x['bytes'] for x in evidence.values()),
                evidence_sha256=sha(OUT / 'evidence.tar.xz'),
                boundary='All original regular files preserved; source archive stored once separately. No new performance measurements.'))
    with (OUT / 'SHA256SUMS').open('x') as stream:
        for path in sorted(OUT.iterdir()):
            if path.is_file() and path.name != 'SHA256SUMS':
                stream.write(sha(path) + '  ' + path.name + '\n')
    print(json.dumps(dict(status='packaged', evidence_files=len(evidence), evidence_bytes=(OUT / 'evidence.tar.xz').stat().st_size,
                          sources_bytes=source['bytes'], raw_unchanged=True)))
