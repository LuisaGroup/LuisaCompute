"""Verify the selected archive offline. Never load native code or run benchmarks."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def main():
    report = Path(sys.argv[1]).resolve()
    inventory = load(report / 'package-inventory.json')
    archive = report / 'evidence.tar.xz'
    need(sha(archive) == inventory['archive_sha256'], 'archive digest mismatch')
    need(sha(report / 'notes.md') == inventory['notes_sha256'], 'report digest mismatch')
    expected = dict(inventory['files'])
    generated = inventory['generated_member']
    expected[generated['name']] = generated
    out = Path(tempfile.mkdtemp(prefix='luisa-profile-offline-'))
    with tarfile.open(archive, 'r:xz') as bundle:
        members = bundle.getmembers()
        need(len(members) == len(expected) and {m.name for m in members} == set(expected), 'archive member inventory mismatch')
        for member in members:
            path = PurePosixPath(member.name)
            need(member.isfile() and not path.is_absolute() and '..' not in path.parts and '\\' not in member.name,
                 'unsafe nonregular archive member')
            need(member.size == expected[member.name]['bytes'], 'archive member size mismatch')
            data = bundle.extractfile(member).read()
            need(hashlib.sha256(data).hexdigest() == expected[member.name]['sha256'], 'member digest mismatch: ' + member.name)
        bundle.extractall(out, filter='data')
    profile = out / 'profile'
    selection = load(profile / 'public-selection.json')
    attempts = []
    for label in selection['labels']:
        trace = profile / ('trace-' + label)
        saved = trace / 'analysis-v3'
        recomputed = trace / 'offline-recomputed'
        argv = [sys.executable, '-B', str(profile / 'parse_profile.py'), '--trace-dir', str(trace),
                '--image-root', str(out / 'native' / ('prepared-' + label)),
                '--clock-anchor', str(saved / 'clock-anchor.json'), '--output', str(recomputed)]
        process = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        (trace / 'offline.stdout').write_bytes(process.stdout)
        (trace / 'offline.stderr').write_bytes(process.stderr)
        need(process.returncode == 0, 'offline parser failed: ' + label + ': ' + process.stderr.decode(errors='replace'))
        for filename in ('profile-summary.json', 'normalized-samples.json', 'leaf-pcs.csv', 'clock-anchor.json', 'xml-privacy-audit.json'):
            need(sha(saved / filename) == sha(recomputed / filename), 'offline recomputation differs: ' + label + '/' + filename)
        if label.endswith('-on'):
            case = label.removesuffix('-on')
            rules = profile / 'linked-analysis.pb43ED'
            if case != 'decode-mha-d64':
                rules /= case
            phase_argv = [sys.executable, '-B', str(profile / 'classify_phases.py'),
                          '--analysis', str(recomputed), '--ranges', str(rules / 'phase-ranges.json'),
                          '--output', str(recomputed / 'phase-summary.json')]
            classified = subprocess.run(phase_argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            (trace / 'offline-phases.stdout').write_bytes(classified.stdout)
            (trace / 'offline-phases.stderr').write_bytes(classified.stderr)
            need(classified.returncode == 0, 'offline phase classification failed: ' + label)
            need(sha(saved / 'phase-summary.json') == sha(recomputed / 'phase-summary.json'), 'phase classification differs: ' + label)
        attempts.append(dict(label=label, command=argv, status='byte_identical'))
    tests = subprocess.run([sys.executable, '-B', str(profile / 'test_parse_profile.py')],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    (out / 'parser-tests.stdout').write_bytes(tests.stdout)
    (out / 'parser-tests.stderr').write_bytes(tests.stderr)
    need(tests.returncode == 0, 'synthetic parser tests failed')
    result = dict(status='passed', archive_sha256=sha(archive), verified_members=len(expected),
                  extracted_to=str(out), parser_recomputations=attempts, synthetic_tests_returncode=tests.returncode,
                  scope='Exact inventory and raw-XML extraction/statistics. No native code, new timing, compiler build or reconstruction of omitted attention inputs.')
    (out / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
