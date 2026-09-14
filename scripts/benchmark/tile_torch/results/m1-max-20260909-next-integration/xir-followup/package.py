"""Archive integration receipts, not performance measurements or binaries."""
import gzip
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = Path('/Users/mike/CLionProjects/luisa')
DEST = ROOT / 'scripts/benchmark/tile_torch/results/m1-max-20260909-next-integration/xir-followup'
assert DEST.is_dir() and not (DEST / 'provenance.json').exists()
verification = json.loads((HERE / 'verification-final.json').read_text())
documentation = json.loads((HERE / 'documentation.json').read_text())
assert verification['passed'] and documentation['passed']


def digest(data):
    return hashlib.sha256(data).hexdigest()


names = ['checkpoint.py', 'recheck.py', 'documentation.py', 'package.py',
         'configure-command.json', 'pinned-repositories.json', 'source-snapshot.json',
         'verification.json', 'verification-final.json', 'documentation.json',
         'ctest-metal-codegen.log', 'metal-codegen.xml']
executed = set()
for group in (verification, documentation):
    for run in group['runs']:
        raw = (HERE / run['log']).read_bytes()
        assert digest(raw) == run['log_sha256'], run
        names.append(run['log'])
        if run.get('junit'):
            name = run['junit']
            assert digest((HERE / name).read_bytes()) == run['junit_sha256']
            cases = list(ET.parse(HERE / name).getroot().iter('testcase'))
            assert len(cases) == run['tests'] and run['failures'] == 0 and run['skipped'] == 0
            for case in cases:
                assert case.find('failure') is None and case.find('error') is None and case.find('skipped') is None
                assert case.attrib['name'] not in executed
                executed.add(case.attrib['name'])
            names.append(name)
assert len(executed) == 116
for p, h in documentation['source_sha256'].items():
    assert digest((ROOT / p).read_bytes()) == h
source = json.loads((HERE / 'source-snapshot.json').read_text())
assert digest((HERE / 'source-snapshot.json').read_bytes()) == verification['source_snapshot_sha256']
for p, h in source['source_sha256'].items():
    assert digest((HERE / 'source' / p).read_bytes()) == h

records = []
for name in sorted(set(names)):
    raw = (HERE / name).read_bytes()
    compressed = name.endswith(('.log', '.xml')) or name == 'source-snapshot.json'
    saved_name = name + '.gz' if compressed else name
    stored = gzip.compress(raw, compresslevel=9, mtime=0) if compressed else raw
    assert not (DEST / saved_name).exists()
    (DEST / saved_name).write_bytes(stored)
    roundtrip = gzip.decompress((DEST / saved_name).read_bytes()) if compressed else (DEST / saved_name).read_bytes()
    assert roundtrip == raw
    records.append(dict(path=saved_name, source_name=name, bytes=len(raw), stored_bytes=len(stored),
                        sha256=digest(raw), stored_sha256=digest(stored)))
report = dict(source_commit=verification['source_commit'], next_commit=verification['next_commit'],
              local_evidence=str(HERE), payloads=records, distinct_ctests=116,
              performance_measurement=False, archived_binaries=False,
              notes_sha256=digest((DEST / 'notes.md').read_bytes()))
(DEST / 'provenance.json').write_text(json.dumps(report, indent=2) + '\n')
print(len(records), 'payloads archived;', sum(r['stored_bytes'] for r in records), 'stored bytes')
