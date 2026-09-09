"""Preserve the closed first phase before reusing its incremental build tree."""
import hashlib
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
source = json.loads((HERE / 'source-snapshot.json').read_text())
for name, sha in source['source_sha256'].items():
    assert digest(HERE / 'frozen-stage-a-source' / name) == sha, name
capture = json.loads((HERE / 'capture/manifest.json').read_text())
relocations = {}
for old, sha in capture['binary_closure_sha256'].items():
    new = HERE / 'frozen-stage-a-bin' / Path(old).name
    assert digest(new) == sha, old
    relocations[old] = dict(path=str(new), sha256=sha)
assert json.loads((HERE / 'diagnostic-host.json').read_text())['exit_code'] == 0
report = dict(finished_unix=time.time(), passed=True, source_files=len(source['source_sha256']),
              source_location=str(HERE / 'frozen-stage-a-source'),
              source_snapshot_sha256=digest(HERE / 'source-snapshot.json'),
              binary_relocations=relocations,
              note='Phase A is closed. Captured native libraries and results remain unchanged; only source/build are reused by a separately versioned next phase.')
(HERE / 'stage-a-frozen.json').write_text(json.dumps(report, indent=2) + '\n')
print('Frozen phase A:', report['source_files'], 'sources,', len(relocations), 'binaries')
