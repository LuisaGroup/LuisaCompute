"""Incremental full-build gate for the separate integer projection experiment."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'source'
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def save(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2) + '\n')
assert json.loads((HERE / 'stage-a-frozen.json').read_text())['passed']
baseline = json.loads((HERE / 'baseline-source.json').read_text())
all_files = {str(p.relative_to(SOURCE)): digest(p) for p in sorted(SOURCE.rglob('*')) if p.is_file()}
generated_names = {'src/backends/metal/metal_builtin_embedded.cpp', 'src/backends/metal/metal_tex_compress_embedded.cpp'}
assert set(all_files) - set(baseline['source_sha256']) == generated_names
assert set(baseline['source_sha256']) <= set(all_files)
generated = {name: all_files[name] for name in generated_names}
for name, sha in generated.items():
    assert digest(HERE / 'frozen-stage-a-source' / name) == sha
hashes = {name: all_files[name] for name in baseline['source_sha256']}
changed = sorted(name for name, sha in hashes.items() if sha != baseline['source_sha256'][name])
assert changed == ['src/backends/simd/llvm/llvm_jit.cpp', 'src/backends/simd/simd_compiler.cpp']
save('projection-source-snapshot.json', dict(source_commit=baseline['source_commit'], source_sha256=hashes, overlay=changed,
                                            generated_metal_cpp_sha256=generated, experimental_only=True))
command = ['cmake', '--build', str(HERE / 'build'), '--parallel', '8']
report = dict(started_unix=time.time(), command=command, source_snapshot_sha256=digest(HERE / 'projection-source-snapshot.json'))
environment = {k: v for k, v in os.environ.items() if not k.startswith(('LUISA_', 'DYLD_'))}
with (HERE / 'projection-build.log').open('x') as log:
    result = subprocess.run(command, cwd=HERE, env=environment, stdout=log, stderr=subprocess.STDOUT)
report.update(exit_code=result.returncode, log_sha256=digest(HERE / 'projection-build.log'),
              finished_unix=time.time(), source_unchanged=all(digest(SOURCE / name) == sha for name, sha in hashes.items()))
report['passed'] = result.returncode == 0 and report['source_unchanged']
save('projection-build.json', report)
print('Projection full build:', report['passed'], flush=True)
assert report['passed']
