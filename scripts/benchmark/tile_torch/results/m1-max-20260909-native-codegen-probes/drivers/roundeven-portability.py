"""Compiler-only target probe; not a kernel timing or numerical test."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
LLC = Path('/opt/homebrew/opt/llvm@21/bin/llc')
source = HERE / 'roundeven-probe.ll'
report = dict(scope='Compiler-only cross-target diagnostic. Proposed precise-math tests were not run.',
              input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), runs=[])
for target, triple, cpu in [('x86-baseline', 'x86_64-unknown-linux-gnu', 'x86-64'),
                            ('apple-m1', 'aarch64-apple-darwin', 'apple-m1')]:
    output = HERE / ('roundeven-' + target + '.s')
    assert not output.exists()
    command = [str(LLC), '-mtriple=' + triple, '-mcpu=' + cpu, str(source), '-o', '-']
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    output.write_text(result.stdout)
    calls = len(re.findall(r'callq\s+roundevenf@PLT', result.stdout))
    frintn = len(re.findall(r'frintn\.4s', result.stdout))
    assert (calls, frintn) == ((4, 0) if target == 'x86-baseline' else (0, 1))
    report['runs'].append(dict(target=target, command=command, exit_code=result.returncode,
                               stderr=result.stderr, assembly_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                               scalar_roundeven_calls=calls, vector_frintn_instructions=frintn))
report['passed'] = True
(HERE / 'roundeven-portability.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
