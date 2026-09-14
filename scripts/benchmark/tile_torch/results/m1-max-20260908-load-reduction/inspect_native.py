#!/usr/bin/env python3
"""Static instruction inventory from actual ORC objects, not a dynamic profile."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
records = []
for directory in sorted((HERE / 'capture-final').glob('rmsnorm-64x256-*')):
    data = gzip.decompress((directory / 'kernel.o.gz').read_bytes())
    with tempfile.TemporaryDirectory(prefix='luisa-fusion-disassemble-') as temporary:
        obj = Path(temporary) / 'kernel.o'
        obj.write_bytes(data)
        text = subprocess.check_output([
            '/opt/homebrew/opt/llvm@21/bin/llvm-objdump', '--macho', '--disassemble',
            '--no-show-raw-insn', str(obj)], text=True)
    if '\n_llm_rows.packet_batch.blocks:\n' not in text:
        raise ValueError('expected actual native block-batch entry')
    # Inlining changes symbol topology: do not compare an outlined body with
    # an entire inlined entry under the misleading label "same function".
    symbols = re.findall(r'^(_llm_rows[^\n]*):$', text, re.M)
    instructions = [line.split(':', 1)[1].strip().split()[0] for line in text.splitlines()
                    if re.match(r'\s*[0-9a-f]+:\s+\S', line)]
    if not instructions:
        raise ValueError('empty disassembly')
    records.append(dict(case=directory.name, object_sha256=hashlib.sha256(data).hexdigest(),
                        symbols=symbols, scope='complete native text section', instructions=len(instructions),
                        conditional_branches=sum(i in ('tbz', 'tbnz', 'cbz', 'cbnz') or i.startswith('b.') for i in instructions),
                        loads=sum(i.startswith('ld') for i in instructions),
                        stores=sum(i.startswith('st') for i in instructions)))
report = dict(metric='static_instruction_sites', dynamic_profile=False, cycles=False,
              caveat='Static sites do not establish executed branch counts, misprediction, stalls or spill traffic.', records=records)
(HERE / 'assembly-summary.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
