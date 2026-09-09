#!/usr/bin/env python3
"""Archive actual object disassembly; static facts are not sampled bottlenecks."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

from measure import NATIVE_SHAPES

HERE = Path(__file__).resolve().parent
OBJDUMP = '/opt/homebrew/opt/llvm@21/bin/llvm-objdump'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    args = parser.parse_args()
    report = dict(static_only=True, hardware_profile=False, observations=[], controls=[])
    for dims in NATIVE_SHAPES:
        shape = 'x'.join(map(str, dims))
        replay = json.loads((args.raw / ('native-' + shape) / 'results.json').read_text())
        identities = {}
        for enabled in (False, True):
            name = f'rmsnorm-{shape}-r0-l8-p{int(enabled)}'
            source = args.raw / 'capture' / name
            target = HERE / 'captures' / name
            objects = list((source / 'object').glob('*.o'))
            assembly = list((source / 'object').glob('*.s'))
            assert len(objects) == len(assembly) == 1
            command = [OBJDUMP, '-d', '--no-show-raw-insn', str(objects[0])]
            disassembly = subprocess.check_output(command, text=True)
            (target / 'object.asm.gz').write_bytes(gzip.compress(disassembly.encode(), mtime=0))
            (target / 'compiler.s.gz').write_bytes(gzip.compress(assembly[0].read_bytes(), mtime=0))
            body = assembly[0].read_text()
            identities[enabled] = digest(objects[0]), digest(source / 'kernel.ll')
            report['observations'].append(dict(shape=dims, enabled=enabled, command=command,
                                                object_sha256=identities[enabled][0], compiler_asm_sha256=digest(assembly[0]),
                                                vector_lane_load_sites=len(re.findall(r'^\s*ld1\.s\s+\{[^}]+\}\[', body, re.M)),
                                                scalar_or_vector_divide_sites=len(re.findall(r'^\s*fdiv(?:\.4s)?\s', body, re.M)),
                                                scope='entire emitted ORC object including residual fallback paths; static sites, not dynamic execution counts'))
        report['controls'].append(dict(shape=dims, object_identity=identities[False][0] == identities[True][0],
                                       llvm_identity=identities[False][1] == identities[True][1]))
        cpp = Path(replay['artifacts']['inductor']['source'])
        command = [OBJDUMP, '--disassemble-symbols=_kernel', '--no-show-raw-insn', str(cpp.with_suffix('.so'))]
        disassembly = subprocess.check_output(command, text=True)
        assert '<_kernel>:' in disassembly
        (HERE / ('native-' + shape) / 'inductor.kernel.asm.gz').write_bytes(gzip.compress(disassembly.encode(), mtime=0))
        report['observations'].append(dict(shape=dims, implementation='inductor', command=command,
                                            source_sha256=digest(cpp), source_tail_partition='C10_UNLIKELY' in cpp.read_text(),
                                            source_cascade_sum='CascadeSumHelper' in cpp.read_text(),
                                            scope='exported kernel disassembly; callees are not flattened'))
    (HERE / 'native-inspection.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'PASS: {2 * len(NATIVE_SHAPES)} ORC and {len(NATIVE_SHAPES)} Inductor disassemblies archived; identities and static observations recorded')


if __name__ == '__main__':
    main()
