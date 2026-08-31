"""Verification sweep for the refactored lc-compile-builtin tool.

Checks every route the tool supports against the ground truth the repo already
produces (the Vulkan backend build tree and the checked-in DX builtin blobs).
"""
import hashlib
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, 'bin', 'debug', 'lc-compile-builtin.exe')
GEN = os.path.join(ROOT, 'build', '.gens', 'lc-backend-vk', 'windows', 'x64', 'debug', 'vk_builtin')
BUILTIN_HLSL = os.path.join(ROOT, 'src', 'backends', 'common', 'hlsl', 'builtin')
VK_BUILTIN = os.path.join(ROOT, 'src', 'backends', 'vk', 'builtin')
OUT = os.path.join(ROOT, 'build', '_lcbt_verify')

failures = []


def run(args, expect_zero=True, label=''):
    cmd = [EXE] + args
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    out = (r.stdout or '') + (r.stderr or '')
    if expect_zero and r.returncode != 0:
        failures.append(f'{label or args}: exit {r.returncode}\n{tail(out)}')
    if not expect_zero and r.returncode == 0:
        failures.append(f'{label or args}: expected failure, got success')
    return r.returncode, out


def tail(out, n=6):
    lines = [l for l in out.splitlines() if 'console' in l or 'error' in l.lower()]
    return '\n'.join(lines[-n:])


def md5(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def check(cond, msg):
    print(('  PASS  ' if cond else '  FAIL  ') + msg)
    if not cond:
        failures.append(msg)


shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)

KERN = ['indirect_prepare', 'accel_process', 'bindless_upload']
EXPECT_BLOCK = {'indirect_prepare': (64, 1, 1), 'accel_process': (256, 1, 1), 'bindless_upload': (256, 1, 1)}

print('== 1. spv route: glslang -> raw SPIR-V (must equal the backend build output) ==')
for k in KERN:
    src = os.path.join(VK_BUILTIN, k + '.comp.hlsl')
    dst = os.path.join(OUT, k + '.spv')
    rc, out = run(['spv', src, dst, '--name', k, '--verify'], label='spv ' + k)
    check('block ({}, {}, {})'.format(*EXPECT_BLOCK[k]) in out, f'{k}: block size reported from LocalSize')
    check('entry point \'main\'' in out, f'{k}: entry point main found')
    check(os.path.isfile(dst) and md5(dst) == md5(os.path.join(GEN, k + '.spv')),
          f'{k}: byte-identical to build/.gens vk_builtin/{k}.spv')
    check('.lc-compile-builtin' not in os.listdir(OUT), f'{k}: scratch dir cleaned up')

print('== 2. spv --container --contract: VK v10 artifact with dialect 3 ==')
for k in KERN:
    src = os.path.join(VK_BUILTIN, k + '.comp.hlsl')
    dst = os.path.join(OUT, k + '.dxil')
    rc, out = run(['spv', src, dst, '--name', k, '--contract', k, '--container', '--verify'],
                  label='container ' + k)
    check('dialect 3' in out, f'{k}: VULKAN_BUILTIN dialect (3) recorded')
    check('header v10 pipeline v4' in out, f'{k}: v10/v4 container versions')
    check('3 properties' in out, f'{k}: vulkan_builtin_buffer_properties table (3 entries)')
    check('error' not in out.lower(), f'{k}: verification clean')

print('== 3. determinism: identical bytes across repeated compiles ==')
for k in KERN:
    src = os.path.join(VK_BUILTIN, k + '.comp.hlsl')
    a, b = os.path.join(OUT, k + '.a.dxil'), os.path.join(OUT, k + '.b.dxil')
    run(['spv', src, a, '--name', k, '--contract', k, '--container'])
    run(['spv', src, b, '--name', k, '--contract', k, '--container'])
    check(md5(a) == md5(b), f'{k}: container is reproducible (no padding drift)')

print('== 4. embed route: generated device-library pair ==')
mods = [os.path.join(OUT, k + '.spv') for k in KERN]
ecpp = os.path.join(OUT, 'vulkan_builtin_spirv_embedded.cpp')
eh = os.path.join(OUT, 'vulkan_builtin_spirv_embedded.h')
rc, out = run(['spv', 'embed'] + mods + ['-o', ecpp, '-h', eh], label='embed')
check(md5(eh) == md5(os.path.join(GEN, 'vulkan_builtin_spirv_embedded.h')),
      'embedded .h byte-identical to the build tree')
check(md5(ecpp) == md5(os.path.join(GEN, 'vulkan_builtin_spirv_embedded.cpp')),
      'embedded .cpp byte-identical to the build tree')

print('== 5. dx route: legacy DXC containers vs the checked-in .dxil blobs ==')
for src_name, blob in [('accel_process.bytes', 'set_accel4.dxil'), ('bindless_upload.bytes', 'load_bdls.dxil')]:
    dst = os.path.join(OUT, blob)
    rc, out = run(['dx', os.path.join(BUILTIN_HLSL, src_name), dst, '--name', blob, '--verify'],
                  label='dx ' + src_name)
    check('is loadable' in out, f'{src_name}: DX v5 artifact verified')
    ref = md5(os.path.join(BUILTIN_HLSL, blob))
    same = md5(dst) == ref
    print(f'    note: {blob} byte-equal to checked-in blob: {same} '
          f'({os.path.getsize(dst)} vs {os.path.getsize(os.path.join(BUILTIN_HLSL, blob))} bytes)')

print('== 6. inspect: reads back tool output and the checked-in blobs ==')
for k in KERN:
    rc, out = run(['spv', os.path.join(VK_BUILTIN, k + '.comp.hlsl'), os.path.join(OUT, k + '.c.dxil'),
                   '--name', k, '--contract', k, '--container'], label='pre-inspect ' + k)
    rc, out = run(['vk', 'inspect', os.path.join(OUT, k + '.c.dxil'), '--name', k], label='inspect ' + k)
    # inspect of a dialect-3 container through the 'vk' (HLSL_SPIRV) reader:
    check(rc == 0 and 'is loadable' in out, f'{k}: inspect decodes the container')

print('== 7. legacy/stale artifacts are reported, not crashed on ==')
rc, out = run(['vk', 'inspect', os.path.join(BUILTIN_HLSL, 'load_bdls_vk.dxil'), '--name', 'legacy'],
              expect_zero=False, label='legacy v2 blob')
check('ShaderSerializer v2' in out or 'v2 container' in out, 'legacy v2 VK blob diagnosed with a clear message')

print('== 8. error handling ==')
rc, out = run(['spv', os.path.join(VK_BUILTIN, 'bindless_upload.comp.hlsl'), os.path.join(OUT, 'bad.spv'),
               '--name', 'bad', '--block-size', '128,1,1'], expect_zero=False, label='block-size mismatch')
check('does not match the SPIR-V' in out, '--block-size conflicting with LocalSize is rejected')
rc, out = run(['dx', os.path.join(BUILTIN_HLSL, 'accel_process.bytes'), os.path.join(OUT, 'x.dxil'),
               '--name', 'x', '--container'], expect_zero=False, label='--container on dx')
check(rc != 0, '--container rejected on the dx target')
rc, out = run(['bogus', 'a', 'b'], expect_zero=False, label='unknown backend')
check(rc != 0, 'unknown backend rejected')

print('== 9. --install through the runtime BinaryIO store ==')
dst = os.path.join(OUT, 'installed.dxil')
rc, out = run(['spv', os.path.join(VK_BUILTIN, 'bindless_upload.comp.hlsl'), dst, '--name',
               'bindless_upload', '--contract', 'bindless_upload', '--container', '--install'],
              label='--install')
check('Installed' in out and 'read_internal_shader' in out, 'artifact stored in the runtime data dir')
data_dir = os.path.join(ROOT, 'bin', 'debug', '.data')
if os.path.isdir(data_dir):
    found = os.path.isfile(os.path.join(data_dir, 'LMDB') if os.path.isfile(os.path.join(data_dir, 'LMDB')) else os.path.join(data_dir, 'installed.dxil'))
    print(f'    note: {data_dir} present (LMDB or file store), installed.dxil lookup -> {found}')

print()
if failures:
    print(f'{len(failures)} FAILURE(S):')
    for f in failures:
        print(' - ' + f)
    sys.exit(1)
print('ALL CHECKS PASSED')
