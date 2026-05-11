import os
import shutil
import subprocess
import sys

def find_dxc():
    # 1. global 'dxc'
    p = shutil.which('dxc')
    if p:
        return p
    # 2. 'dxc.exe' in local dir
    local = os.path.join(os.getcwd(), 'dxc.exe')
    if os.path.isfile(local):
        return local
    # 3. 'bin/debug/dxc.exe'
    debug = os.path.join('bin', 'debug', 'dxc.exe')
    if os.path.isfile(debug):
        return debug
    return None

def main():
    hlsl_path = r'bin\debug\hlsl_output.hlsl'
    output_dir = r'bin\debug'

    dxc_path = find_dxc()
    if dxc_path is None:
        print('Error: dxc not found. Searched: global dxc, ./dxc.exe, bin/debug/dxc.exe')
        sys.exit(1)
    print(f'Using dxc: {dxc_path}')
    delimiter = '#define _INF_f (1.#INF)'

    if not os.path.exists(hlsl_path):
        print(f'Error: {hlsl_path} not found')
        sys.exit(1)

    with open(hlsl_path, 'r', encoding='utf-8') as f:
        content = f.read()

    parts = content.split(delimiter)

    compiled_count = 0
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue

        # Re-add the delimiter so each file is self-contained
        hlsl_code = delimiter + '\n' + part + '\n'
        hlsl_file = os.path.join(output_dir, f'hlsl_part_{i}.hlsl')
        spv_file = os.path.join(output_dir, f'hlsl_part_{i}.spvasm')

        with open(hlsl_file, 'w', encoding='utf-8') as f:
            f.write(hlsl_code)

        cmd = [
            dxc_path,
            '-spirv',
            '/DSPV',
            '-fspv-target-env=vulkan1.1',
            '-all_resources_bound',
            '-enable-16bit-types',
            '-Zpr',
            '-Gfa',
            '-HV', '2021',
            '-T', 'cs_6_6',
            '-E', 'main',
            hlsl_file,
            '-Fc', spv_file,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f'Success: {spv_file}')
            compiled_count += 1
        else:
            print(f'Error compiling {hlsl_file}:')
            if result.stderr:
                print(result.stderr)
            if result.stdout:
                print(result.stdout)

    print(f'\nTotal compiled: {compiled_count}')

if __name__ == '__main__':
    main()
