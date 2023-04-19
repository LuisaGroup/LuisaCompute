import multiprocessing
import os
import sys
from subprocess import Popen
from typing import List


def default_config():
    return {
        'cmake_args': [],
        'xmake_args': [],
        'build_system': 'cmake',
        'features': [],
        'output': 'build',
    }


platform = sys.platform


def install_dep(dep: str):
    if dep == 'rust':
        if platform == 'win32':
            # download https://static.rust-lang.org/rustup/dist/i686-pc-windows-gnu/rustup-init.exe
            os.system(
                'curl -sSf https://static.rust-lang.org/rustup/dist/i686-pc-windows-gnu/rustup-init.exe -o rustup-init.exe')
            os.system('rustup-init.exe -y')
        elif platform == 'linux' or platform == 'darwin':
            os.system('curl https://sh.rustup.rs -sSf | sh -s -- -y')
        else:
            raise ValueError(f'Unknown platform: {platform}')
    else:
        raise ValueError(f'Unknown dependency: {dep}')


def get_config():
    # check if config.json exists
    if os.path.exists('config.json'):
        import json
        with open('config.json', 'r') as f:
            return json.load(f)
    else:
        return default_config()


def print_help():
    print('Usage: python bootstrap.py [build system] [options]')
    print('Build system:')
    print('  cmake                  Use CMake')
    print('  xmake                  Use xmake')
    print('Options:')
    print('  --config               Configure build system')
    print('  --features [features]  Add features')
    print('  --build [N]            Build (N = number of jobs)')
    print('  --clean                Clean build directory')
    print('  --install [deps]       Install dependencies')
    print('  --output               Path to output directory')
    print('  -- [args]              Pass arguments to build system')


def dump_build_system_args(config: dict):
    args = build_system_args(config)
    with open(f"{config['output']}/options.cli", 'w') as f:
        print('\n'.join(args), file=f)


def build_system_args_cmake(config: dict) -> List[str]:
    args = config['cmake_args']
    if 'cuda' in config['features']:
        args.append('-DLUISA_COMPUTE_ENABLE_CUDA=ON')
    if 'cpu' in config['features']:
        args.append('-DLUISA_COMPUTE_ENABLE_CPU=ON')
    if 'remote' in config['features']:
        args.append('-DLUISA_COMPUTE_ENABLE_REMOTE=ON')
    if 'dx' in config['features']:
        args.append('-DLUISA_COMPUTE_ENABLE_DX=ON')
    if 'metal' in config['features']:
        args.append('-DLUISA_COMPUTE_ENABLE_METAL=ON')
    return args


def build_system_args_xmake(config: dict) -> List[str]:
    args = config['xmake_args']
    if 'cuda' in config['features']:
        args.append('-c')
    return args


def build_system_args(config) -> List[str]:
    if config['build_system'] == 'cmake':
        return build_system_args_cmake(config)
    elif config['build_system'] == 'xmake':
        return build_system_args_xmake(config)
    else:
        raise ValueError(f'Unknown build system: {config["build_system"]}')


def main(args: List[str]):
    if len(args) == 1:
        print_help()
        return
    i = 1
    config = get_config()
    if i < len(args) and not args[i].startswith('-'):
        config['build_system'] = args[i]
        i += 1
    run_config = False
    run_build = False
    build_jobs = multiprocessing.cpu_count()
    while i < len(args):
        opt = args[i]
        if opt == '--clean':
            if os.path.exists(config['output']):
                import shutil
                shutil.rmtree(config['output'])
            return
        elif opt == '--help' or opt == '-h':
            print_help()
            return
        elif opt == '--config' or opt == '-c':
            run_config = True
            i += 2
        elif opt == '--build' or opt == '-b':
            run_build = run_config = True
            i += 1
            if i < len(args) and not args[i].startswith('-'):
                build_jobs = int(args[i])
                i += 1
        elif opt == '--features' or opt == '-f':
            i += 1
            while i < len(args) and not args[i].startswith('-'):
                config['backends'].append(args[i])
                i += 1
        elif opt == '--install' or opt == '-i':
            i += 1
            while i < len(args) and not args[i].startswith('-'):
                install_dep(args[i])
                i += 1
        elif opt == '--output' or opt == '-o':
            config['output'] = args[i + 1]
            i += 2
        elif opt == "--":
            if config['build_system'] == 'cmake':
                config['cmake_args'] = args[i + 1:]
            elif config['build_system'] == 'xmake':
                config['xmake_args'] = args[i + 1:]
            else:
                raise ValueError(f'Unknown build system: {config["build_system"]}')
            break
        else:
            raise ValueError(f'Unknown option: {opt}')
    # write config.json
    import json
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)

    output = config['output']
    if not os.path.exists(output):
        os.mkdir(output)
    dump_build_system_args(config)
    # config build system
    if run_config:
        args = build_system_args(config)

        if config['build_system'] == 'cmake':
            p = Popen(['cmake', '..'] + args, cwd=output)
            p.wait()
        elif config['build_system'] == 'xmake':
            p = Popen(['xmake', 'f'] + args)
            p.wait()
        else:
            raise ValueError(f'Unknown build system: {config["build_system"]}')
    if run_build:
        if config['build_system'] == 'cmake':
            p = Popen(['cmake', '--build', '.', '-j', str(build_jobs)], cwd=output)
            p.wait()
        elif config['build_system'] == 'xmake':
            os.system('xmake build')
        else:
            raise ValueError(f'Unknown build system: {config["build_system"]}')


if __name__ == '__main__':
    main(sys.argv)
