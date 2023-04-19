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
        'mode': 'release',
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
    config = default_config()
    # check if config.json exists
    if os.path.exists('config.json'):
        import json
        with open('config.json', 'r') as f:
            config.update(json.load(f))
    return config


def print_help():
    print('Usage: python bootstrap.py [build system] [options]')
    print('Build system:')
    print('  cmake                  Use CMake')
    print('  xmake                  Use xmake')
    print('Options:')
    print('  --config | -c          Configure build system')
    print('  --features | -f [[no-]features]  Add/remove features')
    print('      Features:')
    print('          [no-]cuda          Enable (disable) CUDA backend')
    print('          [no-]cpu           Enable (disable) CPU backend')
    print('          [no-]remote        Enable (disable) remote backend')
    print('          [no-]dx            Enable (disable) DirectX backend')
    print('          [no-]metal         Enable (disable) Metal backend')
    print('  --mode | -m [node]     Build mode')
    print('      Modes:')
    print('          debug              Debug mode')
    print('          release            Release mode')
    print('          relwithdebuginfo   Release with debug infomation mode')
    print('  --build   | -b [N]     Build (N = number of jobs)')
    print('  --clean   | -C         Clean build directory')
    print('  --install | -i [deps]  Install dependencies')
    print('  --output  | -o         Path to output directory')
    print('  -- [args]              Pass arguments to build system')


def dump_build_system_args(config: dict):
    args = build_system_args(config)
    with open(f"options.{config['mode']}.cli", 'w') as f:
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
    if config['mode'] == 'debug':
        args.append('-DCMAKE_BUILD_TYPE=Debug')
    elif config['mode'] == 'release':
        args.append('-DCMAKE_BUILD_TYPE=Release')
    elif config['mode'] == 'relwithdebuginfo':
        args.append('-DCMAKE_BUILD_TYPE=RelWithDebInfo')
    return args


def build_system_args_xmake(config: dict) -> List[str]:
    args = config['xmake_args']
    if 'cuda' in config['features']:
        args.append('-c')
    # TODO: Maxwell handle this pls
    return args


def build_system_args(config) -> List[str]:
    if config['build_system'] == 'cmake':
        return build_system_args_cmake(config)
    elif config['build_system'] == 'xmake':
        return build_system_args_xmake(config)
    else:
        raise ValueError(f'Unknown build system: {config["build_system"]}')


submods = [
    'corrosion',
    'EASTL',
    ## TODO: add more submodules here
]


def init_submodule():
    if os.path.exists('.git'):
        os.system('git submodule update --init --recursive')
    else:
        for s in submods:
            if not os.path.exists(f'src/ext/{s}'):
                print(f'Fatal error: submodule in src/ext/{s} not found.', file=sys.stderr)
                print('Please clone the repository with --recursive option.', file=sys.stderr)
                sys.exit(1)


def main(args: List[str]):
    init_submodule()
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
            i += 1
        elif opt == '--build' or opt == '-b':
            run_build = run_config = True
            i += 1
            if i < len(args) and not args[i].startswith('-'):
                build_jobs = int(args[i])
                i += 1
        elif opt == '--mode' or opt == '-m':
            i += 1
            if i < len(args) and not args[i].lower() in ('debug', 'release', 'relwithdebuginfo'):
                config['mode'] = args[i].lower()
                i += 1
        elif opt == '--features' or opt == '-f':
            i += 1
            while i < len(args) and not args[i].startswith('-'):
                f = args[i]
                if f.startswith('no-'):
                    f = f[3:]
                    if f in config['features']:
                        config['features'].remove(f)
                else:
                    if f not in config['features']:
                        config['features'].append(f)
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
