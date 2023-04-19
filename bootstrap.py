import os
import sys
from subprocess import Popen
from typing import List


def default_config():
    return {
        'cmake_args': [],
        'xmake_args': [],
        'build_system': 'cmake',
        'backends': [],
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
    print('  --config [true|false]  Run config script')
    print('  --backend [backends]   Add backends')
    print('  --install [deps]       Install dependencies')
    print('  -- [args]              Pass arguments to build system')


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
    while i < len(args):
        opt = args[i]
        if opt == '--help' or opt == '-h':
            print_help()
            return
        if opt == '--config' or opt == '-c':
            if i + 1 < len(args):
                if args[i + 1].startswith('-'):
                    run_config = True
                else:
                    v = args[i + 1]
                    if v == 'true':
                        run_config = True
                    elif v == 'false':
                        run_config = False
                    else:
                        raise ValueError(f'Unknown value for --config: {v}')
            else:
                run_config = True
            i += 2
        elif opt == '--backend' or opt == '-b':
            i += 1
            while i < len(args) and not args[i].startswith('-'):
                config['backends'].append(args[i])
                i += 1
        elif opt == '--install' or opt == '-i':
            i += 1
            while i < len(args) and not args[i].startswith('-'):
                install_dep(args[i])
                i += 1
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
    # config build system
    if run_config:
        if config['build_system'] == 'cmake':
            if not os.path.exists('build'):
                os.mkdir('build')
            p = Popen(['cmake', '..'] + config['cmake_args'], cwd='build')
            p.wait()
        elif config['build_system'] == 'xmake':
            os.system('xmake ' + ' '.join(config['xmake_args']))
        else:
            raise ValueError(f'Unknown build system: {config["build_system"]}')


if __name__ == '__main__':
    main(sys.argv)
